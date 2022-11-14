import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import galhalo as g

#import sys
#sys.path.insert(0, '../')

def collapse_realizations(datadir, npdir, keys):

#---get the list of data files
    ana_mass = []
    redshifts = []
    stellar_mass = []

    for filename in os.listdir(datadir):
        if filename.endswith('.npz'): 
            file = os.path.join(datadir, filename)
            mass, z = grab_tree_mass(file, mlres=8, masstype="acc", ana_z=0, plot_evo=False)


    
        ana_mass.append(mass)
        stellar_mass.append(s_mass)
        redshifts.append(z)

    Nreal = len(ana_mass)
    Nhalo = max([len(i) for i in ana_mass])

def accretion_mass(file, plot_evo=False, save=False):

    tree = np.load(file)

    mass = tree["mass"]
    time = tree["CosmicTime"]
    redshift = tree["redshift"]

    mass = np.delete(mass, 1, axis=0) #there is some weird bug for this index

    mask = mass != -99. # converting to NaN values
    mass = np.where(mask, mass, np.nan)  

    ana_mass = np.nanmax(mass, axis=1) #finding the maximum mass
    ana_index = np.nanargmax(mass, axis=1)
    ana_redshift = redshift[ana_index]

    accretion = np.array(ana_mass, ana_redshift)

    if plot_evo == True:

        colors = cm.viridis(np.linspace(0, 1, n_branch))

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(30,20))
        for i in range(n_branch):
            ax1.plot(redshift, mass[i], color=colors[i])
        ax1.set_yscale('log')
        ax1.set_ylabel("halo mass (M$_{\odot}$)", fontsize=30)
        ax1.set_xlabel("z", fontsize=30)

        for i in range(n_branch):
            ax2.plot(time, mass[i], color=colors[i])
        ax2.set_xlabel("Gyr", fontsize=30)

        if save==True:
            plt.savefig("evolution.pdf")
        plt.show()
    
    return accretion

def surviving_mass(file, mlres, plot_evo=False, save=False):

    tree = np.load(file)

    mass = tree["mass"]
    time = tree["CosmicTime"]
    redshift = tree["redshift"]

    mass = np.delete(mass, 1, axis=0) #their is some weird bug for this index

    mask = mass != -99. # converting to NaN values
    mass = np.where(mask, mass, np.nan)  

    min_mass = np.nanmin(mass, axis=1) #finding the minimum mass
    min_index = np.nanargmin(mass, axis=1)
    min_redshift = redshift[min_index]

    ana_mass = min_mass[min_mass >= mlres] #is it above the mass resolution?
    ana_redshift = min_redshift[min_mass >= mlres]

    print("Of the", len(min_mass), "subhalos, only", len(ana_mass), "survived.")

    surviving = np.array(ana_mass, ana_redshift)

    if plot_evo == True:

        colors = cm.viridis(np.linspace(0, 1, n_branch))

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(30,20))
        for i in range(n_branch):
            ax1.plot(redshift, mass[i], color=colors[i])
        ax1.set_yscale('log')
        ax1.set_ylabel("halo mass (M$_{\odot}$)", fontsize=30)
        ax1.set_xlabel("z", fontsize=30)

        for i in range(n_branch):
            ax2.plot(time, mass[i], color=colors[i])
        ax2.set_xlabel("Gyr", fontsize=30)

        if save==True:
            plt.savefig("evolution.pdf")
        plt.show()
    
    return surviving
    

def closest_value(input_list, input_value):
    arr = np.asarray(input_list)

    i = (np.abs(arr - input_value)).argmin()

    return arr[i]
           
def histofunc1(mass, mass_max=0, mass_min=-3, Nbins=30, bins=False):
    
    if bins==True:
        return np.histogram(mass, range=(mass_min, mass_max), bins=Nbins)
    else:
        return np.histogram(mass, range=(mass_min, mass_max), bins=Nbins)[0]
    
def histofunc2(mass, mass_max=-2, mass_min=-9, Nbins=30, bins=False):
    
    if bins==True:
        return np.histogram(mass, range=(mass_min, mass_max), bins=Nbins)
    else:
        return np.histogram(mass, range=(mass_min, mass_max), bins=Nbins)[0]

def SHMF(mass, red, plotmed=False, plotave=True):

        
    mass_frac = mass/np.max(mass)
    mass_frac[:, 0] = 0.0  # removing the host mass from the matrix
    zero_mask = mass_frac != 0.0 
    ana_mass = np.log10(np.where(zero_mask, mass_frac, np.nan))  # up until here the stats are good

    # now to start counting!
    m_counts, bins = histofunc1(ana_mass[0], bins=True)  # to be kep in memory, only needs to be measured once
    binsize = (bins[1] - bins[0])
    bincenters = 0.5 * (bins[1:] + bins[:-1])

    I = np.apply_along_axis(histofunc1, 1, ana_mass)  # this applies the histogram to the whole matrix

    SHMF_quant = np.log10(np.percentile(I, [15, 50, 85], axis=0) / binsize)# this calculates the percentiles for each index

    SHMF_ave = np.average(I, axis=0)

    SHMF_std = np.std(I, axis=0)

    if plotmed == True:

        plt.figure(figsize=(12, 10))

        plt.plot(bincenters, SHMF[0, :], label="15%", marker="o", ls=":", color="grey")
        plt.plot(bincenters, SHMF[1, :], label="50%", marker="o", color="red")
        plt.plot(bincenters, SHMF[2, :], label="85%", marker="o", ls="--", color="grey")
        plt.xlabel("log (m/M)", fontsize=20)
        plt.ylabel("log[ dN / dlog(m/M) ]", fontsize=20)
        plt.legend()
        plt.show()
        
    if plotave == True:
        plt.figure(figsize=(12, 10))

        plt.plot(bincenters, np.log10(SHMF_ave/binsize), label="average", marker="o", color="black")
        plt.plot(bincenters, np.log10((SHMF_ave+SHMF_std)/binsize), label="1 $\sigma$", marker=".", ls=":", color="grey")
        plt.plot(bincenters, np.log10((SHMF_ave-SHMF_std)/binsize), marker=".", ls=":", color="grey")

        #plt.plot(bincenters, np.log10(compare(10**bincenters))+1.55, label="scaled de Lucia 2004 relation")

        plt.xlabel("log (m/M)", fontsize=20)
        plt.ylabel("log[ dN / dlog(m/M) ]", fontsize=20)
        plt.legend()
        #plt.savefig("SHMF.pdf")
        plt.show()

    return I, bincenters, SHMF_ave, SHMF_std
