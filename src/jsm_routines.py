import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import galhalo

#import sys
#sys.path.insert(0, '../')

def data_save(datadir, npdir, mass_type, mlres, Nhalo):
    
    files = []    
    for filename in os.listdir(datadir):
        if filename.startswith('tree') and filename.endswith('evo.npz'): 
            files.append(os.path.join(datadir, filename))
    Nreal = len(files)
    
    print("number of realizations:", Nreal)
    print("number of branches/halos:", Nhalo)

    Mass = np.zeros(shape=(Nreal, Nhalo))
    Redshift = np.zeros(shape=(Nreal, Nhalo))
    
    if mass_type=="acc":

        for i,file in enumerate(files):
        
            if i%100 == 0:
                print(i)

            mass_clean, red_clean = accretion_mass(file) # grabbing the mass type
            acc_mass = np.pad(mass_clean, (0,Nhalo-len(mass_clean)), mode="constant", constant_values=0) 
            acc_red = np.pad(red_clean, (0,Nhalo-len(red_clean)), mode="constant", constant_values=np.nan)
            Mass[i,:] = acc_mass
            Redshift[i,:] = acc_red

        np.save(npdir+"acc_mass.npy", Mass)
        np.save(npdir+"acc_redshift.npy", Redshift)
        
    if mass_type=="surv":
        
        for i,file in enumerate(files):
        
            if i%100 == 0:
                print(i)

            mass_clean = surviving_mass(file, mlres)
            surv_mass = np.pad(mass_clean, (0,Nhalo-len(mass_clean)), mode="constant", constant_values=0)
            Mass[i,:] = surv_mass
            Redshift[i,:] = np.zeros(Nhalo)

        np.save(npdir+"surv_mass.npy", Mass)
        np.save(npdir+"surv_redshift.npy", Redshift)

    if mass_type=="acc_surv":
        
        for i,file in enumerate(files):
        
            if i%100 == 0:
                print(i)
  
            mass_clean, red_clean = surviving_accreation_mass(file, mlres)
            acc_surv_mass = np.pad(mass_clean, (0,Nhalo-len(mass_clean)), mode="constant", constant_values=0)
            acc_surv_red = np.pad(red_clean, (0,Nhalo-len(red_clean)), mode="constant", constant_values=np.nan)
            Mass[i,:] = acc_surv_mass
            Redshift[i,:] = acc_surv_red

        np.save(npdir+"acc_surv_mass.npy", Mass)
        np.save(npdir+"acc_surv_redshift.npy", Redshift)


def accretion_mass(file, plot_evo=False, save=False):

    tree = np.load(file)

    mass = tree["mass"]
    time = tree["CosmicTime"]
    redshift = tree["redshift"]

    mass = np.delete(mass, 1, axis=0) #there is some weird bug for this index
    n_branch = mass.shape[0]

    mask = mass != -99. # converting to NaN values
    mass = np.where(mask, mass, np.nan)  

    ana_mass = np.nanmax(mass, axis=1) #finding the maximum mass
    ana_index = np.nanargmax(mass, axis=1)
    ana_redshift = redshift[ana_index]

    if plot_evo == True:

        colors = cm.viridis(np.linspace(0, 1, n_branch))

        plt.figure(figsize=(10,10))

        for i in range(n_branch):
            plt.plot(time, mass[i], color=colors[i])
        plt.xlabel("Gyr", fontsize=30)
        plt.ylabel("halo mass (M$_{\odot}$)", fontsize=30)
        plt.yscale("log")

        if save==True:
            plt.savefig("evolution.pdf")
        plt.show()
    
    return ana_mass, ana_redshift

def surviving_mass(file, mlres, plot_evo=False, save=False):

    tree = np.load(file)

    mass = tree["mass"]
    time = tree["CosmicTime"]
    redshift = tree["redshift"]
    
    mass = np.delete(mass, 1, axis=0) #there is some weird bug for this index
    n_branch = mass.shape[0]

    mask = mass != -99. # converting to NaN values
    mass = np.where(mask, mass, np.nan)  
    
    min_mass = mass[:,0] # the final index is the redshift we evolve it to. this will be the minimum!
    ana_mass = min_mass[min_mass > mlres] #is it above the mass resolution?

    #print("Of the", len(min_mass), "subhalos, only", len(ana_mass), "survived.")

    if plot_evo == True:

        colors = cm.viridis(np.linspace(0, 1, n_branch))

        plt.figure(figsize=(10,10))

        for i in range(n_branch):
            plt.plot(time, mass[i], color=colors[i])
        plt.xlabel("Gyr", fontsize=30)
        plt.ylabel("halo mass (M$_{\odot}$)", fontsize=30)
        plt.yscale("log")

        if save==True:
            plt.savefig("evolution.pdf")
        plt.show()
    
    return ana_mass


def surviving_accreation_mass(file, mlres, plot_evo=False, save=False):

    tree = np.load(file)

    mass = tree["mass"]
    time = tree["CosmicTime"]
    redshift = tree["redshift"]
    
    mass = np.delete(mass, 1, axis=0) #there is some weird bug for this index
    n_branch = mass.shape[0]

    mask = mass != -99. # converting to NaN values
    mass = np.where(mask, mass, np.nan)  

    mass = tree["mass"]
    time = tree["CosmicTime"]
    redshift = tree["redshift"]

    mass = np.delete(mass, 1, axis=0) #their is some weird bug for this index
    n_branch = mass.shape[0]

    mask = mass != -99. # converting to NaN values
    mass = np.where(mask, mass, np.nan)  

    ana_mass = []
    ana_redshift = []
    for branch in mass:
        if branch[0] > mlres:
            ana_mass.append(np.nanmax(branch)) #finding the maximum mass
            ana_index = np.nanargmax(branch)
            ana_redshift.append(redshift[ana_index]) # finding the corresponding redshift
    
    if plot_evo == True:

        colors = cm.viridis(np.linspace(0, 1, n_branch))

        plt.figure(figsize=(10,10))

        for i in range(n_branch):
            plt.plot(time, mass[i], color=colors[i])
        plt.xlabel("Gyr", fontsize=30)
        plt.ylabel("halo mass (M$_{\odot}$)", fontsize=30)
        plt.yscale("log")

        if save==True:
            plt.savefig("evolution.pdf")
        plt.show()
    
    return np.array(ana_mass), np.array(ana_redshift)

def SHMR(Mh, scatter=None, extra=False, alt=None, gamma=None, red=None, Npix=50): ## neeed to update this!!!!!

    Mh[:, 0] = 0.0  # removing the host mass from the matrix
    zero_mask = Mh != 0.0 
    Mh = np.log10(np.where(zero_mask, Mh, np.nan)) #switching the to nans!

    if scatter != None:
        return galhalo.lgMs_D22_dex(Mh, scatter)

    if extra == True: #simple extrapolations of well constrained SHMRs
        red[:, 0] = np.nan 
        Ms_B13 = galhalo.lgMs_B13(Mh, z=red)
        Ms_RP17 = galhalo.lgMs_RP17(Mh, z=red)
        return Ms_B13, Ms_RP17

    if alt =="s": #the slope changes, no scatter
        red[:, 0] = np.nan 
        return galhalo.lgMs_D22_red(Mh, red, gamma_s=gamma)

    if alt =="i": #the intercept changes, no scatter
        red[:, 0] = np.nan 
        return galhalo.lgMs_D22_red(Mh, red, gamma_i=gamma)

    else: #deterministic
        return galhalo.lgMs_D22_det(Mh)


def CSMF(Ms, Npix=50, down=5, up=95, plot=True, full=False):

    """
    calculates the cumulative satellite mass function given a number of mass ind
    input mass should not be in log space
    """
    # the same x-array for all the CSMFs
    mass_range = np.logspace(3,10,Npix)
    #cleaning up the excessive padding
    #max_real = Ms.shape[1] - np.sum(np.isnan(Ms),axis=1)

    #now to start counting!
    I = np.zeros((Npix, Ms.shape[0]))

    for i,val in enumerate(mass_range):
        I[i] = np.sum(Ms > np.log10(val),axis=1)
            
    CSMF_quant = np.percentile(I.transpose(), [down, 50, up], axis=0) # the percentiles
        
    if plot==True:
        
        plt.figure(figsize=(8, 8))

        plt.plot(mass_range, CSMF_quant[1, :], label="median", color="black")
        plt.fill_between(mass_range, y1=CSMF_quant[0, :], y2=CSMF_quant[2, :], alpha=0.2, color="grey")
        plt.grid(alpha=0.4)
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("log m$_{stellar}$ (M$_\odot$)", fontsize=15)
        plt.ylabel("log N (> m$_{stellar}$)", fontsize=15)
        plt.legend()
        plt.show()

    if full == True:
        return I.transpose()
    else:
        return CSMF_quant

def CSMF_1D(Ms, Npix=50, plot=True):

    """
    calculates the cumulative satellite mass function given a number of mass ind
    input mass should not be in log space
    """
    # the same x-array for all the CSMFs
    mass_range = np.logspace(3,10,Npix)
    I = [np.sum(Ms > np.log10(i)) for i in mass_range]
                    
    if plot==True:
        
        plt.figure(figsize=(8, 8))

        plt.plot(mass_range, I, color="black")
        plt.grid(alpha=0.4)
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("log m$_{stellar}$ (M$_\odot$)", fontsize=15)
        plt.ylabel("log N (> m$_{stellar}$)", fontsize=15)
        plt.show()

    return I

def scatter_stat(CSMF, det_CSMF):

    if len(CSMF.shape) == 4:
        print("averaging over the samples!")
        CSMF = np.average(CSMF,axis=0)
    
    Nreal = CSMF.shape[0]
    Mpix = CSMF.shape[2]

    stat = (CSMF[:,2]-CSMF[:,0])

    print("returning a", Nreal, "by", Mpix, "matrix")
    return stat
           

def SHMF(mass, mass_min=-4, Nbins=50, plot=True):
 
    mass_frac = mass/np.max(mass) #normalizing by host mass
    mass_frac[:, 0] = 0.0  # removing the host mass from the matrix
    zero_mask = mass_frac != 0.0 
    ana_mass = np.log10(np.where(zero_mask, mass_frac, np.nan))  # up until here the stats are good
    
    def histofunc(mass, bins=False): # nested function
        if bins==True:
            return np.histogram(mass, range=(mass_min, 0), bins=Nbins)
        else:
            return np.histogram(mass, range=(mass_min, 0), bins=Nbins)[0]

    # now to start counting!
    m_counts, bins = histofunc(ana_mass[0], bins=True)  # to be keep in memory, only needs to be measured once
    binsize = (bins[1] - bins[0])
    bincenters = 0.5 * (bins[1:] + bins[:-1])

    I = np.apply_along_axis(histofunc, 1, ana_mass)  # this applies the histogram to the whole matrix

    SHMF_ave = np.average(I, axis=0)

    SHMF_std = np.std(I, axis=0)
        
    if plot == True:
        plt.figure(figsize=(8, 8))

        plt.plot(bincenters, np.log10(SHMF_ave/binsize), label="average", marker="o", color="black")
        plt.plot(bincenters, np.log10((SHMF_ave+SHMF_std)/binsize), label="1 $\sigma$", marker=".", ls=":", color="grey")
        plt.plot(bincenters, np.log10((SHMF_ave-SHMF_std)/binsize), marker=".", ls=":", color="grey")

        plt.xlabel("log (m/M)", fontsize=20)
        plt.ylabel("log[ dN / dlog(m/M) ]", fontsize=20)
        plt.legend()
        plt.show()

    return bincenters, np.array([SHMF_ave, SHMF_std])