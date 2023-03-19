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

def prep_data(numpyfile, convert=True, includenan=True):
    Mh = np.load(numpyfile)
    #Mh[:, 0] = 0.0  # masking the host mass in the matrix
    zero_mask = Mh != 0.0 
    Mh = np.log10(np.where(zero_mask, Mh, np.nan)) #switching the to nans!

    if includenan == False:
        max_sub = min(Mh.shape[1] - np.sum(np.isnan(Mh),axis=1))
    else: 
        max_sub = max(Mh.shape[1] - np.sum(np.isnan(Mh),axis=1))

    Mh = Mh[:,1:max_sub]  #excluding the host mass
    if convert==False:
        return Mh
    else:
        return galhalo.lgMs_D22_det(Mh)