import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import galhalo

def accretion_mass(file):

    tree = np.load(file)
    mass = tree["mass"]
    redshift = tree["redshift"]
    mass = np.delete(mass, 1, axis=0) #there is some weird bug for this index
    mask = mass != -99. # converting to NaN values
    mass = np.where(mask, mass, np.nan)  
    ana_mass = np.nanmax(mass, axis=1) #finding the maximum mass
    ana_index = np.nanargmax(mass, axis=1)
    ana_redshift = redshift[ana_index]

    return ana_mass, ana_redshift

def surviving_mass(file, mlres):

    tree = np.load(file)
    mass = tree["mass"]
    mass = np.delete(mass, 1, axis=0) #there is some weird bug for this index
    mask = mass != -99. # converting to NaN values
    mass = np.where(mask, mass, np.nan)  
    min_mass = mass[:,0] # the final index is the redshift we evolve it to. this will be the minimum!
    ana_mass = min_mass[min_mass > mlres] #is it above the mass resolution?
    ana_redshift = np.zeros(ana_mass.shape)
    
    return ana_mass, ana_redshift

def surviving_accreation_mass(file, mlres):

    tree = np.load(file)
    mass = tree["mass"]
    redshift = tree["redshift"]
    mass = np.delete(mass, 1, axis=0) #there is some weird bug for this index
    mask = mass != -99. # converting to NaN values
    mass = np.where(mask, mass, np.nan)  
    ana_mass = []
    ana_redshift = []
    for branch in mass:
        if branch[0] > mlres:
            ana_mass.append(np.nanmax(branch)) #finding the maximum mass
            ana_index = np.nanargmax(branch)
            ana_redshift.append(redshift[ana_index]) # finding the corresponding redshift
    
    return np.array(ana_mass), np.array(ana_redshift)


class Realizations:
        
    def __init__(self, datadir, mlres):
        self.datadir = datadir
        self.mlres = mlres
            
    def grab_mass(self, type, Nhalo=1200): 
        # should fix the hardcoding on the shape later!
        
        files = []    
        for filename in os.listdir(self.datadir):
            if filename.startswith('tree') and filename.endswith('evo.npz'): 
                files.append(os.path.join(self.datadir, filename))

        self.Nreal = len(files)
        self.Nhalo = Nhalo

        print("number of realizations:", self.Nreal)
        print("number of branches/halos:", self.Nhalo)

        Mass = np.zeros(shape=(self.Nreal, self.Nhalo))
        Redshift = np.zeros(shape=(self.Nreal, self.Nhalo))
        
        if type=="acc":
            for i,file in enumerate(files):

                mass_clean, red_clean = accretion_mass(file)
                acc_mass = np.pad(mass_clean, (0,self.Nhalo-len(mass_clean)), mode="constant", constant_values=np.nan) 
                acc_red = np.pad(red_clean, (0,self.Nhalo-len(red_clean)), mode="constant", constant_values=np.nan)
                Mass[i,:] = acc_mass
                Redshift[i,:] = acc_red

            print("saving to numpy files to the same directory")
            np.save(self.datadir+"acc_mass.npy", Mass)
            np.save(self.datadir+"acc_redshift.npy", Redshift)
            self.acc_mass = Mass
            self.acc_redshift = Redshift
            
        if type=="surv":
            for i,file in enumerate(files):

                mass_clean, red_clean = surviving_mass(file, self.mlres)
                surv_mass = np.pad(mass_clean, (0,self.Nhalo-len(mass_clean)), mode="constant", constant_values=np.nan)
                surv_red = np.pad(red_clean, (0,self.Nhalo-len(red_clean)), mode="constant", constant_values=np.nan)
                Mass[i,:] = surv_mass
                Redshift[i,:] = surv_red

            print("saving to numpy files to the same directory")
            np.save(self.datadir+"surv_mass.npy", Mass)
            np.save(self.datadir+"surv_redshift.npy", Redshift)
            self.surv_mass = Mass
            self.surv_redshift = Redshift

        if type=="acc_surv": 
            for i,file in enumerate(files):
    
                mass_clean, red_clean = surviving_accreation_mass(file, self.mlres)
                acc_surv_mass = np.pad(mass_clean, (0,self.Nhalo-len(mass_clean)), mode="constant", constant_values=np.nan)
                acc_surv_red = np.pad(red_clean, (0,self.Nhalo-len(red_clean)), mode="constant", constant_values=np.nan)
                Mass[i,:] = acc_surv_mass
                Redshift[i,:] = acc_surv_red

            print("saving to numpy files to the same directory")
            np.save(self.datadir+"acc_surv_mass.npy", Mass)
            np.save(self.datadir+"acc_surv_redshift.npy", Redshift)
            self.acc_surv_mass = Mass
            self.acc_surv_redshift = Redshift

    def plot_evo(self, save=False):

        colors = cm.viridis(np.linspace(0, 1, self.Nreal))

        plt.figure(figsize=(10,10))

        for i in range(self.Nreal):
            plt.plot(self.acc_redshift, self.acc_mass[i], color=colors[i])
        plt.xlabel("Gyr", fontsize=30)
        plt.ylabel("halo mass (M$_{\odot}$)", fontsize=30)
        plt.yscale("log")

        if save==True:
            plt.savefig("evolution.pdf")
        plt.show()