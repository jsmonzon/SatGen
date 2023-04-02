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

    test = np.sum(np.isnan(mass),axis=1) # in case of the bug...
    ind = np.where(test==mass.shape[1])[0]
    mass = np.delete(mass, ind, axis=0)

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

    """
    A cleaner way of handling the multi-dimensional output data
    """
        
    def __init__(self, datadir, mlres):
        self.datadir = datadir
        self.mlres = mlres
            
    def grab_mass(self, type, Nreal_subset=False, Nhalo=1200): 
        # should fix the hardcoding on the shape later!
        
        files = []    
        for filename in os.listdir(self.datadir):
            if filename.startswith('tree') and filename.endswith('evo.npz'): 
                files.append(os.path.join(self.datadir, filename))
        
        self.Nreal = len(files)


        #if Nreal_subset==False:
        #    self.Nreal = len(files)
        # else: # haven't quite worked this out!
        #     self.Nreal = Nreal_subset
        #     rand_samp = np.random.randint(0, len(files), size=Nreal_subset)
        #     files=files[rand_samp]

        self.Nhalo = Nhalo

        print("number of realizations:", self.Nreal)
        print("number of branches/subhalos:", self.Nhalo)

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


    def plot_single_realization(self, file, nhalo=20, rand=True, i=10):

        tree = np.load(file)

        mass = tree["mass"]
        time = tree["CosmicTime"]

        if rand==True:
            select = np.random.randint(1,mass.shape[0],nhalo)
        elif rand==False:
            select = np.linspace(i,i+nhalo, nhalo).astype("int")

        colors = cm.viridis(np.linspace(0, 1, nhalo))

        plt.figure(figsize=(6,6))

        for i in range(nhalo):
            plt.plot(time, mass[select[i]], color=colors[i])

        plt.plot(time, mass[0], color="red")
        plt.xlabel("Gyr", fontsize=15)
        plt.ylabel("halo mass (M$_{\odot}$)", fontsize=15)
        plt.yscale("log")
        plt.axhline(10**8, ls="--", color="black")
        #plt.ylim(1e6,1e14)
        plt.show()


def cumulative(Ms, mass_bins):
    N = np.histogram(Ms, bins=mass_bins)[0]
    Nsub = np.sum(N)
    stat = Nsub-np.cumsum(N) 
    return np.insert(stat, 0, Nsub) #to add the missing index

def differential(phi, phi_bins, phi_binsize): 
    N = np.histogram(phi, bins=phi_bins)[0]
    return N/phi_binsize


class MassMat:

    """
    A cleaner way of interacting with the condensed mass matricies. One instance should be made for each of the mass_types in the Realizations class
    """
        
    def __init__(self, massfile, Nbins=45, phimin=-4, lgMsmin=3, lgMsmax=10):

        self.massfile = massfile
        self.Nbins = Nbins
        self.lgMsmin = lgMsmin
        self.lgMsmax = lgMsmax
        self.phimin = phimin
        self.phi_bins = np.linspace(self.phimin, 0, Nbins)
        self.phi_binsize = self.phi_bins[1] - self.phi_bins[0]
        self.mass_bins = np.linspace(lgMsmin, lgMsmax, Nbins)
        self.binsize = self.mass_bins[1] - self.mass_bins[0]

    def prep_data(self, redfile=None, includenan=True, a=1.82, log_e=-1.5):

        Mh = np.load(self.massfile)

        Mhosts = np.nanmax(Mh, axis=1)
        lgMh = np.log10(Mh)

        self.shape = Mh.shape

        if includenan == False:
            max_sub = min(lgMh.shape[1] - np.sum(np.isnan(lgMh),axis=1))
        else: 
            max_sub = max(lgMh.shape[1] - np.sum(np.isnan(lgMh),axis=1))

        lgMh = lgMh[:,1:max_sub]  #excluding the host mass
        self.lgMh = lgMh

        phi = np.log10((Mh.T / Mhosts).T)  #excluding the host mass
        self.phi = phi[:,1:max_sub]

        self.Mh = Mh[:,0:max_sub]  #including the host mass

        if redfile!=None:
            reds = np.load(redfile)
            self.z = reds[:,1:max_sub]

        self.lgMs = galhalo.lgMs_D22_det(lgMh, a, log_e) #and the deterministic stellar mass!

    def CSMF(self):

        counts = np.apply_along_axis(cumulative, 1, self.lgMs, mass_bins=self.mass_bins) 
        quant = np.percentile(counts, np.array([5, 50, 95]), axis=0)

        self.CSMF_counts = counts # a CSMF for each of the realizations
        self.quant = quant # the stats across realizations

    def plot_CSMF(self):

        plt.figure(figsize=(8, 8))
        plt.plot(self.mass_bins, self.quant[1], label="median", color="black")
        plt.fill_between(self.mass_bins, y1=self.quant[0], y2=self.quant[2], alpha=0.2, color="grey", label="5% - 95%")
        plt.yscale("log")
        plt.grid(alpha=0.4)
        plt.ylim(0.5,10**4.5)
        plt.xlabel("log m$_{stellar}$ (M$_\odot$)", fontsize=15)
        plt.ylabel("log N (> m$_{stellar}$)", fontsize=15)
        plt.legend()
        plt.show()

    def H2H_CSMF(self, norm=False):

        if norm==True:
            stat = (self.quant[:,2]-self.quant[:,0])/self.quant[:,1]
        else:
            stat = (self.quant[:,2]-self.quant[:,0])
        return stat

    def SHMF(self):
        counts = np.apply_along_axis(differential, 1, self.phi, phi_bins=self.phi_bins, phi_binsize=self.phi_binsize) 

        SHMF_ave = np.average(counts, axis=0)
        SHMF_std = np.std(counts, axis=0)

        self.SHMF_counts = counts
        self.SHMF_werr = np.array([SHMF_ave, SHMF_std])

    def plot_SHMF(self):

        self.phi_bincenters = 0.5 * (self.phi_bins[1:] + self.phi_bins[:-1])
    
        plt.figure(figsize=(8, 8))

        plt.plot(self.phi_bincenters, self.SHMF_werr[0], label="average", color="black")
        plt.fill_between(self.phi_bincenters, y1=self.SHMF_werr[0]-self.SHMF_werr[1], y2=self.SHMF_werr[0]+self.SHMF_werr[1], alpha=0.2, color="grey", label="1$\sigma$")
        plt.yscale("log")
        plt.grid(alpha=0.4)
        plt.xlabel("log (m/M)", fontsize=15)
        plt.ylabel("log[ dN / dlog(m/M) ]", fontsize=15)
        plt.legend()
        plt.show()
    

    # def mass_rank(mass):

    #     rank = np.flip(np.argsort(mass,axis=1), axis=1) # rank the subhalos from largest to smallest
    #     ranked_mass = np.take_along_axis(mass, rank, axis=1) # this is it!!!

    #     return rank, ranked_mass
                
