import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


def find_nearest(values:np.ndarray):

    """_summary_
    an auxilary function to find the closest mass bin to the index where you want to measure some statistics

    Returns:
        np.ndarray: 1D mass indices with the same shape as the input values
    """

    array = np.linspace(4,11,45)
    indices = np.abs(np.subtract.outer(array, values)).argmin(0)
    return indices

def anamass(file, mlres, Print=False):
    tree = np.load(file)
    mass = tree["mass"]
    redshift = tree["redshift"]
    mass = np.delete(mass, 1, axis=0) #there is some weird bug for this index
    mask = mass != -99. # converting to NaN values
    mass = np.where(mask, mass, np.nan)  
    Nhalo = mass.shape[0]

    peak_mass = np.nanmax(mass, axis=1) #finding the maximum mass
    peak_index = np.nanargmax(mass, axis=1)
    peak_red = redshift[peak_index]

    surv_mass = mass[:,0] # the final index is the z=0 time step. this will be the minimum mass for all subhalos
    surv_mask = surv_mass > mlres
    if Print==True:
        print("there are", Nhalo, "subhalos in this tree")
        print("only", sum(surv_mask), "survived")

    return peak_mass, peak_red, surv_mask

    # peak_surv_mass = np.ma.filled(np.ma.masked_array(peak_mass, mask=~surv_mask),fill_value=np.nan) # now selecting only those that are above mlres
    # peak_surv_red = np.ma.filled(np.ma.masked_array(peak_red, mask=~surv_mask),fill_value=np.nan)

    # host_mass = mass[0]
    # f_subhalo = host_mass/np.nansum(mass, axis=0)
    # host_mass_t = mass[0]
    # cumulative_subhalo_mass_t = np.nansum(mass[1:], axis=0)
    #return (np.ma.masked_array(peak_mass, mask=~surv_mask)), np.ma.masked_array(peak_red, mask=~surv_mask)

    #peak_surv_mass, peak_surv_red, surv_mass, surv_mask



class Realizations:

    """
    Condensing each set of realizations into mass matrices that are easy to handle. 
    This class is applied to a directory that holds all the "raw" satgen files. 
    Each directory is a seperate set of realizations.
    """
        
    def __init__(self, datadir, mlres):
        self.datadir = datadir
        self.mlres = mlres

    def grab_mass(self, Nhalo=1600):

        files = []    
        for filename in os.listdir(self.datadir):
            if filename.startswith('tree') and filename.endswith('evo.npz'): 
                files.append(os.path.join(self.datadir, filename))

        self.files = files
        self.Nreal = len(files)
        self.Nhalo = Nhalo

        print("number of realizations:", self.Nreal)
        print("number of branches/subhalos:", self.Nhalo)

        Mass = np.zeros(shape=(self.Nreal, self.Nhalo))
        Redshift = np.zeros(shape=(self.Nreal, self.Nhalo))
        Surv = np.empty(shape=(self.Nreal, self.Nhalo), dtype=bool)

        for i,file in enumerate(files):
            peak_mass, peak_red, surv_mask = anamass(file, self.mlres)
            peak_mass = np.pad(peak_mass, (0,self.Nhalo-len(peak_mass)), mode="constant", constant_values=np.nan) 
            peak_red = np.pad(peak_red, (0,self.Nhalo-len(peak_red)), mode="constant", constant_values=np.nan)
            surv_mask = np.pad(surv_mask, (0,self.Nhalo-len(surv_mask)), mode="constant", constant_values=False)

            Mass[i,:] = peak_mass
            Redshift[i,:] = peak_red
            Surv[i,:] = surv_mask

        self.acc_mass = Mass
        self.acc_redshift = Redshift
        self.surv_mask = Surv

        np.save(self.datadir+"acc_mass.npy", Mass)
        np.save(self.datadir+"acc_redshift.npy", Redshift)
        np.save(self.datadir+"surv_mask.npy", Surv)


    def plot_single_realization(self, nhalo=20, rand=False, i=10):

        random_index = np.random.randint(0,len(self.files)-1)
        tree = np.load(self.files[random_index])

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


# this is the halo mass function from the CLF
# halo = np.load("../etc/halo_mass_PDF_full.npy")
# plt.plot(halo[:,0], halo[:,1])

# def accretion_mass(file):

#     tree = np.load(file)
#     mass = tree["mass"]
#     redshift = tree["redshift"]
#     mass = np.delete(mass, 1, axis=0) #there is some weird bug for this index
#     mask = mass != -99. # converting to NaN values
#     mass = np.where(mask, mass, np.nan)  

#     test = np.sum(np.isnan(mass),axis=1) # in case of the bug...
#     ind = np.where(test==mass.shape[1])[0]
#     mass = np.delete(mass, ind, axis=0)

#     ana_mass = np.nanmax(mass, axis=1) #finding the maximum mass
#     ana_index = np.nanargmax(mass, axis=1)
#     ana_redshift = redshift[ana_index]

#     return ana_mass, ana_redshift

# def surviving_mass(file, mlres):

#     tree = np.load(file)
#     mass = tree["mass"]
#     mass = np.delete(mass, 1, axis=0) #there is some weird bug for this index
#     mask = mass != -99. # converting to NaN values
#     mass = np.where(mask, mass, np.nan)  
#     min_mass = mass[:,0] # the final index is the redshift we evolve it to. this will be the minimum!
#     ana_mass = min_mass[min_mass > mlres] #is it above the mass resolution?
#     ana_redshift = np.zeros(ana_mass.shape)
    
#     return ana_mass, ana_redshift

# def surviving_accreation_mass(file, mlres):

#     tree = np.load(file)
#     mass = tree["mass"]
#     redshift = tree["redshift"]
#     mass = np.delete(mass, 1, axis=0) #there is some weird bug for this index
#     mask = mass != -99. # converting to NaN values
#     mass = np.where(mask, mass, np.nan)  
#     ana_mass = []
#     ana_redshift = []
#     for branch in mass:
#         if branch[0] > mlres:
#             ana_mass.append(np.nanmax(branch)) #finding the maximum mass
#             ana_index = np.nanargmax(branch)
#             ana_redshift.append(redshift[ana_index]) # finding the corresponding redshift
    
#     return np.array(ana_mass), np.array(ana_redshift)

# def assembly_time(file):
#     tree = np.load(file)
#     mass = tree["mass"]
#     redshift = tree["redshift"]

#     def grab_mass_old(self, type, Nhalo=1600): 
#         # should fix the hardcoding on the shape later!
        
#         files = []    
#         for filename in os.listdir(self.datadir):
#             if filename.startswith('tree') and filename.endswith('evo.npz'): 
#                 files.append(os.path.join(self.datadir, filename))

#         self.files = files
#         self.Nreal = len(files)
#         self.Nhalo = Nhalo

#         print("number of realizations:", self.Nreal)
#         print("number of branches/subhalos:", self.Nhalo)

#         Mass = np.zeros(shape=(self.Nreal, self.Nhalo))
#         Redshift = np.zeros(shape=(self.Nreal, self.Nhalo))
        
#         if type=="acc":
#             for i,file in enumerate(files):
#                 mass_clean, red_clean = accretion_mass(file)
#                 acc_mass = np.pad(mass_clean, (0,self.Nhalo-len(mass_clean)), mode="constant", constant_values=np.nan) 
#                 acc_red = np.pad(red_clean, (0,self.Nhalo-len(red_clean)), mode="constant", constant_values=np.nan)
#                 Mass[i,:] = acc_mass
#                 Redshift[i,:] = acc_red


#             print("saving to numpy files to the same directory")
#             np.save(self.datadir+"acc_mass.npy", Mass)
#             np.save(self.datadir+"acc_redshift.npy", Redshift)
#             self.acc_mass = Mass
#             self.acc_redshift = Redshift
            
#         if type=="surv":
#             for i,file in enumerate(files):

#                 mass_clean, red_clean = surviving_mass(file, self.mlres)
#                 surv_mass = np.pad(mass_clean, (0,self.Nhalo-len(mass_clean)), mode="constant", constant_values=np.nan)
#                 surv_red = np.pad(red_clean, (0,self.Nhalo-len(red_clean)), mode="constant", constant_values=np.nan)
#                 Mass[i,:] = surv_mass
#                 Redshift[i,:] = surv_red

#             print("saving to numpy files to the same directory")
#             np.save(self.datadir+"surv_mass.npy", Mass)
#             np.save(self.datadir+"surv_redshift.npy", Redshift)
#             self.surv_mass = Mass
#             self.surv_redshift = Redshift

#         if type=="acc_surv": 
#             for i,file in enumerate(files):
    
#                 mass_clean, red_clean = surviving_accreation_mass(file, self.mlres)
#                 acc_surv_mass = np.pad(mass_clean, (0,self.Nhalo-len(mass_clean)), mode="constant", constant_values=np.nan)
#                 acc_surv_red = np.pad(red_clean, (0,self.Nhalo-len(red_clean)), mode="constant", constant_values=np.nan)
#                 Mass[i,:] = acc_surv_mass
#                 Redshift[i,:] = acc_surv_red

#             print("saving to numpy files to the same directory")
#             np.save(self.datadir+"acc_surv_mass.npy", Mass)
#             np.save(self.datadir+"acc_surv_redshift.npy", Redshift)
#             self.acc_surv_mass = Mass
#             self.acc_surv_redshift = Redshift
