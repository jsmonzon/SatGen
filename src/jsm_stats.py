import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from numpy.random import poisson


def cumulative(lgMs_1D:np.ndarray, mass_bins):
    N = np.histogram(lgMs_1D, bins=mass_bins)[0]
    Nsub = np.sum(N)
    stat = Nsub-np.cumsum(N) 
    return np.insert(stat, 0, Nsub) #to add the missing index

def ecdf(data):
    return np.arange(1, data.shape[0]+1)/float(data.shape[0])

# def pdf(data, max_ind=25):
#     binz = np.arange(-0.5, max_ind+0.5) 
#     return np.histogram(data, bins=binz, density=True)[0]

def pdf(data):
    index, counts = np.unique(data, return_counts=True)
    full = np.zeros(300) # the max number of unique counts across the models
    full[index.astype("int")] = counts/data.shape[0]
    return full

def satfreq(lgMs, Ms_min):
    return np.sum(lgMs > Ms_min, axis = 1)

def maxsatmass(lgMs):
    return np.sort(np.nanmax(lgMs, axis=1)) # since it will be passed to ecdf

class SatStats:

    def __init__(self, lgMs):
        self.lgMs = lgMs

    def Nsat(self, Ms_min, plot=False):
        self.Ms_min = Ms_min
        self.satfreq = satfreq(self.lgMs, self.Ms_min)
        self.Pnsat = pdf(self.satfreq)
        
        if plot==True:
            plt.plot(np.arange(self.Pnsat.shape[0]), self.Pnsat, marker="o")
            plt.xlabel("number of satellites > $10^{"+str(self.Ms_min)+"} \mathrm{M_{\odot}}$", fontsize=15)
            plt.ylabel("PDF", fontsize=15)
            plt.xlim(0,20)
            plt.show()

    def Maxmass(self, plot=False):
        self.Msmax = maxsatmass(self.lgMs)
        self.ecdf_MsMax = ecdf(self.Msmax)

        if plot==True:
            plt.plot(self.Msmax, self.ecdf_MsMax)
            plt.xlabel("stellar mass of most massive satellite ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
            plt.ylabel("CDF", fontsize=15)
            plt.show()        

    # def CSMF(self, mass_bins:np.ndarray=np.linspace(4,11,45)):

    #     self.mass_bins = mass_bins

    #     counts = np.apply_along_axis(cumulative, 1, self.lgMs, mass_bins=self.mass_bins) 
    #     self.CSMF_counts = counts # a CSMF for each of the realizations

    #     quant = np.percentile(counts, np.array([5, 50, 95]), axis=0, method="closest_observation")
    #     self.quant = quant # the stats across realizations

    #     Nsets = int(counts.shape[0]/self.Nsamp) #dividing by the number of samples
    #     set_ind = np.arange(0,Nsets)*self.Nsamp
    #     print("dividing your sample into", Nsets-1, "sets")

    #     quant_split = np.zeros(shape=(Nsets-1, 3, self.mass_bins.shape[0]))
    #     for i in range(Nsets-1):
    #         quant_split[i] = np.percentile(counts[set_ind[i]:set_ind[i+1]], np.array([5, 50, 95]), axis=0, method="closest_observation")

    #     self.quant_split = quant_split # the stats across realizations

    # def plot_CSMF(self, fill=True, lim=False):

    #     plt.figure(figsize=(8, 8))
    #     plt.plot(self.mass_bins, self.quant[1], label="median", color="black")
    #     if fill==True:
    #         plt.fill_between(self.mass_bins, y1=self.quant[0], y2=self.quant[2], alpha=0.2, color="grey", label="5% - 95%")
    #     plt.yscale("log")
    #     plt.grid(alpha=0.4)
    #     if lim==True:
    #         plt.ylim(0.5,10**4.5)
    #     plt.xlim(4.5, 11)
    #     plt.xlabel("log m$_{stellar}$ (M$_\odot$)", fontsize=15)
    #     plt.ylabel("log N (> m$_{stellar}$)", fontsize=15)
    #     plt.legend()
    #     plt.show()

    # def mass_rank(self):

    #     self.rank = np.flip(np.argsort(self.lgMs,axis=1), axis=1) # rank the subhalos from largest to smallest
    #     self.rankedmass =  np.take_along_axis(self.lgMs, self.rank, axis=1) # this is it!!!
                
