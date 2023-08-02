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

def PDF(data, bins):

    count = np.histogram(data, bins)[0]
    return count / sum(count)

def CDF(data, bins):
    
    count = np.histogram(data, bins)[0]
    pdf = count / sum(count)
    return np.cumsum(pdf)

def satfreq(lgMs, Ms_min, satfreq_bins=np.linspace(0,17,18)):

    satfreq_bins = np.linspace(0,17,18)
    satfreq = np.sum(lgMs > Ms_min, axis = 1)
    satfreq_PDF = PDF(satfreq, satfreq_bins)
    return np.nan_to_num(satfreq_PDF) # to fix any nans, happens for extreme values of theta

def maxsatmass(lgMs, maxsatmass_bins=np.linspace(6,11,18)):
    
    maxsatmass = np.max(lgMs, axis=1)
    maxsatmass_CDF = CDF(maxsatmass, maxsatmass_bins)
    return np.nan_to_num(maxsatmass_CDF)


class SatStats:

    def __init__(self, lgMs_mat):
        self.lgMs_mat = lgMs_mat

    def CSMF(self, mass_bins:np.ndarray=np.linspace(4,11,45)):

        self.mass_bins = mass_bins

        counts = np.apply_along_axis(cumulative, 1, self.lgMs, mass_bins=self.mass_bins) 
        self.CSMF_counts = counts # a CSMF for each of the realizations

        quant = np.percentile(counts, np.array([5, 50, 95]), axis=0, method="closest_observation")
        self.quant = quant # the stats across realizations

        Nsets = int(counts.shape[0]/self.Nsamp) #dividing by the number of samples
        set_ind = np.arange(0,Nsets)*self.Nsamp
        print("dividing your sample into", Nsets-1, "sets")

        quant_split = np.zeros(shape=(Nsets-1, 3, self.mass_bins.shape[0]))
        for i in range(Nsets-1):
            quant_split[i] = np.percentile(counts[set_ind[i]:set_ind[i+1]], np.array([5, 50, 95]), axis=0, method="closest_observation")

        self.quant_split = quant_split # the stats across realizations

    def plot_CSMF(self, fill=True, lim=False):

        plt.figure(figsize=(8, 8))
        plt.plot(self.mass_bins, self.quant[1], label="median", color="black")
        if fill==True:
            plt.fill_between(self.mass_bins, y1=self.quant[0], y2=self.quant[2], alpha=0.2, color="grey", label="5% - 95%")
        plt.yscale("log")
        plt.grid(alpha=0.4)
        if lim==True:
            plt.ylim(0.5,10**4.5)
        plt.xlim(4.5, 11)
        plt.xlabel("log m$_{stellar}$ (M$_\odot$)", fontsize=15)
        plt.ylabel("log N (> m$_{stellar}$)", fontsize=15)
        plt.legend()
        plt.show()

    def satfreq(self, Ms_min, satfreq_bins=np.linspace(0,17,18)):

        self.Ms_min = Ms_min
        self.satfreq_bins = satfreq_bins

        self.satfreq_mat = np.sum(self.lgMs_mat > Ms_min, axis = 2)

        self.satfreq_PDF_mat = np.apply_along_axis(PDF, 1, self.satfreq_mat, self.satfreq_bins)
        self.satfreq_PDF_ave = np.average(self.satfreq_PDF_mat, axis=0)

        self.satfreq_CDF_mat = np.apply_along_axis(CDF, 1, self.satfreq_mat, self.satfreq_bins)
        self.satfreq_CDF_ave = np.average(self.satfreq_CDF_mat, axis=0)

    def plot_satfreq(self, poisson=False, pN=30):

        if poisson ==True:
            peak = np.where(self.satfreq_PDF_ave == max(self.satfreq_PDF_ave))[0][0]
            pdata = poisson(self.satfreq_bins[peak], 10000)
            px = np.linspace(0,17,pN)
            pcounts = PDF(pdata, px)
            mask = pcounts > 0.0    
            plt.plot(px[1:][mask], pcounts[mask], label="Poisson", color="red", ls="--")


        plt.step(self.satfreq_bins[1:], self.satfreq_PDF_mat[0], label="example data", color="green")
        plt.step(self.satfreq_bins[1:], self.satfreq_PDF_ave, label="average", color="black")

        plt.xlabel("number of satellites > $10^{6.5} \mathrm{M_{\odot}}$", fontsize=15)
        plt.ylabel("PDF", fontsize=15)
        plt.legend(fontsize=12)
        plt.show()

        if poisson ==True:
            ccounts = CDF(pdata, px)    
            plt.plot(px[1:][mask], ccounts[mask], label="Poisson", color="red", ls="--")

        plt.step(self.satfreq_bins[1:], self.satfreq_CDF_mat[0], label="example data", color="green")
        plt.step(self.satfreq_bins[1:], self.satfreq_CDF_ave, label="average", color="black")

        plt.xlabel("number of satellites > $10^{6.5} \mathrm{M_{\odot}}$", fontsize=15)
        plt.ylabel("CDF", fontsize=15)
        plt.legend(fontsize=12)
        plt.show()


    def maxsatmass(self, maxsatmass_bins = np.linspace(6,11,18)):

        self.maxsatmass_bins = maxsatmass_bins

        self.maxsatmass_mat = np.max(self.lgMs_mat, axis=2)

        self.maxsatmass_PDF_mat = np.apply_along_axis(PDF, 1, self.maxsatmass_mat , self.maxsatmass_bins)
        self.maxsatmass_PDF_ave = np.average(self.maxsatmass_PDF_mat, axis=0)

        self.maxsatmass_CDF_mat = np.apply_along_axis(CDF, 1, self.maxsatmass_mat , self.maxsatmass_bins)
        self.maxsatmass_CDF_ave = np.average(self.maxsatmass_CDF_mat, axis=0)

    def plot_maxsatmass(self, poisson=False, pN=30):

        if poisson ==True:
            peak = np.where(self.maxsatmass_PDF_ave == max(self.maxsatmass_PDF_ave))[0][0]
            pdata = poisson(self.maxsatmass_bins[peak], 10000)
            px = np.linspace(6,11,pN)
            pcounts = PDF(pdata, px)
            mask = pcounts > 0.0    
            plt.plot(px[1:][mask], pcounts[mask], label="Poisson", color="red", ls="--")


        plt.step(self.maxsatmass_bins[1:], self.maxsatmass_PDF_mat[0], label="example data", color="green")
        plt.step(self.maxsatmass_bins[1:], self.maxsatmass_PDF_ave, label="average", color="black")

        plt.xlabel("stellar mass of most massive satellite ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
        plt.ylabel("PDF", fontsize=15)
        plt.legend(fontsize=12)
        plt.show()        

        if poisson ==True:
            ccounts = CDF(pdata, px)   
            plt.plot(px[1:][mask], ccounts[mask], label="Poisson", color="red", ls="--")

        plt.step(self.maxsatmass_bins[1:], self.maxsatmass_CDF_mat[0], label="example data", color="green")
        plt.step(self.maxsatmass_bins[1:], self.maxsatmass_CDF_ave, label="average", color="black")

        plt.xlabel("stellar mass of most massive satellite ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
        plt.ylabel("CDF", fontsize=15)
        plt.legend(fontsize=12)
        plt.show()


    def mass_rank(self):

        self.rank = np.flip(np.argsort(self.lgMs,axis=1), axis=1) # rank the subhalos from largest to smallest
        self.rankedmass =  np.take_along_axis(self.lgMs, self.rank, axis=1) # this is it!!!
                
