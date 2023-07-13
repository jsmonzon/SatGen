
import numpy as np
import galhalo
import config as cfg
import aux
import profiles as pr
import cosmo as co
import matplotlib.pyplot as plt


def prep_data(numpyfile, convert=True, includenan=True):

    """_summary_
    a quick a dirty way of getting satellites statistics. Really should use the MassMat class below

    """
    Mh = np.load(numpyfile)
    #Mh[:, 0] = 0.0  # masking the host mass in the matrix
    #zero_mask = Mh != 0.0 
    #Mh = np.log10(np.where(zero_mask, Mh, np.nan)) #switching the to nans!
    lgMh = np.log10(Mh)

    if includenan == False:
        max_sub = min(Mh.shape[1] - np.sum(np.isnan(Mh),axis=1)) # not padded! hope it doesnt screw up stats
    else: 
        max_sub = max(Mh.shape[1] - np.sum(np.isnan(Mh),axis=1))

    lgMh = lgMh[:,1:max_sub]  #excluding the host mass
    if convert==False:
        return lgMh
    else:
        return galhalo.lgMs_D22_det(lgMh)


def differential(phi, phi_bins, phi_binsize): 
    N = np.histogram(phi, bins=phi_bins)[0]
    return N/phi_binsize

class MassMat:

    """
    An easy way of interacting with the condensed mass matricies.
    One instance of the Realizations class will create several SAGA-like samples.
    This includes the conversion from halo mass to stellar mass
    """
        
    def __init__(self, massfile, Nbins=45, phimin=-4):

        self.massfile = massfile
        self.Nbins = Nbins
        self.phimin = phimin
        self.phi_bins = np.linspace(self.phimin, 0, Nbins)
        self.phi_binsize = self.phi_bins[1] - self.phi_bins[0]

    def prep_data(self, redfile=None, includenan=True):

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


    def SHMR(self, alpha:float=1.85, delta:float=0.2, sigma:float=0.1):

        """_summary_
        Convert from halo mass to stellar mass

        Args:
            lgMh_2D (np.ndarray): 2D halo mass array
            alpha (float): power law slope
            delta (float): quadratic term to cruve relation
            sigma (float): log normal scatter

        Returns:
            np.ndarray: 2D stellar mass array
        """

        M_star_a = 10 # these are the anchor points
        M_halo_a = 11.67

        lgMs_2D = alpha*(self.lgMh-M_halo_a) - delta*(self.lgMh-M_halo_a)**2 + M_star_a
        scatter = np.random.normal(loc=0, scale=sigma, size=(lgMs_2D.shape))
        self.lgMs = lgMs_2D + scatter
