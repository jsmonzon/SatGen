import numpy as np
import matplotlib.pyplot as plt
from numpy.random import poisson
from scipy.stats import ks_2samp
from scipy.special import gamma, loggamma, factorial
from scipy import stats

##### ------------------------------------------------------------------------
## Our stats
##### ------------------------------------------------------------------------

def pdf(data, max):
    index, counts = np.unique(data, return_counts=True)
    full = np.zeros(max) # the max number of unique counts across the models
    # needs to be set sufficiently high such that even extreme models can populate the Pnsat matrix
    full[index.astype("int")] = counts/data.shape[0]
    return full
    
def correlation(stat1, stat2):
    return stats.pearsonr(stat1, stat2)[0]

def ecdf(data):
    return np.arange(1, data.shape[0]+1)/float(data.shape[0])

def N_rank(arr, threshold, fillval=np.nan):
    sorted_arr = np.sort(arr, axis=1) # sort the masses
    mask = (sorted_arr > threshold) & (~np.isnan(sorted_arr)) # are they above the threshold? cut out the nan values
    masked_sorted_arr = np.where(mask, sorted_arr, np.nan)
    uneven = list(map(lambda row: row[~np.isnan(row)], masked_sorted_arr)) #setting up a list of lists
    lens = np.array(list(map(len, uneven))) # which list has the most elements?
    shift = lens[:,None] > np.arange(lens.max())[::-1] #flipping so it starts with the largest
    even = np.full(shift.shape, fillval)
    even[shift] = np.concatenate(uneven)
    full_rank = even[:, ::-1]
    return full_rank[~np.isnan(full_rank).all(axis=1)] # this automatically removes all rows that are filled with nans 

def lnL_PNsat(data, model):
    lnL = np.sum(np.log(model.stat.PNsat[data.stat.Nsat_perhost]))
    if np.isnan(lnL):
        return -np.inf
    else:
        return lnL

def lnL_KS_max(data, model):
    try:
        clean_max_split = list(map(model.stat.max_split.__getitem__, data.stat.model_mask)) # this might yield an index error!
        p_vals = np.array(list(map(lambda x, y: ks_2samp(x, y)[1], data.stat.clean_max_split, clean_max_split)))
        return np.sum(np.log(p_vals))
    except IndexError:
        #print("this model is not preferable!")
        return -np.inf
    
##### ------------------------------------------------------------------------
## To count satellites
##### ------------------------------------------------------------------------


def cumulative(lgMs_1D:np.ndarray, mass_bins, return_bins=False):
    N = np.histogram(lgMs_1D, bins=mass_bins)[0]
    if return_bins:
        return np.cumsum(N[::-1])[::-1], (mass_bins[:-1] + mass_bins[1:]) / 2
    else:
        return np.cumsum(N[::-1])[::-1]
    
def count(lgMs_1D:np.ndarray, mass_bins, return_bins=False):
    N = np.histogram(lgMs_1D, bins=mass_bins)[0]
    if return_bins:
        return N, (mass_bins[:-1] + mass_bins[1:]) / 2
    else:
        return N
    
##### ------------------------------------------------------------------------
## Nadler stats
##### ------------------------------------------------------------------------


def lnprob_i(N_real, n_i, sum_j):
    N_ratio = (N_real+1)/N_real
    fac1 = np.log(N_ratio)
    fac2 = np.log(N_real+1)
    return -(sum_j+1)*fac1 - n_i*fac2 + loggamma(sum_j+n_i+1) - loggamma(n_i+1) - loggamma(sum_j+1) # the version written in log space

def lnL_Nadler(data, model):
    N_real, N_bins = model.stat.stack.shape[0], model.stat.stack.shape[1]
    lnProb = []
    for i_bin in range(N_bins):
        n_obs = data.stat.stack[i_bin]
        n_model = model.stat.stack[:, i_bin]
        n_model_sum = n_model.sum()
        lnProb.append(lnprob_i(N_real, n_obs, n_model_sum))
    return np.sum(lnProb)


##### ------------------------------------------------------------------------
## EXTRA STUFF
##### ------------------------------------------------------------------------

# def prob_i(N_real, n_i, sum_j):
#     return ((N_real+1)/(N_real))**(-(sum_j+1)) * (N_real+1)**n_i * (factorial(sum_j+n_i) / (factorial(n_i) * factorial(sum_j))) # a version written with factorials

# def prob_i(N_real, n_i, sum_j):
#     return ((N_real+1)/(N_real))**(-(sum_j+1)) * (N_real+1)**n_i * (gamma(sum_j+n_i+1) / (gamma(n_i+1) * gamma(sum_j+1))) # a version written in linear space

# def lnL_KS_tot(data, model):
#     try:
#         clean_tot_split = list(map(model.stat.tot_split.__getitem__, data.stat.model_mask)) # this might yield an index error!
#         p_vals = np.array(list(map(lambda x, y: ks_2samp(x, y)[1], data.stat.clean_tot_split, clean_tot_split)))
#         return np.sum(np.log(p_vals))
#     except IndexError:
#         #print("this model is not preferable!")
#         return -np.inf


# def lnL_JSM(data, model):
#     try:
#         clean_max_split = list(map(model.max_split.__getitem__, data.model_mask)) # this might yield an index error!
#         p_vals = np.array(list(map(lambda x, y: ks_2samp(x, y)[1], data.clean_max_split, clean_max_split)))
#         pKS = np.sum(np.log(p_vals))
#     except IndexError:
#         pKS = -np.inf
#     lnL = np.sum(np.log(model.PNsat[data.Nsat_perhost]))
#     if np.isnan(lnL):
#         Pnsat = -np.inf
#     else:
#         Pnsat = lnL
#     return pKS + Pnsat

    
class SatStats_D:

    def __init__(self, lgMs, min_mass, max_N):
        self.lgMs = lgMs
        self.min_mass = min_mass
        self.max_N = max_N

        self.mass_rank = N_rank(self.lgMs, threshold=self.min_mass)
        self.Nsat_perhost = np.sum(~np.isnan(self.mass_rank), axis=1)
        self.PNsat = pdf(self.Nsat_perhost, max_N)
        self.Nsat_unibin, self.Nsat_perbin = np.unique(self.Nsat_perhost, return_counts=True)

        self.Nsat_index = np.insert(np.cumsum(self.Nsat_perbin),0,0)
        self.maxmass = self.mass_rank[:,0] # this is where you can toggle through frames! the second most massive and so on
        self.max_split = np.split(self.maxmass[np.argsort(self.Nsat_perhost)], self.Nsat_index)[1:-1]
        self.Neff_mask = self.Nsat_perbin > 4 # need to feed this to the models in the KS test step
        self.model_mask = self.Nsat_unibin[self.Neff_mask].tolist() 
        self.clean_max_split = list(map(self.max_split.__getitem__, np.where(self.Neff_mask)[0].tolist()))

        self.sigma_N = np.nanstd(self.Nsat_perhost)
        self.correlation = correlation(self.Nsat_perhost[self.Nsat_perhost>0], self.maxmass[self.Nsat_perhost>0])
        self.Nsat_tot = np.sum(~np.isnan(self.mass_rank))

        #just for plotting!
    def Pnsat_plot(self):
        self.PNsat_range = np.arange(self.PNsat.shape[0])
        plt.figure(figsize=(6,6))
        plt.plot(self.PNsat_range, self.PNsat)
        plt.xlabel("N satellites > $10^{"+str(self.min_mass)+"} \mathrm{M_{\odot}}$", fontsize=15)
        plt.ylabel("PDF", fontsize=15)
        plt.xlim(0,35)
        plt.show()

    def Msmax_plot(self):
        self.Msmax_sorted = np.sort(self.maxmass)
        self.ecdf_Msmax = ecdf(self.Msmax_sorted)
        plt.figure(figsize=(6,6))
        plt.plot(np.sort(self.maxmass), ecdf(np.sort(self.maxmass)))
        plt.xlabel("max (M$_*$) ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
        plt.ylabel("CDF", fontsize=15)
        plt.show()

class SatStats_M:

    def __init__(self, lgMs, min_mass, max_N):
        self.lgMs = lgMs
        self.min_mass = min_mass
        self.max_N = max_N

        self.mass_rank = N_rank(self.lgMs, threshold=self.min_mass)
        self.Nsat_perhost = np.sum(~np.isnan(self.mass_rank), axis=1)
        self.PNsat = pdf(self.Nsat_perhost, max_N)
        self.Nsat_unibin, self.Nsat_perbin = np.unique(self.Nsat_perhost, return_counts=True)

        self.Nsat_index = np.insert(np.cumsum(self.Nsat_perbin),0,0)
        self.maxmass = self.mass_rank[:,0] # this is where you can toggle through frames! the second most massive and so on
        self.max_split = np.split(self.maxmass[np.argsort(self.Nsat_perhost)], self.Nsat_index)[1:-1]

        #just for plotting!
    def Pnsat_plot(self):
        self.PNsat_range = np.arange(self.PNsat.shape[0])
        plt.figure(figsize=(6,6))
        plt.plot(self.PNsat_range, self.PNsat)
        plt.xlabel("N satellites > $10^{"+str(self.min_mass)+"} \mathrm{M_{\odot}}$", fontsize=15)
        plt.ylabel("PDF", fontsize=15)
        plt.xlim(0,35)
        plt.show()

    def Msmax_plot(self):
        self.Msmax_sorted = np.sort(self.maxmass)
        self.ecdf_Msmax = ecdf(self.Msmax_sorted)
        plt.figure(figsize=(6,6))
        plt.plot(np.sort(self.maxmass), ecdf(np.sort(self.maxmass)))
        plt.xlabel("max (M$_*$) ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
        plt.ylabel("CDF", fontsize=15)
        plt.show()



##### ------------------------------------------------------------------------
## EXTRA STUFF
##### ------------------------------------------------------------------------


class SatStats_NAD_D:

    def __init__(self, lgMs, min_mass, N_bin):
        self.lgMs = lgMs
        self.min_mass = min_mass
        self.N_bin = N_bin

        self.bins = np.linspace(self.min_mass, 10.5, self.N_bin)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        self.count_mat = np.apply_along_axis(count, 1, self.lgMs, mass_bins=self.bins)
        self.stack = np.sum(self.count_mat, axis=0)

        self.count_mat_cum = np.apply_along_axis(cumulative, 1, self.lgMs, mass_bins=self.bins)
        self.stack_cum = np.sum(self.count_mat_cum, axis=0)

    
    def plot(self):
        plt.figure(figsize=(6,6))
        plt.plot(self.bin_centers, self.stack, color="grey")
        plt.xlabel("stellar mass")
        plt.ylabel("N")
        plt.yscale("log")
        plt.show() 


class SatStats_NAD_M:

    def __init__(self, lgMs_mat, min_mass, N_bin):
        self.lgMs_mat = lgMs_mat
        self.min_mass = min_mass
        self.N_bin = N_bin

        self.bins = np.linspace(self.min_mass, 10, self.N_bin)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        self.N_real = self.lgMs_mat.shape[0]
        self.stack = np.zeros(shape=(self.N_real, self.N_bin-1))

        for i, real in enumerate(self.lgMs_mat):
            self.count_mat_i = np.apply_along_axis(count, 1, real, mass_bins=self.bins)
            self.stack[i] = np.sum(self.count_mat_i, axis=0)

    def plot(self):
        plt.figure(figsize=(6,6))
        for stack in self.stack[0:10]:
            plt.plot(self.bin_centers, stack, color="grey", alpha=0.2)
        plt.xlabel("stellar mass")
        plt.ylabel("N")
        plt.yscale("log")
        plt.show()  

# def prob_i(N_real, n_i, sum_j):
#     return ((N_real+1)/(N_real))**(-(sum_j+1)) * (N_real+1)**n_i * (factorial(sum_j+n_i) / (factorial(n_i) * factorial(sum_j))) # a version written with factorials

# def prob_i(N_real, n_i, sum_j):
#     return ((N_real+1)/(N_real))**(-(sum_j+1)) * (N_real+1)**n_i * (gamma(sum_j+n_i+1) / (gamma(n_i+1) * gamma(sum_j+1))) # a version written in linear space

# def lnL_KS_tot(data, model):
#     try:
#         clean_tot_split = list(map(model.stat.tot_split.__getitem__, data.stat.model_mask)) # this might yield an index error!
#         p_vals = np.array(list(map(lambda x, y: ks_2samp(x, y)[1], data.stat.clean_tot_split, clean_tot_split)))
#         return np.sum(np.log(p_vals))
#     except IndexError:
#         #print("this model is not preferable!")
#         return -np.inf


# def lnL_JSM(data, model):
#     try:
#         clean_max_split = list(map(model.max_split.__getitem__, data.model_mask)) # this might yield an index error!
#         p_vals = np.array(list(map(lambda x, y: ks_2samp(x, y)[1], data.clean_max_split, clean_max_split)))
#         pKS = np.sum(np.log(p_vals))
#     except IndexError:
#         pKS = -np.inf
#     lnL = np.sum(np.log(model.PNsat[data.Nsat_perhost]))
#     if np.isnan(lnL):
#         Pnsat = -np.inf
#     else:
#         Pnsat = lnL
#     return pKS + Pnsat

        # self.totmass = np.log10(np.nansum(10**self.mass_rank, axis=1))
        # self.tot_split = np.split(self.totmass[np.argsort(self.Nsat_perhost)], self.Nsat_index)[1:-1]
        # self.clean_tot_split = list(map(self.tot_split.__getitem__, np.where(self.Neff_mask)[0].tolist()))

        #self.Nsat_completeness = np.sum(~np.isnan(self.mass_rank), axis=0)
        #self.N_grtM = np.arange(0, self.mass_rank.shape[1])

        # self.Mstot_sorted = np.sort(self.totmass)
        # self.ecdf_Mstot = ecdf(self.Mstot_sorted)


        # self.mass_bins = np.linspace(4.5,10.5,45)
        # self.CSMF_counts = np.apply_along_axis(cumulative, 1, self.lgMs, mass_bins=self.mass_bins) 
        # self.quant = np.percentile(self.CSMF_counts, np.array([5, 50, 95]), axis=0, method="closest_observation")
        # self.D23_quant = np.sum(self.CSMF_counts, axis=0)

            # def CSMF_plot(self):
    #     plt.figure(figsize=(6,6))
    #     plt.plot(self.mass_bins, self.quant[1], label="median")
    #     plt.fill_between(self.mass_bins, y1=self.quant[0], y2=self.quant[2], alpha=0.2, label="5% - 95%")
    #     plt.xlabel("log m$_{stellar}$ (M$_\odot$)", fontsize=15)
    #     plt.ylabel("N (> m$_{stellar}$)", fontsize=15)
    #     plt.xlim(6.5, 11)
    #     plt.yscale("log")
    #     plt.legend()
    #     plt.show()

    # def Mstot_plot(self):
    #     plt.figure(figsize=(6,6))
    #     plt.plot(np.sort(self.totmass), ecdf(np.sort(self.totmass)))
    #     plt.xlabel("max (M$_*$) ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
    #     plt.ylabel("CDF", fontsize=15)
    #     plt.show()


##### ------------------------------------------------------------------------


# def lnL_KS(model, data):
#     return np.log(ks_2samp(model, data)[1])


# def lnL_chi2r(model, data):
#     ave = np.average(model)
#     std = np.std(model)
#     chi2 = ((data - ave) / std) ** 2    
#     return (chi2 / -2.0)
    
# def satfreq(lgMs, min_mass):
#     return np.sum(lgMs > min_mass, axis = 1)

# def maxmass(lgMs):
#     return np.nanmax(lgMs, axis=1)

# def totalmass(lgMs):
#     return np.log10(np.nansum(10**lgMs, axis=1))

# def meanmass(lgMs):
#     return np.log10(np.nanmean(10**lgMs, axis=1))

# def correlation(stat1, stat2):
#     return stats.pearsonr(stat1, stat2)[0]

# def ecdf_nan(data):
#     return np.arange(1, data.shape[0]+1)/float(np.sum(~np.isnan(data)))

# def ecdf_plot(data):
#     cdf = np.arange(1, data.shape[0]+1)/float(np.sum(~np.isnan(data)))
#     index = np.sort(data)
#     return index, cdf


    
