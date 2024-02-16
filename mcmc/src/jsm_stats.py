import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import matplotlib.cm as cm
from scipy import stats
from numpy.random import poisson
from scipy.stats import ks_2samp


def pdf(data):
    index, counts = np.unique(data, return_counts=True)
    full = np.zeros(700) # the max number of unique counts across the models
    # needs to be set sufficiently high such that even extreme models can populate the Pnsat matrix
    full[index.astype("int")] = counts/data.shape[0]
    return full

def cumulative(lgMs_1D:np.ndarray, mass_bins):
    N = np.histogram(lgMs_1D, bins=mass_bins)[0]
    Nsub = np.sum(N)
    stat = Nsub-np.cumsum(N) 
    return np.insert(stat, 0, Nsub) #to add the missing index

def ecdf(data):
    return np.arange(1, data.shape[0]+1)/float(data.shape[0])

def N_rank(arr, threshold, fillval=np.nan):
    sorted_arr = np.sort(arr, axis=1)
    mask = (sorted_arr > threshold) & (~np.isnan(sorted_arr))
    masked_sorted_arr = np.where(mask, sorted_arr, np.nan)
    uneven = list(map(lambda row: row[~np.isnan(row)], masked_sorted_arr))
    lens = np.array(list(map(len, uneven)))
    shift = lens[:,None] > np.arange(lens.max())[::-1]
    even = np.full(shift.shape, fillval)
    even[shift] = np.concatenate(uneven)
    return even[:, ::-1]

def lnL_KS_max(data, model):
    try:
        clean_max_split = list(map(model.stat.max_split.__getitem__, data.stat.model_mask)) # this might yield an index error!
        p_vals = np.array(list(map(lambda x, y: ks_2samp(x, y)[1], data.stat.clean_max_split, clean_max_split)))
        return np.sum(np.log(p_vals))
    except IndexError:
        #print("this model is not preferable!")
        return -np.inf
    
def lnL_KS_tot(data, model):
    try:
        clean_tot_split = list(map(model.stat.tot_split.__getitem__, data.stat.model_mask)) # this might yield an index error!
        p_vals = np.array(list(map(lambda x, y: ks_2samp(x, y)[1], data.stat.clean_tot_split, clean_tot_split)))
        return np.sum(np.log(p_vals))
    except IndexError:
        #print("this model is not preferable!")
        return -np.inf

def lnL_PNsat(data, model):
    lnL = np.sum(np.log(model.stat.PNsat[data.stat.Nsat_perhost]))
    if np.isnan(lnL):
        return -np.inf
    else:
        return lnL


class SatStats_D:

    def __init__(self, lgMs, min_mass):
        self.lgMs = lgMs
        self.min_mass = min_mass

        self.Nsat_perhost = np.sum(self.lgMs > self.min_mass, axis=1)
        self.PNsat = pdf(self.Nsat_perhost)
        self.Nsat_unibin, self.Nsat_perbin = np.unique(self.Nsat_perhost, return_counts=True)

        self.mass_rank = N_rank(self.lgMs, threshold=self.min_mass)
        #self.Nsat_completeness = np.sum(~np.isnan(self.mass_rank), axis=0)
        #self.N_grtM = np.arange(0, self.mass_rank.shape[1])

        self.Nsat_index = np.insert(np.cumsum(self.Nsat_perbin),0,0)
        self.maxmass = self.mass_rank[:,0] # this is where you can toggle through frames! the second most massive and so on
        self.max_split = np.split(self.maxmass[np.argsort(self.Nsat_perhost)], self.Nsat_index)[1:-1]

        self.totmass = np.log10(np.nansum(10**self.mass_rank, axis=1))
        self.tot_split = np.split(self.totmass[np.argsort(self.Nsat_perhost)], self.Nsat_index)[1:-1]

        self.Neff_mask = self.Nsat_perbin > 4 # need to feed this to the models in the KS test step
        self.model_mask = self.Nsat_unibin[self.Neff_mask].tolist() 
        self.clean_max_split = list(map(self.max_split.__getitem__, np.where(self.Neff_mask)[0].tolist()))
        self.clean_tot_split = list(map(self.tot_split.__getitem__, np.where(self.Neff_mask)[0].tolist()))

class SatStats_M:

    def __init__(self, lgMs, min_mass):
        self.lgMs = lgMs
        self.min_mass = min_mass

        self.Nsat_perhost = np.sum(self.lgMs > self.min_mass, axis=1)
        self.PNsat = pdf(self.Nsat_perhost)
        self.Nsat_unibin, self.Nsat_perbin = np.unique(self.Nsat_perhost, return_counts=True)

        self.mass_rank = N_rank(self.lgMs, threshold=self.min_mass)
        #self.Nsat_completeness = np.sum(~np.isnan(self.mass_rank), axis=0)
        #self.N_grtM = np.arange(0, self.mass_rank.shape[1])

        self.Nsat_index = np.insert(np.cumsum(self.Nsat_perbin),0,0)
        self.maxmass = self.mass_rank[:,0]
        self.max_split = np.split(self.maxmass[np.argsort(self.Nsat_perhost)], self.Nsat_index)[1:-1]

        self.totmass = np.log10(np.nansum(10**self.mass_rank, axis=1))
        self.tot_split = np.split(self.totmass[np.argsort(self.Nsat_perhost)], self.Nsat_index)[1:-1]

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


    
# class SatStats:

#     def __init__(self, lgMs, min_mass):
#         self.lgMs = lgMs
#         self.min_mass = min_mass

        #self.N_RANK(plot=True)
        # self.SATFREQ()
        # self.MAXMASS()
        #self.CORRELATION()
        # self.TOTALMASS()
        # self.MEANMASS()
        # self.CSMF()

    # def N_RANK(self, plot=False):
    #     self.mass_rank = N_rank(self.lgMs, threshold=self.min_mass)
    #     self.N_per_bin = np.sum(~np.isnan(self.mass_rank), axis=0)
    #     self.N_greater_than = np.arange(0, self.mass_rank.shape[1])
    #     if plot==True:
    #         plt.figure(figsize=(6,6))
    #         plt.plot(self.N_greater_than, self.N_per_bin,color="black", marker="+")
    #         plt.xlabel("N (>M)")
    #         plt.ylabel("number of satellites")
    #         plt.show()

    # def SATFREQ(self, plot=False):
    #     self.satfreq = satfreq(self.lgMs, self.min_mass)
    #     #self.Pnsat = pdf(self.satfreq)

    #     self.satfreq_sorted = np.sort(self.satfreq)
    #     self.ecdf_satfreq = ecdf(self.satfreq_sorted)
        
    #     if plot==True:
    #         plt.figure(figsize=(6,6))
    #         plt.plot(np.arange(self.Pnsat.shape[0]), self.Pnsat)
    #         plt.xlabel("N satellites > $10^{"+str(self.min_mass)+"} \mathrm{M_{\odot}}$", fontsize=15)
    #         plt.ylabel("PDF", fontsize=15)
    #         plt.xlim(0,35)
    #         plt.show()

    #         plt.figure(figsize=(6,6))
    #         plt.plot(self.satfreq_sorted, self.ecdf_satfreq)
    #         plt.xlabel("N satellites > $10^{"+str(self.min_mass)+"} \mathrm{M_{\odot}}$", fontsize=15)
    #         plt.ylabel("CDF", fontsize=15)
    #         plt.show()

    # def MAXMASS(self, plot=False):
    #     self.Msmax = maxmass(self.lgMs)
    #     self.Msmax_sorted = np.sort(self.Msmax)
    #     self.ecdf_Msmax = ecdf(self.Msmax_sorted)

    #     if plot==True:
    #         plt.figure(figsize=(6,6))
    #         plt.plot(self.Msmax_sorted, self.ecdf_Msmax)
    #         plt.xlabel("max (M$_*$) ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
    #         plt.ylabel("CDF", fontsize=15)
    #         plt.show()

    # def CORRELATION(self):
    #     self.r = correlation(self.satfreq, self.Msmax)        

    # def TOTALMASS(self, plot=False):
    #     self.Mstot = totalmass(self.lgMs)
    #     self.Mstot_sorted = np.sort(self.Mstot)
    #     self.ecdf_Mstot = ecdf(self.Mstot_sorted)

    #     if plot==True:
    #         plt.figure(figsize=(6,6))
    #         plt.plot(self.Mstot_sorted, self.ecdf_Mstot)
    #         plt.xlabel("total stellar mass in satellites ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
    #         plt.ylabel("CDF", fontsize=15)
    #         plt.show()

    # def MEANMASS(self, plot=False):
    #     self.Msave = meanmass(self.lgMs)
    #     self.Msave_sorted = np.sort(self.Msave)
    #     self.ecdf_Msave = ecdf(self.Msave_sorted)

    #     if plot==True:
    #         plt.figure(figsize=(6,6))
    #         plt.plot(self.Msave_sorted, self.ecdf_Msave)
    #         plt.xlabel("average stellar mass across satellites ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
    #         plt.ylabel("CDF", fontsize=15)
    #         plt.show()        

    # def CSMF(self, mass_bins:np.ndarray=np.linspace(6,12,45), plotmed=False, plottot=False):

    #     self.mass_bins = mass_bins

    #     self.CSMF_counts = np.apply_along_axis(cumulative, 1, self.lgMs, mass_bins=self.mass_bins) 
    #     self.quant = np.percentile(self.CSMF_counts, np.array([5, 50, 95]), axis=0, method="closest_observation")
    #     self.total = np.sum(self.CSMF_counts, axis=0)

    #     # Nsets = int(counts.shape[0]/self.Nsamp) #dividing by the number of samples
    #     # set_ind = np.arange(0,Nsets)*self.Nsamp
    #     # print("dividing your sample into", Nsets-1, "sets")

    #     # quant_split = np.zeros(shape=(Nsets-1, 3, self.mass_bins.shape[0]))
    #     # for i in range(Nsets-1):
    #     #     quant_split[i] = np.percentile(counts[set_ind[i]:set_ind[i+1]], np.array([5, 50, 95]), axis=0, method="closest_observation")

    #     # self.quant_split = quant_split # the stats across realizations

    #     if plotmed == True:
    #         plt.figure(figsize=(6,6))
    #         plt.plot(self.mass_bins, self.quant[1], label="median")
    #         plt.fill_between(self.mass_bins, y1=self.quant[0], y2=self.quant[2], alpha=0.2, label="5% - 95%")
    #         # plt.xlim(6.5, 10.3)
    #         # plt.ylim(0, 11)
    #         plt.xlabel("log m$_{stellar}$ (M$_\odot$)", fontsize=15)
    #         plt.ylabel("N (> m$_{stellar}$)", fontsize=15)
    #         plt.legend()
    #         plt.show()

    #     if plottot == True:
    #         plt.figure(figsize=(6,6))
    #         plt.plot(self.mass_bins, self.total, label="total")
    #         plt.xlim(4.5, 10.3)
    #         plt.xlabel("log m$_{stellar}$ (M$_\odot$)", fontsize=15)
    #         plt.ylabel("log N (> m$_{stellar}$)", fontsize=15)
    #         plt.yscale("log")
    #         plt.legend()
    #         plt.show()

    # def mass_rank(self):

    #     self.rank = np.flip(np.argsort(self.lgMs,axis=1), axis=1) # rank the subhalos from largest to smallest
    #     self.rankedmass =  np.take_along_axis(self.lgMs, self.rank, axis=1) # this is it!!!
                
