import numpy as np
import matplotlib.pyplot as plt
from numpy.random import poisson
from scipy.stats import ks_2samp
from scipy.special import gamma, loggamma, factorial
from scipy import stats
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.stats import binned_statistic
from scipy.optimize import minimize
import pandas as pd


##### ------------------------------------------------------------------------
## Our stats
##### ------------------------------------------------------------------------

def nan_mask(array):
    return array[~np.isnan(array)]

def pdf(data, max):
    index, counts = np.unique(data, return_counts=True)
    full = np.zeros(max) # the max number of unique counts across the models
    # needs to be set sufficiently high such that even extreme models can populate the Pnsat matrix
    full[index.astype("int")] = counts/data.shape[0]
    return full
    
def correlation(stat1, stat2):
    return stats.spearmanr(stat1, stat2)[0]

def correlation_p(stat1, stat2):
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
    nan_row_mask = ~np.isnan(full_rank).all(axis=1)
    return full_rank[nan_row_mask], nan_row_mask # this automatically removes all rows that are filled with nans 

def lnL_PNsat(data, model):
    lnL = np.sum(np.log(model.stat.PNsat[data.stat.Nsat_perhost]))
    if np.isinf(lnL):
        #print("index error in Pnsat")
        return -np.inf
    else:
        return lnL
    
def lnL_PNsat_test(data, model):
    return model.stat.PNsat[data.stat.Nsat_perhost]

def lnL_KS_max(data, model):
    try:
        clean_max_split = list(map(model.stat.max_split.__getitem__, data.stat.model_mask)) # this might yield an index error!
        p_vals = np.array(list(map(lambda x, y: ks_2samp(x, y)[1], data.stat.clean_max_split, clean_max_split)))
        return np.sum(np.log(p_vals))
    except IndexError:
        #print("this model is not preferable!")
        return -np.inf
    
def lnL_KS_sec(data, model):
    try:
        clean_sec_split = list(map(model.stat.sec_split.__getitem__, data.stat.model_mask)) # this might yield an index error!
        p_vals = np.array(list(map(lambda x, y: ks_2samp(x, y)[1], data.stat.clean_sec_split, clean_sec_split)))
        return np.sum(np.log(p_vals))
    except IndexError:
        #print("this model is not preferable!")
        return -np.inf
    
def lnL_KS_thir(data, model):
    try:
        clean_thir_split = list(map(model.stat.thir_split.__getitem__, data.stat.model_mask)) # this might yield an index error!
        p_vals = np.array(list(map(lambda x, y: ks_2samp(x, y)[1], data.stat.clean_thir_split, clean_thir_split)))
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
    
def lnL_KS_old(data, model):
    return np.log(ks_2samp(data.stat.maxmass, model.stat.maxmass)[1])
    
##### ------------------------------------------------------------------------
## To count satellites
##### ------------------------------------------------------------------------

def cumulative_histogram(values, bin_edges):
    counts, _ = np.histogram(values, bins=bin_edges)
    cumulative_counts = np.cumsum(counts)
    cumulative_fraction = cumulative_counts / cumulative_counts[-1]
    
    return bin_edges[:-1], cumulative_fraction

def cumulative_fbound(fbound_data, bins):
    hist = np.array([np.sum(fbound_data <= b) for b in bins])
    hist = hist / len(fbound_data)
    return hist

def cumulative(lgMs_1D:np.ndarray, mass_bins, return_bins=False):
    N = np.histogram(lgMs_1D, bins=mass_bins)[0]
    if return_bins:
        return np.cumsum(N[::-1])[::-1], (mass_bins[:-1] + mass_bins[1:]) / 2
    else:
        return np.cumsum(N[::-1])[::-1]
    
def count(lgMs_1D:np.ndarray, mass_bins, return_bins=False):
    N = np.histogram(lgMs_1D, bins=mass_bins, density=True)[0]
    if return_bins:
        return N, (mass_bins[:-1] + mass_bins[1:]) / 2
    else:
        return N
    
def count_straight(lgMs_1D:np.ndarray, mass_bins, return_bins=False):
    N = np.histogram(lgMs_1D, bins=mass_bins, density=False)[0]
    if return_bins:
        return N, (mass_bins[:-1] + mass_bins[1:]) / 2
    else:
        return N
    
def grab_mass_ind(mass_array, Nsat_perhost, Nsat_index, Neff_mask):
    m_split = np.split(mass_array[np.argsort(Nsat_perhost)], Nsat_index)[1:-1]
    if type(Neff_mask) == np.ndarray:
        clean_m_split = list(map(m_split.__getitem__, np.where(Neff_mask)[0].tolist()))
        return m_split, clean_m_split
    else:
        return m_split
    
def radii_grthan(radii, bins):
    # Define bins
    # Histogram with cumulative count
    N_grthan_r, bin_edges = np.histogram(radii, bins=bins)
    N_grthan_r_cumulative = np.cumsum(N_grthan_r[::-1])[::-1]

    # Mid-points of the bins for plotting
    bincenters = (bin_edges[:-1] + bin_edges[1:]) / 2
    return N_grthan_r_cumulative


def scatter_color(x, y, c, **kwargs):
    norm = colors.Normalize(vmin=c.min(), vmax=c.max())
    colormap = cm.viridis_r

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))

    # Filter out scatter-specific kwargs (the rest will go to ax configuration)
    scatter_kwargs = {key: kwargs[key] for key in kwargs if key not in ['title', 'xlabel', 'ylabel', 'xscale', 'yscale', 'xlim', 'ylim', 'cbar_label', 'label', 'figsize']}
    
    # Create the scatter plot
    sc = ax.scatter(x, y, color=colormap(norm(c)), **scatter_kwargs)

    # Configure the axes and labels
    ax.set_title(kwargs.get("title", ""))
    ax.set_xlabel(kwargs.get("xlabel", ""))
    ax.set_ylabel(kwargs.get("ylabel", ""))
    ax.set_xscale(kwargs.get("xscale", "linear"))
    ax.set_yscale(kwargs.get("yscale", "linear"))
    ax.set_xlim(kwargs.get("xlim", ax.get_xlim()))
    ax.set_ylim(kwargs.get("ylim", ax.get_ylim()))

    # Create and configure the colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(kwargs.get("cbar_label", ""))

    # Add legend if a label is provided
    if "label" in kwargs:
        ax.legend([kwargs["label"]])

    plt.show()

def quadrant_percentages_plot(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    valid = (x != 0) & (y != 0)
    x, y = x[valid], y[valid]
    total = len(x)

    q1 = np.sum((x > 0) & (y > 0)) / total * 100
    q2 = np.sum((x < 0) & (y > 0)) / total * 100
    q3 = np.sum((x < 0) & (y < 0)) / total * 100
    q4 = np.sum((x > 0) & (y < 0)) / total * 100

    txt = [f"{q1:.1f}%", f"{q4:.1f}%", f"{q3:.1f}%", f"{q2:.1f}%"]

    rho, pval = correlation_p(x, y)

    return txt, rho, pval

def finite_binned_stat(x, y, bins):
    """
    Helper that removes non-finite y values before computing
    mean and std in bins.
    """
    mask = np.isfinite(y)

    mean, _, _ = binned_statistic(
        x[mask], y[mask],
        statistic='median',
        bins=bins)

    std, _, _ = binned_statistic(
        x[mask], y[mask],
        statistic='std',
        bins=bins)

    return mean, std

def fit_line_asym_errors(x, y, yerr_up, yerr_down, p0=(1.0, 0.0)):
    """
    Fit y = m x + b with asymmetric y-errors using weighted least squares.

    Parameters
    ----------
    x : array_like
        Independent variable
    y : array_like
        Dependent variable
    yerr_up : array_like
        Upper (positive) y-uncertainties
    yerr_down : array_like
        Lower (negative) y-uncertainties
    p0 : tuple, optional
        Initial guess for (m, b)

    Returns
    -------
    m, b : float
        Best-fit slope and intercept
    result : OptimizeResult
        Full scipy optimization result
    """

    x = np.asarray(x)
    y = np.asarray(y)
    yerr_up = np.asarray(yerr_up)
    yerr_down = np.asarray(yerr_down)

    def chi2(params):
        m, b = params
        y_model = m * x + b

        # Choose uncertainty based on which side of the point the model is on
        sigma = np.where(y_model > y, yerr_up, yerr_down)

        return np.sum(((y - y_model) / sigma) ** 2)

    result = minimize(chi2, p0, method="Nelder-Mead")
    m_best, b_best = result.x

    return m_best, b_best

def fit_line_sym_errors(x, y, sigma, p0=(1.0, 0.0)):
    """
    Fit y = m x + b with symmetric y-errors using weighted least squares.

    Parameters
    ----------
    x : array_like
        Independent variable
    y : array_like
        Dependent variable
    sigma : array_like
        Symmetric 1-sigma uncertainties on y
    p0 : tuple, optional
        Initial guess for (m, b)

    Returns
    -------
    m, b : float
        Best-fit slope and intercept
    result : OptimizeResult
        Full scipy optimization result
    """

    x = np.asarray(x)
    y = np.asarray(y)
    sigma = np.asarray(sigma)

    # Remove non-finite values (important if sigma contains inf/nan)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    x = x[mask]
    y = y[mask]
    sigma = sigma[mask]

    def chi2(params):
        m, b = params
        y_model = m * x + b
        return np.sum(((y - y_model) / sigma) ** 2)

    result = minimize(chi2, p0, method="Nelder-Mead")
    m_best, b_best = result.x

    return m_best, b_best


    
##### ------------------------------------------------------------------------
## EXTRA STUFF
##### ------------------------------------------------------------------------


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

        self.mass_rank, self.nan_mask = N_rank(self.lgMs, threshold=self.min_mass)
        self.Nsat_perhost = np.sum(~np.isnan(self.mass_rank), axis=1)
        self.PNsat = pdf(self.Nsat_perhost, max_N)
        self.Nsat_unibin, self.Nsat_perbin = np.unique(self.Nsat_perhost, return_counts=True)
        self.Neff_mask = self.Nsat_perbin > 4 # need to feed this to the models in the KS test step
        self.model_mask = self.Nsat_unibin[self.Neff_mask].tolist() 
        self.Nsat_index = np.insert(np.cumsum(self.Nsat_perbin),0,0)


        #### ADDING IN THE NEW STATS
        self.maxmass = self.mass_rank[:,0] # this is where you can toggle through frames! the second most massive and so on
        self.max_split, self.clean_max_split = grab_mass_ind(self.maxmass, self.Nsat_perhost, self.Nsat_index, self.Neff_mask)

        self.secmass = self.mass_rank[:,1] #2nd most massive
        self.sec_split, self.clean_sec_split = grab_mass_ind(self.secmass, self.Nsat_perhost, self.Nsat_index, self.Neff_mask)

        self.thirmass = self.mass_rank[:,2] #3rd most massive
        self.thir_split, self.clean_thir_split = grab_mass_ind(self.thirmass, self.Nsat_perhost, self.Nsat_index, self.Neff_mask)

        self.totmass = np.log10(np.nansum(10**self.mass_rank, axis=1)) #total mass
        self.tot_split, self.clean_tot_split = grab_mass_ind(self.totmass, self.Nsat_perhost, self.Nsat_index, self.Neff_mask)

        ### misc stats
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

        self.mass_rank, _ = N_rank(self.lgMs, threshold=self.min_mass)
        self.Nsat_perhost = np.sum(~np.isnan(self.mass_rank), axis=1)
        self.PNsat = pdf(self.Nsat_perhost, max_N)
        self.Nsat_unibin, self.Nsat_perbin = np.unique(self.Nsat_perhost, return_counts=True)
        self.Nsat_index = np.insert(np.cumsum(self.Nsat_perbin),0,0)
        
        #### ADDING IN THE NEW STATS
        self.maxmass = self.mass_rank[:,0] # this is where you can toggle through frames! the second most massive and so on
        self.max_split = grab_mass_ind(self.maxmass, self.Nsat_perhost, self.Nsat_index, 0)

        self.secmass = self.mass_rank[:,1] #2nd most massive
        self.sec_split = grab_mass_ind(self.secmass, self.Nsat_perhost, self.Nsat_index, 0)

        self.thirmass = self.mass_rank[:,2] #3rd most massive
        self.thir_split = grab_mass_ind(self.thirmass, self.Nsat_perhost, self.Nsat_index, 0)

        self.totmass = np.log10(np.nansum(10**self.mass_rank, axis=1)) #total mass
        self.tot_split = grab_mass_ind(self.totmass, self.Nsat_perhost, self.Nsat_index, 0)

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
## Nadler stats
##### ------------------------------------------------------------------------


def lnprob_i(N_real, n_i, sum_j):
    N_ratio = (N_real+1)/N_real
    fac1 = np.log(N_ratio)
    fac2 = np.log(N_real+1)
    return -(sum_j+1)*fac1 - n_i*fac2 + loggamma(sum_j+n_i+1) - loggamma(n_i+1) - loggamma(sum_j+1) # the version written in log space

def lnL_Nadler(data, model):
    N_real, N_bins = model.stat.stack_mat.shape[0], model.stat.stack_mat.shape[1]
    lnProb = []
    for i_bin in range(N_bins):
        n_obs = data.stat.stack[i_bin]
        n_model = model.stat.stack_mat[:, i_bin]
        n_model_sum = n_model.sum()
        lnProb.append(lnprob_i(N_real, n_obs, n_model_sum))
    return np.sum(lnProb)


class SatStats_D_NADLER:

    def __init__(self, lgMs, min_mass, max_mass, N_bin):
        self.lgMs = lgMs
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.N_bin = N_bin

        self.bins = np.linspace(self.min_mass, self.max_mass, self.N_bin)
        self.self.logMvir_bincenters = (self.bins[:-1] + self.bins[1:]) / 2
        self.count_mat = np.apply_along_axis(count, 1, self.lgMs, mass_bins=self.bins)
        self.stack = np.sum(self.count_mat, axis=0)

    def SMF_plot(self):
        plt.figure(figsize=(6,6))
        plt.step(self.self.logMvir_bincenters, self.stack, color="grey", where="mid")
        plt.xlabel("stellar mass")
        plt.ylabel("stacked N")
        plt.yscale("log")
        plt.show() 


class SatStats_M_NADLER:

    def __init__(self, lgMs_mat, min_mass, max_mass, N_bin):
        self.lgMs_mat = lgMs_mat
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.N_bin = N_bin

        self.bins = np.linspace(self.min_mass, self.max_mass, self.N_bin)
        self.self.logMvir_bincenters = (self.bins[:-1] + self.bins[1:]) / 2
        self.N_real = self.lgMs_mat.shape[0]
        self.stack_mat = np.zeros(shape=(self.N_real, self.N_bin-1))

        for i, realization in enumerate(self.lgMs_mat):
            self.count_mat_i = np.apply_along_axis(count, 1, realization, mass_bins=self.bins)
            self.stack_mat[i] = np.sum(self.count_mat_i, axis=0)

    def SMF_plot(self):
        plt.figure(figsize=(6,6))
        for stack in self.stack_mat:
            plt.step(self.self.logMvir_bincenters, stack, color="grey", alpha=0.2, where="mid")
        plt.xlabel("stellar mass")
        plt.ylabel("stacked N")
        plt.yscale("log")
        plt.show()  

# def prob_i(N_real, n_i, sum_j):
#     return ((N_real+1)/(N_real))**(-(sum_j+1)) * (N_real+1)**n_i * (factorial(sum_j+n_i) / (factorial(n_i) * factorial(sum_j))) # a version written with factorials

# def prob_i(N_real, n_i, sum_j):
#     return ((N_real+1)/(N_real))**(-(sum_j+1)) * (N_real+1)**n_i * (gamma(sum_j+n_i+1) / (gamma(n_i+1) * gamma(sum_j+1))) # a version written in linear space


##### ------------------------------------------------------------------------
##### ------------------------------------------------------------------------
##### ------------------------------------------------------------------------
##### ------------------------------------------------------------------------
## ABUNDANCE MEASUREMENTS FOR PAPER 3
##### ------------------------------------------------------------------------
##### ------------------------------------------------------------------------
##### ------------------------------------------------------------------------
##### ------------------------------------------------------------------------

    

class CorrNorm_satgen:

    def __init__(self, dataframe, **kwargs):

        self.df = dataframe

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.normalize()
        self.fit_lines()
        self.plot_bestfit()
        self.plot_fullcorr()

    def normalize(self):

        median_logz50 = []
        median_logNsub = []
        median_logc = []

        std_logz50 = []
        std_logNsub = []
        std_logc = []

        delta_logz50 = []
        delta_logNsub = []
        delta_logc = []

        self.logMvir_bincenters = np.unique(self.df["logMvir"])
        for mass_bin in self.logMvir_bincenters:

            subsample = self.df["logMvir"] == mass_bin

            # ====================
            # a50
            # ====================
            vals = self.df["logz50"][subsample].values
            med_i = np.median(vals)
            std_i = np.std(vals)

            median_logz50.append(med_i)
            std_logz50.append(std_i)

            delta_logz50.extend(vals - med_i)

            # ====================
            # Nsub
            # ====================
            vals = self.df["logNsub"][subsample].values
            med_i = np.median(vals)
            std_i = np.std(vals)

            median_logNsub.append(med_i)
            std_logNsub.append(std_i)

            delta_logNsub.extend(vals - med_i)

            # ====================
            # concentration
            # ====================
            vals = self.df["logc"][subsample].values
            med_i = np.median(vals)
            std_i = np.std(vals)

            median_logc.append(med_i)
            std_logc.append(std_i)

            delta_logc.extend(vals - med_i)

        # ---- store medians ----
        self.logz50_med = np.array(median_logz50)
        self.logNsub_med = np.array(median_logNsub)
        self.logc_med = np.array(median_logc)

        # ---- store stds ----
        self.logz50_std = np.array(std_logz50)
        self.logNsub_std = np.array(std_logNsub)
        self.logc_std = np.array(std_logc)

        # ---- store residuals ----
        self.delta_logz50 = np.array(delta_logz50)
        self.delta_logNsub = np.array(delta_logNsub)
        self.delta_logc = np.array(delta_logc)

    def fit_lines(self):

        # ---- a50 ----
        self.m_logz50, self.b_logz50 = fit_line_sym_errors(self.logMvir_bincenters, self.logz50_med, self.logz50_std, p0=(0.5, 1.0))
        # ---- Nsub ----
        self.m_logNsub, self.b_logNsub = fit_line_sym_errors(self.logMvir_bincenters, self.logNsub_med, self.logNsub_std, p0=(1.0, 1.0))
        # ---- concentration ----
        self.m_logc, self.b_logc = fit_line_sym_errors(self.logMvir_bincenters, self.logc_med, self.logc_std, p0=(1, 1.0))

        self.logMvir_smooth = np.linspace(12.5, 14, 100)
        self.logz50_smooth = self.m_logz50*self.logMvir_smooth+self.b_logz50
        self.logc_smooth = self.m_logc*self.logMvir_smooth+self.b_logc
        self.logNsub_smooth = self.m_logNsub*self.logMvir_smooth+self.b_logNsub

        self.bestfit_mat = np.array([[self.m_logz50, self.b_logz50], [self.m_logc, self.b_logc], [self.m_logNsub, self.b_logNsub]])

    def plot_bestfit(self):

        fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

        ax[0].scatter(self.df["logMvir"], self.df["logz50"], marker=".", s=1, alpha=0.5)
        ax[1].scatter(self.df["logMvir"], self.df["logc"], marker=".", s=1, alpha=0.5)
        ax[2].scatter(self.df["logMvir"], self.df["logNsub"], marker=".", s=1, alpha=0.5)

        ax[0].errorbar(self.logMvir_bincenters, self.logz50_med, yerr=self.logz50_std, fmt="o", color="k")
        ax[1].errorbar(self.logMvir_bincenters, self.logc_med, yerr=self.logc_std, fmt="o", color="k")
        ax[2].errorbar(self.logMvir_bincenters, self.logNsub_med, yerr=self.logNsub_std, fmt="o", color="k")

        ax[0].plot(self.logMvir_smooth, self.logz50_smooth, color="k")
        ax[1].plot(self.logMvir_smooth, self.logc_smooth, color="k")
        ax[2].plot(self.logMvir_smooth, self.logNsub_smooth, color="k")

        ax[0].text(0.75, 0.2, s=f"y = {self.m_logz50:.2f}x {self.b_logz50:.2f}", fontsize=12, transform=ax[0].transAxes, bbox=dict(boxstyle="round", facecolor="white"))
        ax[1].text(0.75, 0.2, s=f"y = {self.m_logc:.2f}x {self.b_logc:.2f}", fontsize=12, transform=ax[1].transAxes, bbox=dict(boxstyle="round", facecolor="white"))
        ax[2].text(0.75, 0.2, s=f"y = {self.m_logNsub:.2f}x {self.b_logNsub:.2f}", fontsize=12, transform=ax[2].transAxes, bbox=dict(boxstyle="round", facecolor="white"))

        ax[0].set_ylabel("log z$_{50}$")
        ax[1].set_ylabel("log c")
        ax[2].set_ylabel("log N$_{\\rm sub}$")

        # ax[0].set_ylim(-0.6, 0)
        ax[1].set_ylim(0, 1.8)
        ax[2].set_ylim(0, 2.8)

        ax[2].set_xlabel("log M$_{\\rm vir}$ [M$_{\\odot}$]")
        ax[0].set_title(self.dataset_title)
        plt.tight_layout()
        plt.show()

    def plot_fullcorr(self):

        fig, ax = plt.subplots(1, 2, figsize=(8, 6), sharey=True)

        # ===============================
        # ===============================

        ax[0].set_xlabel(r"$\Delta [\log z_{50}]$")
        ax[0].set_ylabel(r"$\Delta [\log N_{\rm sub}]$")

        ax[0].axhline(0, ls="--", color="k", zorder=11)
        ax[0].axvline(0, ls="--", color="k", zorder=11)

        qs_a50, rho_a50, pval_a50 = quadrant_percentages_plot(
            self.delta_logz50,
            self.delta_logNsub)

        sm0 = ax[0].scatter(
            self.delta_logz50,
            self.delta_logNsub,
            c=self.df["logMvir"],
            marker=".")

        # Quadrant labels
        ax[0].text(0.85, 0.95, qs_a50[0], fontsize=12,
                transform=ax[0].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))
        ax[0].text(0.85, 0.03, qs_a50[1], fontsize=12,
                transform=ax[0].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))
        ax[0].text(0.02, 0.03, qs_a50[2], fontsize=12,
                transform=ax[0].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))
        ax[0].text(0.02, 0.95, qs_a50[3], fontsize=12,
                transform=ax[0].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))

        ax[0].set_title(rf"$\rho_S = {rho_a50:.2f}$")

        # ===============================
        # ===============================

        ax[1].set_xlabel(r"$\Delta [\log c]$")
        ax[1].axhline(0, ls="--", color="k", zorder=11)
        ax[1].axvline(0, ls="--", color="k", zorder=11)

        qs_c, rho_c, pval_c = quadrant_percentages_plot(
            self.delta_logc,
            self.delta_logNsub)

        sm1 = ax[1].scatter(
            self.delta_logc,
            self.delta_logNsub,
            c=self.df["logMvir"],
            marker=".")

        # Quadrant labels
        ax[1].text(0.85, 0.95, qs_c[0], fontsize=12,
                transform=ax[1].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))
        ax[1].text(0.85, 0.03, qs_c[1], fontsize=12,
                transform=ax[1].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))
        ax[1].text(0.02, 0.03, qs_c[2], fontsize=12,
                transform=ax[1].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))
        ax[1].text(0.02, 0.95, qs_c[3], fontsize=12,
                transform=ax[1].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))

        ax[1].set_title(rf"$\rho_S = {rho_c:.2f}$")

        # ===============================
        # Shared colorbar
        # ===============================

        cbar = fig.colorbar(
            sm1,
            ax=ax,
            orientation="horizontal",
            pad=0.2,
            fraction=0.05
        )

        cbar.set_label(r"log M$_{\rm vir}$ [M$_{\odot}$]")
        # ax[1].set_ylim(-1, 0.75)

        fig.suptitle(self.dataset_title)
        fig.tight_layout(rect=[0, 0.25, 1, 0.95])
        plt.show()

    def write_summary_tab(self, filepath):

        self.df = pd.DataFrame({"logMvir": self.df["logMvir"], "logfsub": self.df["logfsub"],
                           "logz50": self.df["logz50"], "delta_logz50": self.delta_logz50,
                           "logc": self.df["logc"], "delta_logc": self.delta_logc,
                           "logNsub": self.df["logNsub"], "delta_logNsub": self.delta_logNsub})
        
        self.df.to_csv(filepath+self.dataset_title+".csv", index=False)
        np.save(filepath+self.dataset_title+".npy", self.bestfit_mat)



class CorrNorm_simulations:

    def __init__(self, logMvir, logz50, logc, logNsub, logfsub, **kwargs):

        self.logMvir = logMvir
        self.logz50 = logz50
        self.logc = logc
        self.logNsub = logNsub
        self.logfsub = logfsub

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.bin_data()
        self.fit_lines()
        self.normalize()
        self.plot_bestfit()
        self.plot_fullcorr()

    def bin_data(self):

        self.logMvir_smooth = np.linspace(12.5, 14, 100)
        self.logMvir_bins = np.arange(12.5, 14.1, 0.1)

        # ---- a50 ----
        self.logz50_med, self.logz50_std = finite_binned_stat(self.logMvir, self.logz50, self.logMvir_bins)
        # ---- Nsub (handles -inf safely) ----
        self.logNsub_med, self.logNsub_std = finite_binned_stat(self.logMvir, self.logNsub, self.logMvir_bins)
        # ---- concentration ----
        self.logc_med, self.logc_std = finite_binned_stat(self.logMvir, self.logc, self.logMvir_bins)
        # bin centers
        self.logMvir_bincenters = 0.5 * (self.logMvir_bins[:-1] + self.logMvir_bins[1:])

    def fit_lines(self):

        # ---- a50 ----
        self.m_logz50, self.b_logz50 = fit_line_sym_errors(self.logMvir_bincenters, self.logz50_med, self.logz50_std, p0=(0.5, 1.0))
        # ---- Nsub ----
        self.m_logNsub, self.b_logNsub = fit_line_sym_errors(self.logMvir_bincenters, self.logNsub_med, self.logNsub_std, p0=(1.0, 1.0))
        # ---- concentration ----
        self.m_logc, self.b_logc = fit_line_sym_errors(self.logMvir_bincenters, self.logc_med, self.logc_std, p0=(1, 1.0))
        
        self.bestfit_mat = np.array([[self.m_logz50, self.b_logz50], [self.m_logc, self.b_logc], [self.m_logNsub, self.b_logNsub]])

    def normalize(self):

        self.logz50_smooth = self.m_logz50*self.logMvir_smooth+self.b_logz50
        self.logc_smooth = self.m_logc*self.logMvir_smooth+self.b_logc
        self.logNsub_smooth = self.m_logNsub*self.logMvir_smooth+self.b_logNsub

        self.logz50_full = self.m_logz50*self.logMvir+self.b_logz50
        self.logc_full = self.m_logc*self.logMvir+self.b_logc
        self.logNsub_full = self.m_logNsub*self.logMvir+self.b_logNsub

        self.delta_logz50 = self.logz50 - self.logz50_full
        self.delta_logc = self.logc - self.logc_full
        self.delta_logNsub = self.logNsub - self.logNsub_full

    def plot_bestfit(self):

        fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

        ax[0].scatter(self.logMvir, self.logz50, marker=".", s=1, alpha=0.5)
        ax[1].scatter(self.logMvir, self.logc, marker=".", s=1, alpha=0.5)
        ax[2].scatter(self.logMvir, self.logNsub, marker=".", s=1, alpha=0.5)

        ax[0].errorbar(self.logMvir_bincenters, self.logz50_med, yerr=self.logz50_std, fmt="o", color="k")
        ax[1].errorbar(self.logMvir_bincenters, self.logc_med, yerr=self.logc_std, fmt="o", color="k")
        ax[2].errorbar(self.logMvir_bincenters, self.logNsub_med, yerr=self.logNsub_std, fmt="o", color="k")

        ax[0].plot(self.logMvir_smooth, self.logz50_smooth, color="k")
        ax[1].plot(self.logMvir_smooth, self.logc_smooth, color="k")
        ax[2].plot(self.logMvir_smooth, self.logNsub_smooth, color="k")

        ax[0].text(0.75, 0.2, s=f"y = {self.m_logz50:.2f}x {self.b_logz50:.2f}", fontsize=12, transform=ax[0].transAxes, bbox=dict(boxstyle="round", facecolor="white"))
        ax[1].text(0.75, 0.2, s=f"y = {self.m_logc:.2f}x {self.b_logc:.2f}", fontsize=12, transform=ax[1].transAxes, bbox=dict(boxstyle="round", facecolor="white"))
        ax[2].text(0.75, 0.2, s=f"y = {self.m_logNsub:.2f}x {self.b_logNsub:.2f}", fontsize=12, transform=ax[2].transAxes, bbox=dict(boxstyle="round", facecolor="white"))

        ax[0].set_ylabel("log z$_{50}$")
        ax[1].set_ylabel("log c")
        ax[2].set_ylabel("log N$_{\\rm sub}$")

        ax[1].set_ylim(0, 1.8)
        ax[2].set_ylim(0, 2.8)

        ax[2].set_xlabel("log M$_{\\rm vir}$ [M$_{\\odot}$]")
        ax[0].set_title(self.dataset_title)
        plt.tight_layout()
        plt.show()

    def plot_fullcorr(self):

        fig, ax = plt.subplots(1, 2, figsize=(8, 6), sharey=True)

        # ===============================
        # ===============================

        ax[0].set_xlabel(r"$\Delta [\log z_{50}]$")
        ax[0].set_ylabel(r"$\Delta [\log N_{\rm sub}]$")

        ax[0].axhline(0, ls="--", color="k", zorder=11)
        ax[0].axvline(0, ls="--", color="k", zorder=11)

        qs_a50, rho_a50, pval_a50 = quadrant_percentages_plot(
            self.delta_logz50,
            self.delta_logNsub)

        sm0 = ax[0].scatter(
            self.delta_logz50,
            self.delta_logNsub,
            c=self.logMvir,
            marker=".")

        # Quadrant labels
        ax[0].text(0.85, 0.95, qs_a50[0], fontsize=12,
                transform=ax[0].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))
        ax[0].text(0.85, 0.03, qs_a50[1], fontsize=12,
                transform=ax[0].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))
        ax[0].text(0.02, 0.03, qs_a50[2], fontsize=12,
                transform=ax[0].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))
        ax[0].text(0.02, 0.95, qs_a50[3], fontsize=12,
                transform=ax[0].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))

        ax[0].set_title(rf"$\rho_S = {rho_a50:.2f}$")

        # ===============================
        # ===============================

        ax[1].set_xlabel(r"$\Delta [\log c]$")
        ax[1].axhline(0, ls="--", color="k", zorder=11)
        ax[1].axvline(0, ls="--", color="k", zorder=11)

        qs_c, rho_c, pval_c = quadrant_percentages_plot(
            self.delta_logc,
            self.delta_logNsub)

        sm1 = ax[1].scatter(
            self.delta_logc,
            self.delta_logNsub,
            c=self.logMvir,
            marker=".")

        # Quadrant labels
        ax[1].text(0.85, 0.95, qs_c[0], fontsize=12,
                transform=ax[1].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))
        ax[1].text(0.85, 0.03, qs_c[1], fontsize=12,
                transform=ax[1].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))
        ax[1].text(0.02, 0.03, qs_c[2], fontsize=12,
                transform=ax[1].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))
        ax[1].text(0.02, 0.95, qs_c[3], fontsize=12,
                transform=ax[1].transAxes,
                bbox=dict(boxstyle="round", facecolor="white"))

        ax[1].set_title(rf"$\rho_S = {rho_c:.2f}$")

        # ===============================
        # Shared colorbar
        # ===============================

        cbar = fig.colorbar(
            sm1,
            ax=ax,
            orientation="horizontal",
            pad=0.2,
            fraction=0.05
        )

        cbar.set_label(r"log M$_{\rm vir}$ [M$_{\odot}$]")
        ax[1].set_ylim(-1, 0.75)

        fig.suptitle(self.dataset_title)
        fig.tight_layout(rect=[0, 0.25, 1, 0.95])
        plt.show()

    def write_summary_tab(self, filepath):

        df = pd.DataFrame({"logMvir": self.logMvir, "logfsub": self.logfsub,
                           "logz50": self.logz50, "delta_logz50": self.delta_logz50,
                           "logc": self.logc, "delta_logc": self.delta_logc,
                           "logNsub": self.logNsub, "delta_logNsub": self.delta_logNsub})
        
        df.to_csv(filepath+self.dataset_title+".csv", index=False)
        np.save(filepath+self.dataset_title+".npy", self.bestfit_mat)


def make_summary_rho(rho_mat, labels, xlabel, ylabel):

    k_tot = rho_mat.shape[0]
    logMvir_binned = np.linspace(12.6, 14, 8)

    # get default color cycle
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # repeat cycle if k_tot exceeds default length
    colorz = [default_colors[i % len(default_colors)] for i in range(k_tot)]

    fig, ax = plt.subplots(figsize=(8, 5))
    
    for k, rho_arr in enumerate(rho_mat):
        ax.plot(logMvir_binned, rho_arr, label=labels[k], c=colorz[k], marker=".")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, ls=":", color="grey")
    ax.set_ylim(-1, 1)
    ax.set_xlim(12.6, 14.0)

    ax.legend()
    plt.show()

def make_bestfit_plot(datasets, labels):

    k_tot = len(datasets)
    logMvir_smooth = np.linspace(12.6, 14)

    # get default color cycle
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # repeat cycle if k_tot exceeds default length
    colorz = [default_colors[i % len(default_colors)] for i in range(k_tot)]

    fig, ax = plt.subplots(figsize=(8, 5))
    

    for k, data in enumerate(datasets):
        m, b = data[2, 0], data[2, 1]
        ax.plot(logMvir_smooth, m*logMvir_smooth + b, label=labels[k], c=colorz[k])

    ax.set_xlabel("$\\log \\rm M_{vir}$")
    ax.set_ylabel("$\\log \\rm N_{sub}$")
    ax.set_xlim(12.6, 14.0)
    ax.legend()
    plt.show()


def make_binned_plot(datasets, xkey, ykey, xlabel, ylabel, plot_origin=False):

    lgM_min = [12.5, 12.7, 12.9, 13.1, 13.3, 13.5, 13.7, 13.9]
    lgM_max = [12.7, 12.9, 13.1, 13.3, 13.5, 13.7, 13.9, 14.1]

    k_tot = len(datasets)
    n_bins = len(lgM_min)

    # get default color cycle
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # repeat cycle if k_tot exceeds default length
    colorz = [default_colors[i % len(default_colors)] for i in range(k_tot)]

    # ---------------------------------------
    # Storage for correlation coefficients
    # ---------------------------------------
    rho_array = np.zeros((k_tot, n_bins))

    fig, axes = plt.subplots(
        k_tot, n_bins,
        figsize=(20, 10),
        sharey=True,
        sharex=True
    )

    # Column titles
    for i in range(n_bins):
        label = rf"$10^{{{lgM_min[i]:.1f}}} < M_{{\rm h}} \leq 10^{{{lgM_max[i]:.1f}}}$"
        axes[0, i].set_title(label)

    # Main plotting loop
    for k, dataset in enumerate(datasets):

        for i in range(n_bins):

            subsample = (
                (dataset["logMvir"] > lgM_min[i]) &
                (dataset["logMvir"] <= lgM_max[i])
            )

            x = dataset[xkey][subsample]
            y = dataset[ykey][subsample]

            N = len(x)

            axes[k, i].scatter(
                x,
                y,
                marker=".",
                s=1,
                alpha=1,
                color=colorz[k]
            )

            if plot_origin:
                axes[k, i].axhline(0, ls="--", color="k")
                axes[k, i].axvline(0, ls="--", color="k")

            qs, rho, p_val = quadrant_percentages_plot(x, y)

            # ---------------------------------------
            # SAVE rho here
            # ---------------------------------------
            rho_array[k, i] = float(rho)

            axes[k, i].text(0.7, 0.9,  qs[0], fontsize=10, transform=axes[k, i].transAxes)
            axes[k, i].text(0.7, 0.05, qs[1], fontsize=10, transform=axes[k, i].transAxes)
            axes[k, i].text(0.1, 0.05, qs[2], fontsize=10, transform=axes[k, i].transAxes)
            axes[k, i].text(0.1, 0.9,  qs[3], fontsize=10, transform=axes[k, i].transAxes)

            rho_label = "$\\rho_S$=" + f"{rho:.2f}" + f"\nN={N}"

            axes[k, i].text(
                0.7, 0.2,
                rho_label,
                fontsize=11,
                color="red" if p_val < 0.05 else "grey",
                transform=axes[k, i].transAxes,
                bbox=dict(boxstyle="round", facecolor="white")
            )

        axes[k, 0].set_ylabel(ylabel)

    for i in range(n_bins):
        axes[-1, i].set_xlabel(xlabel)

    plt.tight_layout()

    return rho_array