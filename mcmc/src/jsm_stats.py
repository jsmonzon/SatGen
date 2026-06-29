import numpy as np
import matplotlib.pyplot as plt
from numpy.random import poisson
from scipy.stats import ks_2samp
from scipy.special import gamma, loggamma, factorial
from scipy import stats
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.stats import binned_statistic, poisson
from scipy.optimize import minimize
import pandas as pd


##### ------------------------------------------------------------------------
## Our stats
##### ------------------------------------------------------------------------

def countzero(array):
    return np.sum(array == 0.0) / array.shape[0]

def nan_mask(array):
    return array[~np.isnan(array)]

def pdf(data, max):
    index, counts = np.unique(data, return_counts=True)
    full = np.zeros(max) # the max number of unique counts across the models
    # needs to be set sufficiently high such that even extreme models can populate the Pnsat matrix
    full[index.astype("int")] = counts/data.shape[0]
    return full
    
def correlation(xdat, ydat):
    mask = np.isfinite(xdat) & np.isfinite(ydat)
    return stats.spearmanr(xdat[mask], ydat[mask])[0]

def correlation_with_p(xdat, ydat):
    mask = np.isfinite(xdat) & np.isfinite(ydat)
    return stats.spearmanr(xdat[mask], ydat[mask])

def poisson_approx(Nsub_arr):

    bins = np.arange(Nsub_arr.min() - 0.5, Nsub_arr.max() + 1.5, 1)
    k = np.arange(Nsub_arr.min(), Nsub_arr.max() + 1)
    pdf = poisson.pmf(k, np.mean(Nsub_arr))

    return np.array([k, pdf]), bins

def jackknife_correlation(xdat, ydat, n_jack=10, method='spearman'):
    """
    Estimate uncertainty in Spearman or Pearson correlation via jackknife resampling.
    Randomly removes 1/n_jack of the data each time and recomputes the correlation.

    Parameters:
        xdat, ydat : array-like
        n_jack     : number of jackknife subsets
        method     : 'spearman' (rank-based) or 'pearson' (linear)

    Returns:
        rho     : correlation on full sample
        rho_err : jackknife uncertainty estimate
        p_val   : p-value from full-sample correlation
    """
    method = method.lower()
    if method not in ('spearman', 'pearson'):
        raise ValueError(f"method must be 'spearman' or 'pearson', got '{method}'")

    def correlate(x, y):
        if method == 'spearman':
            return stats.spearmanr(x, y)
        else:
            return stats.pearsonr(x, y)

    x = np.asarray(xdat)
    y = np.asarray(ydat)
    N = len(x)

    # full sample correlation
    rho, p_val = correlate(x, y)

    # shuffle indices and split into n_jack subsets
    indices = np.random.permutation(N)
    subsets = np.array_split(indices, n_jack)

    rho_jack = np.zeros(n_jack)
    for i, subset in enumerate(subsets):
        jack_mask = np.ones(N, dtype=bool)
        jack_mask[subset] = False
        rho_jack[i], _ = correlate(x[jack_mask], y[jack_mask])

    # jackknife error estimate
    rho_err = np.sqrt((n_jack - 1) / n_jack * np.sum((rho_jack - rho_jack.mean()) ** 2))

    return rho, rho_err, p_val

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


##### ------------------------------------------------------------------------
## MISC TOOLS
##### ------------------------------------------------------------------------


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

    rho, pval = correlation_with_p(x, y)

    return txt, rho, pval

def finite_binned_stat(x, y, bins):

    mean, _, _ = binned_statistic(
        x,y,
        statistic='mean',
        bins=bins)

    std, _, _ = binned_statistic(
        x,y,
        statistic='std',
        bins=bins)

    count, _, _ = binned_statistic(
        x,y,
        statistic=countzero,
        bins=bins)

    return mean, std, count

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
## PAPER 1 stats
##### ------------------------------------------------------------------------

    
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


# def prob_i(N_real, n_i, sum_j):
#     return ((N_real+1)/(N_real))**(-(sum_j+1)) * (N_real+1)**n_i * (factorial(sum_j+n_i) / (factorial(n_i) * factorial(sum_j))) # a version written with factorials

# def prob_i(N_real, n_i, sum_j):
#     return ((N_real+1)/(N_real))**(-(sum_j+1)) * (N_real+1)**n_i * (gamma(sum_j+n_i+1) / (gamma(n_i+1) * gamma(sum_j+1))) # a version written in linear space



