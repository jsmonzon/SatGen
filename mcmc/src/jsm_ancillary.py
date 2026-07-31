import numpy as np
import warnings; warnings.simplefilter('ignore')
import sys
import h5py
import pandas as pd
import os
import json
import jsm_stats
# Get the absolute path to config.json relative to this file
config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.json")
config_path = os.path.abspath(config_path)

# Load the config
with open(config_path, "r") as f:
    config_file = json.load(f)

# Use config values
location = config_file["location"]

if location == "server":
    parentdir = "/home/jsm99/SatGen/src/"

elif location == "local":
    parentdir = "/Users/jsmonzon/Research/SatGen/src/"

sys.path.insert(0, parentdir)
import profiles as profiles
import galhalo as gh
import evolve as ev
import astropy.units as u
import cosmo as co
import astropy.constants as const
import astropy.coordinates as crd
from treelib import Node, Tree
from scipy.optimize import brentq

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
### For the concentration measurments
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 1. Analytic profiles
# --------------------------------------------------------------------------
 
def mu(x):
    """NFW enclosed-mass shape function: M(<x*rs) propto mu(x)."""
    return np.log1p(x) - x / (1.0 + x)
 
 
def nfw_rho0(Mvir, rvir, c):
    """Characteristic density normalization from (Mvir, rvir, c)."""
    rs = rvir / c
    return Mvir / (4.0 * np.pi * rs**3 * mu(c))
 
 
def nfw_density(r, rho0, rs):
    x = r / rs
    return rho0 / (x * (1.0 + x) ** 2)
 

def plummer_density(r, a):
    """
    Plummer density profile, normalized to total mass = 1.
    Multiply by M (total mass) to get physical density.

        rho(r) = (3 / 4*pi*a^3) * (1 + (r/a)^2)^(-5/2)

    r : radius (array or scalar)
    a : Plummer scale radius
    """
    return (3.0 / (4.0 * np.pi * a**3)) * (1.0 + (r / a) ** 2) ** (-2.5)

# --------------------------------------------------------------------------
# 2. Sample particle radii from the NFW CDF
# --------------------------------------------------------------------------
 
def sample_nfw_radii(N, rvir, c, rng, x_min=1e-6, n_grid=20000):
    """
    Inverse-transform sampling of r via the NFW enclosed-mass profile.
    Vectorized: builds mu(x) on a log grid once, then inverts via
    interpolation for all N draws at once (fast even for N ~ 1e7).
    """
    rs = rvir / c
    x_grid = np.logspace(np.log10(x_min), np.log10(c), n_grid)
    mu_grid = mu(x_grid)
    mu_grid /= mu_grid[-1]  # normalize enclosed-mass fraction to [0, 1] at x=c
 
    u = rng.random(N)
    x = np.interp(u, mu_grid, x_grid)
    return x * rs

def sample_plummer_radii(N, a, rng):
    """
    Inverse-transform sampling of radii from a Plummer sphere of scale
    radius `a`, centered at the origin.

    CDF:  F(x) = x^3 / (1+x^2)^(3/2),  x = r/a
    Inverted analytically (no grid/interp needed, unlike NFW):
        x = sqrt(u^(2/3) / (1 - u^(2/3)))
    """
    u = rng.random(N)
    u23 = u ** (2.0 / 3.0)
    x = np.sqrt(u23 / (1.0 - u23))
    return x * a
 
 
def sample_isotropic_positions(r, rng):
    """Give each radius r_i a random direction -> 3D positions."""
    costheta = rng.uniform(-1.0, 1.0, size=r.size)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=r.size)
    sintheta = np.sqrt(1.0 - costheta**2)
    x = r * sintheta * np.cos(phi)
    y = r * sintheta * np.sin(phi)
    z = r * costheta
    return np.column_stack([x, y, z]).T



def measure_vmax(pos, Rvir, Nparticles, center=None, r_cut=None, plot=False):
    """
    Compute Vmax and Vvir from the enclosed mass profile of particles.

    Parameters
    ----------
    pos : ndarray, shape (3, N)
        Particle positions.
    mp : float
        Particle mass (assumed equal-mass particles).
    Mvir : float
        Virial mass of the halo.
    rvir : float
        Virial radius of the halo.
    center : ndarray, shape (3,), optional
        Center to measure radii from. Defaults to the origin.
    r_cut : float, optional
        Only include particles with r < r_cut in the enclosed mass profile.
        Defaults to rvir.
    plot : bool, optional
        If True, plot the circular velocity profile Vc(r), marking Vmax/rmax
        and Vvir/rvir.

    Returns
    -------
    Vmax, rmax, Vvir
    """
    G = 1.0
    Mvir = 1
    mp = 1/Nparticles

    if center is None:
        center = np.zeros(3)
    if r_cut is None:
        r_cut = Rvir

    r = np.linalg.norm(pos - center[:, None], axis=0)

    mask = r < r_cut
    rsort = np.sort(r[mask])

    Menc = mp * np.arange(1, len(rsort) + 1)
    min_particles = 10
    Vc = np.sqrt(G * Menc / rsort)
    Vc[:min_particles] = -np.inf  # exclude spuriously noisy inner points from argmax
    imax = np.argmax(Vc)

    Vmax = Vc[imax]
    rmax = rsort[imax]
    Vvir = np.sqrt(G * Mvir / Rvir)

    """
    Recover NFW concentration from the Klypin Vmax/Vvir relation:
    (Vmax/Vvir)^2 = 0.216*c/mu(c).
    """
    y = (Vmax / Vvir)**2
    func = lambda c: 0.216 * c / mu(c) - y
    f1, f2 = func(1), func(1000)
    if f1 * f2 > 0:
        print(f"y={y:.3e} out of bracket range: f(1)={f1:.3e}, f(1000)={f2:.3e}")
        return np.nan  # or handle however makes sense for your pipeline
    concentration = brentq(func, 1, 1000)

    return np.array([concentration, Vmax, rmax, Vvir])

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
### MISC TOOLS
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def measure_mass_frac(tree, mask_list):

    if len(mask_list) > 1:
        final_bool = np.logical_and.reduce(mask_list)
    else:
        final_bool = mask_list[0]

    selected_masses = tree.mass[1:, 0][final_bool]

    Nsub = len(selected_masses)

    if Nsub == 0:
        return 0, 0.0, 0.0

    fsub = np.sum(selected_masses) / tree.mass[0, 0]
    MMs = np.max(selected_masses) / tree.mass[0, 0]

    return Nsub, fsub, MMs

def ave_dmdt(m, M, tau_dyn):
    #Jiang and vdBosch et al 2016
    return -0.81 * (m / tau_dyn) * (m / M)**0.04


def rmax_evo_aPlum(tree, subhalo_ind):

    profile_i = profiles.Green(
        tree.acc_mass[subhalo_ind],
        tree.acc_concentration[subhalo_ind],
        z=tree.acc_redshift[subhalo_ind]
    )  # at accretion

    profile_i.update_mass_jsm(tree.mass[subhalo_ind, 0] / tree.acc_mass[subhalo_ind])  # at z=0

    a_plum = profile_i.rmax / np.sqrt(2)

    return a_plum

def FUNC_ave_mass_loss(tree, subhalo_ind):

    m_evo = np.full(tree.redshift.shape[0], np.nan)

    zsample = tree.redshift
    tsample = tree.CosmicTime
    host_mass = tree.mass[0]

    # z_acc = tree.acc_index[sub_ind]
    z_acc = tree.proper_acc_index[subhalo_ind]

    for z_ind in range(len(zsample)-1, -1, -1):

        z = zsample[z_ind]

        if z_ind > z_acc: #not yet accreted
            continue

        elif z_ind == z_acc: #accreted
            m_evo[z_ind] = tree.acc_mass[subhalo_ind]

        else: #losing mass!
            m_prev = m_evo[z_ind+1]
            if np.isnan(m_prev):
                continue
            dt = tsample[z_ind] - tsample[z_ind+1]
            # m_lost = ave_dmdt(m_prev, host_mass[z_ind], tree.host_profiles[z_ind].tdyn(tree.VirialRadius[0, z_ind]))
            m_lost = ave_dmdt(m_prev, host_mass[z_ind], co.tdyn(z))
            m_evo[z_ind] = m_prev + m_lost * dt

    return m_evo


def FUNC_halo_mass_evo(tree, subhalo_ind):

    #based on Green et al 2019 fitting code using the transfer function
    rmax = tree.acc_rmax[subhalo_ind] * ev.GvdB_19(tree.fb[subhalo_ind], tree.acc_concentration[subhalo_ind], track_type="rad") #Green et al 2019 transfer function tidal track
    Vmax = tree.acc_Vmax[subhalo_ind] * ev.GvdB_19(tree.fb[subhalo_ind], tree.acc_concentration[subhalo_ind], track_type="vel") #Green et al 2019 transfer function tidal track

    return rmax, Vmax

def FUNC_stellar_mass_evo(tree, subhalo_ind):

    #based on Errani et al 2021 fitting code

    R50_by_rmax = tree.acc_R50[subhalo_ind]/tree.acc_rmax[subhalo_ind]
    R50_fb, stellarmass_fb = ev.g_EPW18(tree.fb[subhalo_ind], alpha=1.0, lefflmax=R50_by_rmax) #Errani 2018 tidal tracks for stellar evolution!
    R50 = tree.acc_R50[subhalo_ind]*R50_fb #scale the sizes!
    stellarmass = tree.acc_stellarmass[subhalo_ind]*stellarmass_fb #scale the masses!
    
    return R50, stellarmass
    
def FUNC_in_situ_SFR(tree):

    tree.zmask = ~np.isnan(tree.VirialRadius[0]) 
    tree.Rvir = tree.VirialRadius[0][tree.zmask][::-1] # need to flip so the integration works!
    tree.Mhalo = tree.mass[0][tree.zmask][::-1]
    tree.Vmaxhalo = tree.host_Vmax[tree.zmask][::-1]
    tree.zhalo = tree.redshift[tree.zmask][::-1]
    tree.thalo = tree.CosmicTime[tree.zmask][::-1]

    tree.t_dyn = gh.dynamical_time(tree.Rvir*u.kpc, tree.Mhalo*u.solMass).to(u.Gyr).value
    tree.SFR  = gh.SFR_B19(tree.Vmaxhalo, tree.zhalo)
    tree.Mstar, tree.f_lost = gh.integrate_SFH(tree.SFR, tree.thalo)

    padding = np.zeros(shape=(tree.CosmicTime.shape[0] - tree.Mstar.shape[0],))  # Create the padding array
    return np.concatenate((padding, tree.Mstar))[::-1]

def artificial_disruption(m_acc, c_acc,
         A=3.08, B=-3.26, C=-8.89,
         D=0.38, E=-0.51, F=0.40,
         size=None, return_scale=False, return_draws=False):

    def f(x):
        return np.log(1 + x) - x / (1 + x)

    log_m = np.log10(m_acc)
    mu = np.empty_like(log_m, dtype=float)
    sigma = np.empty_like(log_m, dtype=float)

    mask = log_m < 10

    # mu[mask] = 0.6579463705605844
    sigma[mask] = 0.21760472162764027

    mu = A + B * (1 + (log_m + C)**(-2))**(-0.5)
    #mu[~mask] = A + B * (1 + (log_m[~mask] + C)**(-2))**(-0.5)
    sigma[~mask] = D + E * mu[~mask] + F * mu[~mask]**2

    f_dis_draw = np.random.lognormal(mean=mu, sigma=sigma, size=size)

    scale = f(f_dis_draw) / f(c_acc)
    m_dis = m_acc * scale

    if return_scale:
        return scale
    if return_draws:
        return np.log10(f_dis_draw)
    return m_dis


def MMP(self):

    sat_id = np.argmax(self.acc_stellarmass[1:]) + 1 #the most massive satellite accreted (not the most massive subhalo!)

    str_to_int = {"merged": 0, "surviving": 1, "disrupted": 2}
    fate = np.vectorize(str_to_int.get)(self.subhalo_fates[sat_id])

    properties = [self.acc_mass[sat_id], self.acc_stellarmass[sat_id], #accretion masses
                self.final_mass[sat_id], self.final_stellarmass[sat_id], #final masses  
                float(fate), self.proper_acc_redshift[sat_id], #fate and acc redshift    
                np.nanmin(self.rmags[sat_id]), np.nanmin(self.Vmags[sat_id])] #min R, V
    
    return properties

def N90_cont(tree):

    mass_sorted = np.argsort(tree.contributed)[::-1] # sort the contributions to the stellar halo
    perc_sorted = tree.contributed[mass_sorted]/tree.total_ICL #measure the percentage
    perc_cm = np.cumsum(perc_sorted) #cumulaitve sum to find where the rank hits 90
    N90_rank = np.argmin(perc_cm < 0.9) #where does the rank hit 90
    if N90_rank == 0:
        N90_ids = mass_sorted[0:1] # the subhalos that contribute to that rank
    else:
        N90_ids = mass_sorted[0:N90_rank] # the subhalos that contribute to that rank
    fates = tree.subhalo_fates[N90_ids]
    str_to_int = {"merged": 0, "surviving": 1, "disrupted": 2}
    fates = np.vectorize(str_to_int.get)(fates)
    return N90_ids, perc_cm, fates

def MW_est_criteria(tree):

    # from Nadler et al. 2024
    lower_GSE_index = np.argmin(np.abs(tree.CosmicTime - (13.8-6.5))) #time constraints
    upper_GSE_index = np.argmin(np.abs(tree.CosmicTime - (13.8-11.5)))

    lower_LMC_index = 0
    upper_LMC_index = np.argmin(np.abs(tree.CosmicTime - (13.8-2)))

    mass_ratio_mat = tree.mass / tree.mass[0] #mass ratio!

    potential_GSEs = np.where((lower_GSE_index <= tree.acc_index) & (tree.acc_index <= upper_GSE_index))[0] #everything that was accreted in that window
    GSE_analogs = potential_GSEs[mass_ratio_mat[potential_GSEs, tree.acc_index[potential_GSEs]] >= 1/5]

    potential_LMCs = np.where((lower_LMC_index < tree.acc_index) & (tree.acc_index <= upper_LMC_index))[0]
    LMC_analogs = potential_LMCs[mass_ratio_mat[potential_LMCs, tree.acc_index[potential_LMCs]] >= 1/10]

    host_c = 1 if (7 < tree.concentration[0,0] < 16) else 0 #host concentration!

    GSE = 0
    if GSE_analogs.shape[0] > 0:
        if np.any(tree.acc_order[GSE_analogs] == 1): #first order subhalos!
            GSE = 1

    LMC = 0
    if LMC_analogs.shape[0] > 0:
        if np.any(tree.acc_order[LMC_analogs] == 1):
            LMC = 1

    return np.array([host_c, GSE, LMC])

def fb_surv_frac(tree):
    
    bins = np.linspace(0, 1, 100) # the same as Riley
    # only the surviving subhalos (no fb < 0 already included)
    fb_DM = tree.fb[tree.surviving_subhalos, 0]
    fb_stellar = tree.fb_stellar[tree.surviving_subhalos, 0]

    fraction_DM = jsm_stats.cumulative_fbound(fb_DM, bins)
    fraction_stellar = jsm_stats.cumulative_fbound(fb_stellar, bins)

    return fraction_DM, fraction_stellar

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
### LOADING IN DATA COMPILATIONS
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def select_order(df, k):
    """
    Replace Nsub/fsub/MMs columns with the corresponding k-specific versions.
 
    Note: unlike Nsub (which has a bare k=all column plus Nsub_k1/k2/k3),
    fsub and MMs have no "all" value -- the k=1 values are already written
    out bare (e.g. "fsub" IS "fsub_k1"; there's no "fsub_k1" column in the
    frame). So for k=1, fsub/logfsub/MMs/logMMs are left untouched instead
    of being looked up with a "_k1" suffix that doesn't exist; only Nsub/
    logNsub get swapped in from their "_k1" columns.
 
    Parameters
    ----------
    df : pandas.DataFrame
    k : {1, 2, 3}
        Which k-value to select.
 
    Returns
    -------
    pandas.DataFrame
    """
    if k not in (1, 2, 3):
        raise ValueError(f"k must be 1, 2, or 3, got {k}.")
 
    suffix = f"_k{k}"
 
    # Columns that have k-specific versions
    base_cols = ["Nsub", "logNsub", "fsub", "logfsub", "MMs", "logMMs"]
 
    # fsub/MMs (and their logs) have no "_k1" column -- the bare column
    # already holds the k=1 value, so skip re-mapping those for k=1.
    mass_cols = {"fsub", "logfsub", "MMs", "logMMs"}
 
    new_df = df.copy()
 
    for col in base_cols:
        if k == 1 and col in mass_cols:
            continue
 
        kcol = col + suffix
        if kcol not in df.columns:
            raise KeyError(f"Column '{kcol}' not found.")
        new_df[col] = df[kcol]
 
    # Keep only the unsuffixed columns
    cols_to_drop = [c for c in new_df.columns if "_k" in c]
    new_df = new_df.drop(columns=cols_to_drop)
 
    return new_df

def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx

def make_matrix(dataframe, key): ### should fix this to be 1000 if I am going to compare!

    # Create NxM matrix, padding with NaN
    matrix = np.full((len(dataframe), max(dataframe[key].apply(len))), np.nan)  # Initialize with NaNs

    # Fill the matrix with actual values
    for i, row in enumerate(dataframe[key]):
        matrix[i, :len(row)] = row  # Assign values

    return matrix

def load_sample(filename):
    data = {}
    with h5py.File(filename, "r") as f:   
        for sim_name in f.keys():
            row = {}
            for attr_name in f[sim_name].keys():
                dset = f[sim_name][attr_name]
                if dset.shape == ():  # scalar dataset
                    row[attr_name] = dset[()]  # or dset[()].item()
                else:
                    row[attr_name] = dset[:]
            data[sim_name] = row

    dfh5 = pd.DataFrame.from_dict(data, orient='index')
    return dfh5

def load_massspec(datadir, sub_key, sub_index):

    dfs = []
    for file in os.listdir(datadir):

        if file.endswith("h5"):

            ii = load_sample(datadir + file)
            Nsub = make_matrix(ii, "Nsub_"+sub_key)[:, sub_index]
            fsub = make_matrix(ii, "fsub_"+sub_key)[:, sub_index]
            MMs = make_matrix(ii, "MMs_"+sub_key)[:, sub_index]

            logMvir = np.log10(make_matrix(ii, "MAH")[:, 0])
            logc = np.log10(make_matrix(ii, "host_c")[:, 0])

            df = pd.DataFrame({
                "logMvir":  logMvir,
                "log1pz50": np.log10(1 + ii.host_z50.values),
                "logc":     logc,
                "Nsub":     Nsub,
                "logNsub":  np.log10(Nsub),
                "fsub":     fsub,
                "logfsub":  np.log10(fsub),
                "MMs":      MMs,
                "logMMs":   np.log10(MMs),
            }).replace([np.inf, -np.inf], np.nan)

            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def load_massspec_MW(datadir, sub_key, sub_index, conctype=None):

    dfs = []

    for file in os.listdir(datadir):

        if file.endswith("h5"):

            ii = load_sample(datadir + file)

            Nsub = make_matrix(ii, "N_" + sub_key)[:, sub_index]
            fsub = make_matrix(ii, "f_" + sub_key)[:, sub_index]
            MMs  = make_matrix(ii, "MMs_" + sub_key)[:, sub_index]

            if conctype == "vdb":
                logc = np.log10(ii.host_c_vdb)
            elif conctype == "zhao":
                logc = np.log10(ii.host_c_zhao)
            else:
                logc = np.log10(make_matrix(ii, "host_c")[:, 0])

            df = pd.DataFrame({
                "logMcut": np.log10(float(file.split(("_"))[2][:-3])),
                "log1pz50": np.log10(1 + ii.host_z50.values),
                "logc":     logc,
                "Nsub":     Nsub,
                "logNsub":  np.log10(Nsub),
                "fsub":     fsub,
                "logfsub":  np.log10(fsub),
                "MMs":      MMs,
                "logMMs":   np.log10(MMs),
            }).replace([np.inf, -np.inf], np.nan)

            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def _stack_column(dataframe, key):
    """
    Stacks a per-tree 1D time-series column into an (Ntrees, Ntime) matrix.
    Pads with NaN if array lengths differ across trees (shouldn't happen if
    every tree shares the same time grid, but guards against silently
    misaligned data rather than assuming it).
    """
    arrays = dataframe[key].values
    max_len = max(len(a) for a in arrays)

    matrix = np.full((len(arrays), max_len), np.nan)
    for i, arr in enumerate(arrays):
        matrix[i, :len(arr)] = arr

    return matrix


def load_samples(filename):
    data = {}
    with h5py.File(filename, "r") as f:
        for sim_name in f.keys():
            row = {}
            for attr_name in f[sim_name].keys():
                dset = f[sim_name][attr_name]
                if dset.shape == ():  # scalar dataset
                    row[attr_name] = dset[()]
                else:
                    row[attr_name] = dset[:]
            data[sim_name] = row

    dfh5 = pd.DataFrame.from_dict(data, orient='index')
    return dfh5

def stack_series(df, key):
    """
    Stacks a DataFrame column of per-tree 1D arrays (shape (Ntime,) each)
    into a single 2D array of shape (Ntrees, Ntime), suitable for
    vectorized reductions (median, percentiles, etc.) across trees.
    """
    return np.stack(df[key].values)

def stack_ragged_series(df, key):
    """
    Stacks a DataFrame column of per-tree 1D arrays of DIFFERING length
    (e.g. SHMF arrays, which are ragged since tree-to-tree subhalo counts
    differ) into a single 2D array of shape (Ntrees, max_len), padded with
    NaN past each row's actual length. Use this instead of stack_series
    whenever row lengths aren't guaranteed to match.
    """
    arrays = df[key].values
    max_len = max(len(a) for a in arrays)

    matrix = np.full((len(arrays), max_len), np.nan)
    for i, arr in enumerate(arrays):
        matrix[i, :len(arr)] = arr

    return matrix

def compute_mass_bin_stats(df, key, decimals=1):
    """
    Groups the DataFrame by unique discrete logMvir bins (rounded to `decimals`
    to avoid floating-point artifacts like 12.799999999999999), and for each
    bin computes the mean and std of `key` (a column of per-tree (Ntime,)
    arrays) across all trees in that bin, at every time step.

    Returns a dict keyed by the rounded logMvir bin value (float, e.g. 12.6),
    each entry a dict with:
        "mean" : (Ntime,) array, mean across trees at each timestep
        "std"  : (Ntime,) array, std across trees at each timestep
        "N"    : number of trees in that bin
    """
    results = {}

    rounded_bins = df["logMvir"].round(decimals)

    for mvir_bin in np.unique(rounded_bins):
        subsample = df[rounded_bins == mvir_bin]

        matrix = np.stack(subsample[key].values)  # (Ntrees_in_bin, Ntime)

        results = {
            "mean": np.nanmean(matrix, axis=0),
            "std":  np.nanstd(matrix, axis=0),
            "N":    matrix.shape[0]}

    return results

def split_mass_spec(df, decimals=1):
    """
    Just to seperate the discrete mass intervals!
    """
    results = []

    rounded_bins = df["logMvir"].round(decimals)

    for mvir_bin in np.unique(rounded_bins):
        subsample = df[rounded_bins == mvir_bin]

        results.append(subsample)

    return results


def load_massspec_timeseries(datadir, regime, order="all"):
    """
    Loads Nsub, Msub, fsub as full time series (not sliced to a single
    time index) for a single regime and subhalo order, across every tree
    in every .h5 file in datadir.

    Per tree (row):
      - logMvir, logc, log1pz50 : scalars (z=0 host properties)
      - Nsub, logNsub, fsub, logfsub, Msub, logMsub : 1D arrays, shape (Ntime,)

    regime : one of "total", "massive", "surviving", "rvir", "artificial", "splashback"
    order  : "all", "k1", "k2", or "k3"
             - "all" is a valid Nsub row directly (Nsub_{regime}_all)
             - Msub/fsub have NO "all" row, and can NOT be reconstructed by
               summing k1+k2+k3 mass, since subhalo mass is inclusive of its
               own subhalo hierarchy in some cases (e.g. an order-2 subhalo's
               mass isn't independent of its order-1 host's mass budget in
               the way that would make a naive sum non-double-counting).
               For "all" mass columns, this function selects k1 mass instead,
               which is the standard convention here.
    """
    def clean_row(arr):
        arr = np.asarray(arr, dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        return arr

    def clean_scalar(arr):
        arr = np.asarray(arr, dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        return arr

    dfs = []
    for file in os.listdir(datadir):
        if not file.endswith("h5"):
            continue

        ii = load_sample(datadir + file)

        # --- Nsub (full time series) ---
        Nsub_key = f"Nsub_{regime}_{order}"  # "all" is a real key here
        Nsub_matrix = _stack_column(ii, Nsub_key)  # (Ntrees, Ntime)

        # --- Msub / fsub (full time series) ---
        if order == "all":
            # print("NOTE: Msub/fsub have no 'all' row, since mass is inclusive "
            #       "and summing k1+k2+k3 would double-count. Using k=1 mass "
            #       "instead, which already accounts for all of its children.")
            mass_order = "k1"
        else:
            mass_order = order

        Msub_matrix = _stack_column(ii, f"Msub_{regime}_{mass_order}")  # (Ntrees, Ntime)
        fsub_matrix = _stack_column(ii, f"fsub_{regime}_{mass_order}")  # (Ntrees, Ntime)

        # --- host properties (z=0 scalars) ---
        logMvir  = clean_scalar(np.log10(_stack_column(ii, "MAH")[:, 0]))     # (Ntrees,)
        logc     = clean_scalar(np.log10(_stack_column(ii, "host_c")[:, 0]))  # (Ntrees,)
        log1pz50 = clean_scalar(np.log10(1 + ii["host_z50"].values))          # (Ntrees,)
        # --- MMs at z=0 (per regime, not per order) ---
        MMs_z0 = clean_scalar(np.asarray(ii[f"MMs_z0{regime}"].values))
        MMs_z0[np.isnan(MMs_z0)] = 0.0

        df = pd.DataFrame({
            "logMvir":  logMvir,                                    # scalar per tree
            "log1pz50": log1pz50,                                   # scalar per tree
            "logc":     logc,                                       # scalar per tree
            "logMMs":   np.log10(MMs_z0/(10**logMvir)),
            "Nsub":     [clean_row(row) for row in Nsub_matrix],           # (Ntime,) per tree
            "logNsub":  [clean_row(np.log10(row)) for row in Nsub_matrix], # (Ntime,) per tree
            "fsub":     [clean_row(row) for row in fsub_matrix],           # (Ntime,) per tree
            "logfsub":  [clean_row(np.log10(row)) for row in fsub_matrix], # (Ntime,) per tree
            "Msub":     [clean_row(row) for row in Msub_matrix],           # (Ntime,) per tree
            "logMsub":  [clean_row(np.log10(row)) for row in Msub_matrix], # (Ntime,) per tree
        })

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True).sort_values("logMvir")

def load_massspec_z0(datadir, regime, order="all"):
    """
    Loads Nsub, fsub, MMs at z=0 only for a single regime and subhalo order,
    across every tree in every .h5 file in datadir. Every column is a scalar
    per tree (row = one merger tree).

    regime : one of "total", "massive", "surviving", "rvir", "artificial", "splashback"
    order  : "all", "k1", "k2", or "k3"
             - "all" is a valid Nsub row directly (Nsub_{regime}_all)
             - fsub has NO "all" row, and can NOT be reconstructed by summing
               k1+k2+k3 mass, since subhalo mass is inclusive of its own
               subhalo hierarchy. For "all" fsub, this function selects k1
               fsub instead, which already accounts for all of its children.
             - MMs (max subhalo mass at z=0) is stored per regime only, not
               per order, so it's the same value regardless of `order`.
    """
    def clean_scalar(arr):
        arr = np.asarray(arr, dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        return arr

    dfs = []
    for file in os.listdir(datadir):
        if not file.endswith("h5"):
            continue

        ii = load_sample(datadir + file)

        # --- Nsub at z=0 ---
        Nsub_key = f"Nsub_{regime}_{order}"  # "all" is a real key here
        Nsub_z0 = clean_scalar(_stack_column(ii, Nsub_key)[:, 0])

        # --- fsub at z=0 ---
        if order == "all":
            # fsub has no 'all' row, since mass is inclusive and summing
            # k1+k2+k3 would double-count. Using k=1 instead, which already
            # accounts for all of its children.
            mass_order = "k1"
        else:
            mass_order = order

        fsub_z0 = clean_scalar(_stack_column(ii, f"fsub_{regime}_{mass_order}")[:, 0])

        # --- MMs at z=0 (per regime, not per order) ---
        MMs_z0 = clean_scalar(np.asarray(ii[f"MMs_z0{regime}"].values))
        MMs_z0[np.isnan(MMs_z0)] = 0.0

        # --- host properties at z=0 ---
        logMvir  = clean_scalar(np.log10(_stack_column(ii, "MAH")[:, 0]))
        logc     = clean_scalar(np.log10(_stack_column(ii, "host_c")[:, 0]))
        log1pz50 = clean_scalar(np.log10(1 + ii["host_z50"].values))

        df = pd.DataFrame({
            "logMvir":  logMvir,
            "log1pz50": log1pz50,
            "logc":     logc,
            "Nsub":     Nsub_z0,
            "logNsub":  np.log10(Nsub_z0),
            "fsub":     fsub_z0,
            "logfsub":  np.log10(fsub_z0),
            "MMs":      MMs_z0/(10**logMvir),
            "logMMs":   np.log10(MMs_z0/(10**logMvir)),
        }).replace([np.inf, -np.inf], np.nan)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True).sort_values("logMvir")

def load_shmf_z0(datadir, regimes=("surviving", "rvir_surv", "artificial")):
    """
    Loads the z=0 subhalo mass function (SHMF) for the given regimes,
    across every tree in every .h5 file in datadir. Each row is one merger
    tree; SHMF columns hold a 1D array of subhalo masses (sorted descending,
    NaN-padded) per tree — not a scalar — since tree-to-tree subhalo counts
    differ.

    Per tree (row):
      - logMvir, logc, log1pz50 : scalars (z=0 host properties)
      - shmf_{regime}_all, shmf_{regime}_k1, shmf_{regime}_k2, shmf_{regime}_k3 :
        1D arrays (ragged length across trees) for each regime in `regimes`
    """
    def clean_row(arr):
        arr = np.asarray(arr, dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        return arr

    def clean_scalar(arr):
        arr = np.asarray(arr, dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        return arr

    order_labels = ("all", "k1", "k2", "k3")

    dfs = []
    for file in os.listdir(datadir):
        if not file.endswith("h5"):
            continue

        ii = load_sample(datadir + file)

        # --- host properties at z=0 ---
        logMvir  = clean_scalar(np.log10(_stack_column(ii, "MAH")[:, 0]))
        logc     = clean_scalar(np.log10(_stack_column(ii, "host_c")[:, 0]))
        log1pz50 = clean_scalar(np.log10(1 + ii["host_z50"].values))

        data = {
            "logMvir":  logMvir,
            "log1pz50": log1pz50,
            "logc":     logc,
        }

        # --- SHMF arrays per regime per order ---
        for regime in regimes:
            for label in order_labels:
                shmf_key = f"shmf_{regime}_{label}"
                data[shmf_key] = [clean_row(row) for row in ii[shmf_key].values]

        df = pd.DataFrame(data)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True).sort_values("logMvir")

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
### MERGER TREE STURUCTRES
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def acc_hierarchy(tree):

    tree_hierarchy = Tree()
    tree_hierarchy.create_node("Host Halo", "0", 
                    data={"acc_mass": tree.target_mass, "acc_redshift": 0, "acc_stellarmass": tree.target_stellarmass})  # The root node

    def add_node_with_parents(subhalo_ind):
        parent_id = str(tree.acc_ParentID[subhalo_ind])
        node_id = str(subhalo_ind)
        data_id = {"acc_mass": tree.acc_mass[subhalo_ind], "acc_redshift": tree.acc_redshift[subhalo_ind], "acc_stellarmass": tree.acc_stellarmass[subhalo_ind]}

        # If parent not yet in tree, add it (or recurse)
        if not tree_hierarchy.contains(parent_id):
            # Recursively ensure the parent is added first
            add_node_with_parents(int(parent_id))

        # Finally add the current node (if not already added)
        if not tree_hierarchy.contains(node_id):
            tree_hierarchy.create_node("subID:" + node_id, node_id, parent=parent_id, data=data_id)

    for subhalo_ind in range(1, tree.Nhalo):
        add_node_with_parents(subhalo_ind) 
    return tree_hierarchy


def find_late_events(tree):

    late_mergers = []
    late_disruptions = []

    for node in tree.final_tree.all_nodes():
        if node.is_root():
            continue  # skip root node

        parent = tree.final_tree.parent(node.identifier)

        child_z = node.data["final_redshift"]
        parent_z = parent.data["final_redshift"]

        # Check if child merged/disrupted after parent already merged
        if child_z < parent_z and parent.data["fate"] == "merged":
            child_fate = node.data["fate"]

            entry = {
                "child_id": node.identifier,
                "parent_id": parent.identifier,
                "child_z": child_z,
                "parent_z": parent_z,
                "child_fate": child_fate,
                "parent_fate": parent.data["fate"]
            }

            if child_fate == "merged":
                late_mergers.append(entry)
            elif child_fate == "disrupted":
                late_disruptions.append(entry)

    return late_disruptions, late_mergers


def add_node_with_parents(tree, tree_hierarchy, subhalo_ind, z_ind):
    node_id = str(subhalo_ind)
    parent_id = str(tree.ParentID[subhalo_ind, z_ind])
    
    # Check if parent exists at this time step (not -99)
    if int(parent_id) != -1 and tree.ParentID[int(parent_id), z_ind] == -99:
        # Parent hasn't been born yet, skip adding this node
        return
    
    # If parent not yet in tree, add it (or recurse)
    if not tree_hierarchy.contains(parent_id):
        # Only recurse if parent is not the host halo (-1)
        if int(parent_id) != -1:
            # Recursively ensure the parent is added first
            add_node_with_parents(tree, tree_hierarchy, int(parent_id), z_ind)
            # If parent still not in tree after recursion, skip this node
            if not tree_hierarchy.contains(parent_id):
                return
    
    # Finally add the current node (if not already added)
    if not tree_hierarchy.contains(node_id):
        tree_hierarchy.create_node("subID:" + node_id, node_id, parent=parent_id)

def forest_generator(Tree_Vis):
    # forest because every time step is its own merger tree!
    forest = []
    
    for z_ind in range(0, len(Tree_Vis.redshift)):
        
        tree_hierarchy_z = Tree()
        tree_hierarchy_z.create_node("Host Halo", "0", data={"redshift": Tree_Vis.redshift[z_ind]})
        forest.append(tree_hierarchy_z)
        
        parents = Tree_Vis.ParentID[:, z_ind]
        initialized_subhalos = np.where(parents != -99)[0]
        initialized_subhalos = initialized_subhalos[initialized_subhalos != -1]  # remove the host!
        
        if initialized_subhalos.shape[0] > 1:
            for subhalo_ind in initialized_subhalos:
                try:
                    add_node_with_parents(Tree_Vis, tree_hierarchy_z, subhalo_ind, z_ind)
                except RecursionError:
                    print(f"Recursion error at z_ind {z_ind}, subhalo {subhalo_ind}")
    
    return forest

def tree_walker(Tree_Vis, current_index):

    prev_index = current_index + 1

    current_tree = Tree_Vis.forest[current_index]
    prev_tree = Tree_Vis.forest[prev_index]

    order_jumps = 0
    births = 0
    host_halo_mergers = 0
    host_acc = 0

    for current_subhalo in current_tree.all_nodes_itr():
        current_subhalo_id = current_subhalo.identifier

        #skip the host!
        if current_subhalo_id == "0":
            continue
        
        #skip the subhalos that don't have a previous time-step counter part
        prev_subhalo = prev_tree.get_node(current_subhalo_id)
        if prev_subhalo == None:
            births += 1
            continue

        #just to count the order jumps
        current_parent = current_tree.parent(current_subhalo_id)
        current_parent_id = current_parent.identifier

        prev_parent = prev_tree.parent(current_subhalo_id)
        prev_parent_id = prev_parent.identifier

        if current_parent_id != prev_parent_id:
            order_jumps += 1

        #this should be fine since we have already skipped the subhalos that were just born this time_index
        current_subhalo_id_int = int(current_subhalo_id) # just grabbing as an integers
        current_parent_id_int = int(current_parent_id)

        #the subhalo hasn't been intialized with stellarmass
        if Tree_Vis.acc_index[current_subhalo_id_int] < current_index:
            continue

        current_mass = Tree_Vis.stellarmass[current_subhalo_id_int, current_index]
        prev_mass = Tree_Vis.stellarmass[current_subhalo_id_int, prev_index]
        mass_loss = prev_mass - current_mass

        if mass_loss > 0: # don't need to worry about the parent as much here since ICL is summed across all subhalos
            Tree_Vis.icl[current_parent_id_int, current_index] += mass_loss
            Tree_Vis.contributed[current_subhalo_id_int] += mass_loss

        #if the subhalo is found dead
        if Tree_Vis.final_index[current_subhalo_id_int] == current_index:

            fate = Tree_Vis.subhalo_fates[current_subhalo_id_int]
            is_late_event = Tree_Vis.final_index[current_subhalo_id_int]  < Tree_Vis.final_index[current_parent_id_int]

            if fate == "merged": #break up the mass
                icl_mass = Tree_Vis.fesc * current_mass
                merger_mass = current_mass - icl_mass

                # if the parent hasn't been born yet and is not the host!
                if Tree_Vis.acc_index[current_parent_id_int] != 0 and Tree_Vis.acc_index[current_parent_id_int] < current_index:
                    continue
                else:
                    if is_late_event: # the merger happens with the grandparent!
                        grandparent = current_tree.parent(current_parent_id)

                        if grandparent is None: #the host is the grandparent!
                            current_grandparent_id_int = 0
                            host_halo_mergers += 1
                            host_acc += merger_mass 
                            Tree_Vis.exsitu[current_subhalo_id_int, current_index] = merger_mass 

                        else: #the grandparent exists!
                            current_grandparent_id = grandparent.identifier
                            current_grandparent_id_int = int(current_grandparent_id)

                        #keep track of the ratio - need to measure this first!
                        merger_mass_ratio = merger_mass/Tree_Vis.stellarmass[current_grandparent_id_int, current_index]
                        merger_index = np.where(Tree_Vis.merged_subhalos == current_subhalo_id_int)[0][0]
                        Tree_Vis.merger_ratios[merger_index] = merger_mass_ratio

                        #disribute the mass
                        Tree_Vis.icl[current_grandparent_id_int, current_index] += icl_mass #this only happens at the time index
                        Tree_Vis.contributed[current_subhalo_id_int] += icl_mass

                        Tree_Vis.stellarmass[current_grandparent_id_int, :current_index] += merger_mass #this applies to everywhere after!

                    else: #the merger happens with the parent!!
                        if current_parent_id_int == 0: #the parent is the host!
                            host_halo_mergers += 1
                            host_acc += merger_mass 
                            Tree_Vis.exsitu[current_subhalo_id_int, current_index] = merger_mass 

                        #keep track of the ratio
                        merger_mass_ratio = merger_mass/Tree_Vis.stellarmass[current_parent_id_int, current_index]
                        merger_index = np.where(Tree_Vis.merged_subhalos == current_subhalo_id_int)[0][0]
                        Tree_Vis.merger_ratios[merger_index] = merger_mass_ratio

                        #disribute the mass
                        Tree_Vis.icl[current_parent_id_int, current_index] += icl_mass
                        Tree_Vis.contributed[current_subhalo_id_int] += icl_mass

                        Tree_Vis.stellarmass[current_parent_id_int, :current_index] += merger_mass

                # EXPLICITLY ZERO OUT THE DEAD SUBHALO'S MASS
                Tree_Vis.stellarmass[current_subhalo_id_int, :current_index+1] = 0.0

            if fate == "disrupted": #all of the mass goes to the ICL
                icl_mass = current_mass

                if is_late_event: # the disruption happens while in the grandparent!
                    grandparent = current_tree.parent(current_parent_id)

                    if grandparent is None: #the host is the grandparent!
                        current_grandparent_id_int = 0 
                    else: #the grandparent exists!
                        current_grandparent_id = grandparent.identifier
                        current_grandparent_id_int = int(current_grandparent_id)
                    #disribute the mass
                    Tree_Vis.icl[current_grandparent_id_int, current_index] += icl_mass
                    Tree_Vis.contributed[current_subhalo_id_int] += icl_mass


                else: #the merger happens with the parent!!
                    #disribute the mass
                    Tree_Vis.icl[current_parent_id_int, current_index] += icl_mass
                    Tree_Vis.contributed[current_subhalo_id_int] += icl_mass

                # EXPLICITLY ZERO OUT THE DEAD SUBHALO'S MASS
                Tree_Vis.stellarmass[current_subhalo_id_int, :current_index+1] = 0.0

            if fate == "surviving": #do nothing with it since it already lost mass!
                continue 
                
# def find_associated_subhalos(tree, sub_ind, time_ind):
#     associated_set = []

#     # Checking to see if there are any direct children at this time step
#     direct_parent_merging = tree.ParentID[:, time_ind] == sub_ind 
#     if np.any(direct_parent_merging):
#         associated_subhalos = np.where(direct_parent_merging)[0]  # Any subhalos that have the same parent
#         disrupt_mask = tree.disrupt_index[associated_subhalos] < time_ind  # Disruption must happen after the merger
#         associated_subhalos = associated_subhalos[disrupt_mask]
#         associated_set.extend(associated_subhalos)
        
#         # Recursively collect descendants of each subhalo
#         for subhalo in associated_subhalos:
#             subhalo_descendants = find_associated_subhalos(tree, subhalo, time_ind)
#             if subhalo_descendants:  # Ensure no NoneType is returned
#                 associated_set.extend(subhalo_descendants)

#     return associated_set