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
import jsm_stats
from scipy.spatial import cKDTree
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


class HaloCatalogue:
    """
    Unified halo catalogue reader/analyzer for BolshoiP and VSMDPL hlist-derived
    catalogues. All logic from the original `Bolshoi_HaloCatalogue` class is
    preserved as-is (including order-resolved (k=1,2,3+) substructure statistics).
    Both sims carry pid/upid hierarchy columns, so the only differences between
    them are the particle mass (used to set the mass-resolution cut) and the
    header keywords (column layout / mass column name).

    `has_order_info` is kept as a per-sim config flag (rather than hardcoded)
    so a future sim without pid/upid could still be added and would gracefully
    fall back to "all"-only statistics.

    Parameters
    ----------
    sim_title : str
        Either "bolshoi" or "vsmdpl".
    """

    # ------------------------------------------------------------------ #
    # Per-simulation configuration: column layout, particle mass, mass
    # column name, and whether pid/upid hierarchy info is present.
    # ------------------------------------------------------------------ #
    _SIM_CONFIG = {
        "bolshoi": {
            "particle_mass":  1.55e8,   # Msun
            "mass_col":       "log10Mvir",
            "has_order_info": True,
            "cols": [
                "host_id", "logMh", "ch", "a_50h",
                "Xoff_h", "Spin_h", "Spin_Bullock_h", "ch_K",
                "x_h", "y_h", "z_h", "R_vir", "h_pid", "h_upid",
                "id", "pid", "upid", "log10Mvir", "Rvir", "rs", "vrms", "scale_of_last_MM",
                "vmax", "x", "y", "z", "vx", "vy", "vz",
                "Jx", "Jy", "Jz", "Spin", "Tidal_Force", "Tidal_ID",
                "Mmvir_all", "M200b", "M200c", "M500c",
                "Xoff", "Voff", "Spin_Bullock",
                "b_to_a", "c_to_a",
                "Ax", "Ay", "Az", "T_by_U",
                "M_pe_Behroozi", "M_pe_Diemer",
                "Macc", "Mpeak", "Vacc", "Vpeak", "Halfmass_Scale",
                "Acc_Rate_Inst", "Acc_Rate_100Myr", "Acc_Rate_1Tdyn",
                "Acc_Rate_2Tdyn", "Acc_Rate_Mpeak",
                "Mpeak_Scale", "Acc_Scale", "First_Acc_Scale",
                "First_Acc_Mvir", "First_Acc_Vmax", "Vmax_at_Mpeak",
                "Tidal_Force_Tdyn",
            ],
        },
        "vsmdpl": {
            "particle_mass":  6.2e6,    # Msun
            "mass_col":       "ALOG10(Mvir)",
            "has_order_info": True,
            "cols": [
                "host_id", "logMh", "ch", "a_50h",
                "Xoff_h", "Spin_h", "Spin_Bullock_h", "ch_K",
                "x_h", "y_h", "z_h", "R_vir", "h_pid", "h_upid",
                "id", "pid", "upid", "ALOG10(Mvir)", "Rvir", "rs", "vrms", "scale_of_last_MM",
                "vmax", "x", "y", "z", "vx", "vy", "vz", "Jx", "Jy", "Jz", "Spin", "Tidal_Force", "Tidal_ID",
                "Mmvir_all", "M200b", "M200c", "M500c", "Xoff", "Voff", "Spin_Bullock", "b_to_a",
                "c_to_a", "Ax", "Ay", "Az", "T_by_U", "M_pe_Behroozi", "M_pe_Diemer",
                "Halfmass_Radius", "Macc", "Mpeak", "Vacc", "Vpeak", "Halfmass_Scale",
                "Acc_Rate_Inst", "Acc_Rate_100Myr", "Acc_Rate_1Tdyn",
                "Acc_Rate_2Tdyn", "Acc_Rate_Mpeak", "Acc_Log_Vmax_Inst",
                "Acc_Log_Vmax_1Tdyn", "Mpeak_Scale", "Acc_Scale", "First_Acc_Scale",
                "First_Acc_Mvir", "First_Acc_Vmax", "Vmax_at_Mpeak",
                "Tidal_Force_Tdyn",
            ],
        },
    }

    def __init__(self, sim_title, filepath,mthresh,
        xoff_thresh=0.07,
        spin_thresh=0.07,
        isolation_factor=3):

        sim_title = sim_title.lower()
        if sim_title not in self._SIM_CONFIG:
            raise ValueError(
                f"Unknown sim_title '{sim_title}'. Must be one of {list(self._SIM_CONFIG)}."
            )

        cfg = self._SIM_CONFIG[sim_title]

        self.sim_title        = sim_title
        self._cols            = cfg["cols"]
        self._mass_col        = cfg["mass_col"]
        self._has_order_info  = cfg["has_order_info"]
        self._particle_mass   = cfg["particle_mass"]                # Msun, this sim's particle mass

        self.filepath         = filepath
        self.mass_thresh      = mthresh                             # Msun — mass resolution cut, set directly
        self.log_mass_thresh  = np.log10(mthresh)                   # log10 Msun — for comparisons
        self.npart_thresh     = mthresh / cfg["particle_mass"]      # for reference/printout only
        self.xoff_thresh      = xoff_thresh
        self.spin_thresh      = spin_thresh
        self.isolation_factor = isolation_factor

        self._print_mass_thresh()

        self._load()
        if self._has_order_info:
            self._compute_subhalo_order()
        self._relaxation_cut()
        self._isolation_cut()
        self._print_counts()

    def _print_mass_thresh(self):
        print(
            f"[{self.sim_title}] mass threshold = {self.mass_thresh:.3e} Msun "
            f"-> {self.npart_thresh:.1f} particles "
            f"(particle mass = {self._particle_mass:.3e} Msun)"
        )

    def _load(self):
        raw        = np.loadtxt(self.filepath)
        self._df   = pd.DataFrame(raw, columns=self._cols)

    def _compute_subhalo_order(self, max_order=5):
        """
        Vectorized parent-chain hop-count: assigns each row an integer 'order'
        counting how many pid-hops separate it from its top-level host (upid).
        order = 0        -> host itself (upid == -1)
        order = 1        -> direct child of the host (pid == upid)
        order = 2,3,...  -> each additional intermediate parent
        Rows whose chain doesn't resolve within max_order hops (broken/orphaned
        links, e.g. parent pruned from this snapshot) are left at max_order as
        a safe upper-bound bucket (folds into the k>=3 bucket downstream).

        Requires pid/upid columns, so only called for sims with
        `has_order_info == True` (currently: bolshoi).
        """
        ids   = self._df["id"].values
        pids  = self._df["pid"].values
        upids = self._df["upid"].values

        id_to_pid = pd.Series(pids, index=ids)

        order = np.zeros(len(self._df), dtype=int)
        is_host = (upids == -1)
        order[is_host] = 0

        is_sub = ~is_host
        order[is_sub] = 1
        current = pids.copy()

        unresolved = is_sub & (current != upids)

        hop = 2
        while unresolved.any() and hop <= max_order:
            next_parent = pd.Series(current[unresolved]).map(id_to_pid).values
            broken = pd.isna(next_parent)
            next_parent = np.where(broken, current[unresolved], next_parent)

            current[unresolved] = next_parent
            order[unresolved] = hop

            still_unresolved = unresolved.copy()
            still_unresolved[unresolved] = (current[unresolved] != upids[unresolved]) & (~broken)
            unresolved = still_unresolved
            hop += 1

        self._df["order"] = order

    def _relaxation_cut(self):
        host_props = (
            self._df
            .groupby("host_id")[["Xoff_h", "Spin_h", "R_vir"]]
            .mean()
            .reset_index()
        )
        relaxed_ids = host_props[
            (host_props["Xoff_h"] / host_props["R_vir"] <= self.xoff_thresh) &
            (host_props["Spin_h"] <= self.spin_thresh)
        ]["host_id"].values

        self._df_relaxed = self._df[self._df["host_id"].isin(relaxed_ids)]

    def _isolation_cut(self):
        host_relaxed = (
            self._df_relaxed
            .groupby("host_id")[["x_h", "y_h", "z_h", "R_vir", "logMh"]]
            .mean()
            .reset_index()
        )

        coords   = host_relaxed[["x_h", "y_h", "z_h"]].values
        masses   = host_relaxed["logMh"].values
        r_virial = host_relaxed["R_vir"].values / 1000.0   # kpc/h -> Mpc/h

        tree     = cKDTree(coords)
        isolated = np.ones(len(host_relaxed), dtype=bool)

        # Upper bound search radius: no more massive halo can be further than
        # isolation_factor * max(Rvir) away
        max_search_r = self.isolation_factor * r_virial.max()

        for i in range(len(host_relaxed)):
            # Query all neighbours within the maximum possible isolation radius
            neighbours = tree.query_ball_point(coords[i], r=max_search_r)
            for j in neighbours:
                if j == i:
                    continue
                if masses[j] >= masses[i]:
                    # Use the MORE MASSIVE halo's (j's) Rvir to define isolation
                    isolation_radius = self.isolation_factor * r_virial[j]
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist < isolation_radius:
                        isolated[i] = False
                        break

        isolated_ids      = host_relaxed[isolated]["host_id"].values
        self._df_isolated = self._df_relaxed[self._df_relaxed["host_id"].isin(isolated_ids)]

    def _print_counts(self):
        n_raw     = self._df["host_id"].unique().shape[0]
        n_relaxed = self._df_relaxed["host_id"].unique().shape[0]
        n_final   = self._df_isolated["host_id"].unique().shape[0]
        print(
            f"Hosts: total={n_raw}  "
            f"after relaxation={n_relaxed} ({100*n_relaxed/n_raw:.1f}%)  "
            f"after isolation={n_final} ({100*n_final/n_raw:.1f}%)"
        )

    def _get_groups(self, sample):
        if sample == "isolated":
            host_id_unique = np.sort(self._df_isolated["host_id"].unique())
            groups = self._df_isolated.groupby("host_id")
        elif sample == "relaxed":
            host_id_unique = np.sort(self._df_relaxed["host_id"].unique())
            groups = self._df_relaxed.groupby("host_id")
        elif sample == "all":
            host_id_unique = np.sort(self._df["host_id"].unique())
            groups = self._df.groupby("host_id")
        else:
            raise ValueError(f"Unknown sample type: {sample}")
        return host_id_unique, groups

    def _select_subset1(self, subset):
        """
        Given a host's subhalo subset, return the bound subhalo population:
        mass >= mass threshold AND within Rvir of the host center. Uses
        whichever mass column this sim's catalogue carries (log10Mvir for
        bolshoi, ALOG10(Mvir) for vsmdpl).
        """
        x_h     = subset["x_h"].mean()
        y_h     = subset["y_h"].mean()
        z_h     = subset["z_h"].mean()
        R_vir_i = subset["R_vir"].mean()

        dr = np.sqrt(
            (subset["x"] - x_h)**2 +
            (subset["y"] - y_h)**2 +
            (subset["z"] - z_h)**2
        )

        return subset[(subset[self._mass_col] >= self.log_mass_thresh) & (dr <= R_vir_i)]

    def compute_shmf(self, sample):
        """
        Compute the z=0 subhalo mass function per host halo, split by
        subhalo order (k = all, 1, 2, 3). Both bolshoi and vsmdpl carry
        pid/upid, so this is computed identically for either sim.

        Returns a dict with:
            host_id           : (n_host,)              host halo IDs
            logMvir           : (n_host,)               host log10(Mvir)
            logMsub_all       : (n_host, n_sub_max_all)  NaN-padded, all subhalos
            logMsub_k1        : (n_host, n_sub_max_k1)   NaN-padded, order-1 subhalos
            logMsub_k2        : (n_host, n_sub_max_k2)   NaN-padded, order-2 subhalos
            logMsub_k3        : (n_host, n_sub_max_k3)   NaN-padded, order>=3 subhalos
        """
        host_id_unique, groups = self._get_groups(sample)
        n_host = len(host_id_unique)

        host_logMvir = np.zeros(n_host)

        sub_mass_lists = {"all": []}
        if self._has_order_info:
            sub_mass_lists.update({1: [], 2: [], 3: []})  # k = 3 bucket (order >= 3)

        for i, hid in enumerate(host_id_unique):
            subset  = groups.get_group(hid)
            subset1 = self._select_subset1(subset)

            host_logMvir[i] = subset["logMh"].mean()

            masses = subset1[self._mass_col].values
            sub_mass_lists["all"].append(masses)

            if self._has_order_info:
                order_vals = subset1["order"].values
                sub_mass_lists[1].append(masses[order_vals == 1])
                sub_mass_lists[2].append(masses[order_vals == 2])
                sub_mass_lists[3].append(masses[order_vals >= 3])

        def _pad(mass_list):
            n_sub_max = max((len(a) for a in mass_list), default=0)
            padded = np.full((n_host, n_sub_max), np.nan)
            for i, arr in enumerate(mass_list):
                padded[i, :len(arr)] = arr
            return padded

        result = {
            "host_id":     host_id_unique,
            "logMvir":     host_logMvir,
            "logMsub_all": _pad(sub_mass_lists["all"]),
        }

        if self._has_order_info:
            result["logMsub_k1"] = _pad(sub_mass_lists[1])
            result["logMsub_k2"] = _pad(sub_mass_lists[2])
            result["logMsub_k3"] = _pad(sub_mass_lists[3])

        return result

    def _build_host_table(self, sample):
        """
        Nsub (and the SHMF in compute_shmf) are counts/arrays of individual
        subhalos, so it's meaningful to report 4 subvalues: k=all, k=1, k=2, k=3
        (order>=3).

        fsub and MMs, on the other hand, are built from *mass values*, and
        subhalo masses (log10Mvir / ALOG10(Mvir)) are inclusive of their own
        substructure — i.e. a k=1 subhalo's mass already accounts for any k=2,
        k=3 subhalos living inside it. That makes sum(mass) or max(mass) over
        "all" subhalos double-count mass already contained in k=1 halos (and
        the global max is by construction attained at k=1). So fsub and MMs
        only get 3 subvalues: k=1, k=2, k=3 — there is no separate "all".
        """

        host_id_unique, groups = self._get_groups(sample)
        n_host = len(host_id_unique)

        logMvir    = np.zeros(n_host)
        log1pz50   = np.zeros(n_host)
        logc       = np.zeros(n_host)

        # Nsub: 4 subvalues (all, k1, k2, k3)
        Nsub       = np.zeros(n_host)   # k = all
        logNsub    = np.zeros(n_host)

        if self._has_order_info:
            Nsub_k    = {1: np.zeros(n_host), 2: np.zeros(n_host), 3: np.zeros(n_host)}
            logNsub_k = {1: np.zeros(n_host), 2: np.zeros(n_host), 3: np.zeros(n_host)}

            # fsub/MMs: 3 subvalues only (k1, k2, k3) — no "all"
            fsub_k    = {1: np.zeros(n_host), 2: np.zeros(n_host), 3: np.zeros(n_host)}
            logfsub_k = {1: np.zeros(n_host), 2: np.zeros(n_host), 3: np.zeros(n_host)}
            MMs_k     = {1: np.zeros(n_host), 2: np.zeros(n_host), 3: np.zeros(n_host)}
            logMMs_k  = {1: np.zeros(n_host), 2: np.zeros(n_host), 3: np.zeros(n_host)}

        for i, hid in enumerate(host_id_unique):
            subset   = groups.get_group(hid)

            logMh_i  = subset["logMh"].mean()
            c_h_i    = subset["ch_K"].mean()
            a_half_i = subset["a_50h"].mean()

            subset1  = self._select_subset1(subset)
            host_mass = 10**logMh_i

            z50 = (1.0 / a_half_i) - 1.0
            logMvir[i]  = logMh_i
            log1pz50[i] = np.log10(1.0 + z50)
            logc[i]     = np.log10(c_h_i)

            # ---- Nsub: k = all ----
            Nsub_i = len(subset1)
            Nsub[i]    = Nsub_i
            logNsub[i] = np.log10(Nsub_i)

            # ---- order-resolved: k = 1, 2, 3 ----
            if self._has_order_info:
                order_vals = subset1["order"].values
                for k in (1, 2, 3):
                    sel_mask = (order_vals == k) if k < 3 else (order_vals >= 3)
                    sel      = subset1[sel_mask]
                    n_k      = len(sel)

                    Nsub_k[k][i]    = n_k
                    logNsub_k[k][i] = np.log10(n_k)

                    if n_k > 0:
                        fsub_k[k][i] = np.sum(10**sel[self._mass_col]) / host_mass
                        MMs_k[k][i]  = (10**sel[self._mass_col].max()) / host_mass
                    logfsub_k[k][i] = np.log10(fsub_k[k][i])
                    logMMs_k[k][i]  = np.log10(MMs_k[k][i])

        host_table_dict = {
            "logMvir":  logMvir,
            "log1pz50": log1pz50,
            "logc":     logc,

            "Nsub":     Nsub,      # k = all
            "logNsub":  logNsub,
        }

        if self._has_order_info:
            host_table_dict.update({
                "Nsub_k1":    Nsub_k[1],
                "logNsub_k1": logNsub_k[1],
                "fsub":    fsub_k[1],
                "logfsub": logfsub_k[1],
                "MMs":     MMs_k[1],
                "logMMs":  logMMs_k[1],

                "Nsub_k2":    Nsub_k[2],
                "logNsub_k2": logNsub_k[2],
                "fsub_k2":    fsub_k[2],
                "logfsub_k2": logfsub_k[2],
                "MMs_k2":     MMs_k[2],
                "logMMs_k2":  logMMs_k[2],

                "Nsub_k3":    Nsub_k[3],
                "logNsub_k3": logNsub_k[3],
                "fsub_k3":    fsub_k[3],
                "logfsub_k3": logfsub_k[3],
                "MMs_k3":     MMs_k[3],
                "logMMs_k3":  logMMs_k[3],
            })

        host_table = pd.DataFrame(host_table_dict).replace([np.inf, -np.inf], np.nan)

        return host_table

##### ------------------------------------------------------------------------
##### ------------------------------------------------------------------------
##### ------------------------------------------------------------------------
##### ------------------------------------------------------------------------
## ABUNDANCE MEASUREMENT TOOLS FOR PAPER 3
##### ------------------------------------------------------------------------
##### ------------------------------------------------------------------------
##### ------------------------------------------------------------------------
##### ------------------------------------------------------------------------



class NormalizeData:

    def __init__(self, df, logMvir_binsize=0.1, **kwargs):

        self.df = df
        self.logMvir_binsize = logMvir_binsize

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.bin_data()
        self.fit_lines()
        self.normalize()
        self.HAB_signal()

    def grab_subsample(self, logMvir_min, logMvir_max):

        return self.df[
            (self.df["logMvir"] > logMvir_min)
            & (self.df["logMvir"] <= logMvir_max)
        ]

    def measure_stat(self, column, ignore_nans=False):

        means = []
        stds = []

        for center in self.logMvir_bincenters:

            sample = self.grab_subsample(
                center - self.logMvir_binsize,
                center + self.logMvir_binsize
            )

            vals = sample[column].values

            if ignore_nans:
                means.append(np.nanmean(vals))
                stds.append(np.nanstd(vals))
            else:
                means.append(np.mean(vals))
                stds.append(np.std(vals))               

        return np.array(means), np.array(stds)
    
    def measure_correlation(self, xkey, ykey):

        rhos = []
        rho_errs = []
        Nhosts = []

        for center in self.logMvir_bincenters:

            sample = self.grab_subsample(
                center - self.logMvir_binsize,
                center + self.logMvir_binsize
            )

            x = sample[xkey]
            y = sample[ykey]
            N = len(x)

            rho, rho_err, p_val = jsm_stats.jackknife_correlation(x, y)
            rhos.append(rho)
            rho_errs.append(rho_err)
            Nhosts.append(N)

        return np.array(rhos), np.array(rho_errs), np.array(Nhosts)

    def measure_P0(self):

        P0 = []
        for center in self.logMvir_bincenters:
            sample = self.grab_subsample(
                center - self.logMvir_binsize,
                center + self.logMvir_binsize
            )
            P0.append(jsm_stats.countzero(sample["Nsub"]))

        return np.array(P0)

    def bin_data(self):

        self.logMvir_bincenters = np.linspace(12.6, 14.0, 8)
        self.logMvir_smooth = np.linspace(12.6, 14.0, 100)

        # ---- log1pz50 ----
        self.log1pz50_mean, self.log1pz50_std = \
            self.measure_stat("log1pz50")

        # ---- z50 ----
        self.df["z50"] = (10**self.df["log1pz50"]) - 1
        self.z50_mean, self.z50_std = \
            self.measure_stat("z50")

        # ---- concentration ----
        self.logc_mean, self.logc_std = \
            self.measure_stat("logc")
        
        # ---- logNsub ----
        self.logNsub_mean, self.logNsub_std = \
            self.measure_stat("logNsub", ignore_nans=True)

        # ---- Nsub ----
        self.Nsub_mean, self.Nsub_std = \
            self.measure_stat("Nsub")
        # ---- P(Nsub = 0) ----
        self.P0 = self.measure_P0()

        # ---- logMMs ----
        self.logMMs_mean, self.logMMs_std = \
            self.measure_stat("logMMs", ignore_nans=True)
        
        # ---- MMs ----
        self.MMs_mean, self.MMs_std = \
            self.measure_stat("MMs")

        # ---- logfsub ----                              
        self.logfsub_mean, self.logfsub_std = \
            self.measure_stat("logfsub", ignore_nans=True)  

        # ---- fsub ----                                 
        self.fsub_mean, self.fsub_std = \
            self.measure_stat("fsub")                  

    def fit_lines(self):

        self.m_log1pz50, self.b_log1pz50 = (
            jsm_stats.fit_line_sym_errors(
                self.logMvir_bincenters,
                self.log1pz50_mean,
                self.log1pz50_std,
                p0=(0.5, 1.0)
            )
        )

        self.m_logNsub, self.b_logNsub = (
            jsm_stats.fit_line_sym_errors(
                self.logMvir_bincenters,
                self.logNsub_mean,
                self.logNsub_std,
                p0=(1.0, 1.0)
            )
        )

        self.m_logc, self.b_logc = (
            jsm_stats.fit_line_sym_errors(
                self.logMvir_bincenters,
                self.logc_mean,
                self.logc_std,
                p0=(1.0, 1.0)
            )
        )

        self.m_logMMs, self.b_logMMs = (
            jsm_stats.fit_line_sym_errors(
                self.logMvir_bincenters,
                self.logMMs_mean,
                self.logMMs_std,
                p0=(1.0, 1.0)
            )
        )

        self.m_logfsub, self.b_logfsub = (          
            jsm_stats.fit_line_sym_errors(           
                self.logMvir_bincenters,             
                self.logfsub_mean,                   
                self.logfsub_std,                    
                p0=(1.0, 1.0)                        
            )                                        
        )                                            

        self.bestfit_mat = np.array([
            [self.m_log1pz50, self.b_log1pz50],
            [self.m_logc,      self.b_logc],
            [self.m_logMMs,    self.b_logMMs],
            [self.m_logNsub,   self.b_logNsub],
            [self.m_logfsub,   self.b_logfsub],      
        ])


    def normalize(self):

        self.log1pz50_smooth = (
            self.m_log1pz50 * self.logMvir_smooth
            + self.b_log1pz50
        )

        self.logc_smooth = (
            self.m_logc * self.logMvir_smooth
            + self.b_logc
        )

        self.logNsub_smooth = (
            self.m_logNsub * self.logMvir_smooth
            + self.b_logNsub
        )

        self.logMMs_smooth = (
            self.m_logMMs * self.logMvir_smooth
            + self.b_logMMs
        )

        self.logfsub_smooth = (                      
            self.m_logfsub * self.logMvir_smooth     
            + self.b_logfsub                         
        )                                            

        self.df["delta_log1pz50"] = (
            self.df["log1pz50"]
            - (self.m_log1pz50 * self.df["logMvir"] + self.b_log1pz50)
        )

        self.df["delta_logc"] = (
            self.df["logc"]
            - (self.m_logc * self.df["logMvir"] + self.b_logc)
        )

        self.df["delta_logNsub"] = (
            self.df["logNsub"]
            - (self.m_logNsub * self.df["logMvir"] + self.b_logNsub)
        )

        self.df["delta_logMMs"] = (
            self.df["logMMs"]
            - (self.m_logMMs * self.df["logMvir"] + self.b_logMMs)
        )

        self.df["delta_Nsub"] = (
            self.df["Nsub"]
            - 10 ** (self.m_logNsub * self.df["logMvir"] + self.b_logNsub)
        )

        self.df["delta_MMs"] = (
            self.df["MMs"]
            - 10 ** (self.m_logMMs * self.df["logMvir"] + self.b_logMMs)
        )

        self.df["delta_logfsub"] = (                 
            self.df["logfsub"]                       
            - (self.m_logfsub * self.df["logMvir"] + self.b_logfsub)  
        )                                            

        self.df["delta_fsub"] = (                    
            self.df["fsub"]                          
            - 10 ** (self.m_logfsub * self.df["logMvir"] + self.b_logfsub)  
        )   

    def plot_P0(self):
        plt.subplots(figsize=(3.5, 3.5))
        plt.plot(self.logMvir_bincenters, self.P0, marker=".", lw=1, c="C0", label=self.dataset_title)

        plt.xlim(12.6, 14.0)
        plt.ylim(1e-3, 1)
        plt.xlabel("log M$_{\\rm vir}$ [$\>h^{-1}$ M$_{\\odot}$]")
        plt.ylabel("$P(\\rm N_{sub} = 0)$")

        plt.xticks([12.6, 13.0, 13.4, 13.8])

        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        # plt.savefig("../../figures/Pnsubzero.pdf", bbox_inches="tight")
        plt.show()

    def plot_poisson(self):

        fig, axes = plt.subplots(1, self.logMvir_bincenters.shape[0], sharex=True, sharey=True, figsize=(7*3, 3.5))

        for ii, center in enumerate(self.logMvir_bincenters):
            sample = self.grab_subsample(
                center - self.logMvir_binsize,
                center + self.logMvir_binsize
            )

            Nsub = sample["Nsub"]
            approx_ii, bins_ii = jsm_stats.poisson_approx(Nsub)

            axes[ii].plot(approx_ii[0], approx_ii[1], color="k",  label=f"$\\lambda$={np.mean(Nsub):.2f}", lw=2)
            axes[ii].hist(Nsub, bins=bins_ii, density=True, color="C0", edgecolor="white")
            axes[ii].set_xlabel("N$_{\\rm sub}$")
            axes[ii].legend(loc=1)

        axes[0].set_ylim(0, 0.4)
        axes[0].set_xlim(0, 35)
        axes[0].set_ylabel("PDF")

    def HAB_signal(self):

        self.rho_mat = np.empty(shape=(4, self.logMvir_bincenters.shape[0]))
        self.rho_err_mat = np.empty(shape=(4, self.logMvir_bincenters.shape[0]))
        self.rhonorm_mat = np.empty(shape=(4, self.logMvir_bincenters.shape[0]))
        self.rhonorm_err_mat = np.empty(shape=(4, self.logMvir_bincenters.shape[0]))

        self.rho_mat[0], self.rho_err_mat[0], self.Nhosts_perbin = self.measure_correlation(xkey="Nsub", ykey="log1pz50")
        self.rho_mat[1], self.rho_err_mat[1], _ = self.measure_correlation(xkey="Nsub", ykey="logc")
        self.rho_mat[2], self.rho_err_mat[2], _ = self.measure_correlation(xkey="fsub", ykey="log1pz50")
        self.rho_mat[3], self.rho_err_mat[3], _ = self.measure_correlation(xkey="fsub", ykey="logc")

        self.rhonorm_mat[0], self.rhonorm_err_mat[0], _ = self.measure_correlation(xkey="delta_Nsub", ykey="delta_log1pz50")
        self.rhonorm_mat[1], self.rhonorm_err_mat[1], _ = self.measure_correlation(xkey="delta_Nsub", ykey="delta_logc")
        self.rhonorm_mat[2], self.rhonorm_err_mat[2], _ = self.measure_correlation(xkey="delta_fsub", ykey="delta_log1pz50")
        self.rhonorm_mat[3], self.rhonorm_err_mat[3], _ = self.measure_correlation(xkey="delta_fsub", ykey="delta_logc")

        #fixing the P(Nhost=0) arrays
        # P0_upper_limit = 1/self.Nhosts_perbin
        # upper_limit_mask = self.P0 < P0_upper_limit
        # self.P0[upper_limit_mask] = P0_upper_limit[upper_limit_mask]
        bad = np.where(self.P0[1:] > self.P0[:-1])[0]
        cut = np.minimum(np.min(np.r_[bad, len(self.P0)]), len(self.P0))
        self.P0[cut:] = 0.0

        #extra stats
        self.rhocz_mat, self.rhocz_err_mat, _ = self.measure_correlation(xkey="log1pz50", ykey="logc")
        self.rhocznorm_mat, self.rhocznorm_err_mat, _ = self.measure_correlation(xkey="delta_log1pz50", ykey="delta_logc")
        

    def plot_bestfit(self, savefile=None, col="C0"):

        fig, ax = plt.subplots(4, 1, figsize=(3.5, 7), sharex=True)

        ax[0].scatter(self.df["logMvir"], self.df["log1pz50"], marker=".", s=1, alpha=0.2, c=col, rasterized=True)
        ax[1].scatter(self.df["logMvir"], self.df["logc"],     marker=".", s=1, alpha=0.2, c=col, rasterized=True)
        ax[2].scatter(self.df["logMvir"], self.df["logMMs"],   marker=".", s=1, alpha=0.2, c=col, rasterized=True)
        ax[3].scatter(self.df["logMvir"], self.df["logNsub"],  marker=".", s=1, alpha=0.2, c=col, rasterized=True)

        ax[0].errorbar(self.logMvir_bincenters, self.log1pz50_mean, yerr=self.log1pz50_std, fmt=".", color="k", capsize=3)
        ax[1].errorbar(self.logMvir_bincenters, self.logc_mean,     yerr=self.logc_std,     fmt=".", color="k", capsize=3)
        ax[2].errorbar(self.logMvir_bincenters, self.logfsub_mean,   yerr=self.logfsub_std,   fmt=".", color="k", capsize=3)
        ax[3].errorbar(self.logMvir_bincenters, self.logNsub_mean,  yerr=self.logNsub_std,  fmt=".", color="k", capsize=3)

        ax[0].plot(self.logMvir_smooth, self.log1pz50_smooth, color="k")
        ax[1].plot(self.logMvir_smooth, self.logc_smooth,     color="k")
        ax[2].plot(self.logMvir_smooth, self.logfsub_smooth,   color="k")
        ax[3].plot(self.logMvir_smooth, self.logNsub_smooth,  color="k")

        ax[0].text(0.72, 0.7, s=f"m = {self.m_log1pz50:.2f}\nb = {self.b_log1pz50:.2f}", fontsize=11, transform=ax[0].transAxes, bbox=dict(boxstyle="round", facecolor="white"))
        ax[1].text(0.72, 0.7, s=f"m = {self.m_logc:.2f}\nb = {self.b_logc:.2f}",         fontsize=11, transform=ax[1].transAxes, bbox=dict(boxstyle="round", facecolor="white"))
        ax[2].text(0.72, 0.1, s=f"m = {self.m_logfsub:.2f}\nb = {self.b_logfsub:.2f}",     fontsize=11, transform=ax[2].transAxes, bbox=dict(boxstyle="round", facecolor="white"))
        ax[3].text(0.72, 0.1, s=f"m = {self.m_logNsub:.2f}\nb = {self.b_logNsub:.2f}",   fontsize=11, transform=ax[3].transAxes, bbox=dict(boxstyle="round", facecolor="white"))

        ax[0].set_ylabel("log (1+z$_{50}$)")
        ax[1].set_ylabel("log c$_{\\rm vir}$")
        ax[2].set_ylabel("log f$_{\\rm sub}$")   
        ax[3].set_ylabel("log N$_{\\rm sub}$")

        ax[0].set_ylim(0, 0.62)
        ax[1].set_ylim(0, 1.8)
        ax[2].set_ylim(-3, 0)
        ax[3].set_ylim(-0.2, 2.2)
        ax[3].set_xlim(12.5, 14.1)

        ax[0].set_yticks([0.0, 0.2, 0.4, 0.6])
        ax[1].set_yticks([0.0, 0.6, 1.2, 1.8])
        ax[2].set_yticks([-3.0, -2.0, -1.0, 0.0])
        ax[3].set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])

        for a in ax:
            a.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

        ax[3].set_xlabel("log M$_{\\rm vir}$ [$\>h^{-1}$ M$_{\\odot}$]")
        ax[0].set_title(self.dataset_title, c=col)

        plt.tight_layout()

        if savefile:
            plt.savefig(savefile, bbox_inches="tight")

        plt.show()

    def write_summary_tabs(self, filepath):

        self.df.to_csv(filepath + self.dataset_title + ".csv", index=False)
        np.save(filepath + self.dataset_title + "_bestfitvalues.npy", self.bestfit_mat)
        np.save(filepath + self.dataset_title + "_rhomat.npy", self.rho_mat)
        np.save(filepath + self.dataset_title + "_rhomat_err.npy", self.rho_err_mat)



    # def plot_fullcorr(self):

    #     self.df = self.df.sort_values(by="logMvir")

    #     fig, ax = plt.subplots(1, 3, figsize=(7, 3.5), sharey=True)

    #     # ==================================================
    #     # Shared colormap + normalization
    #     # ==================================================

    #     vmin, vmax = 12.5, 14.1

    #     cmap = plt.cm.viridis  # choose any cmap you like
    #     norm = Normalize(vmin=vmin, vmax=vmax)

    #     # ===============================
    #     # Panel 1
    #     # ===============================

    #     ax[0].set_xlabel(r"$\Delta [\log(1+z_{50})]$")
    #     ax[0].set_ylabel(r"$\Delta [\log N_{\rm sub}]$")

    #     ax[0].axhline(0, ls="--", color="k", zorder=11)
    #     ax[0].axvline(0, ls="--", color="k", zorder=11)

    #     qs_z50, rho_z50, pval_z50 = jsm_stats.quadrant_percentages_plot(
    #         self.df["delta_log1pz50"],
    #         self.df["delta_logNsub"])

    #     sm0 = ax[0].scatter(
    #         self.df["delta_log1pz50"],
    #         self.df["delta_logNsub"],
    #         c=self.df["logMvir"],
    #         cmap=cmap,
    #         norm=norm,
    #         marker="."
    #     )

    #     # Quadrant labels
    #     ax[0].text(0.755, 0.95, qs_z50[0], fontsize=10,
    #             transform=ax[0].transAxes,
    #             bbox=dict(boxstyle="round", facecolor="white"))
    #     ax[0].text(0.755, 0.03, qs_z50[1], fontsize=10,
    #             transform=ax[0].transAxes,
    #             bbox=dict(boxstyle="round", facecolor="white"))
    #     ax[0].text(0.02, 0.03, qs_z50[2], fontsize=10,
    #             transform=ax[0].transAxes,
    #             bbox=dict(boxstyle="round", facecolor="white"))
    #     ax[0].text(0.02, 0.95, qs_z50[3], fontsize=10,
    #             transform=ax[0].transAxes,
    #             bbox=dict(boxstyle="round", facecolor="white"))

    #     ax[0].set_title(rf"$\rho_S = {rho_z50:.2f}$")

    #     # ===============================
    #     # Panel 2
    #     # ===============================

    #     ax[1].set_xlabel(r"$\Delta [\log c]$")
    #     ax[1].axhline(0, ls="--", color="k", zorder=11)
    #     ax[1].axvline(0, ls="--", color="k", zorder=11)

    #     qs_c, rho_c, pval_c = jsm_stats.quadrant_percentages_plot(
    #         self.df["delta_logc"],
    #         self.df["delta_logNsub"])

    #     sm1 = ax[1].scatter(
    #         self.df["delta_logc"],
    #         self.df["delta_logNsub"],
    #         c=self.df["logMvir"],
    #         cmap=cmap,
    #         norm=norm,
    #         marker="."
    #     )

    #     # Quadrant labels
    #     ax[1].text(0.755, 0.95, qs_c[0], fontsize=10,
    #             transform=ax[1].transAxes,
    #             bbox=dict(boxstyle="round", facecolor="white"))
    #     ax[1].text(0.755, 0.03, qs_c[1], fontsize=10,
    #             transform=ax[1].transAxes,
    #             bbox=dict(boxstyle="round", facecolor="white"))
    #     ax[1].text(0.02, 0.03, qs_c[2], fontsize=10,
    #             transform=ax[1].transAxes,
    #             bbox=dict(boxstyle="round", facecolor="white"))
    #     ax[1].text(0.02, 0.95, qs_c[3], fontsize=10,
    #             transform=ax[1].transAxes,
    #             bbox=dict(boxstyle="round", facecolor="white"))

    #     ax[1].set_title(rf"$\rho_S = {rho_c:.2f}$")

    #     # ===============================
    #     # Panel 3
    #     # ===============================

    #     ax[2].set_xlabel(r"$\Delta [\log MMs]$")
    #     ax[2].axhline(0, ls="--", color="k", zorder=11)
    #     ax[2].axvline(0, ls="--", color="k", zorder=11)

    #     qs_MMs, rho_MMs, pval_c = jsm_stats.quadrant_percentages_plot(
    #         self.df["delta_logMMs"],
    #         self.df["delta_logNsub"])

    #     sm2 = ax[2].scatter(
    #         self.df["delta_logMMs"],
    #         self.df["delta_logNsub"],
    #         c=self.df["logMvir"],
    #         cmap=cmap,
    #         norm=norm,
    #         marker="."
    #     )

    #     # Quadrant labels
    #     ax[2].text(0.755, 0.95, qs_MMs[0], fontsize=10,
    #             transform=ax[2].transAxes,
    #             bbox=dict(boxstyle="round", facecolor="white"))
    #     ax[2].text(0.755, 0.03, qs_MMs[1], fontsize=10,
    #             transform=ax[2].transAxes,
    #             bbox=dict(boxstyle="round", facecolor="white"))
    #     ax[2].text(0.02, 0.03, qs_MMs[2], fontsize=10,
    #             transform=ax[2].transAxes,
    #             bbox=dict(boxstyle="round", facecolor="white"))
    #     ax[2].text(0.02, 0.95, qs_MMs[3], fontsize=10,
    #             transform=ax[2].transAxes,
    #             bbox=dict(boxstyle="round", facecolor="white"))

    #     ax[2].set_title(rf"$\rho_S = {rho_MMs:.2f}$")

    #     # ===============================
    #     # Shared colorbar
    #     # ===============================

    #     sm = ScalarMappable(norm=norm, cmap=cmap)
    #     sm.set_array([])

    #     cbar = fig.colorbar(
    #         sm,
    #         ax=ax,
    #         orientation="horizontal",
    #         pad=0.2,
    #         fraction=0.05
    #     )

    #     cbar.set_label(r"log M$_{\rm vir}$ [M$_{\odot}$]")

    #     # Optional fixed ticks
    #     cbar.set_ticks([12.6, 13.0, 13.4, 13.8, 14.1])

    #     ax[0].set_ylim(-1, 1)
    #     ax[0].set_xlim(-1, 1)
    #     ax[1].set_xlim(-1, 1)
    #     ax[2].set_xlim(-1, 1)

    #     # fig.suptitle(self.dataset_title)
    #     fig.tight_layout(rect=[0, 0.25, 1, 0.95])

    #     plt.show()