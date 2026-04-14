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


class Bolshoi_HaloCatalogue:

    def __init__(self,filepath,
        npart_thresh=100,
        xoff_thresh=0.07,
        spin_thresh=0.07,
        isolation_factor=3):
 
        self.filepath         = filepath
        self.npart_thresh     = npart_thresh
        self.mass_thresh      = npart_thresh * (1.5e8)           # Msun — for reference
        self.log_mass_thresh  = np.log10(npart_thresh * (1.5e8)) # log10 Msun — for comparisons
        self.xoff_thresh      = xoff_thresh
        self.spin_thresh      = spin_thresh
        self.isolation_factor = isolation_factor

        self._load()
        self._relaxation_cut()
        self._isolation_cut()
        self._print_counts()

    # ------------------------------------------------------------------ #
    # Private methods                                                      #
    # ------------------------------------------------------------------ #

    def _load(self):

        COLS = [
            "host_id", "logMh", "ch", "a_50h",
            "Xoff_h", "Spin_h", "Spin_Bullock_h", "ch_K",
            "x_h", "y_h", "z_h", "R_vir",
            "id", "log10Mvir", "Rvir", "rs", "vrms", "scale_of_last_MM",
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
            "Tidal_Force_Tdyn"
        ]

        raw        = np.loadtxt(self.filepath)
        self._df   = pd.DataFrame(raw, columns=COLS)

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

        for i in range(len(host_relaxed)):
            search_r   = self.isolation_factor * r_virial[i]
            neighbours = tree.query_ball_point(coords[i], r=search_r)
            for j in neighbours:
                if j == i:
                    continue
                if masses[j] >= masses[i]:
                    isolated[i] = False
                    break

        isolated_ids  = host_relaxed[isolated]["host_id"].values
        self._df_isolated  = self._df_relaxed[self._df_relaxed["host_id"].isin(isolated_ids)]

    def _print_counts(self):
        n_raw     = self._df["host_id"].unique().shape[0]
        n_relaxed = self._df_relaxed["host_id"].unique().shape[0]
        n_final   = self._df_isolated["host_id"].unique().shape[0]
        print(
            f"Hosts: total={n_raw}  "
            f"after relaxation={n_relaxed} ({100*n_relaxed/n_raw:.1f}%)  "
            f"after isolation={n_final} ({100*n_final/n_raw:.1f}%)"
        )

    def _build_host_table(self, sample):

        if sample == "isolated":
            host_id_unique = np.sort(self._df_isolated["host_id"].unique())
            groups = self._df_isolated.groupby("host_id")

        if sample == "relaxed":
            host_id_unique = np.sort(self._df_relaxed["host_id"].unique())
            groups = self._df_relaxed.groupby("host_id")

        if sample == "all":
            host_id_unique = np.sort(self._df["host_id"].unique())
            groups = self._df.groupby("host_id")

        logMvir  = np.zeros(len(host_id_unique))
        log1pz50 = np.zeros(len(host_id_unique))
        logc     = np.zeros(len(host_id_unique))
        logNsub  = np.zeros(len(host_id_unique))
        logfsub  = np.zeros(len(host_id_unique))

        for i, hid in enumerate(host_id_unique):
            subset  = groups.get_group(hid)
            subset1 = subset[subset["log10Mvir"] >= self.log_mass_thresh]

            logMh_i  = subset["logMh"].mean()
            c_h_i    = subset["ch"].mean()
            a_half_i = subset["a_50h"].mean()

            z50            = (1.0 / a_half_i) - 1.0
            Nsub_i         = len(subset1)
            host_mass      = 10**logMh_i
            sub_mass_total = np.sum(10**subset1["log10Mvir"])
            f_sub_i        = sub_mass_total / host_mass

            logMvir[i]  = logMh_i
            log1pz50[i] = np.log10(1.0 + z50)
            logc[i]     = np.log10(c_h_i)
            logNsub[i]  = np.log10(Nsub_i)  if Nsub_i > 0  else np.nan
            logfsub[i]  = np.log10(f_sub_i) if f_sub_i > 0 else np.nan

        host_table = pd.DataFrame({
            "logMvir":  logMvir,
            "log1pz50": log1pz50,
            "logc":     logc,
            "logNsub":  logNsub,
            "logfsub":  logfsub,
        })

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


def mass_binned_rho(datasets, xkey, ykey, xlabel, ylabel, make_plot=True):

    lgM_min = [12.5, 12.7, 12.9, 13.1, 13.3, 13.5, 13.7, 13.9]
    lgM_max = [12.7, 12.9, 13.1, 13.3, 13.5, 13.7, 13.9, 14.1]

    k_tot = len(datasets)
    n_bins = len(lgM_min)

    # ---------------------------------------
    # Storage for correlation coefficients
    # ---------------------------------------
    rho_array = np.zeros((k_tot, n_bins))
    N_hosts = np.zeros([n_bins])

    if make_plot:
        # get default color cycle
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colorz = [default_colors[i % len(default_colors)] for i in range(k_tot)]

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

    # Main loop
    for k, dataset in enumerate(datasets):
        for i in range(n_bins):

            subsample = (
                (dataset["logMvir"] > lgM_min[i]) &
                (dataset["logMvir"] <= lgM_max[i])
            )

            x = dataset[xkey][subsample]
            y = dataset[ykey][subsample]
            N = len(x)

            qs, rho, p_val = jsm_stats.quadrant_percentages_plot(x, y)

            # ---------------------------------------
            # SAVE rho here (always)
            # ---------------------------------------
            rho_array[k, i] = float(rho)
            N_hosts[i] = N

            if make_plot:
                axes[k, i].scatter(x, y, marker=".", s=1, alpha=1, color=colorz[k])

                rho_label = "$\\rho_S$=" + f"{rho:.2f}" + f"\nN={N}"
                axes[k, i].text(
                    0.7, 0.2,
                    rho_label,
                    fontsize=11,
                    color="red" if p_val < 0.05 else "grey",
                    transform=axes[k, i].transAxes,
                    bbox=dict(boxstyle="round", facecolor="white")
                )

        if make_plot:
            axes[k, 0].set_ylabel(ylabel)

    if make_plot:
        for i in range(n_bins):
            axes[-1, i].set_xlabel(xlabel)
        plt.tight_layout()

    return rho_array, N_hosts


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
        self.m_logz50, self.b_logz50 = jsm_stats.fit_line_sym_errors(self.logMvir_bincenters, self.logz50_med, self.logz50_std, p0=(0.5, 1.0))
        # ---- Nsub ----
        self.m_logNsub, self.b_logNsub = jsm_stats.fit_line_sym_errors(self.logMvir_bincenters, self.logNsub_med, self.logNsub_std, p0=(1.0, 1.0))
        # ---- concentration ----
        self.m_logc, self.b_logc = jsm_stats.fit_line_sym_errors(self.logMvir_bincenters, self.logc_med, self.logc_std, p0=(1, 1.0))

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

        qs_a50, rho_a50, pval_a50 = jsm_stats.quadrant_percentages_plot(
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

        qs_c, rho_c, pval_c = jsm_stats.quadrant_percentages_plot(
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
        self.logz50_med, self.logz50_std = jsm_stats.finite_binned_stat(self.logMvir, self.logz50, self.logMvir_bins)
        # ---- Nsub (handles -inf safely) ----
        self.logNsub_med, self.logNsub_std = jsm_stats.finite_binned_stat(self.logMvir, self.logNsub, self.logMvir_bins)
        # ---- concentration ----
        self.logc_med, self.logc_std = jsm_stats.finite_binned_stat(self.logMvir, self.logc, self.logMvir_bins)
        # bin centers
        self.logMvir_bincenters = 0.5 * (self.logMvir_bins[:-1] + self.logMvir_bins[1:])

    def fit_lines(self):

        # ---- a50 ----
        self.m_logz50, self.b_logz50 = jsm_stats.fit_line_sym_errors(self.logMvir_bincenters, self.logz50_med, self.logz50_std, p0=(0.5, 1.0))
        # ---- Nsub ----
        self.m_logNsub, self.b_logNsub = jsm_stats.fit_line_sym_errors(self.logMvir_bincenters, self.logNsub_med, self.logNsub_std, p0=(1.0, 1.0))
        # ---- concentration ----
        self.m_logc, self.b_logc = jsm_stats.fit_line_sym_errors(self.logMvir_bincenters, self.logc_med, self.logc_std, p0=(1, 1.0))
        
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

        qs_a50, rho_a50, pval_a50 = jsm_stats.quadrant_percentages_plot(
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

        qs_c, rho_c, pval_c = jsm_stats.quadrant_percentages_plot(
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
