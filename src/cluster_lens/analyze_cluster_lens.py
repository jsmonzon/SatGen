######################### analyze_cluster_lens ##########################
#
# Reads the evolved cluster_lens trees with jsm_stellarhalo.Tree_Reader
# and plots, following the conventions of the van den Bosch group papers:
#   1. SHMF  -- dN/dlog(psi) vs log(psi), psi = m/M0, M0 = host mass at
#      z_obs; evolved = m at z_obs, all orders, r < r_vir; unevolved =
#      m_acc/M0, all orders (Jiang & vdB 2016; Green, vdB & Jiang 2021).
#      Mean over trees + 16-84 percentile band; thin lines = single trees.
#   2. radial distribution -- dN~/dx^3 vs x = r/r_vir with N~ = N/N(<r_vir),
#      compared to the host NFW rho(r)/rho_vir_mean, plus the bias
#      (ratio) panel (Green+21, Fig. 7). All orders, m > M_floor.
#   3. MAH -- top: M/M0 (linear) vs cosmic time; bottom:
#      log(M/M0) vs log(1+z) (vdB+14, Fig. 2). Median + 16-84 range;
#      formation redshifts z_0.5 and z_0.04 (vdB 2002; vdB+14).
#
# Usage:  python analyze_cluster_lens.py [datadir]
#   datadir is a folder of tree_{i}_evo.npz files, e.g. .../cluster_1/;
#   figures go to figures/<that folder name>/.
#
# NOTE on index 0: Tree_Reader's "z=0" quantities (mass[:,0], shmf_z0,
# ...) are index 0 of the arrays, which for these trees is z_obs = 0.436.
#
# NOTE on the grid: Tree_Reader builds host profiles with cfg.Dvsample,
# which config.py builds from z=0. The script rebuilds cfg's grid to start
# at the files' own z_obs (redshift[0]) before any Tree_Reader is built;
# M_floor is read from the files too.
#
# NOTE on disruption: Tree_Reader flags disruption at m/m_acc < 1e-4, but
# these runs use a hard floor (M_floor = 5e9). A subhalo that dropped below
# the floor is frozen at its last mass (< M_floor) and parked at r = Rres,
# so it can look "alive" to the 1e-4 test. Passing
# mass_threshold = M_floor removes these from every regime used here.
#
# NOTE on metadata: master_cluster_{1,2} save run settings (M_floor,
# lnL_pref, ...) as scalars in each file. Tree_Reader.read_arrays loops
# over every key and np.delete()s row 1, which fails on scalars, so
# ClusterTreeReader hands it the arrays only. Tree_Reader is unchanged.
#########################################################################

import os, sys, glob, gc, re
import numpy as np
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))                   # SatGen/src
sys.path.insert(0, os.path.join(here, "..", "..", "mcmc", "src"))  # SatGen/mcmc/src

import master_cluster_1 as mcl   # only for set_output_grid() (see header)
import config as cfg
import jsm_stellarhalo as sh

datadir = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/jsmonzon/Research/collaborations/for_rebecca/tree_data/cluster_1/"
# figures/<cluster folder name>, e.g. figures/cluster_1/
run_name = os.path.basename(os.path.normpath(datadir))
outdir = os.path.join(here, "figures", run_name)
os.makedirs(outdir, exist_ok=True)

files = sorted(glob.glob(os.path.join(datadir, "tree_*_evo.npz")))
assert files, f"no *_evo.npz files in {datadir}"

# take z_obs and the evolution floor from the files themselves, so the same
# script works for every cluster_lens run (5e14 @ 0.436, 1.8e15 @ 0.439, ...)
with np.load(files[0]) as f0:
    z_file = f0["redshift"]
    M_floor = float(f0["M_floor"])
mcl.set_output_grid(z_file[0])
assert len(z_file) == cfg.Nz and np.allclose(z_file, cfg.zsample), \
    "rebuilt cfg grid does not match the tree files"

######################### read every tree ###############################

class _NumpyArraysOnly:
    """Stands in for numpy inside jsm_stellarhalo during read_arrays():
    np.load returns the pre-filtered arrays, everything else is numpy."""
    def __init__(self, arrays):
        self._arrays = arrays
    def load(self, *args, **kwargs):
        return self._arrays
    def __getattr__(self, name):
        return getattr(np, name)

class ClusterTreeReader(sh.Tree_Reader):
    def read_arrays(self):
        with np.load(self.file) as f:
            arrays = {k: f[k] for k in f.files if f[k].ndim > 0}
            self.run_settings = {k: f[k].item() for k in f.files
                                 if f[k].ndim == 0}
        sh.np = _NumpyArraysOnly(arrays)
        try:
            super().read_arrays()
        finally:
            sh.np = np
        # Tree_Reader guesses the index from the 3rd "_" token of the name;
        # take it from "tree_{i}_evo.npz" (or old "tree_{mass}_{i}_evo.npz")
        self.tree_index = re.findall(r"_(\d+)_evo\.npz$",
                                     os.path.basename(self.file))[0]

def reduce_tree(t):
    """Keep only what the figures need, so each full Tree_Reader can be
    freed before the next one is read (a full one takes several GB)."""
    M0, Rv = t.mass[0, 0], t.VirialRadius[0, 0]
    sel = t.regime_masks["rvir_surv"][:, 0]
    host = t.host_profiles[0]
    xr = np.logspace(-2, 0, 200)
    shmf = t.shmf_z0["rvir_surv"]
    return dict(
        index=t.tree_index, M0=M0, Rv=Rv, Nhalo=t.Nhalo, z50=t.host_z50,
        n_surv=int(t.regime_masks["surviving"][:, 0].sum()),
        n_rvir=int(sel.sum()),
        max_m=t.max_msub_z0["rvir_surv"],
        m_ev=shmf[:, 0][np.isfinite(shmf[:, 0])],
        m_k1=shmf[:, 1][np.isfinite(shmf[:, 1])],
        m_un=t.acc_mass[1:][np.isfinite(t.acc_mass[1:])],
        r=t.rmags_stitched[sel, 0],
        xr=xr, host_Mr=np.array([host.M(x * Rv) / host.Mh for x in xr]),
        redshift=t.redshift, time=t.CosmicTime, mah=t.mass[0],
        c_host=t.concentration[0, 0],
        settings=t.run_settings)

trees = []
for f in files:
    print("reading", os.path.basename(f), flush=True)
    t = ClusterTreeReader(file=f, verbose=False, mass_threshold=M_floor)
    trees.append(reduce_tree(t))
    del t; gc.collect()

colors = plt.cm.viridis(np.linspace(0, 0.9, len(trees)))
labels = [f"tree {t['index']}" for t in trees]
zobs = trees[0]["redshift"][0]
print("run settings in files:", trees[0]["settings"])

#---summary table
print(f"\n{'tree':>5} {'Nbranch':>8} {'N_surv':>7} {'N_rvir':>7} "
      f"{'z50':>6} {'Rvir[kpc]':>10} {'max m/M':>8}")
for t in trees:
    print(f"{t['index']:>5} {t['Nhalo']:>8} {t['n_surv']:>7} "
          f"{t['n_rvir']:>7} {t['z50']:>6.2f} {t['Rv']:>10.0f} "
          f"{t['max_m']/t['M0']:>8.3f}")

def band(arr, stat="mean"):
    """centre (mean or median) and 16/84 percentiles across trees"""
    centre = np.nanmean(arr, 0) if stat == "mean" else np.nanmedian(arr, 0)
    lo, hi = np.nanpercentile(arr, [16, 84], axis=0)
    return centre, lo, hi

M0 = trees[0]["M0"]
lpsi_floor = np.log10(M_floor / M0)

############################# 1. SHMF ###################################
# dN/dlog(psi), psi = m/M0, 0.2-dex bins (Jiang & vdB 2016)

bins = np.arange(np.floor(lpsi_floor*5)/5, 0.0 + 1e-9, 0.2)
bc = 0.5*(bins[1:] + bins[:-1])
dlog = np.diff(bins)

def dNdlog(m, M):
    return np.histogram(np.log10(m / M), bins)[0] / dlog

ev = np.array([dNdlog(t["m_ev"], t["M0"]) for t in trees])
un = np.array([dNdlog(t["m_un"], t["M0"]) for t in trees])
ok = bc > lpsi_floor            # bins fully above the floor

fig, ax = plt.subplots(figsize=(6, 5))
for arr, col, lab in [(un, "C0", "unevolved ($m_{\\rm acc}/M_0$)"),
                      (ev, "C3", "evolved ($m/M_0$, $r<r_{\\rm vir}$)")]:
    for row in arr:
        ax.plot(bc[ok], np.where(row[ok] > 0, row[ok], np.nan), color=col,
                lw=0.6, alpha=0.35)
    c, lo, hi = band(arr)
    ax.fill_between(bc[ok], lo[ok], hi[ok], color=col, alpha=0.2, lw=0)
    ax.plot(bc[ok], c[ok], color=col, lw=2, label=lab)
ax.set_yscale("log")
ax.set_xlim(-5, 0)
ax.set_xlabel(r"$\log(\psi)$,   $\psi = m/M_0$")
ax.set_ylabel(r"${\rm d}N/{\rm d}\log\psi$")
ax.set_title(f"SHMF, all orders, $M_0$ = {M0:.1e} M$_\\odot$, "
             f"z = {zobs:.3f}\nmean of {len(trees)} trees, band: 16-84%")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(outdir, "shmf.pdf"))

####################### 2. radial distribution ##########################
# dN~/dx^3, x = r/r_vir, N~ = N/N(<r_vir)  vs  host rho(r)/<rho>_vir

xb = np.logspace(-2, 0, 13)
xc = np.sqrt(xb[1:] * xb[:-1])
dx3 = np.diff(xb**3)

def dNdx3(r, Rv):
    x = r / Rv
    return np.histogram(x, xb)[0] / dx3 / x.size

prof = np.array([dNdx3(t["r"], t["Rv"]) for t in trees])

def nfw_rho_over_mean(x, c):
    # rho(x)/<rho>_vir = (c^2/3) / [m(c) * cx (1+cx)^2],  m(c)=ln(1+c)-c/(1+c)
    mc = np.log(1 + c) - c/(1 + c)
    return (c**2 / 3.) / (mc * (c*x) * (1 + c*x)**2) * c

nfw = np.array([nfw_rho_over_mean(xc, t["c_host"]) for t in trees])

fig, (ax, axb) = plt.subplots(2, 1, figsize=(6, 6.5), sharex=True,
                              gridspec_kw=dict(height_ratios=[2.2, 1],
                                               hspace=0))
for row in prof:
    ax.plot(xc, np.where(row > 0, row, np.nan), color="C3", lw=0.6,
            alpha=0.35)
c, lo, hi = band(prof)
ax.fill_between(xc, lo, hi, color="C3", alpha=0.2, lw=0)
ax.plot(xc, c, color="C3", lw=2, label=f"subhaloes, $m > {M_floor:.0e}$")
cn = nfw.mean(0)
ax.plot(xc, cn, color="k", ls="--", lw=1.5, label="host NFW")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylabel(r"${\rm d}\tilde N/{\rm d}x^3$")
ax.set_title(f"radial distribution, all orders, z = {zobs:.3f}\n"
             f"mean of {len(trees)} trees, band: 16-84%")
ax.legend(frameon=False)

bias = prof / nfw
for row in bias:
    axb.plot(xc, np.where(row > 0, row, np.nan), color="C3", lw=0.6,
             alpha=0.35)
c, lo, hi = band(bias)
axb.fill_between(xc, lo, hi, color="C3", alpha=0.2, lw=0)
axb.plot(xc, c, color="C3", lw=2)
axb.axhline(1, color="k", ls="--", lw=1)
axb.set_yscale("log")
axb.set_xlabel(r"$x = r/r_{\rm vir}$")
axb.set_ylabel("bias (sub / NFW)")
fig.tight_layout()
fig.savefig(os.path.join(outdir, "radial_distribution.pdf"))

############################## 3. MAHs ##################################
# top: M/M0 vs cosmic time; bottom: log M/M0 vs log(1+z); both run from
# z_obs (left) to z = 9 (right) so the curves start and stop together

z = trees[0]["redshift"]
tcos = trees[0]["time"]   # cosmic time [Gyr]
psi = np.array([t["mah"] / t["M0"] for t in trees])

def zf(t, f):
    above = np.where(t["mah"] >= f * t["M0"])[0]
    return t["redshift"][above.max()]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
for row in psi:
    ax1.plot(tcos, row, color="C0", lw=0.6, alpha=0.35)
    ax2.plot(np.log10(1 + z), np.log10(row), color="C0", lw=0.6, alpha=0.35)
c, lo, hi = band(psi, "median")
ax1.fill_between(tcos, lo, hi, color="C0", alpha=0.2, lw=0)
ax1.plot(tcos, c, color="C0", lw=2, label="median")
ax2.fill_between(np.log10(1 + z), np.log10(lo), np.log10(hi), color="C0",
                 alpha=0.2, lw=0)
ax2.plot(np.log10(1 + z), np.log10(c), color="C0", lw=2)

for f, ls in [(0.5, "--"), (0.04, ":")]:
    zs = [zf(t, f) for t in trees]
    ax2.axhline(np.log10(f), color="k", ls=ls, lw=1)
    lab = f"$z_{{{f:g}}}$ = " + ", ".join(f"{v:.2f}" for v in zs)
    ax2.plot([], [], color="k", ls=ls, label=lab)
    print(f"z_{f:g} per tree:", np.round(zs, 2))

ax1.set_xlabel(r"cosmic time $t$ [Gyr]")
ax1.set_ylabel(r"$M(t)/M_0$")
# same span AND direction as the bottom panel: left = z_obs, right = z = 9
# (log(1+z) = 1), so cosmic time decreases to the right
ax1.set_xlim(tcos[0], tcos[np.log10(1 + z) <= 1.0].min())
ax1.set_ylim(0, 1.02)
ax1.set_title(f"host MAHs, $M_0$ = {M0:.1e} M$_\\odot$ at z = {zobs:.3f}\n"
              f"{len(trees)} trees, band: 16-84%")
ax1.legend(frameon=False)
ax2.set_xlabel(r"$\log(1+z)$")
ax2.set_ylabel(r"$\log[M(z)/M_0]$")
ax2.set_xlim(np.log10(1 + z[0]), 1.0)
in_range = np.log10(1 + z) <= 1.0
ax2.set_ylim(np.floor(np.nanmin(np.log10(psi[:, in_range]))*2)/2, 0.1)
ax2.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(outdir, "mah.pdf"))

print(f"\nfigures written to {outdir}")
plt.show()
