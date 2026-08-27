"""
jsm_processh5.py

Consolidates the h5-directory loading/derivation routines scattered across
jsm_ancillary.py (load_sample, load_massspec_z0, load_shmf_z0,
load_massspec_timeseries) and the single-file z=0 loader written for
MassSpec/notebooks/paper3/epsilon_vs_fiducial.ipynb, into one class:
ProcessH5.

Point it at a directory of Tree_Reader.write_out_abundance() h5 files --
one file per mass bin (e.g. MassSpec/data/zhao/N1000/*.h5) or a
single-mass-bin directory with one or more files (e.g.
MassSpec/data/epsilon/{fiducial,epsilon}.h5) -- and it builds tidy
one-row-per-tree tables:

    z0_table          every column a z=0 scalar               -> csv
    shmf_table        SHMF columns are ragged per-tree arrays  -> npz
    timeseries_table  columns are (Ntime,) per-tree arrays     -> npz

The z0 table is the primary product and the one meant for "just load this
into a notebook" -- plain scalars, one row per tree, a straight
pd.read_csv(). shmf/timeseries cells hold whole arrays, which don't
serialize cleanly to CSV (each cell would need to be a stringified list),
so those go to .npz instead -- the same convention already used elsewhere
in this project for array-shaped MAH-like data (see
MassSpec/data/zhao/N1000/{median,mean,ratio}_MAH/).

When a directory holds more than one h5 file, tree_index values can repeat
across files (each file's Tree_Reader assigns its own tree_index), so every
table also carries a `source_file` column to disambiguate.

NOT reproduced here:
  - jsm_ancillary.load_massspec() / load_massspec_MW() -- both read an
    older h5 key schema (Nsub_{sub_key} / N_{sub_key}) that predates the
    current Nsub_{regime}_{order} convention Tree_Reader.write_out_abundance()
    now writes. Reproducing them here would silently read the wrong keys
    against current output.
  - jsm_simload.HaloCatalogue -- unrelated: it reads real observational
    catalogues (BolshoiP/VSMDPL hlists), not SatGen merger-tree h5 output.
  - jsm_ancillary.compute_mass_bin_stats() as originally written has a bug
    (its per-bin result dict is overwritten every loop iteration, so only
    the last mass bin survives) -- see mass_bin_stats() below for the fixed
    version instead.

Usage:
    from jsm_processh5 import ProcessH5

    proc = ProcessH5("/path/to/N1000")                 # multi-mass-bin directory
    proc.process(which=("z0",))                        # -> N1000_z0.csv next to the h5 files

    proc = ProcessH5("/path/to/epsilon", label="epsilon_vs_fiducial")
    proc.process(which=("z0", "shmf"))                 # -> epsilon_vs_fiducial_z0.csv + _shmf.npz
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py


class ProcessH5:

    def __init__(self, datadir, files=None, regime="artificial", order="all",
                 conctype="measured", shmf_regimes=("surviving", "rvir_surv", "artificial"),
                 label=None, verbose=True):
        """
        datadir      : directory containing Tree_Reader.write_out_abundance() h5 files
        files        : optional explicit list of h5 paths, overriding the datadir.glob("*.h5") scan
        regime       : one of "total", "massive", "surviving", "rvir", "rvir_surv", "artificial"
        order        : "all", "k1", "k2", or "k3" -- subhalo order selection for Nsub/logNsub
        conctype     : "measured" (c_measured_fixed_COM, from compute_concentration), "ludlow"
                       (ludlow_c), or None (the analytic host_c)
        shmf_regimes : regimes to include when building the SHMF table
        label        : filename prefix used when saving tables; defaults to datadir's own name
        """
        self.datadir = Path(datadir)
        self.regime = regime
        self.order = order
        self.conctype = conctype
        self.shmf_regimes = shmf_regimes
        self.verbose = verbose
        self.label = label or self.datadir.name

        if files is not None:
            self.files = [Path(f) for f in files]
        else:
            self.files = sorted(self.datadir.glob("*.h5"))

        if not self.files:
            raise FileNotFoundError(f"no .h5 files found in {self.datadir}")

        self.z0_table = None
        self.shmf_table = None
        self.timeseries_table = None

        if self.verbose:
            print(f"ProcessH5: found {len(self.files)} h5 file(s) in {self.datadir}")

    # ------------------------------------------------------------------
    # core h5 reading primitives (jsm_ancillary.load_sample / _stack_column)
    # ------------------------------------------------------------------

    @staticmethod
    def load_sample(filename):
        """Read one h5 file into a DataFrame, one row per tree (group);
        each column is either a scalar or a 1D array."""
        data = {}
        with h5py.File(filename, "r") as f:
            for sim_name in f.keys():
                row = {}
                for attr_name in f[sim_name].keys():
                    dset = f[sim_name][attr_name]
                    row[attr_name] = dset[()] if dset.shape == () else dset[:]
                data[sim_name] = row
        return pd.DataFrame.from_dict(data, orient="index")

    @staticmethod
    def _stack_column(dataframe, key):
        """Stack a per-tree 1D array column into an (Ntrees, Ntime) matrix,
        NaN-padded if array lengths differ across trees."""
        arrays = dataframe[key].values
        max_len = max(len(a) for a in arrays)
        matrix = np.full((len(arrays), max_len), np.nan)
        for i, arr in enumerate(arrays):
            matrix[i, :len(arr)] = arr
        return matrix

    @staticmethod
    def _clean_scalar(arr):
        arr = np.array(arr, dtype=float)  # np.array (not asarray) forces a writable copy
        arr[~np.isfinite(arr)] = np.nan
        return arr

    def _logc(self, ii, conctype):
        if conctype == "ludlow":
            return self._clean_scalar(np.log10(1 + ii["ludlow_c"].values))
        elif conctype == "measured":
            return self._clean_scalar(np.log10(self._stack_column(ii, "c_measured_fixed_COM")[:, 0]))
        else:
            return self._clean_scalar(np.log10(self._stack_column(ii, "host_c")[:, 0]))

    # ------------------------------------------------------------------
    # z=0 summary table (jsm_ancillary.load_massspec_z0's per-file logic,
    # generalized in the epsilon_vs_fiducial.ipynb notebook's load_z0_table,
    # looped across every file in the directory here)
    # ------------------------------------------------------------------

    def _z0_from_file(self, h5_path):
        regime, order, conctype = self.regime, self.order, self.conctype
        ii = self.load_sample(h5_path)

        Nsub_key = f"Nsub_{regime}_{order}"
        Nsub_z0 = self._clean_scalar(self._stack_column(ii, Nsub_key)[:, 0])

        # fsub has no 'all' row (mass is inclusive, so k1+k2+k3 would double-count);
        # k1 already accounts for all of its children, so use that for order="all"
        mass_order = "k1" if order == "all" else order
        fsub_z0 = self._clean_scalar(self._stack_column(ii, f"fsub_{regime}_{mass_order}")[:, 0])

        MMs_z0 = self._clean_scalar(np.asarray(ii[f"MMs_z0{regime}"].values))
        MMs_z0[np.isnan(MMs_z0)] = 0.0

        logMvir = self._clean_scalar(np.log10(self._stack_column(ii, "MAH")[:, 0]))
        log1pz50 = self._clean_scalar(np.log10(1 + ii["host_z50"].values))
        logc = self._logc(ii, conctype)

        df = pd.DataFrame({
            "tree_index": ii.index.values,
            "source_file": h5_path.name,
            "logMvir":  logMvir,
            "log1pz50": log1pz50,
            "logc":     logc,
            "Nsub":     Nsub_z0,
            "logNsub":  np.log10(Nsub_z0),
            "fsub":     fsub_z0,
            "logfsub":  np.log10(fsub_z0),
            "MMs":      MMs_z0 / (10 ** logMvir),
            "logMMs":   np.log10(MMs_z0 / (10 ** logMvir)),
        }).replace([np.inf, -np.inf], np.nan)
        return df

    def build_z0_table(self):
        frames = []
        for h5_path in self.files:
            t0 = time.time()
            df = self._z0_from_file(h5_path)
            frames.append(df)
            if self.verbose:
                print(f"  z0: {h5_path.name} -> {len(df)} trees ({time.time() - t0:.1f}s)")
        self.z0_table = pd.concat(frames, ignore_index=True).sort_values("logMvir").reset_index(drop=True)
        return self.z0_table

    # ------------------------------------------------------------------
    # full time-series table (jsm_ancillary.load_massspec_timeseries's
    # per-file logic, with conctype generalized to match the z0 builder
    # above rather than being fixed to host_c)
    # ------------------------------------------------------------------

    def _timeseries_from_file(self, h5_path):
        regime, order, conctype = self.regime, self.order, self.conctype
        ii = self.load_sample(h5_path)

        mass_order = "k1" if order == "all" else order
        Nsub_ts = self._stack_column(ii, f"Nsub_{regime}_{order}")
        Msub_ts = self._stack_column(ii, f"Msub_{regime}_{mass_order}")
        fsub_ts = self._stack_column(ii, f"fsub_{regime}_{mass_order}")

        logMvir = self._clean_scalar(np.log10(self._stack_column(ii, "MAH")[:, 0]))
        log1pz50 = self._clean_scalar(np.log10(1 + ii["host_z50"].values))
        logc = self._logc(ii, conctype)
        MMs_z0 = self._clean_scalar(np.asarray(ii[f"MMs_z0{regime}"].values))
        MMs_z0[np.isnan(MMs_z0)] = 0.0
        logMMs = np.log10(MMs_z0 / (10 ** logMvir))

        df = pd.DataFrame({
            "tree_index": ii.index.values,
            "source_file": h5_path.name,
            "logMvir": logMvir, "log1pz50": log1pz50, "logc": logc, "logMMs": logMMs,
            "Nsub": list(Nsub_ts), "logNsub": list(np.log10(Nsub_ts)),
            "Msub": list(Msub_ts), "logMsub": list(np.log10(Msub_ts)),
            "fsub": list(fsub_ts), "logfsub": list(np.log10(fsub_ts)),
        })
        return df

    def build_timeseries_table(self):
        frames = []
        for h5_path in self.files:
            t0 = time.time()
            df = self._timeseries_from_file(h5_path)
            frames.append(df)
            if self.verbose:
                print(f"  timeseries: {h5_path.name} -> {len(df)} trees ({time.time() - t0:.1f}s)")
        self.timeseries_table = pd.concat(frames, ignore_index=True).sort_values("logMvir").reset_index(drop=True)
        return self.timeseries_table

    # ------------------------------------------------------------------
    # z=0 SHMF table (jsm_ancillary.load_shmf_z0's per-file logic)
    # ------------------------------------------------------------------

    def _shmf_from_file(self, h5_path):
        ii = self.load_sample(h5_path)
        logMvir = self._clean_scalar(np.log10(self._stack_column(ii, "MAH")[:, 0]))
        log1pz50 = self._clean_scalar(np.log10(1 + ii["host_z50"].values))
        logc = self._logc(ii, self.conctype)

        data = {
            "tree_index": ii.index.values,
            "source_file": h5_path.name,
            "logMvir": logMvir, "log1pz50": log1pz50, "logc": logc,
        }
        for regime in self.shmf_regimes:
            for order_label in ("all", "k1", "k2", "k3"):
                key = f"shmf_{regime}_{order_label}"
                data[key] = list(ii[key].values)  # ragged per-tree arrays, kept at native length
        return pd.DataFrame(data)

    def build_shmf_table(self):
        frames = []
        for h5_path in self.files:
            t0 = time.time()
            df = self._shmf_from_file(h5_path)
            frames.append(df)
            if self.verbose:
                print(f"  shmf: {h5_path.name} -> {len(df)} trees ({time.time() - t0:.1f}s)")
        self.shmf_table = pd.concat(frames, ignore_index=True).sort_values("logMvir").reset_index(drop=True)
        return self.shmf_table

    # ------------------------------------------------------------------
    # convenience: per-mass-bin summary stats
    # (jsm_ancillary.compute_mass_bin_stats, with its per-bin-overwrite bug
    # fixed -- this version genuinely accumulates one entry per bin)
    # ------------------------------------------------------------------

    def mass_bin_stats(self, key, decimals=1, table=None):
        df = table if table is not None else self.z0_table
        if df is None:
            raise ValueError("no table available -- call build_z0_table() first, or pass table=")
        bins = df["logMvir"].round(decimals)
        results = {}
        for b in sorted(bins.unique()):
            vals = df.loc[bins == b, key].values
            results[b] = {"mean": np.nanmean(vals), "std": np.nanstd(vals), "N": len(vals)}
        return results

    def split_by_mass_bin(self, decimals=1, table=None):
        df = table if table is not None else self.z0_table
        if df is None:
            raise ValueError("no table available -- call build_z0_table() first, or pass table=")
        bins = df["logMvir"].round(decimals)
        return {b: df.loc[bins == b].reset_index(drop=True) for b in sorted(bins.unique())}

    # ------------------------------------------------------------------
    # saving
    # ------------------------------------------------------------------

    def save_z0_csv(self, outdir=None, filename=None):
        if self.z0_table is None:
            self.build_z0_table()
        outdir = Path(outdir) if outdir else self.datadir
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / (filename or f"{self.label}_z0.csv")
        self.z0_table.to_csv(path, index=False)
        if self.verbose:
            print(f"saved {len(self.z0_table)} rows -> {path}")
        return path

    def save_timeseries_npz(self, outdir=None, filename=None):
        if self.timeseries_table is None:
            self.build_timeseries_table()
        outdir = Path(outdir) if outdir else self.datadir
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / (filename or f"{self.label}_timeseries.npz")
        df = self.timeseries_table
        arrays = {c: np.stack(df[c].values) for c in
                  ["Nsub", "logNsub", "Msub", "logMsub", "fsub", "logfsub"]}
        np.savez(path, tree_index=df["tree_index"].values, source_file=df["source_file"].values,
                 logMvir=df["logMvir"].values, log1pz50=df["log1pz50"].values,
                 logc=df["logc"].values, logMMs=df["logMMs"].values, **arrays)
        if self.verbose:
            print(f"saved {len(df)} rows -> {path}")
        return path

    def save_shmf_npz(self, outdir=None, filename=None):
        if self.shmf_table is None:
            self.build_shmf_table()
        outdir = Path(outdir) if outdir else self.datadir
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / (filename or f"{self.label}_shmf.npz")
        df = self.shmf_table
        shmf_keys = [c for c in df.columns if c.startswith("shmf_")]
        padded = {k: self._stack_column(df, k) for k in shmf_keys}  # NaN-pad ragged arrays for npz storage
        np.savez(path, tree_index=df["tree_index"].values, source_file=df["source_file"].values,
                 logMvir=df["logMvir"].values, log1pz50=df["log1pz50"].values, logc=df["logc"].values,
                 **padded)
        if self.verbose:
            print(f"saved {len(df)} rows -> {path}")
        return path

    def process(self, which=("z0",), outdir=None):
        """Convenience one-liner: build + save the requested table(s).
        which : subset of ("z0", "timeseries", "shmf")."""
        saved = {}
        if "z0" in which:
            saved["z0"] = self.save_z0_csv(outdir=outdir)
        if "timeseries" in which:
            saved["timeseries"] = self.save_timeseries_npz(outdir=outdir)
        if "shmf" in which:
            saved["shmf"] = self.save_shmf_npz(outdir=outdir)
        return saved
