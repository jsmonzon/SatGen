#!/usr/bin/env python3
"""
run_ASH.py

Batch-runs jsm_ASH.Tree_Reader over a directory of evolved SatGen merger
trees (one .npz file per tree) and writes the accreted-stellar-halo
(ASH/ICL) bookkeeping -- including the bound-stellar-mass-fraction "cumsum"
-- to a single output HDF5 file, one group per tree.

For each tree file this script:
  1. builds jsm_ASH.Tree_Reader(file=..., verbose=..., merger_crit=...,
     scatter=..., [ALPHA=...])              -- runs read_arrays through
     satellites() automatically (see Tree_Reader.__init__)
  2. calls .disk() then .stellarhalo()       -- NOT auto-run by __init__;
     this is what actually builds the ICL/ASH tree-walk (icl_MAH,
     total_ICL, MW_est, etc.)
  3. writes create_survsat_dict() -- which now also carries merger_crit/
     scatter/ALPHA, the free parameters this tree was run with -- plus
     "cumsum" (tree.frac_fb_stellar from ancil.fb_surv_frac) to its own
     HDF5 group.

See the accompanying parameter walkthrough for what merger_crit/scatter/
ALPHA actually mean and how to choose them -- they are not given defaults
here on purpose.

Usage:

    python run_ASH.py \
        --input-dir /path/to/evolved_trees \
        --output /path/to/abundance_output.h5 \
        --merger-crit -0.5 \
        --scatter \
        --nproc 8
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np
import h5py

# jsm_ASH.py itself resolves the SatGen src/ modules relative to config.json
# (see the NOTE at the top of jsm_ASH.py) -- we only need it importable,
# which it is as long as this script stays next to it in mcmc/src/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jsm_ASH


def _build_output_dict(tree):
    """create_survsat_dict() (which as of the jsm_ASH.py port now also
    carries merger_crit/scatter/ALPHA -- the free parameters this tree was
    run with) plus "cumsum" (tree.frac_fb_stellar, the bound-stellar-mass-
    fraction CDF from ancil.fb_surv_frac -- same name write_out_disc() uses
    for it). frac_fb_DM is intentionally left out here; see fb_surv_frac /
    the module docstring above if you need it after all."""
    d = tree.create_survsat_dict()
    d["cumsum"] = tree.frac_fb_stellar
    return d


def _process_one(args):
    """Runs in a worker process. Returns a plain dict (picklable) rather
    than raising, so one bad/oddly-named tree file doesn't kill the batch."""
    filepath, tree_kwargs = args
    try:
        kwargs = dict(tree_kwargs)
        kwargs["file"] = filepath
        tree = jsm_ASH.Tree_Reader(**kwargs)
        tree.disk()
        tree.stellarhalo()
        data = _build_output_dict(tree)
        return dict(filepath=filepath, ash_tree_index=str(tree.tree_index),
                    data=data, error=None)
    except Exception as e:
        return dict(filepath=filepath, ash_tree_index=None, data=None,
                    error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def _write_group(h5file, group_name, result):
    grp = h5file.create_group(group_name)
    grp.attrs["source_file"] = os.path.basename(result["filepath"])
    for key, value in result["data"].items():
        arr = np.asarray(value)
        if arr.dtype.kind in ("U", "S", "O"):  # strings/objects need explicit h5py dtype
            grp.create_dataset(key, data=arr.astype(h5py.string_dtype()))
        else:
            grp.create_dataset(key, data=arr)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True,
        help="directory containing evolved-tree .npz files")
    parser.add_argument("--pattern", default="*.npz",
        help="glob pattern for tree files within --input-dir (default: *.npz)")
    parser.add_argument("--output", required=True,
        help="path to the output HDF5 file")
    parser.add_argument("--overwrite", action="store_true",
        help="overwrite --output if it already exists (default: error out)")
    parser.add_argument("--merger-crit", type=float, default=-2.0,
        help="log10 threshold on r/parent_rmax and V/parent_Vmax below "
             "which a subhalo is flagged as merged into its parent "
             "(default: -2.0, the calibrated value)")
    parser.add_argument("--scatter", action="store_true",
        help="inject dex scatter into the accretion-time SHMR/size-mass/"
             "mass-metallicity draws (default: off, pure mean relations)")
    parser.add_argument("--alpha", type=float, default=None,
        help="override the Behroozi+18 SHMR low-mass-end slope ALPHA "
             "passed to gh.lgMs_B18 (default: its built-in value, 1.963342)")
    parser.add_argument("--fesc", type=float, default=0.2,
        help="fraction of a MERGED satellite's stellar mass deposited into "
             "the ICL at merger, with the remainder (1-fesc) going to the "
             "parent as smooth accretion -- required by ancil.tree_walker() "
             "inside .stellarhalo() (default: 0.2, the calibrated value). A "
             "DISRUPTED satellite always sends 100%% of its mass to the "
             "ICL regardless of fesc.")
    parser.add_argument("--verbose", action="store_true",
        help="pass verbose=True to Tree_Reader (noisy -- best with --nproc 1)")
    parser.add_argument("--nproc", type=int, default=1,
        help="number of worker processes (default: 1, no multiprocessing)")
    parser.add_argument("--seed", type=int, default=None,
        help="np.random.seed() before running -- the scatter/artificial-"
             "disruption draws use the global numpy RNG, not a per-"
             "instance one, so this is the only way to make --scatter "
             "runs reproducible")
    parser.add_argument("--flush-every", type=int, default=50,
        help="flush the HDF5 file to disk every N completed trees, so a "
             "killed/preempted job still leaves a readable partial output "
             "(default: 50)")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        parser.error(f"{args.output} already exists (use --overwrite)")

    if args.seed is not None:
        np.random.seed(args.seed)

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        parser.error(f"no files matching {args.pattern!r} in {args.input_dir}")
    print(f"found {len(files)} tree files in {args.input_dir}", flush=True)

    tree_kwargs = dict(verbose=args.verbose, merger_crit=args.merger_crit,
                        scatter=args.scatter, fesc=args.fesc)
    if args.alpha is not None:
        tree_kwargs["ALPHA"] = args.alpha

    work = [(f, tree_kwargs) for f in files]

    t0 = time.time()
    n_done, failures = 0, []
    with h5py.File(args.output, "w") as h5file:
        h5file.attrs["input_dir"] = os.path.abspath(args.input_dir)
        h5file.attrs["pattern"] = args.pattern
        h5file.attrs["merger_crit"] = args.merger_crit
        h5file.attrs["scatter"] = args.scatter
        h5file.attrs["alpha"] = args.alpha if args.alpha is not None else np.nan
        h5file.attrs["seed"] = args.seed if args.seed is not None else -1
        h5file.attrs["n_files_total"] = len(files)
        h5file.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

        def handle(result):
            nonlocal n_done
            n_done += 1
            if result["error"] is not None:
                failures.append(result)
            else:
                _write_group(h5file, f"tree_{n_done - 1:05d}", result)
            if n_done % args.flush_every == 0 or n_done == len(files):
                h5file.flush()
                elapsed = time.time() - t0
                print(f"  {n_done}/{len(files)} done "
                      f"({len(failures)} failed) -- {elapsed:.0f}s elapsed",
                      flush=True)

        if args.nproc > 1:
            # jsm_ASH is imported (and pays its ~20s one-time cosmology-grid
            # build, see config.py) before the Pool is created, so on
            # Linux's default fork() start method, workers inherit that
            # already-initialized state instead of redoing it each.
            with mp.Pool(args.nproc) as pool:
                for result in pool.imap_unordered(_process_one, work):
                    handle(result)
        else:
            for w in work:
                handle(_process_one(w))

        h5file.attrs["n_files_succeeded"] = len(files) - len(failures)
        h5file.attrs["n_files_failed"] = len(failures)
        if failures:
            h5file.attrs["failed_files"] = np.array(
                [os.path.basename(r["filepath"]) for r in failures],
                dtype=h5py.string_dtype())

    print(f"wrote {len(files) - len(failures)} trees to {args.output} "
          f"in {time.time()-t0:.1f}s", flush=True)
    if failures:
        print(f"{len(failures)} file(s) failed:")
        for r in failures:
            print(f"  {os.path.basename(r['filepath'])}: "
                  f"{r['error'].splitlines()[0]}")


if __name__ == "__main__":
    main()
