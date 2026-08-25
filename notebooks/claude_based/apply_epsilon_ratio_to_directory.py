"""
apply_epsilon_ratio_to_directory.py

Batch-applies EpsilonRatioOrbitInit to every raw (un-evolved) tree file in
a directory, writing the scaled trees into a new output directory under
the SAME filenames as the originals. Trees are processed in parallel
across multiple cores (same multiprocessing.Pool pattern jsm_SubEvo.py
uses), with progress printed as trees complete.

Why same filenames, new directory: jsm_SubEvo.py discovers trees by
filename convention within whatever datadir it's pointed at (files
starting with "tree" and NOT ending in "evo.npz"; it then writes each
tree's evolved output as "<name>_evo.npz" alongside it in that same
directory). Preserving filenames means the output directory here is a
drop-in datadir for jsm_SubEvo.py -- point it there and it evolves the
epsilon-scaled trees exactly as it would the originals. The original tree
files in input_dir are left untouched, so you can run jsm_SubEvo.py on
input_dir (baseline) and on this script's output_dir (epsilon-scaled) and
compare the two "_evo.npz" sets 1:1, tree by tree.

Each tree's own mass bin (and therefore which mean_MAH reference backs
its epsilon(z)) is inferred from its own z=0 host mass -- see
EpsilonRatioOrbitInit for details. Trees were originally sampled from a
wider mass range (1e11-1e14) than mean_MAH actually covers (12.6-14.0),
so a tree whose host mass falls outside mean_mah_dir's coverage is
skipped here -- not silently matched to the nearest available bin, and
not a fatal error for the whole batch. Skipped trees are tallied by mass
bin and reported in the final summary, so a genuine mean_mah_dir/input_dir
mismatch (e.g. wrong directory entirely) is still obvious rather than
quietly swallowed. Likewise, an unexpected per-tree failure (corrupted
file, etc.) is caught, reported, and counted rather than killing the
whole pool -- mirroring how jsm_SubEvo.py's own loop() catches
AttributeError per-tree instead of dying mid-batch.

Cluster deployment
-------------------
The raw N1000 zhao trees live on the cluster, outside this git repo, at
DEFAULT_INPUT_DIR below. DEFAULT_OUTPUT_DIR is the "epsilon_orbits"
sibling directory already created there to receive the scaled copies.
mean_MAH/*.npz, by contrast, now lives inside the repo at
SatGen/etc/mean_MAH/ -- DEFAULT_MEAN_MAH_DIR resolves it relative to this
file's own location (the same trick config.py uses for config.json), so
it's correct on the cluster once these changes are pushed, regardless of
where the repo happens to be cloned.

Usage (CLI):

    # uses the cluster defaults below -- no arguments needed once pushed
    python apply_epsilon_ratio_to_directory.py

    # or override any of the three explicitly, e.g. for a local test run
    python apply_epsilon_ratio_to_directory.py INPUT_DIR OUTPUT_DIR \\
        MEAN_MAH_DIR --order-filter 1 --eps-min 0.5 --eps-max 1.5 --ncores 16

    # then point jsm_SubEvo.py's datadir at OUTPUT_DIR and run it
"""

import argparse
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

from epsilon_ratio_orbit_init import EpsilonRatioOrbitInit, infer_logM0

DEFAULT_INPUT_DIR = "/netb/vdbosch/jsm99/data/mass_spec_zhao/"
DEFAULT_OUTPUT_DIR = "/netb/vdbosch/jsm99/data/mass_spec_zhao/epsilon_orbits/"
DEFAULT_MEAN_MAH_DIR = str(Path(__file__).resolve().parent.parent.parent / "etc" / "mean_MAH")
DEFAULT_NCORES = 16  # matches jsm_SubGen_masspec.py / jsm_SubEvo.py's ncores


def find_raw_tree_files(input_dir):
    """
    Returns the raw (un-evolved) tree files in input_dir, using the same
    file-discovery convention as jsm_SubEvo.py: filenames starting with
    "tree" and NOT ending in "evo.npz".
    """
    input_dir = Path(input_dir)
    return sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.name.startswith("tree") and not f.name.endswith("evo.npz")
    )


def _process_one(job):
    """
    Worker function run in each Pool process for a single tree file.
    Must stay a plain module-level function (not a closure) so it can be
    pickled out to worker processes. Returns a small, cheap-to-pickle
    status tuple -- (status, logM0, filename, detail) -- rather than the
    EpsilonRatioOrbitInit object itself, since that carries the full
    coordinates array and shipping that back across process boundaries
    for every tree would be wasteful.

    detail is only populated for "error" (always -- rare, and you want to
    know why) and for "done" when verbose=True. Skipping the per-subhalo
    summary string (and the print it would otherwise cause in the main
    process) is the point of verbose=False: at a few thousand trees, one
    print per tree is real overhead, not just log noise.
    """
    tree_file, output_dir, mean_mah_dir, order_filter, eps_min, eps_max, verbose = job
    tree_file = Path(tree_file)

    with np.load(tree_file) as d:
        logM0 = infer_logM0(d["mass"])
    if not (Path(mean_mah_dir) / f"{logM0:.1f}_files_mean_MAH.npz").exists():
        return ("skipped", logM0, tree_file.name, None)

    try:
        erv = EpsilonRatioOrbitInit(
            tree_file=tree_file,
            output_dir=output_dir,
            mean_mah_dir=mean_mah_dir,
            order_filter=order_filter,
            eps_min=eps_min,
            eps_max=eps_max,
            suffix="",  # preserve the original filename for jsm_SubEvo.py
        )
        summary = erv.summary() if verbose else None
        erv.run()
        return ("done", logM0, tree_file.name, summary)
    except Exception as e:
        return ("error", logM0, tree_file.name, str(e))


def apply_epsilon_ratio_to_directory(input_dir=DEFAULT_INPUT_DIR,
                                      output_dir=DEFAULT_OUTPUT_DIR,
                                      mean_mah_dir=DEFAULT_MEAN_MAH_DIR,
                                      order_filter=1, eps_min=0.5, eps_max=1.5,
                                      ncores=DEFAULT_NCORES,
                                      verbose=False, progress_every=200):
    """
    Applies EpsilonRatioOrbitInit to every raw tree file in input_dir,
    writing each modified tree into output_dir under its ORIGINAL
    filename (no suffix), so the resulting directory is a drop-in datadir
    for jsm_SubEvo.py. Trees are processed across `ncores` worker
    processes; progress is printed as each tree completes.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing raw (un-evolved) tree_*.npz files.
        Default: the cluster path holding the N1000 zhao trees.
    output_dir : str or Path
        Directory the modified trees are written into (created if needed).
        Default: the "epsilon_orbits" sibling directory on the cluster.
    mean_mah_dir : str or Path
        Directory holding "{logM0:.1f}_files_mean_MAH.npz" reference files.
        Default: SatGen/etc/mean_MAH/, resolved relative to this file.
    order_filter : int
        Instantaneous order at accretion to target (default: 1).
    eps_min, eps_max : float
        epsilon(z) clip range (default: 0.5, 1.5).
    ncores : int
        Number of worker processes (default: 16).
    verbose : bool
        If True, print a per-tree summary line (epsilon min/max/mean,
        subhalo counts, etc.) as each tree finishes. Default False --
        with a few thousand trees this print is real per-tree overhead,
        not just noise, so it's off unless you're debugging a small run.
        The periodic progress line (every `progress_every` trees) prints
        regardless, and errors always print.
    progress_every : int
        Print a one-line progress rollup after every this-many trees
        finish (and once more at the end). Default 200.

    Returns
    -------
    dict with counts: {"processed": int, "skipped": int, "errors": int,
    "skipped_by_bin": {logM0: count}}.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    mean_mah_dir = Path(mean_mah_dir)
    tree_files = find_raw_tree_files(input_dir)
    total = len(tree_files)

    if not tree_files:
        print(f"no raw tree files found in {input_dir}")
        return {"processed": 0, "skipped": 0, "errors": 0, "skipped_by_bin": {}}

    if input_dir.resolve() == output_dir.resolve():
        raise ValueError(
            "input_dir and output_dir are the same directory -- refusing to "
            "write modified trees on top of the raw ones. Pick a different "
            "output_dir (e.g. a subdirectory of input_dir)."
        )

    # create it up front so worker processes never race each other to do so
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"found {total} raw tree file(s) in {input_dir}; "
          f"processing with {ncores} core(s)")

    jobs = [(f, output_dir, mean_mah_dir, order_filter, eps_min, eps_max, verbose) for f in tree_files]

    n_done = n_skipped = n_error = 0
    skipped_by_bin = {}
    time_start = time.time()

    with Pool(ncores) as pool:
        for i, (status, logM0, name, detail) in enumerate(
                pool.imap_unordered(_process_one, jobs), start=1):

            if status == "done":
                n_done += 1
                if verbose:
                    print(f"[{i}/{total}] {detail}")
            elif status == "skipped":
                n_skipped += 1
                skipped_by_bin[logM0] = skipped_by_bin.get(logM0, 0) + 1
            elif status == "error":
                n_error += 1
                # always printed, regardless of verbose -- rare and actionable
                print(f"[{i}/{total}] ERROR on {name}: {detail}")

            if i % progress_every == 0 or i == total:
                elapsed_min = (time.time() - time_start) / 60.0
                print(f"--- progress: {i}/{total} tree(s) handled "
                      f"({n_done} scaled, {n_skipped} skipped, {n_error} errored), "
                      f"{elapsed_min:.1f} min elapsed ---")

    print(f"\nwrote {n_done} modified tree(s) to {output_dir}")
    if skipped_by_bin:
        by_bin = ", ".join(f"{k:.1f} ({v})" for k, v in sorted(skipped_by_bin.items()))
        print(f"skipped {n_skipped} tree(s) outside mean_MAH coverage, by bin: {by_bin}")
    if n_error:
        print(f"{n_error} tree(s) errored -- see ERROR lines above")

    return {"processed": n_done, "skipped": n_skipped, "errors": n_error,
            "skipped_by_bin": skipped_by_bin}


def _parse_args():
    p = argparse.ArgumentParser(
        description="Apply EpsilonRatioOrbitInit to every raw tree file in a "
                     "directory, writing epsilon(z)-scaled copies (same "
                     "filenames) into a new directory usable directly by "
                     "jsm_SubEvo.py. With no arguments, uses the cluster "
                     "N1000 zhao paths below."
    )
    p.add_argument("input_dir", type=str, nargs="?", default=DEFAULT_INPUT_DIR,
                    help=f"directory of raw (un-evolved) tree_*.npz files "
                         f"(default: {DEFAULT_INPUT_DIR})")
    p.add_argument("output_dir", type=str, nargs="?", default=DEFAULT_OUTPUT_DIR,
                    help=f"directory to write the modified trees into "
                         f"(default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("mean_mah_dir", type=str, nargs="?", default=DEFAULT_MEAN_MAH_DIR,
                    help=f"directory holding '{{logM0:.1f}}_files_mean_MAH.npz' "
                         f"reference files (default: {DEFAULT_MEAN_MAH_DIR})")
    p.add_argument("--order-filter", type=int, default=1,
                    help="instantaneous order at accretion to target (default: 1)")
    p.add_argument("--eps-min", type=float, default=0.5, help="lower epsilon clip (default: 0.5)")
    p.add_argument("--eps-max", type=float, default=1.5, help="upper epsilon clip (default: 1.5)")
    p.add_argument("--ncores", type=int, default=DEFAULT_NCORES,
                    help=f"number of worker processes (default: {DEFAULT_NCORES})")
    p.add_argument("--verbose", action="store_true",
                    help="print a per-tree summary line as each tree finishes "
                         "(default: off -- at scale this print is real overhead, "
                         "not just noise; the periodic progress line and any "
                         "errors print regardless)")
    p.add_argument("--progress-every", type=int, default=200,
                    help="print a progress rollup every this-many trees (default: 200)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    apply_epsilon_ratio_to_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mean_mah_dir=args.mean_mah_dir,
        order_filter=args.order_filter,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
        ncores=args.ncores,
        verbose=args.verbose,
        progress_every=args.progress_every,
    )
