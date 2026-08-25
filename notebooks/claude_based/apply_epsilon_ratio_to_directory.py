"""
apply_epsilon_ratio_to_directory.py

Batch-applies EpsilonRatioOrbitInit to every raw (un-evolved) tree file in
a directory, writing the scaled trees into a new output directory under
the SAME filenames as the originals.

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
EpsilonRatioOrbitInit for details. A tree whose host mass doesn't match
any file in mean_mah_dir raises (rather than being silently skipped or
matched to the nearest available bin), since that mismatch usually means
mean_mah_dir doesn't cover this tree's sample.

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
        MEAN_MAH_DIR --order-filter 1 --eps-min 0.5 --eps-max 1.5

    # then point jsm_SubEvo.py's datadir at OUTPUT_DIR and run it
"""

import argparse
from pathlib import Path

from epsilon_ratio_orbit_init import EpsilonRatioOrbitInit

DEFAULT_INPUT_DIR = "/netb/vdbosch/jsm99/data/mass_spec_zhao/"
DEFAULT_OUTPUT_DIR = "/netb/vdbosch/jsm99/data/mass_spec_zhao/epsilon_orbits/"
DEFAULT_MEAN_MAH_DIR = str(Path(__file__).resolve().parent.parent.parent / "etc" / "mean_MAH")


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


def apply_epsilon_ratio_to_directory(input_dir=DEFAULT_INPUT_DIR,
                                      output_dir=DEFAULT_OUTPUT_DIR,
                                      mean_mah_dir=DEFAULT_MEAN_MAH_DIR,
                                      order_filter=1, eps_min=0.5, eps_max=1.5):
    """
    Applies EpsilonRatioOrbitInit to every raw tree file in input_dir,
    writing each modified tree into output_dir under its ORIGINAL
    filename (no suffix), so the resulting directory is a drop-in datadir
    for jsm_SubEvo.py.

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

    Returns
    -------
    list of (input_file, output_file, EpsilonRatioOrbitInit) tuples, one
    per tree processed.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    tree_files = find_raw_tree_files(input_dir)

    if not tree_files:
        print(f"no raw tree files found in {input_dir}")
        return []

    if input_dir.resolve() == output_dir.resolve():
        raise ValueError(
            "input_dir and output_dir are the same directory -- refusing to "
            "write modified trees on top of the raw ones. Pick a different "
            "output_dir (e.g. a subdirectory of input_dir)."
        )

    results = []
    for tree_file in tree_files:
        erv = EpsilonRatioOrbitInit(
            tree_file=tree_file,
            output_dir=output_dir,
            mean_mah_dir=mean_mah_dir,
            order_filter=order_filter,
            eps_min=eps_min,
            eps_max=eps_max,
            suffix="",  # preserve the original filename for jsm_SubEvo.py
        )
        print(erv.summary())
        out_file = erv.run()
        results.append((tree_file, out_file, erv))

    print(f"\nwrote {len(results)} modified tree(s) to {output_dir}")
    return results


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
    )
