"""
apply_epsilon_to_directory.py

Batch-applies EpsilonOrbitInit to every raw (un-evolved) tree file in a
directory, writing the perturbed trees into a new output directory under
the SAME filenames as the originals.

Why same filenames, new directory: jsm_SubEvo.py discovers trees by
filename convention within whatever datadir it's pointed at (files
starting with "tree" and NOT ending in "evo.npz"; it then writes each
tree's evolved output as "<name>_evo.npz" alongside it in that same
directory). Preserving filenames means the output directory here is a
drop-in datadir for jsm_SubEvo.py -- point it there and it evolves the
perturbed trees exactly as it would the originals. The original tree
files in input_dir are left untouched, so you can run jsm_SubEvo.py on
input_dir (baseline) and on this script's output_dir (perturbed) and
compare the two "_evo.npz" sets 1:1, tree by tree.

Usage (CLI):

    python apply_epsilon_to_directory.py INPUT_DIR OUTPUT_DIR \\
        --epsilon 5.0 --order-filter 1

    # then point jsm_SubEvo.py's datadir at OUTPUT_DIR and run it
"""

import argparse
from pathlib import Path

from epsilon_orbit_init import EpsilonOrbitInit


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


def apply_epsilon_to_directory(input_dir, output_dir, epsilon=0.0, order_filter=1):
    """
    Applies EpsilonOrbitInit to every raw tree file in input_dir, writing
    each modified tree into output_dir under its ORIGINAL filename (no
    suffix), so the resulting directory is a drop-in datadir for
    jsm_SubEvo.py.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing raw (un-evolved) tree_*.npz files.
    output_dir : str or Path
        Directory the modified trees are written into (created if needed).
    epsilon : float or callable(z)
        Perturbation added to VR [kpc/Gyr]. See EpsilonOrbitInit.
    order_filter : int
        Instantaneous order at accretion to target (default: 1).

    Returns
    -------
    list of (input_file, output_file, EpsilonOrbitInit) tuples, one per
    tree processed.
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
        eoi = EpsilonOrbitInit(
            tree_file=tree_file,
            output_dir=output_dir,
            epsilon=epsilon,
            order_filter=order_filter,
            suffix="",  # preserve the original filename for jsm_SubEvo.py
        )
        print(eoi.summary())
        out_file = eoi.run()
        results.append((tree_file, out_file, eoi))

    print(f"\nwrote {len(results)} modified tree(s) to {output_dir}")
    return results


def _parse_args():
    p = argparse.ArgumentParser(
        description="Apply EpsilonOrbitInit to every raw tree file in a "
                     "directory, writing perturbed copies (same filenames) "
                     "into a new directory usable directly by jsm_SubEvo.py."
    )
    p.add_argument("input_dir", type=str,
                    help="directory of raw (un-evolved) tree_*.npz files")
    p.add_argument("output_dir", type=str,
                    help="directory to write the modified trees into "
                         "(e.g. a subdirectory of input_dir)")
    p.add_argument("--epsilon", type=float, default=0.0,
                    help="constant perturbation added to VR [kpc/Gyr] (default: 0.0)")
    p.add_argument("--order-filter", type=int, default=1,
                    help="instantaneous order at accretion to target (default: 1)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    apply_epsilon_to_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        epsilon=args.epsilon,
        order_filter=args.order_filter,
    )
