"""
epsilon_orbit_init.py

Perturbs the initial orbit of first-order subhalos in a raw (un-evolved)
SatGen merger tree.

Background
----------
Raw tree files (as produced by TreeGen, e.g. via the jsm_SubGen_*.py driver
scripts) store a "coordinates" array of shape (Nhalo, Ntime, 6), where each
subhalo's phase-space vector [R, phi, z, VR, Vphi, Vz] is nonzero only at
its accretion snapshot -- everything before/after is zero until SubEvo
evolves the tree. The fiducial orbit at accretion is drawn from the
Li et al. (2020) distributions in src/init.py (ZZLi2020 + orbit_from_Li2020),
called from src/treegen.py.

This script does NOT redraw or resample that orbit. It reads the orbit
TreeGen already assigned and adds a perturbation, epsilon, directly to the
radial velocity component (VR, index 3) at each targeted subhalo's
accretion snapshot. Because nothing is redrawn, there's no random seed
involved and the rest of the tree (mass, order, ParentID, VirialRadius,
concentration, all other coordinates) is passed through unchanged.

epsilon is currently just a placeholder scalar; how it should depend on
redshift is still TBD. To ease that transition, epsilon may already be
passed as a callable epsilon(z) instead of a constant -- see EpsilonOrbitInit.
"""

import argparse
from pathlib import Path

import numpy as np


class EpsilonOrbitInit:
    """
    Reads a raw (un-evolved) SatGen merger tree .npz file and writes out a
    copy in which a perturbation, epsilon, has been added directly to the
    radial velocity component (VR) of the initial orbit, for subhalos of a
    given instantaneous order at accretion (default: first-order,
    order == 1).

    Syntax:

        eoi = EpsilonOrbitInit(tree_file, output_dir, epsilon=5.0)
        eoi.run()

    where

        tree_file: path to a raw tree .npz file, as produced by TreeGen
            (str or pathlib.Path)
        output_dir: directory the modified tree is written into; created
            if it doesn't already exist (str or pathlib.Path)
        epsilon: perturbation added to VR [kpc/Gyr]. Either a constant
            (float, default 0.) or a callable epsilon(z) returning the
            perturbation given the redshift at each subhalo's accretion
            snapshot (float or callable, default 0.)
        order_filter: only subhalos with this instantaneous order at their
            accretion snapshot are perturbed (int, default 1, i.e.
            first-order subhalos)
        suffix: appended to the input file's stem to build the output
            filename (str, default "_modified")

    Attributes (all set in __init__, before any perturbation is applied):

        self.data: the raw NpzFile object for the input tree (unmodified)
        self.coordinates: a modified COPY of data["coordinates"]; identical
            to the input until self.apply_epsilon() is called
        self.mass, self.order, self.redshift: convenience references into
            self.data
        self.Nhalo, self.Ntime: tree dimensions
        self.acc_index: (Nhalo,) accretion-snapshot index per subhalo, or
            -1 where no valid initial orbit was found (host, or the rare
            NaN-coordinate edge case noted in orbit.resample_orbit)
        self.acc_order: (Nhalo,) instantaneous order at accretion, or -1
            where self.acc_index is -1
        self.target_ids: subhalo indices selected for perturbation
        self.output_file: path the modified tree will be/was written to

    Methods:

        eoi.apply_epsilon(): perturbs self.coordinates in place. Calling it
            more than once will double (or triple, ...) apply epsilon, so
            call it exactly once before eoi.write().
        eoi.write(): saves the modified tree to self.output_file, creating
            output_dir if needed. Returns self.output_file.
        eoi.run(): apply_epsilon() + write(), returns self.output_file.
    """

    VR_INDEX = 3  # [R, phi, z, VR, Vphi, Vz]

    def __init__(self, tree_file, output_dir, epsilon=0.0, order_filter=1,
                 suffix="_modified"):
        self.tree_file = Path(tree_file)
        self.output_dir = Path(output_dir)
        self.epsilon = epsilon
        self.order_filter = order_filter
        self.suffix = suffix

        self.output_file = (
            self.output_dir / f"{self.tree_file.stem}{self.suffix}{self.tree_file.suffix}"
        )

        self.data = np.load(self.tree_file)
        self.coordinates = np.copy(self.data["coordinates"])
        self.mass = self.data["mass"]
        self.order = self.data["order"]
        self.redshift = self.data["redshift"]
        self.Nhalo, self.Ntime = self.mass.shape

        self._find_accretion()
        self._select_targets()
        self._applied = False

    def _find_accretion(self):
        """
        Locates each subhalo's accretion snapshot: the first (highest-z)
        time index where its coordinates entry is nonzero. Mirrors the
        approach used in orbit.resample_orbit. sub_ii = 0 (the host) is
        always left at -1, since the host has no orbit of its own.
        """
        acc_index = np.full(self.Nhalo, -1, dtype=int)
        for sub_ii in range(1, self.Nhalo):
            nz = np.nonzero(self.coordinates[sub_ii])[0]
            if len(nz) == 0:
                continue  # no valid initial orbit (rare edge case)
            acc_index[sub_ii] = nz[0]
        self.acc_index = acc_index
        self.valid = acc_index >= 0

    def _select_targets(self):
        """
        Determines each valid subhalo's instantaneous order at its own
        accretion snapshot, then selects the subset matching order_filter.
        """
        acc_order = np.full(self.Nhalo, -1, dtype=int)
        acc_order[self.valid] = self.order[
            np.arange(self.Nhalo)[self.valid], self.acc_index[self.valid]
        ]
        self.acc_order = acc_order
        self.target_ids = np.nonzero(self.valid & (acc_order == self.order_filter))[0]

    def _epsilon_at(self, sub_ii):
        if callable(self.epsilon):
            z = self.redshift[self.acc_index[sub_ii]]
            return self.epsilon(z)
        return self.epsilon

    def apply_epsilon(self):
        """Adds epsilon to VR at the accretion snapshot of each targeted subhalo."""
        for sub_ii in self.target_ids:
            iz = self.acc_index[sub_ii]
            self.coordinates[sub_ii, iz, self.VR_INDEX] += self._epsilon_at(sub_ii)
        self._applied = True
        return self.coordinates

    def write(self):
        """Writes the (possibly perturbed) tree to self.output_file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.output_file,
            redshift=self.data["redshift"],
            CosmicTime=self.data["CosmicTime"],
            mass=self.data["mass"],
            order=self.data["order"],
            ParentID=self.data["ParentID"],
            VirialRadius=self.data["VirialRadius"],
            concentration=self.data["concentration"],
            coordinates=self.coordinates,
        )
        return self.output_file

    def run(self):
        """apply_epsilon() + write(), returns the output path."""
        if not self._applied:
            self.apply_epsilon()
        return self.write()

    def summary(self):
        return (
            f"{self.tree_file.name}: Nhalo={self.Nhalo}, "
            f"valid subhalos={int(self.valid.sum())}, "
            f"order=={self.order_filter} at accretion={len(self.target_ids)}"
        )


def _parse_args():
    p = argparse.ArgumentParser(
        description="Add a radial-velocity perturbation to first-order "
                     "subhalos' initial orbits in a raw SatGen tree file."
    )
    p.add_argument("tree_file", type=str, help="path to a raw (un-evolved) tree .npz file")
    p.add_argument("output_dir", type=str, help="directory to write the modified tree into")
    p.add_argument("--epsilon", type=float, default=0.0,
                    help="constant perturbation added to VR [kpc/Gyr] (default: 0.0)")
    p.add_argument("--order-filter", type=int, default=1,
                    help="instantaneous order at accretion to target (default: 1)")
    p.add_argument("--suffix", type=str, default="_modified",
                    help="suffix appended to the output filename stem (default: _modified)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    eoi = EpsilonOrbitInit(
        tree_file=args.tree_file,
        output_dir=args.output_dir,
        epsilon=args.epsilon,
        order_filter=args.order_filter,
        suffix=args.suffix,
    )
    print(eoi.summary())
    out = eoi.run()
    print(f"wrote: {out}")
