import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import Normalize
from matplotlib.collections import EllipseCollection
from matplotlib.lines import Line2D
import os
import warnings; warnings.simplefilter('ignore')
import sys
import pandas as pd
import os
import json

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
import config as cfg
import galhalo as gh
import evolve as ev
import imageio
import networkx as nx
from jsm_stellarhalo import Tree_Reader
import jsm_ancillary as ancil
from treelib import Node, Tree


class Tree_Vis(Tree_Reader):

    def plot_host_properties(self):

            fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True, constrained_layout=True)

            axes[0].plot(1+self.redshift, self.host_Vmax, color="k")
            axes[0].set_ylabel("Vmax (km/s)") 
            axes[0].axvline(1+2, ls="-.", color="grey")
            axes[0].axvline(1+3, ls="-.", color="grey")
            axes[0].set_yscale("log")

            axes[1].plot(1+self.redshift, self.mass[0], color="k", label="$M_{\\rm h}$")
            axes[1].plot(1+self.redshift, self.stellarmass[0], color="r", label="$M_{*}$")
            axes[1].plot(1+self.redshift, self.icl_MAH, color="C0", label="ICL")

            axes[1].axhline(self.target_mass, color="grey", ls="-.", alpha=0.2)
            axes[1].axhline(self.target_stellarmass, color="r", ls="-.", alpha=0.2)
            axes[1].axhline(self.total_ICL, color="C0", ls="-.", alpha=0.2)

            axes[1].legend(loc=4, framealpha=1)
            axes[1].set_ylabel("$M\ (\mathrm{M}_{\odot})$")
            axes[1].set_yscale("log")
            axes[1].set_xscale("log")
            axes[1].set_ylim(1e4)

            axes[1].set_yscale("log")
            axes[1].set_xscale("log")
            axes[1].set_xlim(13, 1)
            axes[1].set_xlabel("1 + z")
            plt.tight_layout()
            plt.show()

    def plot_subhalo_properties(self, subhalo_ind):

        start = self.acc_index[subhalo_ind] # just to have a reference!
        stop = self.final_index[subhalo_ind]
        fate = self.subhalo_fates[subhalo_ind]

        order_jumps = np.where(self.order[subhalo_ind][:-1] != self.order[subhalo_ind][1:])[0] + 1

        fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=False)
        axes_r = axes.ravel()
        for ax in axes_r:
            ax.axvline(self.CosmicTime[stop], ls=":", color="grey")
            ax.axvline(self.CosmicTime[start], ls=":", color="grey")
            for jump in order_jumps[:-1]:
                ax.axvline(self.CosmicTime[jump], ls="--", color="green")

        plt.suptitle(f" subhalo ID: {subhalo_ind} \n intial order: {self.acc_order[subhalo_ind]} \n number of jumps: {order_jumps.shape[0] -1} \n final fate: {fate}")
        axes[0,0].plot(self.CosmicTime, self.mass[subhalo_ind])
        axes[0,0].plot(self.CosmicTime, self.mass[0], color="k")
        axes[0,0].set_yscale("log")
        axes[0,0].set_ylabel("M$_{\\rm H}$ (M$_{\odot}$)")
        axes[0,0].set_title("halo mass")

        axes[0,1].plot(self.CosmicTime, self.rmags[subhalo_ind])
        axes[0,1].set_ylabel("$| \\vec{r} |$ (kpc)")
        axes[0,1].set_title("position")

        axes[0,2].plot(self.CosmicTime, self.Vmags[subhalo_ind])
        axes[0,2].set_ylabel("$| \\vec{v} |$ (kpc / Gyr)")
        axes[0,2].set_title("velocity")

        axes[1,0].plot(self.CosmicTime, self.stellarmass[subhalo_ind])
        axes[1,0].set_yscale("log")
        axes[1,0].set_ylabel("M$_{*}$ (M$_{\odot}$)")
        axes[1,0].set_xlabel("Cosmic Time (Gyr)")
        axes[1,0].set_title("stellar mass")

        axes[1,1].plot(self.CosmicTime, self.rmax_kscaled[subhalo_ind])
        axes[1,1].set_ylabel("log (r$^k$ / r$_{max}^{k-1}$)")
        axes[1,1].set_xlabel("Cosmic Time (Gyr)")
        axes[1,1].set_title("Scaled Position")
        axes[1,1].set_ylim(-3,0)
        axes[1,1].axhline(self.merger_crit, ls="--", color="red")

        axes[1,2].plot(self.CosmicTime, self.Vmax_kscaled[subhalo_ind])
        axes[1,2].set_ylabel("log (V$^k$ / V$_{max}^{k-1}$)")
        axes[1,2].set_xlabel("Cosmic Time (Gyr)")
        axes[1,2].set_title("Scaled Velocity")
        axes[1,2].set_ylim(-3,0)
        axes[1,2].axhline(self.merger_crit, ls="--", color="red")

        plt.tight_layout()
        plt.show()

    def make_massloss_movie(self, subhalo_indices=None, video_path=None):

            if type(subhalo_indices) == type(None):
                print("plotting all subhalos!")
            else:
                print("plotting a subset of the subhalos in the tree!")

            output_dir = 'temp_frames'
            os.makedirs(output_dir, exist_ok=True)

            # List to hold paths of all saved frames for the video
            frame_paths = []

            # Loop over each time step to save individual frames
            for time_index in range(self.CosmicTime.shape[0]-1, 0, -1):
                fig, ax = plt.subplots(figsize=(8,6))            
                ax.set_title(f"t = {self.CosmicTime[time_index]:.2f} (Gyrs)")

                ax.scatter(np.log10(self.fb[subhalo_indices, time_index]), np.log10(self.mass[subhalo_indices, time_index]), marker=".", s=6.5, color="k")

                ax.set_xlabel("log f$_{\\rm bound}$")
                ax.set_ylabel("log m$_{\\rm inst}$")
                ax.set_xlim(-4, 0)
                ax.set_ylim(5, 13)
                ax.legend(loc=4)

                # Save each frame as a PNG file
                frame_path = f"{output_dir}/frame_{time_index:03d}.png"
                plt.savefig(frame_path)
                frame_paths.append(frame_path)  # Add frame path to list
                plt.close(fig)  # Close the figure to free up memory

            # Now create a video from the frames
            with imageio.get_writer(video_path, fps=30) as writer:
                for frame_path in frame_paths:
                    image = imageio.imread(frame_path)
                    writer.append_data(image)

            print("Movie created successfully!")

            for frame_path in frame_paths:
                os.remove(frame_path)
            os.rmdir(output_dir)

    def make_fb_movie(self, subhalo_indices=None, video_path=None):

        if type(subhalo_indices) == type(None):
            print("plotting all subhalos!")
        else:
            print("plotting a subset of the subhalos in the tree!")

        x_array = np.logspace(0, -4, 100)
        leff_rat_020, Mstar_rat_020 = ev.g_EPW18(x_array, alpha=0, lefflmax=1/20)
        leff_rat_010, Mstar_rat_010 = ev.g_EPW18(x_array, alpha=0, lefflmax=1/10)

        leff_rat_120, Mstar_rat_120 = ev.g_EPW18(x_array, alpha=1.0, lefflmax=1/20)
        leff_rat_110, Mstar_rat_110 = ev.g_EPW18(x_array, alpha=1.0, lefflmax=1/10)

        output_dir = 'temp_frames'
        os.makedirs(output_dir, exist_ok=True)

        # List to hold paths of all saved frames for the video
        frame_paths = []

        # Loop over each time step to save individual frames
        for time_index in range(self.CosmicTime.shape[0]-1, 0, -1):
            fig, ax = plt.subplots(figsize=(8,6))            
            ax.set_title(f"t = {self.CosmicTime[time_index]:.2f} (Gyrs)")

            ax.plot(np.log10(x_array), np.log10(Mstar_rat_020), color="mediumblue", label="core: 1/20", alpha=0.3)
            ax.plot(np.log10(x_array), np.log10(Mstar_rat_010), color="mediumblue", ls="--", label="core: 1/10", alpha=0.3)

            ax.plot(np.log10(x_array), np.log10(Mstar_rat_120), color="red", label="cusp: 1/20", alpha=0.3)
            ax.plot(np.log10(x_array), np.log10(Mstar_rat_110), color="red", ls="--", label="cusp: 1/10", alpha=0.3)

            if type(subhalo_indices) == type(None):
                ax.scatter(np.log10(self.fb[:, time_index]), np.log10(self.fb_stellar[:, time_index]), marker=".", s=6.5, color="k")
            else:
                ax.scatter(np.log10(self.fb[subhalo_indices, time_index]), np.log10(self.fb_stellar[subhalo_indices, time_index]), marker=".", s=6.5, color="k")

            ax.set_ylabel("log M$_{*}$/M$_{*, 0}$")
            ax.set_xlabel("log M/M$_0$")
            ax.set_xlim(-4.2, 0.2)
            ax.set_ylim(-3.6, 0.2)
            ax.axhline(0, ls=":", color="grey")
            ax.axvline(0, ls=":", color="grey")
            ax.axvline(-4, ls="--", color="green")

            ax.legend(loc=4)

            # Save each frame as a PNG file
            frame_path = f"{output_dir}/frame_{time_index:03d}.png"
            plt.savefig(frame_path)
            frame_paths.append(frame_path)  # Add frame path to list
            plt.close(fig)  # Close the figure to free up memory

        # Now create a video from the frames
        with imageio.get_writer(video_path, fps=50) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)

        print("Movie created successfully!")

        for frame_path in frame_paths:
            os.remove(frame_path)
        os.rmdir(output_dir)

    def make_mergercloud_movie(self, subhalo_indices=None, video_path=None):

        if type(subhalo_indices) == type(None):
            print("plotting all subhalos!")
        else:
            print("plotting a subset of the subhalos in the tree!")

        output_dir = 'temp_frames'
        os.makedirs(output_dir, exist_ok=True)

        # List to hold paths of all saved frames for the video
        frame_paths = []

        # Loop over each time step to save individual frames
        for time_index in range(self.CosmicTime.shape[0]-1, 0, -1):
            fig, ax = plt.subplots(figsize=(8,6))            
            ax.set_title(f"t = {self.CosmicTime[time_index]:.2f} (Gyrs)")

            if type(subhalo_indices) == type(None):
                ax.scatter(self.rmax_kscaled[:, time_index], self.Vmax_kscaled[:, time_index], marker=".", s=1.5, color="k")
            else:
                ax.scatter(self.rmax_kscaled[subhalo_indices, time_index], self.Vmax_kscaled[subhalo_indices, time_index], marker=".", s=1.5, color="k")

            ax.set_ylabel("log (V$^k$ / V$_{max}^{k-1}$)")
            ax.set_xlabel("log (R$^k$ / R$_{max}^{k-1}$)")
            ax.set_ylim(-3, 2)
            ax.set_xlim(-3, 2)

            ax.axvline(self.merger_crit, ls="--", color="red")
            ax.axhline(self.merger_crit, ls="--", color="red", label="merging criteria")
            ax.legend(loc=2)
            # Save each frame as a PNG file
            frame_path = f"{output_dir}/frame_{time_index:03d}.png"
            plt.savefig(frame_path)
            frame_paths.append(frame_path)  # Add frame path to list
            plt.close(fig)  # Close the figure to free up memory

        # Now create a video from the frames
        with imageio.get_writer(video_path, fps=50) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)

        print("Movie created successfully!")

        for frame_path in frame_paths:
            os.remove(frame_path)
        os.rmdir(output_dir)

    def make_SHMR_movie(self, subhalo_indices=None, video_path=None):

        if type(subhalo_indices) == type(None):
            print("plotting all subhalos!")
        else:
            print("plotting a subset of the subhalos in the tree!")

        output_dir = 'temp_frames'
        os.makedirs(output_dir, exist_ok=True)

        # List to hold paths of all saved frames for the video
        frame_paths = []
        halo_smooth = np.linspace(4, 13, 100)

        # Loop over each time step to save individual frames
        for time_index in range(self.CosmicTime.shape[0]-1, 0, -1):
            fig, ax = plt.subplots(figsize=(8,6))            
            ax.set_title(f"t = {self.CosmicTime[time_index]:.2f} (Gyrs)") 

            if type(subhalo_indices) == type(None):
                ax.scatter(self.mass[:, time_index], self.stellarmass[:, time_index], marker=".", s=1.5, color="k")
            else:
                ax.scatter(self.mass[subhalo_indices, time_index], self.stellarmass[subhalo_indices, time_index], marker=".", s=1.5, color="k")

            ax.plot(10**halo_smooth, 10**gh.lgMs_B18(halo_smooth, self.redshift[time_index]), label=f"Behroozi et al 2018 (UM): z={self.redshift[time_index]:.2f}")
            ax.legend(loc=2)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlabel("Mhalo")
            ax.set_ylabel("Mstar")
            ax.set_xlim(1e3, 1e12)
            ax.set_ylim(0.1, 1e9)
            
            # Save each frame as a PNG file
            frame_path = f"{output_dir}/frame_{time_index:03d}.png"
            plt.savefig(frame_path)
            frame_paths.append(frame_path)  # Add frame path to list
            plt.close(fig)  # Close the figure to free up memory

        # Now create a video from the frames
        with imageio.get_writer(video_path, fps=50) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)

        print("Movie created successfully!")

        for frame_path in frame_paths:
            os.remove(frame_path)
        os.rmdir(output_dir)

    def make_Rvir_movie(self, video_path=None, scale=300):

            # Create a temporary directory to save frames
            output_dir = 'temp_frames'
            os.makedirs(output_dir, exist_ok=True)

            # List to hold paths of all saved frames for the video
            frame_paths = []

            # --- pull masks straight from compute_regimes, no re-derivation here ---
            keep_mask    = self.regime_masks["massive"]  # subhalo_mask_all & massive_mask
            within_mask  = self.regime_masks["rvir"]      # subhalo_massive & inside_rvir_mask

            # Loop over each time step to save individual frames
            for time_index in range(self.CosmicTime.shape[0]-50, 0, -1):
                fig, ax = plt.subplots(figsize=(8,6))
                ax.set_title(f"t = {self.CosmicTime[time_index]:.2f} (Gyrs)")

                keep = keep_mask[:, time_index]

                x = self.cartesian_stitched[keep, time_index, 0]
                y = self.cartesian_stitched[keep, time_index, 1]
                within = within_mask[keep, time_index]

                rvir = self.VirialRadius[0, time_index]

                ax.scatter(x[within], y[within], c="blue", marker=".", s=10, label="within $R_{\\rm vir}$")
                ax.scatter(x[~within], y[~within], c="red", marker=".", s=10, label="outside $R_{\\rm vir}$")

                # use the same fraction that's already computed in compute_regimes
                frac_within = self.Nsub_within_Rvir[time_index]

                ax.text(0.97, 0.97, f"$f_{{\\rm sub}}(<R_{{\\rm vir}})$ = {frac_within:.2f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

                # Add a circle with a radius of Rvir centered at (0, 0)
                circle1 = plt.Circle((0, 0), rvir, color='black', fill=False, linewidth=1, ls="--")
                ax.add_patch(circle1)

                # Set limits and show plot
                ax.set_xlim(-scale, scale)
                ax.set_ylim(-scale, scale)
                ax.set_xlabel("X Coordinate (kpc)")
                ax.set_ylabel("Y Coordinate (kpc)")
                ax.legend(loc="lower right", fontsize=8, markerscale=2)

                # Save each frame as a PNG file
                frame_path = f"{output_dir}/frame_{time_index:03d}.png"
                plt.savefig(frame_path)
                frame_paths.append(frame_path)  # Add frame path to list
                plt.close(fig)  # Close the figure to free up memory

            # Now create a video from the frames
            with imageio.get_writer(video_path, fps=10) as writer:
                for frame_path in frame_paths:
                    image = imageio.imread(frame_path)
                    writer.append_data(image)
            # delete temp frames!
            for frame_path in frame_paths:
                os.remove(frame_path)
            os.rmdir(output_dir)

    def make_orbit_movie(self, subhalo_indices=None, video_path=None, scale=300):

        if type(subhalo_indices) == type(None):
            print("plotting all subhalos!")
        else:
            print("plotting a subset of the subhalos in the tree!")

        # Create a temporary directory to save frames
        output_dir = 'temp_frames'
        os.makedirs(output_dir, exist_ok=True)

        # List to hold paths of all saved frames for the video
        frame_paths = []
        max_levels=5
        # Use the Pastel1 colormap
        cmap = plt.cm.get_cmap("Set1", max_levels + 1)
        # bounds = np.arange(0.5, 7-0.5, 1)  # Boundaries for each color category
        # norm = BoundaryNorm(bounds, cmap.N)

        # Loop over each time step to save individual frames
        for time_index in range(self.CosmicTime.shape[0]-50, 0, -1):
            fig, ax = plt.subplots(figsize=(8,6))
            ax.set_title(f"t = {self.CosmicTime[time_index]:.2f} (Gyrs)")

            if type(subhalo_indices) == type(None):
                k = self.order[:, time_index]
                sc = ax.scatter(self.cartesian_stitched[:, time_index, 0], self.cartesian_stitched[:, time_index, 1], c=k, cmap=cmap, vmin=0, vmax=max_levels, marker=".", s=10)
            else:
                k = self.order[subhalo_indices, time_index]
                sc = ax.scatter(self.cartesian_stitched[subhalo_indices, time_index, 0], self.cartesian_stitched[subhalo_indices, time_index, 1], c=k, cmap=cmap, vmin=0, vmax=max_levels, marker=".", s=10)

            # Create a color bar, specifying the figure and axis for placement
            cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(0, max_levels))
            cbar.set_label('subhalo order')

            # Add a circle with a radius of 250 centered at (0, 0)
            circle1 = plt.Circle((0, 0), self.VirialRadius[0, time_index], color='red', fill=False, linewidth=1, ls="--")
            # circle2 = plt.Circle((0, 0), 2*self.VirialRadius[0, time_index], color='red', fill=False, linewidth=1, ls="--")

            ax.add_patch(circle1)
            # ax.add_patch(circle2)

            # Set limits and show plot
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_xlabel("X Coordinate (kpc)")
            ax.set_ylabel("Y Coordinate (kpc)")
            
            # Save each frame as a PNG file
            frame_path = f"{output_dir}/frame_{time_index:03d}.png"
            plt.savefig(frame_path)
            frame_paths.append(frame_path)  # Add frame path to list
            plt.close(fig)  # Close the figure to free up memory

        # Now create a video from the frames
        with imageio.get_writer(video_path, fps=10) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)
            #delete temp frames!       
        for frame_path in frame_paths:
            os.remove(frame_path)
        os.rmdir(output_dir)

    def plot_acc_tree(self, save_path=None):

        acc_tree = ancil.acc_hierarchy(self)

        G = nx.DiGraph()

        # Build graph
        for node in acc_tree.all_nodes_itr():
            node_id = node.identifier
            data = node.data or {}
            acc_z = data.get('acc_redshift', 0.0)
            G.add_node(node_id, acc_redshift=acc_z, label=node.tag)
            parent = acc_tree.parent(node_id)
            if parent:
                G.add_edge(parent.identifier, node_id)

        # Assign positions: Y = acc_redshift, X = branch offset
        pos = {}
        x_counter = [0]  # mutable counter

        def assign_pos(node_id, x=0):
            acc_z = G.nodes[node_id]['acc_redshift']
            pos[node_id] = (x, acc_z)
            children = list(G.successors(node_id))
            if not children:
                x_counter[0] += 5
            for child in children:
                assign_pos(child, x_counter[0])

        assign_pos("0", x=-30)

        masses = []
        for n in G.nodes:
            data = acc_tree.get_node(n).data or {}
            mass = data["acc_mass"]  # default: host mass if missing
            masses.append(mass)

        masses = np.array(masses)

        # Normalize: mass fraction relative to host
        mass_frac = masses / self.target_mass

        # Optional: log scale if mass_frac varies a lot
        log_mass_frac = np.log10(mass_frac + 1e-6)  # avoid log(0)

        # Map to node size
        min_size = 1
        max_size = 2000
        node_sizes = min_size + (log_mass_frac - log_mass_frac.min()) / (log_mass_frac.max() - log_mass_frac.min()) * (max_size - min_size)

        max_levels=5
        # Compute branch levels (distance from root)
        branch_levels = dict(nx.single_source_shortest_path_length(G, source="0"))

        # Map each node to its level
        node_colors = [branch_levels[n] for n in G.nodes]

        # Use a discrete colormap with exactly 8 colors (0 through 7)
        cmap = plt.cm.get_cmap("Set1", max_levels + 1)  # or "Set1", "plasma", etc.

        plt.figure(figsize=(8, 8))
        nx.draw(
            G, pos,
            with_labels=True,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=cmap,
            vmin=0, vmax=max_levels, 
            edge_color='lightgrey',
            arrows=False,
            arrowsize=2,
            font_size=8
        )

        plt.show()

    ##### host-centric projections, sized by tidal radius or bound mass #####

    PROJECTIONS = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    AXIS_LABELS = ['x', 'y', 'z']
    MAX_LEVELS = 5

    def subhalo_green(self, ind, iz):
        """Green profile of subhalo `ind`, built at accretion and stripped to its mass at snapshot `iz`."""
        s = profiles.Green(self.acc_mass[ind], self.acc_concentration[ind],
                           Delta=cfg.Dvsample[self.acc_index[ind]], z=self.acc_redshift[ind])
        s.update_mass(self.mass[ind, iz])
        return s

    def compute_tidal_radius(self, iz=0, mask=None, choice='King62', cap=True):
        """
        Tidal radius [kpc] at snapshot `iz` for every subhalo alive there (NaN otherwise),
        optionally restricted to `mask` (Nhalo,). Same definition as jsm_SubEvo_masscut:
        the subhalo Green profile in its immediate parent's potential (host NFW for k=1,
        the parent's stripped Green profile for k>1), at the parent-relative coordinates.

        cap=True limits r_t to the subhalo's own virial radius at accretion (Green.rh).
        When ev.ltidal finds no root because the tides are too weak (m(l)/l^3 > RHS all the
        way out to its 9.999 rh bracket) it returns cfg.Rres; those cases are set to rh too.
        """
        rhs_func = ev.lt_King62_RHS if choice == 'King62' else ev.lt_Tormen98_RHS
        host = profiles.NFW(self.mass[0, iz], self.concentration[0, iz],
                            Delta=cfg.Dvsample[iz], z=self.redshift[iz])
        parents = {}
        rt = np.full(self.Nhalo, np.nan)
        sel = self.orbit_mask[:, iz] if mask is None else self.orbit_mask[:, iz] & mask
        for ind in np.where(sel)[0]:
            ip = self.ParentID[ind, iz]
            if ip == 0:
                p = host
            else:
                if ip not in parents:
                    parents[ip] = self.subhalo_green(ip, iz)
                p = parents[ip]
            s = self.subhalo_green(ind, iz)
            xv = self.coordinates[ind, iz]
            lt = ev.ltidal(s, p, xv, choice=choice)
            if cap:
                if lt <= cfg.Rres and ev.Findlt(9.999 * s.rh, s, rhs_func(p, xv)) > 0:
                    lt = s.rh  # weak tides: no root inside the bracket
                lt = min(lt, s.rh)
            rt[ind] = lt
        return rt

    def tidal_radius_history(self, mask, choice='King62', cap=True):
        """
        Tidal radius [kpc], shape (Nhalo, Nsnap), NaN outside `mask`. Results are cached,
        so a later call with an overlapping mask (e.g. another regime) only solves the new entries.
        """
        if getattr(self, '_rt_cache_key', None) != (choice, cap):
            self._rt_hist = np.full(self.mass.shape, np.nan)
            self._rt_done = np.zeros(self.mass.shape, dtype=bool)
            self._rt_cache_key = (choice, cap)

        todo = mask & self.orbit_mask & ~self._rt_done
        for iz in np.where(todo.any(axis=0))[0]:
            rt_iz = self.compute_tidal_radius(iz=iz, mask=todo[:, iz], choice=choice, cap=cap)
            self._rt_hist[todo[:, iz], iz] = rt_iz[todo[:, iz]]
        self._rt_done |= todo

        return np.where(mask, self._rt_hist, np.nan)

    def plot_subhalo_projection(self, rt=None, proj='xy', iz=0, regime='total', size_mode='physical',
                                scale=1.5, extent=None, alpha=0.5, min_size=0.5, max_size=60,
                                mass_ref=1e9, size_ref=1.0, choice='King62', cap=True, ax=None):
        """
        Host-centric positions (cartesian_stitched) of the subhalos in `regime` at snapshot `iz`,
        colored by instantaneous order, with the host Rvir as a dashed circle.

        size_mode:
            'physical': circles of radius r_t [kpc], drawn to scale in data units
            'scaled':   scatter markers, area linear in log r_t between min_size and max_size [pt^2]
            'mass':     scatter markers, area = size_ref [pt^2] * m(t) / mass_ref (fixed normalization)
        rt: (Nhalo,) tidal radii at `iz`; computed here if needed and not given.
        scale: half-width in units of Rvir(iz); extent overrides it with a fixed half-width [kpc].
        """
        i, j = self.PROJECTIONS[proj]
        sel = self.regime_masks[regime][:, iz]
        if size_mode in ('physical', 'scaled'):
            if rt is None:
                rt = self.compute_tidal_radius(iz=iz, mask=sel, choice=choice, cap=cap)
            sel = sel & np.isfinite(rt)
        inds = np.where(sel)[0]
        m = self.mass[inds, iz]
        order_by = m if size_mode == 'mass' else rt[inds]
        draw = np.argsort(order_by)[::-1]  # draw the largest first so small ones sit on top
        inds, m = inds[draw], m[draw]

        x = self.cartesian_stitched[inds, iz, i]
        y = self.cartesian_stitched[inds, iz, j]
        k = self.order[inds, iz]
        cmap = plt.cm.get_cmap("Set1", self.MAX_LEVELS + 1)
        colors = cmap(k)
        rvir = self.VirialRadius[0, iz]

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        if size_mode == 'physical':
            d = 2 * rt[inds]
            ax.add_collection(EllipseCollection(d, d, np.zeros_like(d), units='xy', offsets=np.c_[x, y],
                                                offset_transform=ax.transData, facecolors=colors,
                                                edgecolors=colors, linewidths=0.3, alpha=alpha))
        elif size_mode == 'scaled':
            lr = np.log10(rt[inds])
            span = np.ptp(lr) if lr.size else 0.0
            s = min_size + (lr - lr.min()) / span * (max_size - min_size) if span > 0 else np.full(lr.size, max_size)
            ax.scatter(x, y, s=s, c=colors, alpha=alpha, edgecolors='none')
        elif size_mode == 'mass':
            ax.scatter(x, y, s=size_ref * m / mass_ref, c=colors, alpha=alpha, edgecolors='none')
        else:
            raise ValueError("size_mode must be 'physical', 'scaled' or 'mass'")

        ax.add_patch(plt.Circle((0, 0), rvir, color='k', fill=False, lw=1.2, ls='--'))

        half = scale * rvir if extent is None else extent
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect('equal')
        ax.set_xlabel(f'{self.AXIS_LABELS[i]} (kpc)')
        ax.set_ylabel(f'{self.AXIS_LABELS[j]} (kpc)')
        ax.set_title(f'z = {self.redshift[iz]:.2f}   {proj} projection   ({regime}, N = {inds.size})')

        handles = [Line2D([], [], marker='o', ls='', color=cmap(kk), label=f'k = {kk}') for kk in np.unique(k)]
        handles.append(Line2D([], [], ls='--', color='k', label=f'$R_{{\\rm vir}}$ = {rvir:.0f} kpc'))
        ax.add_artist(ax.legend(handles=handles, loc='upper right', fontsize=9, framealpha=0.9))

        if size_mode == 'mass':
            size_handles = [Line2D([], [], marker='o', ls='', color='grey', alpha=alpha,
                                   markersize=np.sqrt(size_ref * mm / mass_ref),
                                   label=f'$10^{{{np.log10(mm):.0f}}}\\,M_\\odot$')
                            for mm in (1e9, 1e10, 1e11, 1e12)]
            ax.legend(handles=size_handles, loc='lower right', fontsize=9, framealpha=0.9,
                      labelspacing=2.0, borderpad=1.2, handletextpad=1.6)
        return ax

    def make_projection_movie(self, video_path, regime='massive', size_mode='physical', projs=('xy', 'xz', 'yz'),
                              box='fixed', scale=1.5, stride=1, fps=15, dpi=100, choice='King62', cap=True, **plot_kw):
        """
        Movie of plot_subhalo_projection from the first snapshot with a subhalo in `regime` down to z=0,
        one panel per projection in `projs`.

        box='fixed': every frame spans scale * Rvir(z=0); box='rvir': each frame spans scale * Rvir(t).
        stride: use every `stride`-th snapshot (z=0 is always included).
        """
        mask = self.regime_masks[regime]
        if not mask.any():
            print(f"no subhalos in regime '{regime}', skipping {video_path}")
            return

        rt_hist = self.tidal_radius_history(mask, choice=choice, cap=cap) if size_mode != 'mass' else None

        snaps = np.where(mask.any(axis=0))[0][::-1][::stride]  # high z -> z=0
        if snaps[-1] != 0:
            snaps = np.append(snaps, 0)

        extent0 = scale * self.VirialRadius[0, 0]
        fig, axes = plt.subplots(1, len(projs), figsize=(6.4 * len(projs), 7.2), dpi=dpi)
        axes = np.atleast_1d(axes)
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.88, wspace=0.25)

        with imageio.get_writer(video_path, fps=fps) as writer:
            for iz in snaps:
                for ax, proj in zip(axes, projs):
                    ax.clear()
                    self.plot_subhalo_projection(rt=None if rt_hist is None else rt_hist[:, iz], proj=proj, iz=iz,
                                                 regime=regime, size_mode=size_mode, scale=scale,
                                                 extent=extent0 if box == 'fixed' else None, ax=ax, **plot_kw)
                    ax.set_title(f'{proj}   (N = {mask[:, iz].sum()})')
                fig.suptitle(f'{regime}    t = {self.CosmicTime[iz]:.2f} Gyr    z = {self.redshift[iz]:.2f}', fontsize=14)
                fig.canvas.draw()
                writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])

        plt.close(fig)
        print(f"wrote {len(snaps)} frames to {video_path}")

def make_RVmag_movie(self, subhalo_indices=None, video_path=None):

        if type(subhalo_indices) == type(None):
            print("plotting all subhalos!")
        else:
            print("plotting a subset of the subhalos in the tree!")

        output_dir = 'temp_frames'
        os.makedirs(output_dir, exist_ok=True)

        # List to hold paths of all saved frames for the video
        frame_paths = []

        # Loop over each time step to save individual frames
        for time_index in range(self.CosmicTime.shape[0]-1, 0, -1):
            fig, ax = plt.subplots(figsize=(8,6))            
            ax.set_title(f"t = {self.CosmicTime[time_index]:.2f} (Gyrs)") 

            if type(subhalo_indices) == type(None):
                ax.scatter(np.log10(self.rmags[:, time_index]), np.log10(self.Vmags[:, time_index]), marker=".", s=1.5, color="k")
            else:
                ax.scatter(np.log10(self.rmags[subhalo_indices, time_index]), np.log10(self.Vmags[subhalo_indices, time_index]), marker=".", s=1.5, color="k")

            ax.set_ylabel("log |$\\vec{V}$|")
            ax.set_xlabel("log |$\\vec{R}$|")
            ax.set_ylim(0, 4)
            ax.set_xlim(0, 4)

            # Save each frame as a PNG file
            frame_path = f"{output_dir}/frame_{time_index:03d}.png"
            plt.savefig(frame_path)
            frame_paths.append(frame_path)  # Add frame path to list
            plt.close(fig)  # Close the figure to free up memory

        # Now create a video from the frames
        with imageio.get_writer(video_path, fps=50) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)

        print("Movie created successfully!")

        for frame_path in frame_paths:
            os.remove(frame_path)
        os.rmdir(output_dir)

####################################################
####################################################
####################################################

class Arborist(Tree_Reader):

    def plant_roots(self):

        self.forest = []
        for z_ind in range(self.redshift.shape[0]):

            tree = Tree()

            valid = (self.ParentID[:, z_ind] != -1) & \
                    (self.ParentID[:, z_ind] != -99)

            initialized_subhalos = np.where(valid)[0]

            tree.create_node(
                "Host Halo",
                "0",
                data={
                    "redshift":            self.redshift[z_ind],
                    "z_ind":               z_ind,
                    "initialized_subhalos": initialized_subhalos,
                    "mass":                self.mass[0, z_ind],
                    "host_rvir":           self.VirialRadius[0, z_ind],
                    })

            self.forest.append(tree)

    # -----------------------------------------------------

    def add_branch(self, tree, subhalo_ind, z_ind):

        node_id = str(subhalo_ind)
        parent_ind = self.ParentID[subhalo_ind, z_ind]

        if parent_ind == -99:
            return

        parent_id = str(parent_ind)

        if not tree.contains(parent_id):
            self.add_branch(tree, parent_ind, z_ind)

        if not tree.contains(parent_id):
            return

        if not tree.contains(node_id):
            tree.create_node(
                f"subID:{node_id}",
                node_id,
                parent=parent_id,
                data={
                    "mass":          self.mass[subhalo_ind, z_ind],
                    "rmag":          self.rmags_stitched[subhalo_ind, z_ind],
                    })

    # -----------------------------------------------------

    def water_roots(self):

        for tree in self.forest:

            root = tree.get_node("0")
            z_ind = root.data["z_ind"]
            initialized_subhalos = root.data["initialized_subhalos"]

            for subhalo_ind in initialized_subhalos:
                self.add_branch(tree, subhalo_ind, z_ind)

    # -----------------------------------------------------

    def dendrochronology(self, mass_threshold, kmax, verbose=False):

        n_t = len(self.forest)
        self.kmax = kmax

        n_cols_N  = kmax + 1
        n_cols_fm = kmax

        # --------------------------------------------------
        # Precompute time-independent masks (both indexed into n_sub, 0-based)
        # --------------------------------------------------
        artdisrupt_mass  = ancil.artificial_disruption(self.acc_mass[1:], self.acc_concentration[1:])
        artdisrupt_mask  = self.mass[1:, 0] > artdisrupt_mass   # (n_sub,) — checked at z_acc
        thresh_mask      = self.mass[1:, 0] >= mass_threshold    # (n_sub,) — checked at z=0 only

        self.Nsub_total       = np.zeros((n_t, n_cols_N), dtype=int)
        self.Nsub_surviving   = np.zeros((n_t, n_cols_N), dtype=int)
        self.Nsub_rvir        = np.zeros((n_t, n_cols_N), dtype=int)
        self.Nsub_artificial  = np.zeros((n_t, n_cols_N), dtype=int)

        self.fsub_surviving   = np.zeros((n_t, n_cols_fm))
        self.fsub_rvir        = np.zeros((n_t, n_cols_fm))
        self.fsub_artificial  = np.zeros((n_t, n_cols_fm))

        self.submass_surviving   = None
        self.submass_rvir        = None
        self.submass_artificial  = None

        for t, tree in enumerate(self.forest):

            host_node = tree.get_node("0")
            host_mass = host_node.data["mass"]
            host_rvir = host_node.data["host_rvir"]

            # --------------------------------------------------
            # total: trivially read from tree structure
            # --------------------------------------------------
            self.Nsub_total[t, 0] = tree.size() - 1  # exclude root
            for k in range(1, kmax + 1):
                self.Nsub_total[t, k] = tree.size(level=k + 1)

            # --------------------------------------------------
            # collect all active subhalos at this timestep from the tree
            # sub_inds are 1-based (matching self.mass, self.order, etc.)
            # --------------------------------------------------
            active_nodes = [n for n in tree.all_nodes() if n.identifier != "0"]
            if not active_nodes:
                continue

            sub_inds  = np.array([int(n.identifier) for n in active_nodes])  # (n_active,)
            sub_mass  = self.mass[sub_inds, t]                                # (n_active,)
            sub_order = self.order[sub_inds, t]                               # (n_active,)
            sub_rmag  = self.rmags_stitched[sub_inds, t]                      # (n_active,)

            # --------------------------------------------------
            # regime masks — sub_inds are 1-based so offset by 1 for 0-based arrays
            # --------------------------------------------------
            thresh_here     = thresh_mask[sub_inds - 1]       # z=0 mass cut
            artdisrupt_here = artdisrupt_mask[sub_inds - 1]   # z_acc disruption cut
            rvir_here       = sub_rmag < host_rvir             # recomputed every t

            regime_masks = {
                "surviving":  thresh_here,
                "rvir":       thresh_here & rvir_here,
                "artificial": thresh_here & rvir_here & artdisrupt_here,
            }

            # order broadcast: (n_active, kmax)
            ks          = np.arange(1, kmax + 1)
            order_masks = (sub_order[:, None] == ks[None, :])

            for regime, base_mask in regime_masks.items():

                Nmat = getattr(self, f"Nsub_{regime}")
                fmat = getattr(self, f"fsub_{regime}")

                k_masks = base_mask[:, None] & order_masks   # (n_active, kmax)

                Nmat[t, 0]  = np.sum(base_mask & (sub_order > 0))
                Nmat[t, 1:] = k_masks.sum(axis=0)

                if verbose:
                    print(regime, host_node.data["redshift"], (sub_mass[:, None] * k_masks).sum(axis=0), host_mass)
                    print("--------------------------------")

                fmat[t, :]  = (sub_mass[:, None] * k_masks).sum(axis=0) / host_mass

                if t == 0:
                    shmf = [sub_mass[base_mask & (sub_order > 0)]]
                    for k in range(1, kmax + 1):
                        shmf.append(sub_mass[k_masks[:, k - 1]])
                    setattr(self, f"submass_{regime}", shmf)

    # -----------------------------------------------------

    # def canopy(self, verbose=False):

    #     for regime in ("withering", "rvir", "artificial"):
    #         setattr(self, f"Nsub_{regime}_z0", getattr(self, f"Nsub_{regime}")[0, :])
    #         setattr(self, f"fsub_{regime}_z0", getattr(self, f"fsub_{regime}")[0, :])

    #         # MMs_z0 derived from SHMF arrays: max mass per order slot
    #         shmf = getattr(self, f"submass_{regime}")
    #         MMs_z0 = np.array([np.max(arr) if len(arr) > 0 else 0.0 for arr in shmf])
    #         setattr(self, f"MMs_{regime}_z0", MMs_z0)

    #     if verbose:
    #         print("-----------------")
    #         for regime in ("withering", "rvir", "artificial"):
    #             print(f"at z=0, {regime}: "
    #                 f"Nsub={getattr(self, f'Nsub_{regime}_z0')}, "
    #                 f"fsub={getattr(self, f'fsub_{regime}_z0')}, "
    #                 f"MMs={getattr(self, f'MMs_{regime}_z0')}")
                            
    # def canopy(self, mass_threshold):

    #     # ----- order masks -----
    #     order_masks = [
    #         None,  # all subhalos #0
    #         self.acc_order[1:] == 1, #1
    #         self.order[1:, 0] == 1, #2
    #         self.acc_order[1:] == 2, #3
    #         self.order[1:, 0] == 2, #4
    #         self.acc_order[1:] >= 3, #5 
    #         self.order[1:, 0] >= 3,] #6

    #     # ----- base masks -----
    #     self.within_Rvir_mask = self.rmags_stitched[1:, 0] < self.VirialRadius[0, 0]

    #     self.artdisrupt_mass = ancil.artificial_disruption(self.acc_mass[1:], self.acc_concentration[1:])
    #     self.artdisrupt_mask = self.mass[1:, 0] > self.artdisrupt_mass

    #     self.withering_mask = self.mass[1:, 0] > mass_threshold

    #     # ----- base mask regimes -----
    #     regimes = {"withering_mat": [self.withering_mask], 
    #                "liberal_mat": [self.withering_mask, self.within_Rvir_mask],
    #                "artificial_mat": [self.withering_mask, self.within_Rvir_mask, self.artdisrupt_mask]}

    #     # ----- compute matrices -----
    #     for name, base_masks in regimes.items():

    #         rows = []

    #         for order_mask in order_masks:

    #             mask_list = base_masks.copy()

    #             if order_mask is not None:
    #                 mask_list.append(order_mask)

    #             Nsub, fsub, MMs = ancil.measure_mass_frac(self, mask_list)

    #             rows.append([Nsub, fsub, MMs])

    #         setattr(self, name, np.array(rows))

    # def ave_canopy(self, mass_threshold):

    #     # ----- order masks -----
    #     order_masks = [
    #         None,  # all subhalos
    #         self.acc_order[1:] == 1,
    #         self.order[1:, 0] == 1,
    #         self.acc_order[1:] == 2,
    #         self.order[1:, 0] == 2,
    #         self.acc_order[1:] >= 3,
    #         self.order[1:, 0] >= 3,]

    #     # ----- base masks -----
    #     self.within_Rvir_mask = self.rmags_stitched[1:, 0] < self.VirialRadius[0, 0]

    #     self.artdisrupt_mass = ancil.artificial_disruption(self.acc_mass[1:], self.acc_concentration[1:])
    #     self.artdisrupt_mask = self.ave_mass[1:, 0] > self.artdisrupt_mass

    #     self.withering_mask = self.ave_mass[1:, 0] > mass_threshold

    #     # ----- base mask regimes -----
    #     regimes = {"withering_mat_ave": [self.withering_mask], 
    #                "liberal_mat_ave": [self.withering_mask, self.within_Rvir_mask],
    #                "artificial_mat_ave": [self.withering_mask, self.within_Rvir_mask, self.artdisrupt_mask]}

    #     # ----- compute matrices -----
    #     for name, base_masks in regimes.items():

    #         rows = []

    #         for order_mask in order_masks:

    #             mask_list = base_masks.copy()

    #             if order_mask is not None:
    #                 mask_list.append(order_mask)

    #             Nsub, fsub = ancil.measure_mass_frac(self, mask_list)

    #             rows.append([Nsub, fsub])

    #         setattr(self, name, np.array(rows))

