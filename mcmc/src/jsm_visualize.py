import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import Normalize
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


class Tree_Vis(Tree_Reader):

    def plot_host_properties(self):

            fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True, constrained_layout=True)

            axes[0].plot(1+self.redshift, self.host_Vmax, color="k")
            axes[0].set_ylabel("Vmax (km/s)") 

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
        with imageio.get_writer(video_path, fps=10) as writer:
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
        with imageio.get_writer(video_path, fps=10) as writer:
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
        with imageio.get_writer(video_path, fps=10) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)

        print("Movie created successfully!")

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

        # Use the Pastel1 colormap
        cmap = plt.get_cmap("tab10")
        bounds = np.arange(0.5, self.order.max()-0.5, 1)  # Boundaries for each color category
        norm = BoundaryNorm(bounds, cmap.N)

        # Loop over each time step to save individual frames
        for time_index in range(self.CosmicTime.shape[0]-1, 0, -1):
            fig, ax = plt.subplots(figsize=(8,6))
            ax.set_title(f"t = {self.CosmicTime[time_index]:.2f} (Gyrs)")

            if type(subhalo_indices) == type(None):
                k = self.order[:, time_index]
                sc = ax.scatter(self.cartesian_stitched[:, time_index, 0], self.cartesian_stitched[:, time_index, 1], c=k, cmap=cmap, norm=norm, marker=".", s=10)
            else:
                k = self.order[subhalo_indices, time_index]
                sc = ax.scatter(self.cartesian_stitched[subhalo_indices, time_index, 0], self.cartesian_stitched[subhalo_indices, time_index, 1], c=k, cmap=cmap, norm=norm, marker=".", s=10)

            # Create a color bar, specifying the figure and axis for placement
            cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(0, self.order.max()))
            cbar.set_label('subhalo order')

            # Add a circle with a radius of 250 centered at (0, 0)
            circle1 = plt.Circle((0, 0), self.VirialRadius[0, time_index], color='grey', fill=False, linewidth=1, ls="--")
            circle2 = plt.Circle((0, 0), 2*self.VirialRadius[0, time_index], color='grey', fill=False, linewidth=1, ls="--")

            ax.add_patch(circle1)
            ax.add_patch(circle2)

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
        branch_levels = dict(nx.single_source_shortest_path_length(G, source="0"))

        node_colors = [branch_levels[n] for n in G.nodes]


        plt.figure(figsize=(12,12))
        nx.draw(G, pos,
                with_labels=True,
                labels={n: G.nodes[n]['label'] for n in G.nodes},
                node_size=node_sizes,
                node_color=node_colors,
                cmap=plt.cm.viridis_r,   # or plasma, magma, etc.        
                edge_color='lightgrey',
                arrows=False,
                arrowsize=2,
                # connectionstyle="arc3,rad=0.0",
                font_size=8)
        
        plt.show()

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
        with imageio.get_writer(video_path, fps=10) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)

        print("Movie created successfully!")

        for frame_path in frame_paths:
            os.remove(frame_path)
        os.rmdir(output_dir)