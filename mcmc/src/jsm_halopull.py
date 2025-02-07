import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib
from matplotlib.colors import BoundaryNorm

from astropy.table import Table
import os
import warnings; warnings.simplefilter('ignore')
import jsm_SHMR
import sys

location = "server"
if location == "server":
    parentdir = "/home/jsm99/SatGen/src/"
    
elif location == "local":
    parentdir = "/Users/jsmonzon/Research/SatGen/src/"

sys.path.insert(0, parentdir)
import profiles as profiles
import config as cfg
import galhalo as gh
import evolve as ev

import astropy.units as u
import astropy.constants as const
import astropy.coordinates as crd

import imageio

##################################################
## FOR INTERFACING WITH THE "RAW" SATGEN OUTPUT ##
##################################################


class Tree_Reader:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.read_arrays()
        self.convert_to_cartesian()
        self.tides()
        self.mergers()
        self.baryons()
        self.stellarhalo()
        self.summary_stats()

    def read_arrays(self):
        self.full = np.load(self.file) #open file and read

        if self.verbose:
            print("reading in the tree!")

        for key in self.full.keys():
            if key in ["CosmicTime", "redshift"]:
                setattr(self, key, self.full[key])
            else:
                arr = np.delete(self.full[key], 1, axis=0) #there is some weird bug for this first index!
                if key in ["mass", "concentration", "VirialRadius"]:
                    masked_arr = np.where(arr == -99, np.nan, arr) #replacing dummy variable with nans
                    setattr(self, key, masked_arr)
                else:
                    setattr(self, key, arr)

        self.ParentID[self.ParentID > 0] -= 1 # to correct for the removed index!
        self.Nhalo = self.mass.shape[0] # count the number of subhalos

        #Host halo properties!
        self.target_mass = self.mass[0,0]
        self.target_redshift = self.redshift[0]

        NFW_vectorized = np.vectorize(profiles.NFW) # grabbing the potential of the host at all times
        self.host_profiles = NFW_vectorized(self.mass[0, :], self.concentration[0,:], Delta=cfg.Dvsample, z=self.redshift)
        self.host_rmax = np.array([profile.rmax for profile in self.host_profiles])
        self.host_Vmax = np.array([profile.Vmax for profile in self.host_profiles])

        self.host_z50 = self.redshift[find_nearest1(self.mass[0], self.target_mass/2)] #the formation time of the host!``
        self.host_z10 = self.redshift[find_nearest1(self.mass[0], self.target_mass/10)]

        #subhalo properties!
        self.acc_index = np.nanargmax(self.mass, axis=1) #finding the accertion index for each
        self.acc_mass = self.mass[np.arange(self.acc_index.shape[0]), self.acc_index] # max mass
        self.acc_concentration = self.concentration[np.arange(self.acc_index.shape[0]), self.acc_index]
        self.acc_redshift = self.redshift[self.acc_index]
        self.acc_order = self.order[np.arange(self.acc_index.shape[0]), self.acc_index]
        self.acc_ParentID = self.ParentID[np.arange(self.acc_index.shape[0]), self.acc_index]

        self.fb = self.mass/self.acc_mass[:, None] #the bound fraction of halo mass
        self.order_jump = np.where(np.array([np.unique(subhalo).shape[0] for subhalo in self.order]) > 2)[0] # which halos undergo an order jump?

        Green_vec = np.vectorize(profiles.Green) # grabbing the peak potentials of all subhalos!
        self.acc_profiles = Green_vec(self.acc_mass, self.acc_concentration, Delta=cfg.Dvsample[self.acc_index],z=self.acc_redshift)
        self.acc_Vmax = np.array([profile.Vmax for profile in self.acc_profiles])
        self.acc_rmax = np.array([profile.rmax for profile in self.acc_profiles])
        
        # Create a mask for times before the acc_index for each subhalo
        self.time_indices = np.arange(self.CosmicTime.shape[0])
        self.orbit_mask = self.time_indices[None, :] < self.acc_index[:, None] #anytime before accretion is not valid
        self.orbit_mask &= np.log10(self.fb) > -4 #anytime after disuption is not valid
        self.disrupt_index = np.argmax(self.orbit_mask, axis=1) #when do they disrupt, 0 if they never do!
        self.disrupted_subhalos = np.where(self.disrupt_index !=0)[0] #which ones disrupt

        self.initalized = np.copy(self.orbit_mask) # so we don't throw away the information before accretion onto the main progenitor!
        self.fb = np.where(self.orbit_mask, self.fb, np.nan) #throw out fb values that aren't valid

    def convert_to_cartesian(self):

        if self.verbose:
            print("converting cyldrical coordinates to cartesian!")

        self.coordinates[~self.initalized] = np.tile([0, 0, 0, 0, 0, 0], (np.count_nonzero(~self.initalized),1)) # setting coodinates to zero for uninitalized orbits or if the subhalo is disrupted

        # transform to cartesian
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid value encountered in divide')
            skyobj = crd.SkyCoord(frame='galactocentric', representation_type='cylindrical', rho=self.coordinates[:,:,0] * u.kpc, phi=self.coordinates[:,:,1] * u.rad, z=self.coordinates[:,:,2]* u.kpc,
                            d_rho = self.coordinates[:,:,3] * u.kpc/u.Gyr, d_phi = np.where(self.coordinates[:,:,0], self.coordinates[:,:,4]/self.coordinates[:,:,0], self.coordinates[:,:,0]) * u.rad/u.Gyr, d_z = self.coordinates[:,:,5] * u.kpc/u.Gyr)
            xyz = skyobj.cartesian.xyz.to(u.kpc).value
            vel = skyobj.cartesian.differentials['s'].d_xyz.to(u.kpc/u.Gyr).value

        # this is the same thing as SatGen `coordinates`, i.e. [branch, redshift, xv], but in cartesian coords
        self.cartesian = np.moveaxis(np.r_[xyz, vel], 0, 2)
        self.cartesian_stitched = np.copy(self.cartesian)

        self.proper_acc_index = np.copy(self.acc_index) #lets grab the index that denotes when a subhalos fall into the host (not just their direct parent!)

        # start at the top of the self and propagate to children (first-order subhalos are already okay)
        for kk in range(2, self.order.max() + 1):
            to_fix = (self.order == kk)
            _, redshift = np.where(to_fix)
            self.cartesian_stitched[to_fix] = self.cartesian_stitched[to_fix] + self.cartesian_stitched[self.ParentID[to_fix], redshift]    
            if self.initalized is not None: # this masks out the orbits of the subhalos before they are accreted onto the main progenitor
                self.initalized[to_fix] = self.initalized[to_fix] & self.initalized[self.ParentID[to_fix], redshift]

            subhalo_ind = np.where(self.acc_order == kk)
            for ind in subhalo_ind: #just so we know when the subhalo falls into the main progenitor
                self.proper_acc_index[ind] = self.proper_acc_index[self.acc_ParentID[ind]]

        # masking the coordinates with initialized mask - use this for any movies!
        self.masked_cartesian_stitched = np.where(np.repeat(np.expand_dims(self.initalized, axis=-1), 6, axis=-1), self.cartesian_stitched, np.nan)

        # doing the same for the non stitched array
        self.masked_cartesian = np.where(np.repeat(np.expand_dims(self.initalized, axis=-1), 6, axis=-1), self.cartesian, np.nan)

        #to decide which subhalos merge!
        self.rmags = np.linalg.norm(self.masked_cartesian[:,:,0:3], axis=2)
        self.Vmags = np.linalg.norm(self.masked_cartesian[:,:,3:6], axis=2)
    
    def halo_mass_evo(self, subhalo_ind):

        #based on Green et al 2019 fitting code using the transfer function
        Vmax = self.acc_Vmax[subhalo_ind] * ev.GvdB_19(self.fb[subhalo_ind], self.acc_concentration[subhalo_ind], track_type="vel") #Green et al 2019 transfer function tidal track
        rmax = self.acc_rmax[subhalo_ind] * ev.GvdB_19(self.fb[subhalo_ind], self.acc_concentration[subhalo_ind], track_type="rad") #Green et al 2019 transfer function tidal track

        return rmax, Vmax
    
    def tides(self):

        if self.verbose:
            print("evolving subhalo profiles based on bound fractions")

        self.rmax = np.full(shape=self.mass.shape, fill_value=np.nan) #empty arrays
        self.Vmax = np.full(shape=self.mass.shape, fill_value=np.nan)

        for subhalo_ind in range(self.Nhalo): #each tidal track is based on fb = m(t)/m(t_acc)
            rmax, Vmax = self.halo_mass_evo(subhalo_ind)
            self.rmax[subhalo_ind] = rmax
            self.Vmax[subhalo_ind] = Vmax

        self.rmax[0] = self.host_rmax #cleaning up the empty host row with the precomuted values!
        self.Vmax[0] = self.host_Vmax
    
        self.parent_rmax = np.full(shape=self.mass.shape, fill_value=np.nan)  #empty arrays
        self.parent_Vmax = np.full(shape=self.mass.shape, fill_value=np.nan)

        for subhalo_ind in range(self.Nhalo): #reorganizing so that we have rmax and vmax of the parents!
            for time_ind, time in enumerate(self.CosmicTime):
                parent_ID = self.ParentID[subhalo_ind, time_ind]
                if parent_ID != -99: #the parent hasnt been born yet!
                    self.parent_rmax[subhalo_ind, time_ind] = self.rmax[parent_ID, time_ind]
                    self.parent_Vmax[subhalo_ind, time_ind] = self.Vmax[parent_ID, time_ind]

    def mergers(self):

        #what we use to account for mergers!
        self.rmax_kscaled = np.log10(self.rmags/self.parent_rmax)
        self.Vmax_kscaled = np.log10(self.Vmags/self.parent_Vmax)

        R_mask = self.rmax_kscaled < self.merger_crit
        V_mask = self.Vmax_kscaled < self.merger_crit
        self.merged_mask_2D = R_mask + V_mask # both limits need to be satisified

        self.merger_index = np.argmax(self.merged_mask_2D, axis=1) #this first time this happens along each time axis
        self.merged_subhalos = np.where(self.merger_index !=0)[0] #which subhalos have a non zero merge index!

        self.merged_order = self.order[self.merged_subhalos, self.merger_index[self.merged_subhalos]]
        self.merged_parents = self.ParentID[self.merged_subhalos, self.merger_index[self.merged_subhalos]] #grabbing the parents they merge into
        #self.masked_ParentID = np.where(self.orbit_mask, self.ParentID, np.nan) #otherwise there will be too much higher order systems to account for
    
        self.account_for_higher_order_merging = False
        self.merger_hierarchy = []
        #for sub_ind, time_ind in enumerate(self.merger_index):
        for ii, sub_ind in enumerate(self.merged_subhalos):
            time_ind = self.merger_index[sub_ind]
            if time_ind != 0:
                # Find all associated subhalos for this subhalo ID
                substructure = find_associated_subhalos(self, sub_ind, time_ind)
                if substructure != None:
                    self.merger_hierarchy.append([sub_ind, substructure])
                    self.account_for_higher_order_merging = True
                else:
                    self.merger_hierarchy.append([sub_ind, []])

        self.final_index = np.maximum(self.merger_index, self.disrupt_index) #which one happens first? disruption or merging?
        if self.verbose:
            print(f"{self.merged_subhalos.shape[0]} subhalos satisfied the merging criteria!")

    def stellar_mass_evo(self, subhalo_ind):

        #based on Errani et al 2021 fitting code

        R50_by_rmax = self.acc_R50[subhalo_ind]/self.acc_rmax[subhalo_ind]
        R50_fb, stellarmass_fb = ev.g_EPW18(self.fb[subhalo_ind], alpha=1.0, lefflmax=R50_by_rmax) #Errani 2018 tidal tracks for stellar evolution!
        R50 = self.acc_R50[subhalo_ind]*R50_fb #scale the sizes!
        stellarmass = self.acc_stellarmass[subhalo_ind]*stellarmass_fb #scale the masses!
        
        return R50, stellarmass

    def baryons(self):

        if self.verbose:
            print("using empirical relations to account for baryons")

        self.acc_stellarmass = 10**gh.lgMs_B18(lgMv=np.log10(self.acc_mass), z=self.acc_redshift, scatter=self.scatter) # the SHMR
        self.acc_R50 = 10**gh.Reff_A24(lgMs=np.log10(self.acc_stellarmass), scatter=self.scatter) # the size mass relation from SAGA

        self.R50 = np.full(shape=self.mass.shape, fill_value=np.nan) # empty arrays
        self.stellarmass = np.full(shape=self.mass.shape, fill_value=np.nan)

        for subhalo_ind in range(self.Nhalo): #each tidal track is based on fb = m(t)/m(t_acc)
            R50, stellarmass = self.stellar_mass_evo(subhalo_ind)
            self.R50[subhalo_ind] = R50
            self.stellarmass[subhalo_ind] = stellarmass

        self.stellarmass = np.where(self.orbit_mask, self.stellarmass, np.nan) # cleaing up the places where the orbit was disrupted!
        self.R50 = np.where(self.orbit_mask, self.R50, np.nan)
        self.stellarmass[0] = 10**gh.lgMs_B18(lgMv=np.log10(self.mass[0]), z=self.redshift, scatter=self.scatter) #adding in the stellar mass of the host

    def stellarhalo(self):

        if self.verbose:
            print("counting up the stellar mass in the halo")

        self.delta_stellarmass = np.full(shape=self.merged_subhalos.shape, fill_value=np.nan)

        for ii, sub_ind in enumerate(self.merged_subhalos):
            time_ind = self.merger_index[sub_ind]
            self.delta_stellarmass[ii] = self.stellarmass[sub_ind, time_ind]*(1-self.fesc) #the fraction that ends up in the descendant
            
            if self.account_for_higher_order_merging: #if there is some mass to add from the higher order substructure
                chain = self.merger_hierarchy[ii]
                collapsed_hierarchy = [chain[0]] + chain[1] #just a list of subhalo indices
                self.final_index[chain[1]] == self.merger_index[chain[0]] #updating so that the whole branch has the same final index: the merger of the parent
                self.delta_stellarmass[ii] += np.sum(self.stellarmass[collapsed_hierarchy, time_ind])*(1-self.fesc)

            self.stellarmass[self.merged_parents[ii], :time_ind] = self.stellarmass[self.merged_parents[ii], :time_ind] + self.delta_stellarmass[ii]

        self.final_mass = self.mass[np.arange(self.final_index.shape[0]), self.final_index]
        self.final_stellarmass = self.stellarmass[np.arange(self.final_index.shape[0]), self.final_index]
        self.fb_stellar = self.stellarmass/self.acc_stellarmass[:, None] #the bound fraction of stellar mass accounting for mergers!

        self.contributed = np.zeros_like(self.final_stellarmass)
        for sub_ind, time_ind in enumerate(self.final_index):

            contributed_i = self.acc_stellarmass[sub_ind] - self.final_stellarmass[sub_ind]
            if time_ind !=0: #didn't survive until z=0
                if time_ind == self.merger_index[sub_ind]: # if it merged!
                    contributed_i += self.final_stellarmass[sub_ind]*self.fesc

                elif time_ind == self.disrupt_index[sub_ind]: # if it disrupted!
                    contributed_i += self.final_stellarmass[sub_ind]

            self.contributed[sub_ind] = contributed_i

        self.total_ICL = self.contributed[1:].sum() #sum across all subhalos, exclude the host

    def summary_stats(self):

        #surviving satellites!
        self.surviving_subhalos = np.where(self.final_index == 0)[0]
        self.surviving_subhalos = np.delete(self.surviving_subhalos, 0) # get rid of the host!
        self.stellarmass_in_satellites = self.stellarmass[self.surviving_subhalos, 0].sum()

        #acccretion times!
        self.disrupted_zacc = self.redshift[self.proper_acc_index[self.disrupted_subhalos]]
        self.merged_zacc = self.redshift[self.proper_acc_index[self.merged_subhalos]]
        self.surviving_zacc = self.redshift[self.proper_acc_index[self.surviving_subhalos]]
        self.avez_disrupted = self.disrupted_zacc.mean()
        self.avez_merged = self.merged_zacc.mean()
        self.avez_surviving = self.surviving_zacc.mean()

        #counts
        self.N_disrupted = self.disrupted_subhalos.shape[0]
        self.N_merged = self.merged_subhalos.shape[0]
        self.N_surviving = self.surviving_subhalos.shape[0]

        #mass ranks!
        self.fracs = []
        acc_mass_mask = np.flip(np.argsort(self.acc_mass)) # sorting by the accretion mass of the subhalos!
        for Nprog in range(2, 16):
            frac_i = np.sum(self.contributed[acc_mass_mask][1:Nprog])/self.total_ICL
            self.fracs.append(frac_i)
            setattr(self, "frac_"+str(Nprog), frac_i)

        #accretion onto central!
        self.central_accreted = np.sum(self.delta_stellarmass[self.merged_parents == 0]) #the stellar mass accreted by the central galaxy
        self.mostmassive_accreted = np.max(self.delta_stellarmass[self.merged_parents == 0]) #the most massive satellite accreted by the central galaxy
        self.single_merger_frac = self.mostmassive_accreted/self.central_accreted
        self.target_stellarmass = self.stellarmass[0,0]

        if self.verbose:
            print("------------------------------------")
            print(f"log Msol stellar halo at z=0: {np.log10(self.total_ICL):.4f}")
            print(f"log Msol accreted by the central galaxy at z=0: {np.log10(self.central_accreted):.4f}")
            print(f"log Msol in the central galaxy at z=0: {np.log10(self.stellarmass[0,0]):.4f}")
            print(f"log Msol in surviving satellites at z=0: {np.log10(self.stellarmass_in_satellites):.4f}")
            print("------------------------------------")
            print(f"N satellites disrtuped: {self.N_disrupted}")
            print(f"N satellites merged with direct parents: {self.N_merged}")
            print(f"N satellites survived to z=0: {self.N_surviving}")

    def create_summarystats_array(self, keys):
        #note that this only works for attributes that have a single value! not for array-like attributes
        val_list = []
        for key in keys:
            val_list.append(getattr(self, key, np.nan))  # Return NaN if key does not exist
        return np.array(val_list)
    
    def create_survsat_dict(self):

        # Get the index of the minimum non-NaN value for each row
        self.rmin_mask = np.nanargmin(self.rmags[self.surviving_subhalos], axis=1)

        # Use the obtained indices to extract the corresponding minimum values and their order
        self.rmin = self.rmags[self.surviving_subhalos, self.rmin_mask]
        self.rmin_order = self.order[self.surviving_subhalos, self.rmin_mask]

        #the positions and velocity
        self.surviving_rmag = self.rmags[self.surviving_subhalos, 0]
        self.surviving_Vmag = self.Vmags[self.surviving_subhalos, 0]

        # now the orders!
        self.kmax = np.nanmax(self.order[self.surviving_subhalos], axis=1)
        self.kfinal = self.order[self.surviving_subhalos, 0]

        #the masses
        self.surviving_final_mass = self.final_mass[self.surviving_subhalos]
        self.surviving_acc_mass = self.acc_mass[self.surviving_subhalos]
        self.surviving_final_stellarmass = self.final_stellarmass[self.surviving_subhalos]
        self.surviving_acc_stellarmass = self.acc_stellarmass[self.surviving_subhalos]

        dictionary = {"tree_index": self.file.split("/")[-1], #just to give us the file index
                    "mass": self.surviving_final_mass,  # final halo mass
                    "acc_mass": self.surviving_acc_mass,  # halo mass @ accretion halo mass
                    "stellarmass":  self.surviving_final_stellarmass,  # final stellar mass
                    "acc_stellarmass": self.surviving_acc_stellarmass, # stellar mass @ accretion halo mass
                    "z_acc": self.surviving_zacc, # accretion redshift
                    "self.Rmag": self.surviving_rmag, #position
                    "self.Vmag": self.surviving_Vmag, #velocity
                    "Rperi": self.rmin, #rperi with respect to direct parent!
                    "k_Rperi": self.rmin_order, # the order associated with rperi
                    "k_max": self.kmax, #max order across history
                    "k_final": self.kfinal} # final order
            
        return dictionary
        
    def plot_merged_satellites(self):

        plt.figure(figsize=(10,6))

        for sub_ind in range(1, self.Nhalo):
            if np.isin(sub_ind, self.merged_subhalos):        
                plt.plot(self.CosmicTime, self.stellarmass[sub_ind], color="grey", alpha=0.5)
                plt.scatter(self.CosmicTime[self.merger_index[sub_ind]], self.stellarmass[sub_ind, self.merger_index[sub_ind]], marker="*", color="red")
                plt.scatter(self.CosmicTime[self.disrupt_index[sub_ind]], self.stellarmass[sub_ind, self.disrupt_index[sub_ind]], marker="+", color="k")

        plt.title(f"% of satellites merged: {100*self.merged_subhalos.shape[0]/(self.Nhalo-1):.2f} \n % of satellites disrupted: {100*self.disrupted_subhalos.shape[0]/(self.Nhalo-1):.2f} \n")
        plt.yscale("log")
        plt.xlabel("cosmic time (Gyr)")
        plt.ylabel("stellar mass")
        plt.show()

        
    def plot_subhalo_properties(self, subhalo_ind):

        start = self.acc_index[subhalo_ind] # just to have a reference!
        stop = self.disrupt_index[subhalo_ind]

        order_jumps = np.where(self.order[subhalo_ind][:-1] != self.order[subhalo_ind][1:])[0] + 1

        fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey="col")
        axes_r = axes.ravel()
        for ax in axes_r:
            ax.axvline(self.CosmicTime[stop], ls=":", color="grey")
            ax.axvline(self.CosmicTime[start], ls=":", color="grey")
            for jump in order_jumps[:-1]:
                ax.axvline(self.CosmicTime[jump], ls="--", color="green")

        axes[0,0].plot(self.CosmicTime, self.mass[subhalo_ind], label=f" subhalo ID: {subhalo_ind} \n intial order: {self.acc_order[subhalo_ind]} \n number of jumps: {order_jumps.shape[0] -1}")
        axes[0,0].plot(self.CosmicTime, self.mass[0], color="k")
        axes[0,0].set_yscale("log")
        axes[0,0].set_ylabel("M$_{\\rm H}$ (M$_{\odot}$)")
        axes[0,0].set_title("halo mass")
        axes[0,0].legend(loc=3, framealpha=1)

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

        axes[1,1].plot(self.CosmicTime, self.parent_rmax[subhalo_ind])
        axes[1,1].set_ylabel("r$_{\\rm max}$ (kpc)")
        axes[1,1].set_xlabel("Cosmic Time (Gyr)")
        axes[1,1].set_title("(k-1) radius of maximum V$_{\\rm circ}$")

        axes[1,2].plot(self.CosmicTime, self.parent_Vmax[subhalo_ind])
        axes[1,2].set_ylabel("V$_{\\rm max}$ (kpc / Gyr)")
        axes[1,2].set_xlabel("Cosmic Time (Gyr)")
        axes[1,2].set_title("(k-1) maximum V$_{\\rm circ}$")
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
            ax.set_ylim(0, 3)
            ax.set_xlim(0, 3)
            # ax.set_yscale("log")
            # ax.axvline(0, ls="--", color="grey")
            # ax.axhline(0, ls="--", color="grey")
            # ax.axhline(-1, ls="--", color="red", label="merging criteria")
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

            rect1 = matplotlib.patches.Rectangle((-4,-4), 3, 3, color='red', alpha=0.3)

            if type(subhalo_indices) == type(None):
                ax.scatter(self.rmax_kscaled[:, time_index], self.Vmax_kscaled[:, time_index], marker=".", s=1.5, color="k")
            else:
                ax.scatter(self.rmax_kscaled[subhalo_indices, time_index], self.Vmax_kscaled[subhalo_indices, time_index], marker=".", s=1.5, color="k")

            ax.set_ylabel("log (V$^k$ / V$_{max}^{k-1}$)")
            ax.set_xlabel("log (R$^k$ / R$_{max}^{k-1}$)")
            ax.set_ylim(-3, 2)
            ax.set_xlim(-3, 2)

            rect1 = matplotlib.patches.Rectangle((-4,-4), 3, 3, color='red', alpha=0.3)
            ax.add_patch(rect1)

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

            ax.plot(10**halo_smooth, 10**gh.lgMs_B18(halo_smooth, self.redshift[time_index], scatter=0), label=f"Behroozi et al 2018 (UM): z={self.redshift[time_index]:.2f}")
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
                sc = ax.scatter(self.masked_cartesian_stitched[:, time_index, 0], self.masked_cartesian_stitched[:, time_index, 1], c=k, cmap=cmap, norm=norm, marker=".", s=10)
            else:
                k = self.order[subhalo_indices, time_index]
                sc = ax.scatter(self.masked_cartesian_stitched[subhalo_indices, time_index, 0], self.masked_cartesian_stitched[subhalo_indices, time_index, 1], c=k, cmap=cmap, norm=norm, marker=".", s=10)

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


def find_associated_subhalos(tree, sub_ind, time_ind):
    associated_set = []

    # Checking to see if there are any direct children at this time step
    direct_parent_merging = tree.ParentID[:, time_ind] == sub_ind 
    if np.any(direct_parent_merging):
        associated_subhalos = np.where(direct_parent_merging)[0]  # Any subhalos that have the same parent
        disrupt_mask = tree.disrupt_index[associated_subhalos] < time_ind  # Disruption must happen after the merger
        associated_subhalos = associated_subhalos[disrupt_mask]
        associated_set.extend(associated_subhalos)
        
        # Recursively collect descendants of each subhalo
        for subhalo in associated_subhalos:
            subhalo_descendants = find_associated_subhalos(tree, subhalo, time_ind)
            if subhalo_descendants:  # Ensure no NoneType is returned
                associated_set.extend(subhalo_descendants)

    return associated_set



################################
###### FIRST PAPER FUNCS #######
################################

def anamass(file):
    self = np.load(file) #open file and read
    mass = self["mass"]
    redshift = self["redshift"]
    coords = self["coordinates"]
    orders = self["order"]

    mass = np.delete(mass, 1, axis=0) #there is some weird bug for this index!
    coords = np.delete(coords, 1, axis=0)
    orders = np.delete(orders, 1, axis=0)

    mask = mass != -99. # converting to NaN values
    mass = np.where(mask, mass, np.nan)  
    orders = np.where(mask, orders, np.nan)
    Nhalo = mass.shape[0]
    try:
        peak_index = np.nanargmax(mass, axis=1) #finding the maximum mass
        peak_mass = mass[np.arange(peak_index.shape[0]), peak_index]
        peak_red = redshift[peak_index]
        peak_order = orders[np.arange(peak_index.shape[0]), peak_index]
        final_mass = mass[:,0] # the final index is the z=0 time step. this will be the minimum mass for all subhalos
        final_order = orders[:,0]
        final_coord = coords[:,0,:] # this will be the final 6D positons

        return peak_mass, peak_red, peak_order, final_mass, final_order, final_coord
    
    except ValueError:
        print("bad run, returning empty arrays!")
        return np.zeros(Nhalo), np.zeros(Nhalo), np.zeros(Nhalo), np.zeros(Nhalo), np.zeros(Nhalo), np.zeros(shape=(Nhalo, 6))

    
def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx

def hostmass(file):
    openself = np.load(file) #open file and read
    z50 = openself["redshift"][find_nearest1(openself["mass"][0], openself["mass"][0,0]/2)]
    z10 = openself["redshift"][find_nearest1(openself["mass"][0], openself["mass"][0,0]/10)]
    return np.array([np.log10(openself["mass"][0,0]), z50, z10, openself["mass"].shape[0]])

def main_progenitor_history(datadir, Nself):
    thin = 25 

    files = []    
    for filename in os.listdir(datadir):
        if filename.startswith('self') and filename.endswith('evo.npz'): 
            files.append(os.path.join(datadir, filename))

    host_mat = np.zeros(shape=(Nself,354))
    N_sub = np.zeros(shape=Nself)
    for i, file in enumerate(files[0:Nself]):
        self_data_i = np.load(file)
        if self_data_i["mass"][0,:].shape[0] == 354:
            host_mat[i] = np.log10(self_data_i["mass"][0,:])
            surv = []
            for j, val in enumerate(self_data_i["mass"]):
                final_mass = val[0]
                peak_mass = val.max()
                if np.log10(final_mass) - np.log10(peak_mass) > -4:
                    surv.append(j)
            N_sub[i] = len(surv)

    quant = np.percentile(host_mat, np.array([5, 50, 95]), axis=0, method="closest_observation")
    error = np.array([quant[1][::thin] - quant[0][::thin], quant[2][::thin] - quant[1][::thin]])

    return host_mat, quant, error, N_sub


class Realizations:

    """
    Condensing each set of realizations into mass matrices that are easy to handle. 
    This class is applied to a directory that holds all the "raw" satgen files. 
    Each data directory should be a seperate set of realizations.
    """
        
    def __init__(self, datadir, Nhalo=1600):
        self.datadir = datadir
        self.Nhalo = Nhalo  #Nhalo is set rather high to accomodate for larger numbers of subhalos
        self.grab_anadata()

    def grab_anadata(self):
        files = []    
        for filename in os.listdir(self.datadir):
            if filename.startswith('self') and filename.endswith('evo.npz'): 
                files.append(os.path.join(self.datadir, filename))

        self.files = files
        self.Nreal = len(files)
        # will get mad at you if you set it too low

        print("number of realizations:", self.Nreal)
        print("number of branches/subhalos:", self.Nhalo)

        acc_Mass = np.zeros(shape=(self.Nreal, self.Nhalo))
        acc_Redshift = np.zeros(shape=(self.Nreal, self.Nhalo))
        acc_Order = np.zeros(shape=(self.Nreal, self.Nhalo))
        final_Mass = np.zeros(shape=(self.Nreal, self.Nhalo))
        final_Order = np.zeros(shape=(self.Nreal, self.Nhalo))
        final_Coord = np.zeros(shape=(self.Nreal, self.Nhalo, 6)) 
        host_Prop = np.zeros(shape=(self.Nreal, 4))

        for i,file in enumerate(files): # this part takes a while if you have a lot of selfs
            peak_mass, peak_red, peak_order, final_mass, final_order, final_coord = anamass(file)
            temp_Nhalo = peak_mass.shape[0]

            # need to pad them so they can be written to a single final matrix
            peak_mass = np.pad(peak_mass, (0,self.Nhalo-temp_Nhalo), mode="constant", constant_values=np.nan) 
            peak_red = np.pad(peak_red, (0,self.Nhalo-temp_Nhalo), mode="constant", constant_values=np.nan)
            peak_order = np.pad(peak_order, (0,self.Nhalo-temp_Nhalo), mode="constant", constant_values=np.nan)
            final_mass = np.pad(final_mass, (0,self.Nhalo-temp_Nhalo), mode="constant", constant_values=np.nan)
            final_order = np.pad(final_order, (0,self.Nhalo-temp_Nhalo), mode="constant", constant_values=np.nan)
            coord_pad = np.zeros(shape=(self.Nhalo - temp_Nhalo, 6))
            final_coord = np.append(final_coord, coord_pad, axis=0) #appends it to the end!

            acc_Mass[i,:] = peak_mass
            acc_Redshift[i,:] = peak_red
            acc_Order[i,:] = peak_order
            final_Mass[i,:] = final_mass
            final_Order[i,:] = final_order
            final_Coord[i,:,:] = final_coord

            host_Prop[i,:] = hostmass(file) # now just to grab the host halo properties

        bad_run_ind = np.where(np.sum(acc_Mass, axis=1) == 0)[0] # the all zero cuts
        if bad_run_ind != None:
            print("++++++++++++++++++++")
            print(bad_run_ind)
            print("++++++++++++++++++++")


        self.metadir = self.datadir+"meta_data/"
        os.mkdir(self.metadir)
        analysis = np.dstack((acc_Mass, acc_Redshift, acc_Order, final_Mass, final_Order, final_Coord)).transpose((2,0,1))
        np.save(self.metadir+"subhalo_anadata.npy", analysis)
        np.save(self.metadir+"host_properties.npy", host_Prop) # make another folder one stage up!!!

    def plot_single_realization(self, nhalo=20, rand=False, nstart=1):

        random_index = np.random.randint(0,len(self.files)-1)
        self = np.load(self.files[random_index])

        mass = self["mass"]
        time = self["CosmicTime"]

        if rand==True:
            select = np.random.randint(1,mass.shape[0],nhalo)
        elif rand==False:
            select = np.linspace(nstart,nstart+nhalo, nhalo).astype("int")

        colors = cm.viridis(np.linspace(0, 1, nhalo))

        plt.figure(figsize=(6,6))
        for i in range(nhalo):
            plt.plot(time, mass[select[i]], color=colors[i])

        plt.plot(time, mass[0], color="red")
        plt.xlabel("Gyr", fontsize=15)
        plt.ylabel("halo mass (M$_{\odot}$)", fontsize=15)
        plt.yscale("log")
        plt.axhline(10**8, ls="--", color="black")
        plt.show()


###############################################
### FOR CLEANING THE CONDENSED SUBHALO DATA ###
###############################################

def differential(rat, rat_bins, rat_binsize): 
    N = np.histogram(rat, bins=rat_bins)[0]
    return N/rat_binsize

class MassMat:

    """
    An easy way of interacting with the condensed mass matricies.
    One instance of the Realizations class will create several SAGA-like samples.
    """
        
    def __init__(self, metadir, cut_radius=False, save=False, plot=False, **kwargs):

        self.metadir = metadir
        self.save = save
        self.plot = plot
        self.cut_radius = cut_radius
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.prep_data()
        #self.SHMF_paper()
        self.SAGA_break()
        #self.write_to_FORTRAN()

    def prep_data(self):

        #, clean_host=False):
        self.subdata = np.load(self.metadir+"subhalo_anadata.npy")
        acc_mass = self.subdata[0]
        acc_red = self.subdata[1]
        acc_order = self.subdata[2]
        final_mass = self.subdata[3]
        final_order = self.subdata[4]
        #final_coords = self.subdata[5:11]
        fx = self.subdata[5]
        fy = self.subdata[6]
        fz = self.subdata[7]
        fvx = self.subdata[8]
        fvy = self.subdata[9]
        fvz = self.subdata[10]

        # if clean_host == True:
        #     mask = acc_mass[:,0] == 1e12 # only exactly the same host halo mass, also clears dead runs
        #     if sum(~mask) > 0:
        #         print("excluding some runs!")
        #         acc_mass = acc_mass[mask]
        #         final_mass = final_mass[mask]
        #         acc_red = acc_red[mask]
        #         final_coords = final_coordinates[mask]
        #     else:
        #         print("no host cleaning needed!")

        self.acc_mass = np.delete(acc_mass, 0, axis=1) # removing the host from the datajsm
        self.acc_red = np.delete(acc_red, 0, axis=1)
        self.acc_order = np.delete(acc_order, 0, axis=1)
        self.final_mass = np.delete(final_mass, 0, axis=1)
        self.final_order = np.delete(final_order, 0, axis=1)
        self.fx = np.delete(fx, 0, axis=1)
        self.fy = np.delete(fy, 0, axis=1)
        self.fz = np.delete(fz, 0, axis=1)
        self.fvx = np.delete(fvx, 0, axis=1)
        self.fvy = np.delete(fvy, 0, axis=1)
        self.fvz = np.delete(fvz, 0, axis=1)
        self.r = np.sqrt(self.fx**2 + self.fz**2)
        self.Rvir_host = 291.61264 #kpc

        self.surv_mask = np.log10(self.final_mass/self.acc_mass) > self.phi_res # now selecting only the survivers
        self.virial_mask = self.r < self.Rvir_host

        if self.cut_radius == True:
            self.combined_mask = np.logical_and(self.surv_mask, self.virial_mask)
            self.acc_surv_mass = np.ma.filled(np.ma.masked_array(self.acc_mass, mask=~self.combined_mask),fill_value=np.nan)
            self.acc_mass = np.ma.filled(np.ma.masked_array(self.acc_mass, mask=~self.virial_mask),fill_value=np.nan)
            self.final_mass = np.ma.filled(np.ma.masked_array(self.final_mass, mask=~self.virial_mask),fill_value=np.nan)
        else:
            self.acc_surv_mass = np.ma.filled(np.ma.masked_array(self.acc_mass, mask=~self.surv_mask),fill_value=np.nan)

        self.lgMh_acc = np.log10(self.acc_mass) # accretion
        self.lgMh_final = np.log10(self.final_mass) # final mass
        self.lgMh_acc_surv = np.log10(self.acc_surv_mass) # the accretion mass of surviving halos

        self.hostprop = np.load(self.metadir+"host_properties.npy")
        self.Mhosts = 10**self.hostprop[:,0]
        self.z_50 = self.hostprop[:,1]
        self.z_10 = self.hostprop[:,2]

        self.acc_rat = np.log10((self.acc_mass.T / self.Mhosts).T)  
        self.final_rat = np.log10((self.final_mass.T / self.Mhosts).T) 
        self.acc_surv_rat = np.log10((self.acc_surv_mass.T / self.Mhosts).T)

    def SHMF_paper(self):

        self.Nsamp = self.lgMh_acc.shape[0]

        self.lgMh_bins = np.linspace(self.min_mass, self.max_mass, self.Nbins)
        self.lgMh_binsize = self.lgMh_bins[1] - self.lgMh_bins[0]
        self.lgMh_bincenters = 0.5 * (self.lgMh_bins[1:] + self.lgMh_bins[:-1])

        self.acc_counts = np.apply_along_axis(differential, 1, self.lgMh_acc, rat_bins=self.lgMh_bins, rat_binsize=self.lgMh_binsize) # the accretion mass of the surviving halos
        acc_counts_ave = np.average(self.acc_counts, axis=0)
        acc_counts_std = np.std(self.acc_counts, axis=0)
        self.acc_SHMF_werr = np.array([acc_counts_ave, acc_counts_std])

        self.acc_surv_counts = np.apply_along_axis(differential, 1, self.lgMh_acc_surv, rat_bins=self.lgMh_bins, rat_binsize=self.lgMh_binsize) # the accretion mass of the surviving halos
        acc_surv_counts_ave = np.average(self.acc_surv_counts, axis=0)
        acc_surv_counts_std = np.std(self.acc_surv_counts, axis=0)
        self.acc_surv_SHMF_werr = np.array([acc_surv_counts_ave, acc_surv_counts_std])

        self.surv_counts = np.apply_along_axis(differential, 1, self.lgMh_final, rat_bins=self.lgMh_bins, rat_binsize=self.lgMh_binsize) # the accretion mass of the surviving halos
        surv_counts_ave = np.average(self.surv_counts, axis=0)
        surv_counts_std = np.std(self.surv_counts, axis=0)
        self.surv_SHMF_werr = np.array([surv_counts_ave, surv_counts_std])

        if self.plot==True:
        
            fig, ax = plt.subplots(figsize=(8, 8))

            ax.plot(self.lgMh_bincenters, self.acc_SHMF_werr[0], label="$\mathrm{Unevolved}$", color="darkcyan", ls="-.")
            ax.plot(self.lgMh_bincenters, self.acc_surv_SHMF_werr[0], label="$\mathrm{Unevolved}$, $\mathrm{Surviving}$", color="black")
            ax.fill_between(self.lgMh_bincenters, y1=self.acc_surv_SHMF_werr[0]-self.acc_surv_SHMF_werr[1], y2=self.acc_surv_SHMF_werr[0]+self.acc_surv_SHMF_werr[1], alpha=0.1, color="black")
            ax.plot(self.lgMh_bincenters, self.surv_SHMF_werr[0], label="$\mathrm{Evolved}$, $\mathrm{Surviving}$", color="darkmagenta", ls="-.")

            ax.set_xlabel("$\log (M_{\mathrm{sub}})$")
            ax.set_ylabel("$dN / d\ \log (M_{\mathrm{sub}})$")
            ax.set_ylim(0.03, 1000)
            ax.set_yscale("log")
            ax.legend()
        
            if self.save==True:
                plt.savefig(self.metadir+"SHMF.pdf")

            plt.show()

    def SAGA_break(self):

        self.snip = self.lgMh_acc_surv.shape[0]%self.Nsamp
        if self.snip != 0.0:
            print("Cannot evenly divide your sample by the number of samples!")
            # lgMh_snip = np.delete(self.lgMh_acc_surv, np.arange(self.snip), axis=0)
            # self.Nsets = int(lgMh_snip.shape[0]/self.Nsamp) #dividing by the number of samples
            # print("dividing your sample into", self.Nsets, "sets.", self.snip, "selfs were discarded")
            # self.lgMh_mat = np.array(np.split(lgMh_snip, self.Nsets, axis=0))
        else:
            self.Nsets = int(self.lgMh_acc_surv.shape[0]/self.Nsamp) #dividing by the number of samples
            self.Mhosts_mat = np.array(np.split(self.Mhosts, self.Nsets))
            self.z50_mat = np.array(np.split(self.z_50, self.Nsets))
            self.acc_surv_lgMh_mat = np.array(np.split(self.lgMh_acc_surv, self.Nsets, axis=0))
            self.acc_red_mat = np.array(np.split(self.acc_red, self.Nsets, axis=0))
            self.final_lgMh_mat = np.array(np.split(self.lgMh_final, self.Nsets, axis=0))
            self.final_order_mat = np.array(np.split(self.final_order, self.Nsets, axis=0))
            self.acc_order_mat = np.array(np.split(self.acc_order, self.Nsets, axis=0))
            self.fx_mat = np.array(np.split(self.fx, self.Nsets, axis=0))
            self.fy_mat = np.array(np.split(self.fy, self.Nsets, axis=0))
            self.fz_mat = np.array(np.split(self.fz, self.Nsets, axis=0))
            self.fvx_mat = np.array(np.split(self.fvx, self.Nsets, axis=0))
            self.fvy_mat = np.array(np.split(self.fvy, self.Nsets, axis=0))
            self.fvz_mat = np.array(np.split(self.fvz, self.Nsets, axis=0))
            
            np.savez(self.metadir+"models_updated.npz",
                    host_mass = self.Mhosts_mat,
                    z50 = self.z50_mat,
                    acc_mass = self.acc_surv_lgMh_mat,
                    final_mass = self.final_lgMh_mat,
                    acc_redshift = self.acc_red_mat)

        # if self.save==True:
        #     print("saving the accretion masses!")
        #     np.savez(self.metadir+"models_updated.npz",
        #             host_mass = self.Mhosts_mat,
        #             z50 = self.z50_mat,
        #             acc_mass = self.acc_surv_lgMh_mat,
        #             final_mass = self.final_lgMh_mat,
        #             acc_redshift = self.acc_red_mat)
            

    def write_to_FORTRAN(self):
        Nsub = []
        z_50 = []
        z_10 = []
        M_acc = []
        z_acc = []
        M_star = []
        M_final = []
        x_final = []
        y_final = []
        z_final = []
        acc_order = []
        final_order = []
        vx_final = []
        vy_final = []
        vz_final = []
        self_id = []
        sat_id = []

        for iself in range(self.Nsamp):
            Nsub_i = np.argwhere(~np.isnan(self.lgMh_acc_surv[iself]))[:,0]
            for j, isat in enumerate(Nsub_i):
                Nsub.append(len(Nsub_i))
                self_id.append(iself+1)
                z_50.append(self.z_50[iself])
                z_10.append(self.z_10[iself])
                sat_id.append(j+1)
                M_acc.append(self.lgMh_acc_surv[iself][isat])
                z_acc.append(self.acc_red[iself][isat])
                M_star.append(jsm_SHMR.lgMs_RP17(self.lgMh_acc_surv[iself][isat], self.acc_red[iself][isat]))
                M_final.append(self.lgMh_final[iself][isat])
                x_final.append(self.fx[iself][isat])
                y_final.append(self.fy[iself][isat])
                z_final.append(self.fz[iself][isat])
                vx_final.append(self.fvx[iself][isat])
                vy_final.append(self.fvy[iself][isat])
                vz_final.append(self.fvz[iself][isat])
                acc_order.append(self.acc_order[iself][isat].astype("int"))
                final_order.append(self.final_order[iself][isat].astype("int"))

        keys = ("sat_id", "self_id", "Nsub", "z_50", "z_10",
                "M_acc", "z_acc", "M_star", "M_final",
                "R(kpc)", "rat(rad)", "z(kpc)", "VR(kpc/Gyr)", "Vrat(kpc/Gyr)" ,"Vz(kpc/Gyr)",
                "k_acc", "k_final")
        
        data = Table([sat_id, self_id, Nsub, z_50, z_10,
                      M_acc, z_acc, M_star, M_final,
                      x_final, y_final, z_final, vx_final, vy_final, vz_final,
                      acc_order, final_order], names=keys)
        
        print("writing out the subhalo data")
        data.write(self.metadir+"subhalos.dat", format="ascii", overwrite=True)
    
        # Hnpy = np.load(self.metadir+"host_properties.npz")
        # Hkeys = ("lgMh", "z_50", "z_10", "Nsub_total")
        # Hdata = Table([Hnpy[:,0], Hnpy[:,1], Hnpy[:,2], Hnpy[:,3]], names=Hkeys)

        # print("writing out the host data")
        # Hdata.write(self.metadir+"host_prop.dat", format="ascii", overwrite=True)


# def prep_data(numpyfile, convert=False, includenan=True):

#     """_summary_
#     a quick a dirty way of getting satellites statistics. Really should use the MassMat class below

#     """
#     Mh = np.load(numpyfile)
#     #Mh[:, 0] = 0.0  # masking the host mass in the matrix
#     #zero_mask = Mh != 0.0 
#     #Mh = np.log10(np.where(zero_mask, Mh, np.nan)) #switching the to nans!
#     lgMh = np.log10(Mh)

#     if includenan == False:
#         max_sub = min(Mh.shape[1] - np.sum(np.isnan(Mh),axis=1)) # not padded! hope it doesnt screw up stats
#     else: 
#         max_sub = max(Mh.shape[1] - np.sum(np.isnan(Mh),axis=1))

#     lgMh = lgMh[:,1:max_sub]  #excluding the host mass
#     if convert==False:
#         return lgMh
#     # else:
#     #     return galhalo.lgMs_B13(lgMh)
 
    # def SHMF(self):
    #     self.acc_surv_rat_counts = np.apply_along_axis(differential, 1, self.acc_surv_rat, rat_bins=self.rat_bins, rat_binsize=self.rat_binsize) # the accretion mass of the surviving halos
    #     acc_surv_rat_SHMF_ave = np.average(self.acc_surv_rat_counts, axis=0)
    #     acc_surv_rat_SHMF_std = np.std(self.acc_surv_rat_counts, axis=0)
    #     self.acc_surv_SHMF_werr = np.array([acc_surv_rat_SHMF_ave, acc_surv_rat_SHMF_std])

    #     self.final_rat_counts = np.apply_along_axis(differential, 1, self.final_rat, rat_bins=self.rat_bins, rat_binsize=self.rat_binsize) # the final mass of all halos
    #     final_rat_SHMF_ave = np.average(self.final_rat_counts, axis=0)
    #     final_rat_SHMF_std = np.std(self.final_rat_counts, axis=0)
    #     self.final_SHMF_werr = np.array([final_rat_SHMF_ave, final_rat_SHMF_std])

    #     self.acc_rat_counts = np.apply_along_axis(differential, 1, self.acc_rat, rat_bins=self.rat_bins, rat_binsize=self.rat_binsize) # the accretion mass of all the halos
    #     acc_rat_SHMF_ave = np.average(self.acc_rat_counts, axis=0)
    #     acc_rat_SHMF_std = np.std(self.acc_rat_counts, axis=0)
    #     self.acc_SHMF_werr = np.array([acc_rat_SHMF_ave, acc_rat_SHMF_std])

    #     self.rat_bincenters = 0.5 * (self.rat_bins[1:] + self.rat_bins[:-1])
    #     if self.plot==True:
        
    #         fig, ax = plt.subplots(figsize=(8, 8))

    #         ax.plot(self.rat_bincenters, self.acc_SHMF_werr[0], label="Total population (z = z$_{\mathrm{acc}}$)", color="green", ls="-.")
    #         #plt.fill_between(self.rat_bincenters, y1=self.acc_SHMF_werr[0]-self.acc_SHMF_werr[1], y2=self.acc_SHMF_werr[0]+self.acc_SHMF_werr[1], alpha=0.1, color="grey")

    #         ax.plot(self.rat_bincenters, self.acc_surv_SHMF_werr[0], label="Surviving population (z = z$_{\mathrm{acc}}$)", color="cornflowerblue")
    #         ax.fill_between(self.rat_bincenters, y1=self.acc_surv_SHMF_werr[0]-self.acc_surv_SHMF_werr[1], y2=self.acc_surv_SHMF_werr[0]+self.acc_surv_SHMF_werr[1], alpha=0.2, color="cornflowerblue")

    #         ax.plot(self.rat_bincenters, self.final_SHMF_werr[0],  label="Surviving population (z = 0)", color="red", ls="-.")
    #         #plt.fill_between(self.rat_bincenters, y1=self.final_SHMF_werr[0]-self.final_SHMF_werr[1], y2=self.final_SHMF_werr[0]+self.final_SHMF_werr[1], alpha=0.1, color="grey")

    #         ax.axvline(self.phi_res, ls="--", color="black")
    #         ax.text(self.phi_res+0.05, 0.1, "resolution limit", rotation=90, color="black", fontsize=15)
            
    #         ax.set_xlabel("log (m/M$_{\mathrm{host}}$)", fontsize=15)
    #         ax.set_yscale("log")
    #         ax.set_ylabel("dN / dlog(m/M)", fontsize=15)

    #         if self.save==True:
    #             plt.savefig(self.metadir+"SHMF.pdf")

    #         plt.show()

    # def write_to_FORTRAN(self):
    #     Nsub = []
    #     z_50 = []
    #     z_10 = []
    #     M_acc = []
    #     z_acc = []
    #     M_star = []
    #     M_final = []
    #     x_final = []
    #     y_final = []
    #     z_final = []
    #     acc_order = []
    #     final_order = []
    #     vx_final = []
    #     vy_final = []
    #     vz_final = []
    #     self_id = []
    #     sat_id = []

    #     for iself in range(self.Nsamp):
    #         Nsub_i = np.argwhere(~np.isnan(self.lgMh_acc_surv[iself]))[:,0]
    #         for j, isat in enumerate(Nsub_i):
    #             Nsub.append(len(Nsub_i))
    #             self_id.append(iself+1)
    #             z_50.append(self.z_50[iself])
    #             z_10.append(self.z_10[iself])
    #             sat_id.append(j+1)
    #             M_acc.append(self.lgMh_acc_surv[iself][isat])
    #             z_acc.append(self.acc_red[iself][isat])
    #             M_star.append(jsm_SHMR.lgMs_RP17(self.lgMh_acc_surv[iself][isat], self.acc_red[iself][isat]))
    #             M_final.append(self.lgMh_final[iself][isat])
    #             x_final.append(self.fx[iself][isat])
    #             y_final.append(self.fy[iself][isat])
    #             z_final.append(self.fz[iself][isat])
    #             vx_final.append(self.fvx[iself][isat])
    #             vy_final.append(self.fvy[iself][isat])
    #             vz_final.append(self.fvz[iself][isat])
    #             acc_order.append(self.acc_order[iself][isat].astype("int"))
    #             final_order.append(self.final_order[iself][isat].astype("int"))

    #     keys = ("sat_id", "self_id", "Nsub", "z_50", "z_10",
    #             "M_acc", "z_acc", "M_star", "M_final",
    #             "R(kpc)", "rat(rad)", "z(kpc)", "VR(kpc/Gyr)", "Vrat(kpc/Gyr)" ,"Vz(kpc/Gyr)",
    #             "k_acc", "k_final")
        
    #     data = Table([sat_id, self_id, Nsub, z_50, z_10,
    #                   M_acc, z_acc, M_star, M_final,
    #                   x_final, y_final, z_final, vx_final, vy_final, vz_final,
    #                   acc_order, final_order], names=keys)
        
    #     print("writing out the subhalo data")
    #     data.write(self.metadir+"subhalos.dat", format="ascii", overwrite=True)
    
    #     # Hnpy = np.load(self.metadir+"host_properties.npz")
    #     # Hkeys = ("lgMh", "z_50", "z_10", "Nsub_total")
    #     # Hdata = Table([Hnpy[:,0], Hnpy[:,1], Hnpy[:,2], Hnpy[:,3]], names=Hkeys)

    #     # print("writing out the host data")
    #     # Hdata.write(self.metadir+"host_prop.dat", format="ascii", overwrite=True)


    # def write_to_FORTRAN_SAGA(self):
    #     Nsub = []
    #     M_acc = []
    #     z_acc = []
    #     M_final = []
    #     x_final = []
    #     y_final = []
    #     z_final = []
    #     acc_order = []
    #     final_order = []
    #     vx_final = []
    #     vy_final = []
    #     vz_final = []
    #     SAGA_id = []
    #     self_id = []
    #     sat_id = []

    #     for isaga in range(self.Nsets):
    #         for iself in range(self.Nsamp):
    #             Nsub_i = np.argwhere(~np.isnan(self.acc_surv_lgMh[iself]))[:,0]
    #             for j, isat in enumerate(Nsub_i):
    #                 Nsub.append(len(Nsub_i))
    #                 SAGA_id.append(isaga)
    #                 self_id.append(iself+1)
    #                 sat_id.append(j+1)
    #                 M_acc.append(self.acc_surv_lgMh[iself][isat])
    #                 z_acc.append(self.acc_red[iself][isat])
    #                 M_final.append(self.final_lgMh[iself][isat])
    #                 x_final.append(self.fx[iself][isat])
    #                 y_final.append(self.fy[iself][isat])
    #                 z_final.append(self.fz[iself][isat])
    #                 vx_final.append(self.fvx[iself][isat])
    #                 vy_final.append(self.fvy[iself][isat])
    #                 vz_final.append(self.fvz[iself][isat])
    #                 acc_order.append(self.acc_order[iself][isat].astype("int"))
    #                 final_order.append(self.final_order[iself][isat].astype("int"))

    #     keys = ("sat_id", "self_id", "SAGA_id", "Nsub",
    #             "M_acc", "z_acc", "M_final",
    #             "R(kpc)", "rat(rad)", "z(kpc)", "VR(kpc/Gyr)", "Vrat(kpc/Gyr)" ,"Vz(kpc/Gyr)",
    #             "k_acc", "k_final") # why are these units weird?
        
    #     data = Table([sat_id, self_id, SAGA_id, Nsub,
    #                   M_acc, z_acc, M_final,
    #                   x_final, y_final, z_final, vx_final, vy_final, vz_final,
    #                   acc_order, final_order], names=keys)
        
    #     print("writing out the subhalo data")
    #     data.write(self.metadir+"FvdB_MCMC.dat", format="ascii", overwrite=True)
    
    #     Hnpy = np.load(self.metadir+"host_properties.npz")
    #     Hkeys = ("lgMh", "z_50", "z_10", "Nsub_total")
    #     Hdata = Table([Hnpy[:,0], Hnpy[:,1], Hnpy[:,2], Hnpy[:,3]], names=Hkeys)

    #     print("writing out the host data")
    #     Hdata.write(self.metadir+"FvdB_hostdata.dat", format="ascii", overwrite=True)

    # def sort_subhalos(self):

    #     self.order_meta = np.full((self.coordinates.shape[0], 4), 0.0)
    #     for sub_id, subhalo_hist in enumerate(self.order): 

    #         unique = jsm_stats.nan_mask(np.unique(subhalo_hist)).astype('int') # dont count the order switch from accretion
    #         kmin, kmax = unique.min(), unique.max()
    #         Nswitch = kmax - kmin # number of times the order swithes
    #         k_removed = kmin - 1 # how many orders removed it is from the reference frame!
    #         self.order_meta[sub_id] = np.array([Nswitch, k_removed, kmin, kmax])
    
    #     self.order_rank = np.lexsort((self.order_meta[:, 1], self.order_meta[:, 0]))           
    #     self.order_meta_r = self.order_meta[self.order_rank, :]
    #     self.order_r = self.order[self.order_rank, :]
    #     self.ParentID_r = self.ParentID[self.order_rank, :]
    #     self.cartesian_mat_r = self.cartesian_mat[self.order_rank, :]
    #     self.kmax = np.nanmax(self.order_meta[:, 3])
    #     self.Nswitch_max = np.nanmax(self.order_meta[:, 0])

# def ana_self(file, return_peak=False):
#     self = np.load(file) #open file and read
#     mass = self["mass"]
#     redshift = self["redshift"]
#     time = self["CosmicTime"]
#     coords = self["coordinates"]
#     orders = self["order"]
#     pID = self["ParentID"]
#     size = self["VirialRadius"]

#     mass = np.delete(mass, 1, axis=0) #there is some weird bug for this index!
#     coords = np.delete(coords, 1, axis=0)
#     orders = np.delete(orders, 1, axis=0)
#     size = np.delete(size, 1, axis=0)

#     mask = mass != -99. # converting to NaN values
#     mass = np.where(mask, mass, np.nan)
#     smask = size != -99. # converting to NaN values
#     size =  np.where(smask, size, np.nan)  
#     orders = np.where(mask, orders, np.nan)
#     Nhalo = mass.shape[0]

#     if return_peak==True:
#         try:
#             peak_index = np.nanargmax(mass, axis=1) #finding the maximum mass
#             peak_mass = mass[np.arange(peak_index.shape[0]), peak_index]
#             peak_red = redshift[peak_index]
#             peak_order = orders[np.arange(peak_index.shape[0]), peak_index]
#             final_mass = mass[:,0] # the final index is the z=0 time step. this will be the minimum mass for all subhalos
#             final_order = orders[:,0]
#             final_coord = coordinates[:,0,:] # this will be the final 6D positons
#             return peak_mass, peak_red, peak_order, final_mass
#         except ValueError:
#             print("bad run, returning empty arrays!")
#         return np.zeros(Nhalo), np.zeros(Nhalo), np.zeros(Nhalo), np.zeros(Nhalo)

#     else:
#         return mass, redshift, time, coords, orders


##### STELLAR HALO STUFF

            #self.merged_subhalos = np.unique(np.where(R_mask + V_mask)[0])

        # life_time = self.CosmicTime[self.disrupt_index[self.merged_subhalos]] - self.CosmicTime[self.acc_index[self.merged_subhalos]]
        # acc_red = self.acc_redshift[self.merged_subhalos]
        # merged_acc_masses = np.log10(self.acc_mass[self.merged_subhalos] / self.acc_mass[self.acc_ParentID[self.merged_subhalos]])
        # total_mass = np.log10(np.sum(self.acc_stellarmass[self.merged_subhalos]))

        # if make_plot == True:
        #     plt.figure(figsize=(8,6))
        #     plt.title(f"{self.merged_subhalos.shape[0]} merged out of {self.Nhalo-1} systems \n log Mstar at accretion: {total_mass:.3f} log Mstar")
        #     plt.scatter(merged_acc_masses, life_time, marker=".", c=acc_red)
        #     plt.ylabel("orbital lifetime (Gyr)")
        #     plt.xlabel("log (m$_{k}$ / M$_{k-1}$) @ z$_{\\rm acc}$")
        #     plt.colorbar(label="z$_{\\rm acc}$")
        #     plt.show()

    # def process_mass_evo(self, subhalo_ind):
        
    #     rmax = np.full(shape=self.CosmicTime.shape, fill_value=np.nan) #empty time arrays to fill
    #     Vmax = np.full(shape=self.CosmicTime.shape, fill_value=np.nan)

    #     profile = self.acc_profiles[subhalo_ind] #grabbing the inital density profile!
    #     R50_by_rmax = self.acc_R50[subhalo_ind]/profile.rmax

    #     fb = np.where(self.orbit_mask[subhalo_ind], self.fb[subhalo_ind], np.nan) #masking out where the orbit is not initalized
    #     R50_fb, stellarmass_fb = ev.g_EPW18(fb, alpha=1.0, lefflmax=R50_by_rmax) #Errani 2018 tidal tracks for stellar evolution!

    #     R50 = self.acc_R50[subhalo_ind]*R50_fb #scale the sizes!
    #     stellarmass = self.acc_stellarmass[subhalo_ind]*stellarmass_fb #scale the masses!

    #     for time_ind, bound_fraction in enumerate(fb): # compute the evolved density profiles using Green transfer function!
    #         if ~np.isnan(bound_fraction): #only for the initialized
    #             profile.update_mass_jsm(bound_fraction)
    #             rmax[time_ind] = profile.rmax
    #             Vmax[time_ind] = profile.Vmax
    #         else:
    #             rmax[time_ind] = profile.rmax
    #             Vmax[time_ind] = profile.Vmax

    #     return rmax, Vmax, R50, stellarmass


    # def measure_energy(self):

    #     self.rmags = np.linalg.norm(self.cartesian[:,:,0:3], axis=2)
    #     self.Vmags = np.linalg.norm(self.cartesian[:,:,3:6], axis=2)
    #     self.KE = 0.5 * (self.Vmags**2)
    #     self.KE = np.where(self.orbit_mask, self.KE, np.nan) #masking out the dead orbits, this might fuck it up because now the proper index is used!!!
    #     self.PE = np.empty(shape=self.rmags.shape)
    #     self.Phi0s = np.empty(shape=self.rmags.shape)

    #     for subhalo_ind in range(self.Nhalo):
    #         for t in range(self.CosmicTime.shape[0]):
    #             parent_potential = self.acc_profiles[self.ParentID[subhalo_ind, t]] #grabbing the parent potential
    #             self.PE[subhalo_ind, t] = parent_potential.Phi(self.rmags[subhalo_ind, t]) #measure Phi with respect to the parent
    #             self.Phi0s[subhalo_ind, t] = parent_potential.Phi0

    #     self.Espec = self.KE + self.PE
    #     self.Erat = self.Espec/self.Phi0s

    #     self.KE_init = self.KE[np.arange(self.acc_index.shape[0]), self.acc_index] # the time step right when the orbit starts!
    #     self.PE_init = self.PE[np.arange(self.acc_index.shape[0]), self.acc_index]
    #     self.E_init = self.KE_init + self.PE_init

    #     self.unbound = np.unique(np.where(self.Erat < 0)[0]) # which subhalos have unbound orbits!

    # def plot_energies(self, kk=1): 

    #     fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8,12))

    #     self.mass_rat = np.log10(self.acc_mass/self.mass[0, self.acc_index])
    #     # Normalize the masses for colormap mapping
    #     norm = colors.Normalize(vmin=self.mass_rat.min(), vmax=self.mass_rat.max())
    #     colormap = cm.viridis_r  # You can choose a different colormap if preferred

    #     ax[0].set_title(f"Orbital Energies of the k={kk} Order Subhalos")
    #     for subhalo_ind in range(self.Nhalo):

    #         if self.order[subhalo_ind, self.acc_index[subhalo_ind]] == kk:  # Only first-order subhalos
    #             if np.isin(subhalo_ind, self.unbound):
    #                 pass
    #             else:
    #                 line_color = colormap(norm(self.mass_rat[subhalo_ind]))

    #                 ax[0].plot(self.CosmicTime, self.KE[subhalo_ind], color=line_color, alpha=0.5)
    #                 ax[1].plot(self.CosmicTime, np.abs(self.PE[subhalo_ind]), color=line_color, alpha=0.5)
    #                 ax[2].plot(self.CosmicTime, np.abs(self.Espec[subhalo_ind]), color=line_color, alpha=0.5)
    #                 ax[3].plot(self.CosmicTime, self.Erat[subhalo_ind], color=line_color, alpha=0.5)

    #     ax[0].set_yscale("log")
    #     ax[1].set_yscale("log")
    #     ax[2].set_yscale("log")

    #     ax[0].set_ylabel("KE (kpc$^2$ / Gyr$^2$)")
    #     ax[1].set_ylabel("|PE| (kpc$^2$ / Gyr$^2$)")
    #     ax[2].set_ylabel("|E$_{\\rm specific}$| (kpc$^2$ / Gyr$^2$)")
    #     ax[3].set_ylabel("E$_{\\rm specific}$/$\Phi_0$")

    #     ax[3].set_xlabel("Cosmic Time (Gyr)")

    #     sm = cm.ScalarMappable(cmap=colormap, norm=norm) 
    #     cbar_ax = fig.add_axes([1.01, 0.15, 0.05, 0.7])
    #     cbar = fig.colorbar(sm, cax=cbar_ax)  # Explicitly associate colorbar with the axis
    #     cbar.set_label("log (m/M) @ $z_{\\rm acc}$")

    #     plt.tight_layout()
    #     plt.show()

  # def compare_coordinates(self, subhalo_ind):

    #     plt.plot(self.CosmicTime, self.rmags_cyl[subhalo_ind], label="Cyldrical")
    #     plt.plot(self.CosmicTime, self.rmags[subhalo_ind], label="Cartesian", ls="-.")
    #     plt.plot(self.CosmicTime, self.rmags_stitched[subhalo_ind], label="Cartesian stitched", ls=":")

    #     plt.axvline(self.CosmicTime[self.proper_acc_index[subhalo_ind]], color="grey", ls="--")
    #     plt.ylabel("|r| (kpc)")
    #     plt.xlabel("time (Gyr)")
    #     plt.legend()
    #     plt.show()

# def potential_energy_integrand(r, profile):
#     """Integrand for the gravitational binding energy."""
#     M_enc = profile.M(r)
#     rho = profile.rho(r)
#     return rho * M_enc * r

# def binding_energy(profile):
#     """Compute the gravitational binding energy of the halo."""
#     G = 4.4985e-06 # gravitational constant [kpc^3 Gyr^-2 Msun^-1]
#     r_max = profile.rh #
#     result, _ = quad(potential_energy_integrand, 0, r_max, args=(profile))
#     return -4 * np.pi * G * result

# E_bind = np.array([binding_energy(profile) for profile in self.host_profiles])

#$E_{bind} = -4\pi G \int_0^{R_{vir}} \rho(r) M(<r) r dr $




        # self.stellar_halo_accreted = np.sum(self.acc_stellarmass[1:] - self.final_stellarmass[1:]) #the amount of stellar mass lost across time due to tides
        # if self.verbose:
        #     print(f"log Msol stripped from surviving satellites: {np.log10(self.stellar_halo_accreted):.4f}")

        # self.stellar_halo_disrtupted = np.sum(self.final_stellarmass[self.disrupted_subhalos]) #the amount of stellar mass accreted from disrupted systems
        # if self.verbose:
        #     print(f"log Msol in disrupted systems: {np.log10(self.stellar_halo_disrtupted):.4f}")
            
        # self.stellar_halo_esc = np.sum(self.final_stellarmass[self.merged_subhalos]*self.fesc) #the stellar mass in satellites about to merge that ends up in the halo
        # if self.verbose:
        #     print(f"log Msol accreted during mergers: {np.log10(self.stellar_halo_esc):.4f}")

        # self.stellar_halo = self.stellar_halo_esc + self.stellar_halo_disrtupted + self.stellar_halo_accreted # the final stellar halo
        # if self.verbose:
        #     print(f"log Msol in the ICL at z=0: {np.log10(self.stellar_halo):.4f}")