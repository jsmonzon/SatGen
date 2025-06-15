import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import Normalize
from astropy.table import Table
import os
import warnings; warnings.simplefilter('ignore')
import jsm_SHMR
import sys
import h5py
import pandas as pd

location = "local"
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
        self.fate_timing()
        self.higher_order_merging()
        self.baryons()
        self.stellarhalo()

    def read_arrays(self):
        self.full = np.load(self.file) #open file and read
        self.tree_index = self.file.split("/")[-1].split("_")[1]

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

        mass_fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.host_zx = np.array([self.redshift[find_nearest1(self.mass[0], self.target_mass*mf)] for mf in mass_fracs])

        self.host_z90 = self.host_zx[4]
        self.host_z50 = self.host_zx[2] #the formation time of the host!``
        self.host_z10 = self.host_zx[0]

        #subhalo properties!
        self.acc_index = np.nanargmax(self.mass, axis=1) #finding the accertion index for each
        self.acc_mass = self.mass[np.arange(self.acc_index.shape[0]), self.acc_index] # max mass
        self.acc_concentration = self.concentration[np.arange(self.acc_index.shape[0]), self.acc_index]
        self.acc_redshift = self.redshift[self.acc_index]
        self.acc_order = self.order[np.arange(self.acc_index.shape[0]), self.acc_index]
        self.acc_ParentID = self.ParentID[np.arange(self.acc_index.shape[0]), self.acc_index]

        #is it like the MW?!?!?!
        self.MW_est = MW_est_criteria(self)

        # Identify halos that undergo an order jump (more than 2 unique subhalo states)
        self.order_jump = np.where([len(set(subhalo)) > 2 for subhalo in self.order])[0]

        # Compute accretion-time profiles using Green potentials
        Green_vec = np.vectorize(profiles.Green)
        self.acc_profiles = Green_vec(
            self.acc_mass,
            self.acc_concentration,
            Delta=cfg.Dvsample[self.acc_index],
            z=self.acc_redshift
        )
        self.acc_Vmax = np.array([p.Vmax for p in self.acc_profiles])
        self.acc_rmax = np.array([p.rmax for p in self.acc_profiles])

        # Compute bound fraction of halo mass
        self.fb_og = self.mass / self.acc_mass[:, None]

        # Create mask: times after accretion and with fb above disruption threshold
        self.time_indices = np.arange(self.CosmicTime.shape[0])
        self.valid_fbs = np.log10(self.fb_og) > -4 #this excludes the fb=-4 index

        self.disrupt_index = np.zeros_like(self.acc_index)
        for subind in range(self.Nhalo):
            if self.valid_fbs[subind, 0]: #true at z=0 then the subhalo never disrupts!!!
                self.disrupt_index[subind] = 0
            else: 
                self.disrupt_index[subind] = np.min(np.where(self.valid_fbs[subind])[0]) - 1
        assert np.all(self.disrupt_index <= self.acc_index), "the disruption index is before the accretion index!"

        self.orbit_mask1 = self.time_indices[None, :] <= self.acc_index[:, None] #anytime before accretion is not valid
        self.orbit_mask2 = self.time_indices[None, :] >= self.disrupt_index[:, None] #anytime after disruption is not valid

        self.orbit_mask = self.orbit_mask1 & self.orbit_mask2
        self.fb = np.where(self.orbit_mask, self.fb_og, np.nan)
        self.initialized = np.copy(self.orbit_mask)

    def convert_to_cartesian(self):

        if self.verbose:
            print("converting cyldrical coordinates to cartesian!")

        self.coordinates[~self.initialized] = np.tile([0, 0, 0, 0, 0, 0], (np.count_nonzero(~self.initialized),1)) # setting coodinates to zero for uninitialized orbits or if the subhalo is disrupted

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
            if self.initialized is not None: # this masks out the orbits of the subhalos before they are accreted onto the main progenitor
                self.initialized[to_fix] = self.initialized[to_fix] & self.initialized[self.ParentID[to_fix], redshift]

            subhalo_ind = np.where(self.acc_order == kk)
            for ind in subhalo_ind: #just so we know when the subhalo falls into the main progenitor
                self.proper_acc_index[ind] = self.proper_acc_index[self.acc_ParentID[ind]]

        # masking the coordinates with initialized mask - use this for any movies!
        self.masked_cartesian_stitched = np.where(np.repeat(np.expand_dims(self.initialized, axis=-1), 6, axis=-1), self.cartesian_stitched, np.nan)

        # doing the same for the non stitched array
        self.masked_cartesian = np.where(np.repeat(np.expand_dims(self.initialized, axis=-1), 6, axis=-1), self.cartesian, np.nan)

        #to decide which subhalos merge!
        self.rmags = np.linalg.norm(self.masked_cartesian[:,:,0:3], axis=2)
        self.Vmags = np.linalg.norm(self.masked_cartesian[:,:,3:6], axis=2)

        #to write out for the surving subhalos!
        self.rmags_stitched = np.linalg.norm(self.masked_cartesian_stitched[:,:,0:3], axis=2)
        self.Vmags_stitched = np.linalg.norm(self.masked_cartesian_stitched[:,:,3:6], axis=2)

    def tides(self):

        if self.verbose:
            print("evolving subhalo profiles based on bound fractions")

        self.rmax = np.full(shape=self.mass.shape, fill_value=np.nan) #empty arrays
        self.Vmax = np.full(shape=self.mass.shape, fill_value=np.nan)

        for subhalo_ind in range(self.Nhalo): #each tidal track is based on fb = m(t)/m(t_acc)
            rmax, Vmax = self.FUNC_halo_mass_evo(subhalo_ind)
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

        self.R_mask = self.rmax_kscaled < self.merger_crit
        self.V_mask = self.Vmax_kscaled < self.merger_crit

        x_mer, y_mer = np.where(self.R_mask + self.V_mask)
        self.merger_index = np.zeros(self.Nhalo, dtype=int)
        np.maximum.at(self.merger_index, x_mer, y_mer)
        self.merger_index[0] = 0  # Ensure host is never disrupted

    def fate_timing(self):
        self.final_index = np.zeros(shape=self.Nhalo, dtype=int)

        # Subhalos with both merger and disruption fates
        self.both = (self.merger_index != 0) & (self.disrupt_index != 0)
        self.merge_first = (self.merger_index > self.disrupt_index) & self.both
        self.disrupt_first = (self.disrupt_index > self.merger_index) & self.both
        self.same_time = (self.merger_index == self.disrupt_index) & self.both

        # Assign final_index based on priority rules
        if np.any(self.merge_first):
            self.final_index[self.merge_first] = self.merger_index[self.merge_first]
        if np.any(self.disrupt_first):
            self.final_index[self.disrupt_first] = self.disrupt_index[self.disrupt_first]
        if np.any(self.same_time):
            self.final_index[self.same_time] = self.merger_index[self.same_time]  # tie-breaker

        # Handle one-sided cases
        self.only_merged = (self.merger_index != 0) & (self.disrupt_index == 0)
        self.only_disrupted = (self.disrupt_index != 0) & (self.merger_index == 0)

        if np.any(self.only_merged):
            self.final_index[self.only_merged] = self.merger_index[self.only_merged]
        if np.any(self.only_disrupted):
            self.final_index[self.only_disrupted] = self.disrupt_index[self.only_disrupted]

        # Now explicitly assign fate categories using known indices
        self.surviving_subhalos = np.where((self.merger_index == 0) & (self.disrupt_index == 0))[0]
        self.surviving_subhalos = self.surviving_subhalos[1:]  # mask out the host!

        # Use the original condition masks to ensure clean partitioning
        self.merged_subhalos = np.concatenate([
            np.where(self.only_merged)[0],
            np.where(self.merge_first)[0],
            np.where(self.same_time)[0],
        ])
        self.disrupted_subhalos = np.concatenate([
            np.where(self.only_disrupted)[0],
            np.where(self.disrupt_first)[0],
        ])

        # Final counts
        self.N_disrupted = self.disrupted_subhalos.shape[0]
        self.N_merged = self.merged_subhalos.shape[0]
        self.N_surviving = self.surviving_subhalos.shape[0]

        # Final assertion check
        assert self.N_disrupted + self.N_merged + self.N_surviving == self.Nhalo - 1, \
            f"a subhalo was lost to the winds of time! (counted: {self.N_disrupted + self.N_merged + self.N_surviving}, expected: {self.Nhalo - 1})"

    def higher_order_merging(self):
    
        self.account_for_higher_order_merging = False
        self.merger_hierarchy = [] # this just to check if a higher order subhalo merges onto a parent that has already been merged
        # this should not alter the number of subhalos in each fate bin

        for ii, sub_ind in enumerate(self.merged_subhalos):
            time_ind = self.merger_index[sub_ind]
            if time_ind != 0:
                # Find all associated subhalos for this subhalo ID
                substructure = find_associated_subhalos(self, sub_ind, time_ind)
                if substructure != None:
                    self.merger_hierarchy.append([sub_ind] + substructure)
                    self.account_for_higher_order_merging = True
                else:
                    self.merger_hierarchy.append([sub_ind])
                    

        if self.account_for_higher_order_merging:
            self.merged_with_parent = []
            self.merged_with_child = []

            for chain in self.merger_hierarchy:
                merger_index = self.final_index[chain[0]]

                if len(chain) > 1: # if there is higher order subhalos associated with the subhalo
                    for subind in chain[1:]: #for subhalo in chain
                        if np.isin(subind, self.merged_subhalos): # if the higher order subhalo is found to merge with its parent
                            if self.final_index[subind] <  merger_index: # if the merging event happens after its parent has already merged
                                if self.ParentID[subind, merger_index] == chain[0]: #if it merges with the ghost

                                    self.merged_with_parent.append(subind)
                                    self.merged_with_child.append(chain[0])

                                    self.final_index[subind] = merger_index #update the final index
                                    self.fb[subind, :merger_index] = np.nan
                                    self.initialized[sub_ind, :merger_index] = False

                self.fb[chain[0], :merger_index] = np.nan #reset values to reflect a merged subhalo
                self.initialized[chain[0], :merger_index] = False

        self.merged_parents = self.ParentID[self.merged_subhalos, self.final_index[self.merged_subhalos]] #grabbing the parents they merge into

        for ii in range(len(self.merged_with_parent)): 
            child_i = np.where(self.merged_subhalos == self.merged_with_parent[ii])[0][0]
            parent_i = np.where(self.merged_subhalos == self.merged_with_child[ii])[0][0]
            self.merged_parents[child_i] = self.merged_parents[parent_i]

    def baryons(self):

        if self.verbose:
            print("using empirical relations to account for baryons")

        if hasattr(self, "ALPHA"):
            self.acc_stellarmass = 10**gh.lgMs_B18(lgMv=np.log10(self.acc_mass), z=self.acc_redshift, ALPHA=self.ALPHA) # the SHMR with the updated slopes!!
        else:
            self.acc_stellarmass = 10**gh.lgMs_B18(lgMv=np.log10(self.acc_mass), z=self.acc_redshift) # the SHMR with the updated slopes!!

        if self.scatter==True:
            self.acc_stellarmass = 10**(gh.dex_sampler(np.log10(self.acc_stellarmass)))

        if np.any(self.acc_stellarmass <= 0): #should only happen for the most extreme SHMRs
            self.negative_masses = np.where(self.acc_stellarmass <= 0)[0]
            self.acc_stellarmass[self.negative_masses] = 100 #100 solar masses hard lower limit

        self.acc_R50 = 10**gh.Reff_A24(lgMs=np.log10(self.acc_stellarmass)) # the size mass relation from SAGA
        self.FeH = gh.MZR(self.acc_stellarmass) # the mass metalicity relation!

        if self.scatter==True:
            self.acc_R50 = 10**(gh.dex_sampler(np.log10(self.acc_R50)))
            self.FeH = gh.dex_sampler(self.FeH, dex=0.17)
        
        if hasattr(self, "size_multi"): # play with the stellar tidal track!!
            self.acc_R50 = self.size_multi * self.acc_R50
        
        self.R50 = np.full(shape=self.mass.shape, fill_value=np.nan) # empty arrays
        self.stellarmass = np.full(shape=self.mass.shape, fill_value=np.nan)

        for subhalo_ind in range(self.Nhalo): #each tidal track is based on fb = m(t)/m(t_acc)
            R50, stellarmass = self.FUNC_stellar_mass_evo(subhalo_ind)
            self.R50[subhalo_ind] = R50
            self.stellarmass[subhalo_ind] = stellarmass

        self.insitu = self.FUNC_in_situ_SFR()
        self.stellarmass[0] = self.insitu #the SFR from the UM model
        self.acc_stellarmass[0] = self.stellarmass[0,0] #updating so its not based on the SHMR
        self.target_stellarmass = self.stellarmass[0,0]

        self.total_stellarmass_acc = np.sum(self.acc_stellarmass[1:])

        self.final_mass = np.array([self.mass[subhalo_ind, self.final_index[subhalo_ind]] 
                                    for subhalo_ind in range(self.Nhalo)])

        # Step 2: Add stellar mass from merged subhalos to their final parent
        self.extra_stellarmass = self.stellarmass[self.merged_subhalos, self.final_index[self.merged_subhalos]] * (1 - self.fesc)
        for ii, subhalo in enumerate(self.merged_subhalos):
            time_ind = self.final_index[subhalo]
            self.stellarmass[self.merged_parents[ii], :time_ind] += self.extra_stellarmass[ii]

        # Step 3: Now compute final_stellarmass using updated self.stellarmass
        self.final_stellarmass = np.array([self.stellarmass[subhalo_ind, self.final_index[subhalo_ind]] 
                                        for subhalo_ind in range(self.Nhalo)])

        self.fb_stellar = self.stellarmass / self.acc_stellarmass[:, None]

    def stellarhalo(self):

        if self.verbose:
            print("counting up the stellar mass in the halo")

        self.diff_stellarmass = np.zeros_like(self.stellarmass)

        for subhalo_ind in range(1, self.Nhalo): 
            for time_ind in range(len(self.CosmicTime) - 1):  # avoid overflow at time_ind + 1
                m0 = self.stellarmass[subhalo_ind, time_ind]
                m1 = self.stellarmass[subhalo_ind, time_ind + 1]

                # Skip if either value is NaN (i.e., subhalo disrupted/merged)
                if np.isnan(m0) or np.isnan(m1):
                    continue

                dM = m1 - m0
                if dM > 0:
                    self.diff_stellarmass[subhalo_ind, time_ind] = dM

            # Final mass contributions for disrupted or merged subhalos
            final_t = self.final_index[subhalo_ind]

            if subhalo_ind in self.disrupted_subhalos:
                self.diff_stellarmass[subhalo_ind, final_t] += self.final_stellarmass[subhalo_ind]

            elif subhalo_ind in self.merged_subhalos:
                self.diff_stellarmass[subhalo_ind, final_t] += self.final_stellarmass[subhalo_ind] * (self.fesc)

        self.diff_stellarmass_MP = np.zeros_like(self.diff_stellarmass)

        for sub_ID in range(1, self.Nhalo): #only counting the stellar mass that was lost inside the main progenitor
            acc_ind = self.proper_acc_index[sub_ID]
            self.diff_stellarmass_MP[sub_ID, :acc_ind] += self.diff_stellarmass[sub_ID, :acc_ind] # after the accretion should be the same
            self.diff_stellarmass_MP[sub_ID, acc_ind] += np.nansum(self.diff_stellarmass[sub_ID, acc_ind:]) #at the accretion give a boost!!

        # ICL contributions
        self.contributed = np.nansum(self.diff_stellarmass_MP, axis=1)  #should be MP!!!
        self.ICL_deltaMAH = np.nansum(self.diff_stellarmass_MP, axis=0)
        self.ICL_MAH = np.cumsum(self.ICL_deltaMAH[::-1])[::-1]
        self.total_ICL = self.ICL_MAH[0]

        # Breakdown by satellite type
        self.ICL_fmerged = np.sum(self.contributed[self.merged_subhalos])
        self.ICL_fdisrupted = np.sum(self.contributed[self.disrupted_subhalos])
        self.ICL_fsurviving = np.sum(self.contributed[self.surviving_subhalos])

        assert np.abs(((self.ICL_fmerged + self.ICL_fdisrupted + self.ICL_fsurviving) - self.total_ICL) / self.total_ICL) < 1e-12, "mass loss in the IHL exceeds criteria"

        #surviving satellites!
        self.stellarmass_in_satellites = np.sum(self.final_stellarmass[self.surviving_subhalos])

        #accretion onto central!
        if np.any(self.merged_parents == 0):
            self.central_accreted = np.sum(self.extra_stellarmass[self.merged_parents == 0]) #the stellar mass accreted by the central galaxy
            self.mostmassive_accreted = np.max(self.extra_stellarmass[self.merged_parents == 0]) #the most massive satellite accreted by the central galaxy
            self.single_merger_frac = self.mostmassive_accreted/self.central_accreted
        else:
            self.central_accreted = 0
            self.mostmassive_accreted = 0
            self.single_merger_frac = 0

        #now the final tally
        self.mass_loss = self.total_stellarmass_acc - (self.central_accreted + self.stellarmass_in_satellites + self.total_ICL)
        #assert np.abs(((self.central_accreted + self.stellarmass_in_satellites + self.total_ICL) - self.total_stellarmass_acc) / self.total_stellarmass_acc) < 1e-1, "mass loss in the Mtot exceeds criteria"

        if self.verbose:
            print("-----------------------------")
            print("=== SUBHALO POPULATIONS ===")

            print(f"Total satellites: {self.Nhalo-1}")
            print(f"Satellites disrtuped: {self.N_disrupted}")
            print(f"Satellites merged with direct parents: {self.N_merged}")
            print(f"Satellites survived to z=0: {self.N_surviving}")

            print("=== STELLAR MASS BUDGET ===")
            print(f"Total Accreted Stellar Mass     : {self.total_stellarmass_acc:.3e}")
            print(f"  -> Central Accreted           : {self.central_accreted:.3e}")
            print(f"  -> In Surviving Satellites    : {self.stellarmass_in_satellites:.3e}")
            print(f"  -> In ICL                     : {self.total_ICL:.3e}")
            print(f"  -> Accounted (sum)            : {(self.central_accreted + self.stellarmass_in_satellites + self.total_ICL):.3e}")
            print(f"  -> Missing                    : {(self.total_stellarmass_acc - (self.central_accreted + self.stellarmass_in_satellites + self.total_ICL)):.3e}")

    
    def create_survsat_dict(self):

        # Get the index of the minimum non-NaN value for each row
        self.rmin_mask = np.nanargmin(self.rmags_stitched[self.surviving_subhalos], axis=1)

        # Use the obtained indices to extract the corresponding minimum values and their order
        self.rmin = self.rmags_stitched[self.surviving_subhalos, self.rmin_mask]
        self.rmin_order = self.order[self.surviving_subhalos, self.rmin_mask]

        #the present day positions and velocity
        self.surviving_rmag = self.rmags_stitched[self.surviving_subhalos, 0]
        self.surviving_Vmag = self.Vmags_stitched[self.surviving_subhalos, 0]

        # now the orders!
        self.kmax = np.nanmax(self.order[self.surviving_subhalos], axis=1)
        self.kfinal = self.order[self.surviving_subhalos, 0]

        #the masses
        self.surviving_final_mass = self.final_mass[self.surviving_subhalos]
        self.surviving_acc_mass = self.acc_mass[self.surviving_subhalos]
        self.surviving_final_stellarmass = self.final_stellarmass[self.surviving_subhalos]
        self.surviving_acc_stellarmass = self.acc_stellarmass[self.surviving_subhalos]
        self.surviving_FeH = self.FeH[self.surviving_subhalos]

        dictionary = {"tree_index": self.tree_index, #just to give us the file index
                    "Nhalo": self.Nhalo - 1, #total number of subhalos accreted
                    "MW_est": self.MW_est, #[c, GSE, LMC] all three would be [1,1,1]
                    "MAH": self.mass[0], # the host halo mass across time! (N time indices)
                    "MAH_stellar": self.stellarmass[0], # the central stellar mass across time!
                    "MAH_ICL": self.ICL_MAH, # the build of ICL
                    "target_mass": self.mass[0,0], # the target halo mass (single values from here!)
                    "target_stellarmass": self.stellarmass[0,0], #the target stellar mass including Mstar acc
                    "host_z50": self.host_z50,  #"host_z10": self.host_z10, "host_z90": self.host_z90, 
                    "Mstar_tot": self.total_stellarmass_acc, #total ever accreted (sum from the SHMR sample)
                    "Mstar_lost": self.mass_loss, #this should be less than 0.01 percent of Mstar tot
                    "Mstar_ICL": self.total_ICL, #ICL 
                    "Mstar_sat": self.stellarmass_in_satellites, #total mass in surviving satellites
                    "Mstar_acc": self.central_accreted, # the stellar mass that is accreted onto the central
                    "Mstar_acc_max": self.mostmassive_accreted, # the most massive satellite accreted
                    "N_disrupted": self.N_disrupted, # Number of disrupted halos
                    "N_merged": self.N_merged, # number that merge onto the central
                    "N_surviving": self.N_surviving, # the number of surviving halos
                    "fICL_disrupted": self.ICL_fdisrupted, # the amount of stellar mass contirubted to ICL from disrupted halos
                    "fICL_merged": self.ICL_fmerged, # " merged systems
                    "fICL_surviving": self.ICL_fsurviving, # " from surviving systems
                    "mass": self.surviving_final_mass,  # final halo masses (N halos from here)
                    "acc_mass": self.surviving_acc_mass,  # halo mass @ accretion halo masses
                    "stellarmass":  self.surviving_final_stellarmass,  # final stellar mass
                    "acc_stellarmass": self.surviving_acc_stellarmass, # stellar mass @ accretion halo mass
                    "z_acc": self.surviving_zacc, # proper accretion redshift onto the main progenitor
                    "FeH": self.surviving_FeH, # the metallicity of satellites
                    "Rmag": self.surviving_rmag, #position with respect to the main progenitor
                    "Vmag": self.surviving_Vmag, #velocity "
                    "Rperi": self.rmin, #rperi with respect to the main progenitor
                    "k_Rperi": self.rmin_order, # the order associated with rperi
                    "k_max": self.kmax, #max order across history
                    "k_final": self.kfinal} # final order
            
        return dictionary

### ---------------------------------------------------------------
### ---------------------------------------------------------------
### ---------------------------------------------------------------
    

    def FUNC_halo_mass_evo(self, subhalo_ind):

        #based on Green et al 2019 fitting code using the transfer function
        Vmax = self.acc_Vmax[subhalo_ind] * ev.GvdB_19(self.fb[subhalo_ind], self.acc_concentration[subhalo_ind], track_type="vel") #Green et al 2019 transfer function tidal track
        rmax = self.acc_rmax[subhalo_ind] * ev.GvdB_19(self.fb[subhalo_ind], self.acc_concentration[subhalo_ind], track_type="rad") #Green et al 2019 transfer function tidal track

        return rmax, Vmax
    
    def FUNC_stellar_mass_evo(self, subhalo_ind):

        #based on Errani et al 2021 fitting code

        R50_by_rmax = self.acc_R50[subhalo_ind]/self.acc_rmax[subhalo_ind]
        R50_fb, stellarmass_fb = ev.g_EPW18(self.fb[subhalo_ind], alpha=1.0, lefflmax=R50_by_rmax) #Errani 2018 tidal tracks for stellar evolution!
        R50 = self.acc_R50[subhalo_ind]*R50_fb #scale the sizes!
        stellarmass = self.acc_stellarmass[subhalo_ind]*stellarmass_fb #scale the masses!
        
        return R50, stellarmass
        
    def FUNC_in_situ_SFR(self):

        self.zmask = ~np.isnan(self.VirialRadius[0]) 
        self.Rvir = self.VirialRadius[0][self.zmask][::-1] # need to flip so the integration works!
        self.Mhalo = self.mass[0][self.zmask][::-1]
        self.Vmaxhalo = self.host_Vmax[self.zmask][::-1]
        self.zhalo = self.redshift[self.zmask][::-1]
        self.thalo = self.CosmicTime[self.zmask][::-1]

        self.t_dyn = gh.dynamical_time(self.Rvir*u.kpc, self.Mhalo*u.solMass).to(u.Gyr).value
        self.SFR  = gh.SFR_B19(self.Vmaxhalo, self.zhalo)
        self.Mstar, self.f_lost = gh.integrate_SFH(self.SFR, self.thalo)

        padding = np.zeros(shape=(self.CosmicTime.shape[0] - self.Mstar.shape[0],))  # Create the padding array
        return np.concatenate((padding, self.Mstar))[::-1]
    

### ---------------------------------------------------------------
### ---------------------------------------------------------------
### ---------------------------------------------------------------

    def plot_insitu_SFR(self):

        fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True, constrained_layout=True)

        axes[0].plot(1+self.zhalo, self.Vmaxhalo, color="k")
        axes[0].set_ylabel("Vmax (km/s)") 

        axes[1].plot(1+self.zhalo, self.Mhalo, color="k", label="$M_{\\rm h}$")
        axes[1].plot(1+self.zhalo[:-1], self.Mstar, color="r", label="$M_{*}$")
        axes[1].plot(1+self.redshift, self.ICL_MAH, color="C0", label="ICL")
        axes[1].axhline(self.target_mass, color="grey", ls="-.", alpha=0.5)
        axes[1].axhline(self.total_ICL, color="C0", ls="-.", alpha=0.2)
        axes[1].axhline(self.target_stellarmass, color="r", ls="-.", alpha=0.2)

        axes[1].legend(loc=4, framealpha=1)
        axes[1].set_ylabel("$M\ (\mathrm{M}_{\odot})$")
        axes[1].set_yscale("log")
        axes[1].set_xscale("log")
        axes[1].set_ylim(1e4)

        axes[2].plot(1+self.zhalo, self.SFR, color="r", label="SFR")
        axes[2].plot(1+self.redshift, self.ICL_deltaMAH, label="accretion events", color="C0")
        axes[2].legend(loc=4, framealpha=1)
        axes[2].set_yscale("log")
        axes[2].set_xscale("log")
        axes[2].set_ylabel("$\mathrm{d} M / \mathrm{d} t\ (\mathrm{M}_{\odot} / \mathrm{Gyr})$")
        axes[2].set_xlim(13, 1)
        axes[2].set_xlabel("1 + z")
        plt.tight_layout()
        plt.show()

    
    def plot_ICLsplit(self):

        # Create the figure and axes
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(7, 7))

        # Setup shared color normalization
        all_colors = np.log10(1 + self.redshift[self.proper_acc_index])
        norm = Normalize(vmin=np.min(all_colors), vmax=np.max(all_colors))
        cmap = plt.get_cmap("viridis")

        # Plot each scatter and keep one for the colorbar
        sc0 = axes[0, 0].scatter(
            self.acc_stellarmass[1:], 
            self.final_stellarmass[1:] / self.acc_stellarmass[1:], 
            c=np.log10(1 + self.redshift[self.proper_acc_index][1:]), 
            cmap=cmap, norm=norm, marker="."
        )

        axes[0,0].text(x=0.65, y=0.1, s="log M$_{\\rm ICL}=$"+f"{np.log10(self.total_ICL):.2f}", transform=axes[0,0].transAxes, bbox=dict(facecolor='white', alpha=1, edgecolor="violet"))


        # Surviving subhalos
        colorz1 = np.log10(1 + self.redshift[self.proper_acc_index][self.surviving_subhalos])
        axes[0, 1].scatter(
            self.acc_stellarmass[self.surviving_subhalos], 
            self.final_stellarmass[self.surviving_subhalos] / self.acc_stellarmass[self.surviving_subhalos], 
            c=colorz1, cmap=cmap, norm=norm, marker="*"
        )

        axes[0,1].text(x=0.75, y=0.1, s="surviving \nsystems: \nf$_{\\rm ICL}$ = "+f"{self.ICL_fsurviving/self.total_ICL:.2f}", transform=axes[0,1].transAxes, bbox=dict(facecolor='white', alpha=1, edgecolor="violet"))

        # Disrupted subhalos
        colorz2 = np.log10(1 + self.redshift[self.proper_acc_index][self.disrupted_subhalos])
        axes[1, 0].scatter(
            self.acc_stellarmass[self.disrupted_subhalos], 
            self.final_stellarmass[self.disrupted_subhalos] / self.acc_stellarmass[self.disrupted_subhalos], 
            c=colorz2, cmap=cmap, norm=norm, marker="x", s=20
        )

        axes[1,0].text(x=0.75, y=0.1, s="disrupted \nsystems: \nf$_{\\rm ICL}$ = "+f"{self.ICL_fdisrupted/self.total_ICL:.2f}", transform=axes[1,0].transAxes, bbox=dict(facecolor='white', alpha=1, edgecolor="violet"))

        # Merged subhalos
        colorz3 = np.log10(1 + self.redshift[self.proper_acc_index][self.merged_subhalos])
        axes[1, 1].scatter(
            self.acc_stellarmass[self.merged_subhalos], 
            self.final_stellarmass[self.merged_subhalos] / self.acc_stellarmass[self.merged_subhalos], 
            c=colorz3, cmap=cmap, norm=norm, marker="d", s=20
        )

        axes[1,1].text(x=0.75, y=0.1, s="merged \nsystems: \nf$_{\\rm ICL}$ = "+f"{self.ICL_fmerged/self.total_ICL:.2f}", transform=axes[1,1].transAxes, bbox=dict(facecolor='white', alpha=1, edgecolor="violet"))

        # (Other subplots here as before...)
        plt.tight_layout()
        # Adjust space for colorbar above the plot
        plt.subplots_adjust(top=0.85)  # Leaves space at the top of the figure

        # Manually add a new axes for the colorbar (relative to figure size)
        cbar_ax = fig.add_axes([0.17, 0.93, 0.7, 0.02])  # [left, bottom, width, height]

        # Create the horizontal colorbar
        cbar = fig.colorbar(sc0, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("log(1 + z$_{\\rm acc}$)")

        # Axis settings
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_ylim(0.0002, 1.2)  # changed from -9000 to 1e-5 to work with log scale
        axes[0, 0].set_xlim(50, 1e9)

        axes[1, 1].set_xlabel("m$_{\\rm *, 0}$ [M$_{\odot}$]")
        axes[1, 0].set_xlabel("m$_{\\rm *, 0}$ [M$_{\odot}$]")
        axes[1, 0].set_ylabel("f$_{\\rm b}$") 
        axes[0, 0].set_ylabel("f$_{\\rm b}$")

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

def MW_est_criteria(tree):

    # from Nadler et al. 2024
    lower_GSE_index = np.argmin(np.abs(tree.CosmicTime - (13.8-6.5))) #time constraints
    upper_GSE_index = np.argmin(np.abs(tree.CosmicTime - (13.8-11.5)))

    lower_LMC_index = 0
    upper_LMC_index = np.argmin(np.abs(tree.CosmicTime - (13.8-2)))

    mass_ratio_mat = tree.mass / tree.mass[0] #mass ratio!

    potential_GSEs = np.where((lower_GSE_index <= tree.acc_index) & (tree.acc_index <= upper_GSE_index))[0] #everything that was accreted in that window
    GSE_analogs = potential_GSEs[mass_ratio_mat[potential_GSEs, tree.acc_index[potential_GSEs]] >= 1/5]

    potential_LMCs = np.where((lower_LMC_index < tree.acc_index) & (tree.acc_index <= upper_LMC_index))[0]
    LMC_analogs = potential_LMCs[mass_ratio_mat[potential_LMCs, tree.acc_index[potential_LMCs]] >= 1/10]

    host_c = 1 if (7 < tree.concentration[0,0] < 16) else 0 #host concentration!

    GSE = 0
    if GSE_analogs.shape[0] > 0:
        if np.any(tree.acc_order[GSE_analogs] == 1): #first order subhalos!
            GSE = 1

    LMC = 0
    if LMC_analogs.shape[0] > 0:
        if np.any(tree.acc_order[LMC_analogs] == 1):
            LMC = 1

    return np.array([host_c, GSE, LMC])

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

def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx

def make_matrix(dataframe, key): ### should fix this to be 1000 if I am going to compare!

    # Create NxM matrix, padding with NaN
    matrix = np.full((len(dataframe), max(dataframe[key].apply(len))), np.nan)  # Initialize with NaNs

    # Fill the matrix with actual values
    for i, row in enumerate(dataframe[key]):
        matrix[i, :len(row)] = row  # Assign values

    return matrix

def load_sample(filename):
    data = {}
    with h5py.File(filename, "r") as f:   
        for sim_name in f.keys():
            row = {}
            for attr_name in f[sim_name].keys():
                dset = f[sim_name][attr_name]
                if dset.shape == ():  # scalar dataset
                    row[attr_name] = dset[()]  # or dset[()].item()
                else:
                    row[attr_name] = dset[:]
            data[sim_name] = row

    dfh5 = pd.DataFrame.from_dict(data, orient='index')
    return dfh5



        # # Loop over time steps, skipping the last time index
        # for t in range(self.stellarmass.shape[1] - 1):
        #     # Difference from t to t+1
        #     delta_mass = self.stellarmass[:, t+1] - self.stellarmass[:, t]
            
        #     # Host only grows in mass
        #     delta_mass[0] = max(delta_mass[0], 0.0)
            
        #     # # Optionally ignore negative mass changes in subhalos (i.e., stripping/loss to ICL)
        #     # delta_mass[1:] = np.where(delta_mass[1:] > 0, delta_mass[1:], 0.0)
            
        #     self.diff_stellarmass[:, t] = delta_mass

        # # Add final stellar mass at disruption/merger to diff_stellarmass
        # self.diff_stellarmass[self.merged_subhalos, self.final_index[self.merged_subhalos]] += self.final_stellarmass[self.merged_subhalos] * self.fesc
        # self.diff_stellarmass[self.disrupted_subhalos, self.final_index[self.disrupted_subhalos]] += self.final_stellarmass[self.disrupted_subhalos]