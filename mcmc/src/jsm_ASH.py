import numpy as np
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

# NOTE: parentdir is derived from config_path (already resolved above,
# relative to this file's own location) instead of hardcoding a path per
# "location" string -- this way it resolves correctly on any machine this
# repo is cloned to, with no new location branch needed. (jsm2026-08-29)
repo_root = os.path.dirname(config_path)
parentdir = os.path.join(repo_root, "src") + "/"

sys.path.insert(0, parentdir)
import profiles as profiles
import config as cfg
import cosmo as co
import galhalo as gh
import evolve as ev
import ludlow

import astropy.units as u
import astropy.constants as const
import astropy.coordinates as crd
from treelib import Node, Tree
import networkx as nx
import jsm_ancillary as ancil

#############################################################################################
## jsm_ASH.py -- "Accreted Stellar Halo" reader                                            ##
##                                                                                          ##
## This resurrects the stellar-halo / ICL bookkeeping that lived in                        ##
## mcmc/src/jsm_stellarhalo.py::Tree_Reader as of commit 1e407f3                           ##
## ("pushign to save the cumsum for riley and to make the radial profile").                ##
## At some point after that commit, Tree_Reader in jsm_stellarhalo.py was repurposed for    ##
## concentration/SHMF measurements and the tides -> mergers -> fate_timing ->               ##
## abundance_counts -> satellites -> disk -> stellarhalo pipeline was removed entirely --   ##
## it no longer exists anywhere in the current codebase (the helper functions it calls in   ##
## jsm_ancillary.py, e.g. fb_surv_frac, FUNC_halo_mass_evo, tree_walker, etc. are all still ##
## present and unchanged, just orphaned).                                                   ##
##                                                                                          ##
## read_arrays() and convert_to_cartesian() below are copied from the CURRENT               ##
## jsm_stellarhalo.Tree_Reader instead of the 1e407f3 version, because the current version   ##
## fixes a real bug (orbit_mask1 used acc_index instead of proper_acc_index for order>=2    ##
## subhalos, so higher-order subhalos could be marked "on orbit" one step before their      ##
## branch actually existed) and matches the tree_index parsing your current output files    ##
## use. Every method from tides() onward is a straight port of the 1e407f3 logic, including ##
## the                                                                                       ##
##     self.frac_fb_DM, self.frac_fb_stellar = ancil.fb_surv_frac(self)                     ##
## call at the end of satellites() -- that line runs in this file exactly as it did at      ##
## 1e407f3 (it is currently commented out in jsm_stellarhalo.py's satellites()... except     ##
## satellites() itself no longer exists there at all).                                       ##
#############################################################################################


class Tree_Reader:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.read_arrays()
        self.convert_to_cartesian()
        self.tides()
        self.mergers()
        self.fate_timing()
        self.abundance_counts()
        self.satellites()
        # self.disk()          # NOTE: not auto-run at 1e407f3 either. Call these two
        # self.stellarhalo()   # manually after init if you want the ICL/ASH tree-walk.

    # -----------------------------------------------------------------
    # data loading -- current (bug-fixed) version, not the 1e407f3 version
    # -----------------------------------------------------------------

    def read_arrays(self):
        self.full = np.load(self.file) #open file and read
        # NOTE (jsm 2026-08-29): the old split("_")[2] assumed a 3-part
        # "tree_<hostmass>_<itree>.npz" filename (masspec/bolshoi_rep) and
        # raised IndexError on 2-part conventions used elsewhere in this
        # project -- "tree_<itree>.npz" (MW_analog/crosshost) and
        # "tree_<lgMhost>.npz" (MassSpec/data/local_trees/evolved_trees,
        # one tree per host-mass file). Taking the LAST underscore- and
        # extension-stripped segment instead gives the itree for the first
        # two conventions (same as before) and the host-mass tag for the
        # third, and works regardless of how many "_"-separated parts
        # precede it.
        stem = os.path.splitext(self.file.split("/")[-1])[0]
        self.tree_index = stem.split("_")[-1] # check to see which index is unique in the name

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
        # NOTE (jsm 2026-08-29): Delta used to be cfg.Dvsample, which is
        # DeltaBN(z) precomputed on config.py's OWN time grid (cfg.zsample)
        # and only correct here if self.redshift happens to equal that
        # exact grid index-for-index. That assumption silently breaks for
        # any tree generated under an earlier config.py state (cosmology
        # params in config.py have been swapped more than once -- see e.g.
        # commit c1b78cf) even when the tree's own z0/zmax range looks the
        # same, and previously failed loudly at least (shape mismatch)
        # rather than misaligning silently -- computing Delta directly
        # from this tree's own self.redshift removes the dependency on
        # cfg.zsample lining up at all.
        self.host_profiles = NFW_vectorized(self.mass[0, :], self.concentration[0,:], Delta=co.DeltaBN(self.redshift, cfg.Om, cfg.OL), z=self.redshift)
        self.host_rmax = np.array([profile.rmax for profile in self.host_profiles])
        self.host_Vmax = np.array([profile.Vmax for profile in self.host_profiles])

        mass_fracs = [0.1, 0.5, 0.9]
        self.host_zx = np.array([self.redshift[ancil.find_nearest1(self.mass[0], self.target_mass*mf)] for mf in mass_fracs])

        self.host_z90 = self.host_zx[2]
        self.host_z50 = self.host_zx[1] #the formation time of the host!``
        self.host_z10 = self.host_zx[0]

        #and finally the ludlow model, use the zhao model to guess c
        # NOTE: Delta must match the overdensity that actually defines this
        # tree's own Rvir/M0 -- i.e. the Bryan & Norman (1998) virial
        # overdensity at z=0, the SAME Delta used to build self.host_profiles
        # just above. Passing a fixed Delta=200 here silently mismatches that
        # convention and biases the recovered ludlow_c/ludlow_z2. Computed
        # from self.redshift[0] (always 0. by construction) rather than
        # cfg.Dvsample[0] for the same reason as host_profiles above.
        self.ludlow_c, self.ludlow_z2, self.ludlow_CMH = ludlow.concentration_Ludlow2016(self.mass, self.order, self.ParentID,
                                                                                        z0=0., Delta=co.DeltaBN(self.redshift[0], cfg.Om, cfg.OL),c0=self.concentration[0,0])

        #subhalo properties!
        self.acc_index = np.nanargmax(self.mass, axis=1) #finding the accertion index for each
        self.acc_mass = self.mass[np.arange(self.acc_index.shape[0]), self.acc_index] # max mass
        self.acc_concentration = self.concentration[np.arange(self.acc_index.shape[0]), self.acc_index]
        self.acc_redshift = self.redshift[self.acc_index]
        self.acc_order = self.order[np.arange(self.acc_index.shape[0]), self.acc_index]
        self.acc_ParentID = self.ParentID[np.arange(self.acc_index.shape[0]), self.acc_index]

        self.proper_acc_index = np.copy(self.acc_index)
        for kk in range(2, self.order.max() + 1):
            subhalo_ind = np.where(self.acc_order == kk)
            for ind in subhalo_ind: #just so we know when the subhalo falls into the main progenitor
                self.proper_acc_index[ind] = self.proper_acc_index[self.acc_ParentID[ind]]

        self.proper_acc_redshift = self.redshift[self.proper_acc_index]

        # Compute accretion-time profiles using Green potentials
        self.acc_profiles = NFW_vectorized(
            self.acc_mass,
            self.acc_concentration,
            Delta=co.DeltaBN(self.acc_redshift, cfg.Om, cfg.OL), # see NOTE on host_profiles above
            z=self.acc_redshift)

        self.acc_Vmax = np.array([p.Vmax for p in self.acc_profiles])
        self.acc_rmax = np.array([p.rmax for p in self.acc_profiles])

        # Compute bound fraction of halo mass
        self.fb_og = self.mass / self.acc_mass[:, None]

        # Create mask: times after accretion and with fb above disruption threshold
        self.time_indices = np.arange(self.CosmicTime.shape[0])
        self.valid_fbs = np.log10(self.fb_og) > -4 #this excludes the fb=-4 index

        self.disrupt_index = np.zeros_like(self.acc_index)
        for subhalo_ind in range(self.Nhalo):
            if self.valid_fbs[subhalo_ind, 0]: #true at z=0 then the subhalo never disrupts!!!
                self.disrupt_index[subhalo_ind] = 0
            else:
                self.disrupt_index[subhalo_ind] = np.min(np.where(self.valid_fbs[subhalo_ind])[0]) - 1
        assert np.all(self.disrupt_index <= self.acc_index), "the disruption index is before the accretion index!"

        self.orbit_mask1 = self.time_indices[None, :] <= self.proper_acc_index[:, None] #anytime before accretion is not valid !!!!!!!!
        self.orbit_mask2 = self.time_indices[None, :] >= self.disrupt_index[:, None] #anytime after disruption is not valid
        self.orbit_mask = self.orbit_mask1 & self.orbit_mask2
        self.orbit_mask[0, :] = False #the host never moves!!
        self.fb = np.where(self.orbit_mask, self.fb_og, 0.0)
        self.orbit_masked_coordinates = np.where(self.orbit_mask[:, :, np.newaxis], self.coordinates, np.nan) # I want to do nan mask because 0.0 is techinically a valid coordinate

    def convert_to_cartesian(self, use_orbit_mask=True):

        if self.verbose:
            print("converting cyldrical coordinates to cartesian!")

        if use_orbit_mask:
            # transform to cartesian
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='invalid value encountered in divide')
                skyobj = crd.SkyCoord(frame='galactocentric', representation_type='cylindrical', rho=self.orbit_masked_coordinates[:,:,0] * u.kpc, phi=self.orbit_masked_coordinates[:,:,1] * u.rad, z=self.orbit_masked_coordinates[:,:,2]* u.kpc,
                                d_rho = self.orbit_masked_coordinates[:,:,3] * u.kpc/u.Gyr, d_phi = np.where(self.orbit_masked_coordinates[:,:,0], self.orbit_masked_coordinates[:,:,4]/self.orbit_masked_coordinates[:,:,0], self.orbit_masked_coordinates[:,:,0]) * u.rad/u.Gyr, d_z = self.orbit_masked_coordinates[:,:,5] * u.kpc/u.Gyr)
                xyz = skyobj.cartesian.xyz.to(u.kpc).value
                vel = skyobj.cartesian.differentials['s'].d_xyz.to(u.kpc/u.Gyr).value

        else:
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

        # start at the top of the self and propagate to children (first-order subhalos are already okay)
        for kk in range(2, self.order.max() + 1):
            to_fix = (self.order == kk)
            _, redshift = np.where(to_fix)
            self.cartesian_stitched[to_fix] = self.cartesian_stitched[to_fix] + self.cartesian_stitched[self.ParentID[to_fix], redshift]

        #to write out for the surving subhalos and to make movies with!
        self.rmags_stitched = np.linalg.norm(self.cartesian_stitched[:,:,0:3], axis=2)
        self.Vmags_stitched = np.linalg.norm(self.cartesian_stitched[:,:,3:6], axis=2)

        #to decide which subhalos merge!
        self.rmags = np.linalg.norm(self.cartesian[:,:,0:3], axis=2)
        self.Vmags = np.linalg.norm(self.cartesian[:,:,3:6], axis=2)

        #lets clean up the zero velocity indices
        rres = 0.001
        velres = rres/0.06 #kpc/Gyrvelres, 0, self.velres[0])
        self.Vmags[self.Vmags == 0.0] = velres

    # -----------------------------------------------------------------
    # everything below is a straight port of jsm_stellarhalo.Tree_Reader
    # as it stood at commit 1e407f3 -- this pipeline was later deleted
    # from jsm_stellarhalo.py and does not exist anywhere else in the
    # current codebase.
    # -----------------------------------------------------------------

    def tides(self):

        if self.verbose:
            print("evolving subhalo profiles based on bound fractions")

        self.rmax = np.full(shape=self.mass.shape, fill_value=np.nan) #empty arrays
        self.Vmax = np.full(shape=self.mass.shape, fill_value=np.nan)

        for subhalo_ind in range(self.Nhalo): #each tidal track is based on fb = m(t)/m(t_acc)
            rmax, Vmax = ancil.FUNC_halo_mass_evo(self, subhalo_ind)
            self.rmax[subhalo_ind] = rmax
            self.Vmax[subhalo_ind] = Vmax

        self.rmax[0] = self.host_rmax #cleaning up the empty host row with the precomuted values!
        self.Vmax[0] = self.host_Vmax

        self.parent_rmax = np.full(shape=self.mass.shape, fill_value=np.nan)  #empty arrays
        self.parent_Vmax = np.full(shape=self.mass.shape, fill_value=np.nan)

        for subhalo_ind in range(self.Nhalo): #reorganizing so that we have rmax and vmax of the parents!

            for time_ind in self.time_indices:
                parent_ID = self.ParentID[subhalo_ind, time_ind]
                if parent_ID != -99: #the parent branch has not been initalized

                    if self.orbit_mask[parent_ID, time_ind]: #the parent has been born and can its properties can evolve
                        self.parent_rmax[subhalo_ind, time_ind] = self.rmax[parent_ID, time_ind]
                        self.parent_Vmax[subhalo_ind, time_ind] = self.Vmax[parent_ID, time_ind]

                    elif time_ind > self.disrupt_index[parent_ID]: #the parent hasn't been born yet but also hasn't disrupted
                        self.parent_rmax[subhalo_ind, time_ind] = self.acc_rmax[parent_ID]
                        self.parent_Vmax[subhalo_ind, time_ind] = self.acc_Vmax[parent_ID]

    def mergers(self):

        #what we use to account for mergers!
        self.rmax_kscaled = np.log10(self.rmags/self.parent_rmax)
        self.Vmax_kscaled = np.log10(self.Vmags/self.parent_Vmax)

        self.R_mask = self.rmax_kscaled < self.merger_crit
        self.V_mask = self.Vmax_kscaled < self.merger_crit

        x_mer, y_mer = np.where(self.R_mask & self.V_mask)
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
            self.final_index[self.same_time] = self.merger_index[self.same_time]  # tie-breaker goes to mergers

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

        #assigning fates
        self.subhalo_fates = ["host"]
        for subhalo_ind in range(self.Nhalo):
            if np.isin(subhalo_ind, self.merged_subhalos):
                self.subhalo_fates.append("merged")
            elif np.isin(subhalo_ind, self.disrupted_subhalos):
                self.subhalo_fates.append("disrupted")
            elif np.isin(subhalo_ind, self.surviving_subhalos):
                self.subhalo_fates.append("surviving")

        self.subhalo_fates = np.array(self.subhalo_fates)
        self.merger_ratios = np.full(shape=self.merged_subhalos.shape, fill_value=0.0)
        str_to_int = {"merged": 0, "surviving": 1, "disrupted": 2}
        self.int_fates =  np.vectorize(str_to_int.get)(self.subhalo_fates) #includes the host!!
        self.int_fates[0] = -1 #masks the host!!

        #final subhalo properties!
        self.final_mass = self.mass[np.arange(self.final_index.shape[0]), self.final_index]
        self.final_concentration = self.concentration[np.arange(self.final_index.shape[0]), self.final_index]
        self.final_redshift = self.redshift[self.final_index]
        self.final_order = self.order[np.arange(self.final_index.shape[0]), self.final_index]
        self.final_ParentID = self.ParentID[np.arange(self.final_index.shape[0]), self.final_index]

    def abundance_counts(self):

        #within Virial Radius
        self.within_Rvir = self.rmags_stitched[1:, 0] < self.VirialRadius[0,0]

        #the artificial disruption criteia
        self.artdisrupt_mass = ancil.artificial_disruption(self.acc_mass[1:], self.acc_concentration[1:])
        self.artdisrupt_mask = self.final_mass[1:] > self.artdisrupt_mass #does not artificially disrupt

        #all subhalos above an arbitrary mass limit
        self.mass_cut88 = self.final_mass[1:] > 10**8.8
        self.mass_cut92 = self.final_mass[1:] > 10**9.2
        self.mass_cut96 = self.final_mass[1:] > 10**9.6

        #most conservative
        self.N_art88 = np.sum(self.mass_cut88 & self.within_Rvir & self.artdisrupt_mask)
        self.N_art92 = np.sum(self.mass_cut92 & self.within_Rvir & self.artdisrupt_mask)
        self.N_art96 = np.sum(self.mass_cut96 & self.within_Rvir & self.artdisrupt_mask)

        #somewhere in the middle
        self.N_Rvir88 = np.sum(self.mass_cut88 & self.within_Rvir)
        self.N_Rvir92 = np.sum(self.mass_cut92 & self.within_Rvir)
        self.N_Rvir96 = np.sum(self.mass_cut96 & self.within_Rvir)

        #most lax
        self.N_88 = np.sum(self.mass_cut88)
        self.N_92 = np.sum(self.mass_cut92)
        self.N_96 = np.sum(self.mass_cut96)

    def satellites(self):

        if self.verbose:
            print("using empirical relations to account for baryons")

        if hasattr(self, "ALPHA"):
            self.acc_stellarmass = 10**gh.lgMs_B18(lgMv=np.log10(self.acc_mass), z=self.acc_redshift, ALPHA=self.ALPHA) # the SHMR with the updated slopes!!
        else:
            self.acc_stellarmass = 10**gh.lgMs_B18(lgMv=np.log10(self.acc_mass), z=self.acc_redshift)
            # NOTE (jsm 2026-08-29): record that lgMs_B18's own built-in
            # ALPHA default was used (currently 1.963342, defined inside
            # galhalo.lgMs_B18 itself) rather than hardcoding that number
            # here where it could silently drift out of sync. self.ALPHA
            # is always defined after this point either way, for
            # create_survsat_dict() to report the free parameters used.
            self.ALPHA = np.nan

        if self.scatter==True:
            self.acc_stellarmass = 10**(gh.dex_sampler(np.log10(self.acc_stellarmass)))

        #the sizes and metallicities
        self.acc_R50 = 10**gh.Reff_A24(lgMs=np.log10(self.acc_stellarmass)) # the size mass relation from SAGA
        self.FeH = gh.MZR(self.acc_stellarmass) # the mass metalicity relation!

        if self.scatter==True:
            self.acc_R50 = 10**(gh.dex_sampler(np.log10(self.acc_R50)))
            self.FeH = gh.dex_sampler(self.FeH, dex=0.17)

        self.R50 = np.full(shape=self.mass.shape, fill_value=0.0) # empty arrays
        self.stellarmass = np.full(shape=self.mass.shape, fill_value=0.0)

        for subhalo_ind in range(self.Nhalo): #each tidal track is based on fb = m(t)/m(t_acc)
            R50, stellarmass = ancil.FUNC_stellar_mass_evo(self,subhalo_ind)
            self.R50[subhalo_ind] = R50
            self.stellarmass[subhalo_ind] = stellarmass

        self.stellarmass_og = np.copy(self.stellarmass)

        self.final_stellarmass = self.stellarmass[np.arange(self.final_index.shape[0]), self.final_index]
        self.total_stellarmass_acc = np.sum(self.acc_stellarmass[1:])
        self.fb_stellar = self.stellarmass / self.acc_stellarmass[:, None]

        self.icl = np.full(shape=self.mass.shape, fill_value=0.0)
        self.contributed = np.full(shape=self.mass.shape[0], fill_value=0.0)

        # NOTE (flagged by Sebastian): this is the line that must survive the port intact.
        self.frac_fb_DM, self.frac_fb_stellar = ancil.fb_surv_frac(self)

    def disk(self):

        self.insitu = ancil.FUNC_in_situ_SFR(self)
        self.stellarmass[0] = self.insitu #the SFR from the UM model
        self.acc_stellarmass[0] = self.stellarmass[0,0] #updating so its not based on the SHMR
        self.target_stellarmass = self.acc_stellarmass[0] #just to have the same nomenclature as the DM
        self.exsitu = np.full(shape=self.mass.shape, fill_value=0.0)

    def stellarhalo(self):
        self.forest = ancil.forest_generator(self)
        for current_index in range(len(self.forest) - 2, -1, -1):
            ancil.tree_walker(self, current_index)

        #the ICL
        self.icl_across_systems = np.sum(self.icl, axis=0)
        self.icl_MAH = np.cumsum(self.icl_across_systems[::-1])[::-1]
        self.total_ICL = self.icl_MAH[0]

        # NOTE (jsm, 2026-08-29): fold in "pre-processing" stellar mass loss.
        # self.orbit_mask only allows self.fb/self.stellarmass to be nonzero
        # for time indices in [disrupt_index, proper_acc_index]. For order>=2
        # subhalos this window can be EMPTY (disrupt_index > proper_acc_index)
        # whenever a subhalo fully tidally disrupts, per its own unmasked
        # tidal track, before the group it belongs to ever properly accretes
        # onto the main host. When that happens self.stellarmass is 0 at
        # every timestep that subhalo ever existed, so ancil.tree_walker()
        # never sees any nonzero mass to hand to icl/contributed/exsitu -- the
        # subhalo's entire accreted stellar mass silently disappears from the
        # budget instead of being counted. Physically this is stellar mass
        # stripped off a satellite while it's still embedded in a smaller
        # group, prior to that group's infall -- by the time (if ever) the
        # group reaches the main host that mass is unbound intracluster
        # light, so we credit it to total_ICL (and to self.contributed, so
        # the merged/disrupted/surviving breakdown below stays consistent).
        # self.acc_stellarmass here is still the true SHMR-based value from
        # satellites()/disk() -- it isn't overwritten until further below.
        self.preprocessed_mask = self.disrupt_index > self.proper_acc_index
        self.preprocessed_mask[0] = False #the host is never "pre-processed"
        self.preprocessed_ICL = np.sum(self.acc_stellarmass[self.preprocessed_mask])
        self.contributed[self.preprocessed_mask] += self.acc_stellarmass[self.preprocessed_mask]
        self.total_ICL += self.preprocessed_ICL
        self.icl_MAH[0] = self.total_ICL #keep the z=0 point of the MAH consistent with the correction

        #accretion onto the central
        self.exsitu_across_systems = np.sum(self.exsitu, axis=0)
        self.exsitu_MAH = np.cumsum(self.exsitu_across_systems[::-1])[::-1]
        self.total_exsitu = self.exsitu_MAH[0]
        self.MW_est = ancil.MW_est_criteria(self)

        #the satellites
        self.stellarmass_in_satellites = np.sum(self.stellarmass[self.surviving_subhalos, 0])
        self.N90_ids, self.cumsum_perc, self.N90_fates = ancil.N90_cont(self)
        self.most_massive = ancil.MMP(self)

        #update after the merger shuffle!
        self.acc_stellarmass = self.stellarmass[np.arange(self.acc_index.shape[0]), self.acc_index]
        self.final_stellarmass = self.stellarmass[np.arange(self.final_index.shape[0]), self.final_index]
        self.target_stellarmass = self.acc_stellarmass[0] #update the host as well!

        # Breakdown by satellite type
        self.ICL_fmerged = np.sum(self.contributed[self.merged_subhalos])
        self.ICL_fdisrupted = np.sum(self.contributed[self.disrupted_subhalos])
        self.ICL_fsurviving = np.sum(self.contributed[self.surviving_subhalos])

        #now the final tally
        self.mass_loss = self.total_stellarmass_acc - (self.total_exsitu + self.stellarmass_in_satellites + self.total_ICL)

        if self.verbose:
            print("-----------------------------")
            print("=== SUBHALO POPULATIONS ===")

            print(f"Total satellites: {self.Nhalo-1}")
            print(f"Satellites disrtuped: {self.N_disrupted}")
            print(f"Satellites merged with direct parents: {self.N_merged}")
            print(f"Satellites survived to z=0: {self.N_surviving}")

            print("=== STELLAR MASS BUDGET ===")
            print(f"Total Accreted Stellar Mass     : {self.total_stellarmass_acc:.3e}")
            print(f"  -> Central Accreted           : {self.total_exsitu:.3e}")
            print(f"  -> In Surviving Satellites    : {self.stellarmass_in_satellites:.3e}")
            print(f"  -> In ICL (incl. pre-proc.)   : {self.total_ICL:.3e}")
            print(f"       of which pre-processed   : {self.preprocessed_ICL:.3e}")
            print(f"  -> Accounted (sum)            : {(self.total_exsitu + self.stellarmass_in_satellites + self.total_ICL):.3e}")
            print(f"  -> Missing                    : {(self.total_stellarmass_acc - (self.total_exsitu + self.stellarmass_in_satellites + self.total_ICL)):.3e}")

    def create_survsat_dict(self):

        dictionary = {"tree_index": self.tree_index, #this gets shuffled around because of the multiprocessing!
                    "merger_crit": self.merger_crit, # free parameters used for this run --
                    "scatter": self.scatter,         # see the run_abundance.py walkthrough
                    "ALPHA": self.ALPHA,             # (nan == lgMs_B18's built-in default)
                    "fesc": self.fesc,                # fraction of a merged satellite's stars -> ICL
                    "Nhalo": self.Nhalo - 1, #total number of subhalos accreted
                    "MW_est": self.MW_est, #[c, GSE, LMC] all three would be [1,1,1]
                    "MAH": self.mass[0], # the host halo mass across time! (N time indices)
                    "MAH_stellar": self.stellarmass[0], # the central stellar mass across time!
                    "MAH_ICL": self.icl_MAH, # the build of ICL
                    "host_mass": self.mass[0,0], # the target halo mass (single values from here!)
                    "host_stellarmass": self.stellarmass[0,0], #the target stellar mass including Mstar acc
                    "host_Rvir": self.VirialRadius[0,0],
                    "host_Vcirc": self.host_Vmax[0],
                    "host_z50": self.host_z50,
                    "host_z10": self.host_z10,
                    "host_z90": self.host_z90,
                    "Mstar_tot": self.total_stellarmass_acc, #total ever accreted (sum from the SHMR sample)
                    "Mstar_lost": self.mass_loss, #this should be less than 0.01 percent of Mstar tot
                    "Mstar_ICL": self.total_ICL, #ICL, now includes pre-processing loss folded in below
                    "Mstar_ICL_preprocessed": self.preprocessed_ICL, #subset of Mstar_ICL from subhalos that
                                                                       #fully disrupted before their group
                                                                       #properly joined the main host
                    "Mstar_sat": self.stellarmass_in_satellites, #total mass in surviving satellites
                    "Mstar_acc": self.total_exsitu, # the stellar mass that is accreted onto the central
                    "sat_N90": self.acc_stellarmass[self.N90_ids], #the accretion stellar mass and the number!
                    "Nrank": self.cumsum_perc, #should be able to find the contributions using this!
                    "sat_cumsum": self.cumsum_perc,
                    "sat_fates": self.int_fates[1:].astype('int'), #0 survives, 1 merges, 2 disrupts
                    "sat_mass": self.final_mass[1:], # the final halo masses which depend on fate
                    "sat_acc_mass": self.acc_mass[1:], # the acc mass
                    "sat_stellarmass": self.final_stellarmass[1:],
                    "sat_acc_stellarmass": self.acc_stellarmass[1:],
                    "sat_order": self.final_order[1:],
                    "sat_acc_order": self.acc_order[1:],
                    "sat_zacc": self.acc_redshift[1:],
                    "sat_zacc_proper": self.proper_acc_redshift[1:],
                    "sat_final_rmag": self.rmags_stitched[1:, 0],
                    "sat_final_vmag": self.Vmags_stitched[1:, 0],
                    "sat_acc_c": self.acc_concentration, #the accretion concentration of the satellites
                    "sat_zfinal": self.final_redshift[1:]}
        return dictionary

    def write_out_disc(self):

        dictionary = {"tree_index": self.tree_index, #this gets shuffled around because of the multiprocessing!
                    "Nhalo": self.Nhalo - 1, #total number of subhalos accreted
                    "host_z50": self.host_z50,
                    "host_concentration": self.concentration[0,0],
                    "N_disrupted": self.N_disrupted, # Number of disrupted halos
                    "N_merged": self.N_merged, # number that merge onto the central
                    "N_surviving": self.N_surviving, # the number of surviving halos
                    "sat_fates": self.int_fates[1:].astype('int'), #0 survives, 1 merges, 2 disrupts
                    "sat_mass": self.final_mass[1:], # the final halo masses which depend on fate
                    "sat_acc_mass": self.acc_mass[1:], # the acc mass
                    "sat_art_mass": self.artdisrupt_mass,
                    "sat_stellarmass": self.final_stellarmass[1:],
                    "sat_acc_stellarmass": self.acc_stellarmass[1:],
                    "sat_zacc": self.acc_redshift[1:],
                    "sat_zacc_proper": self.proper_acc_redshift[1:],
                    "sat_final_rmag": self.rmags_stitched[1:, 0],
                    "sat_final_vmag": self.Vmags_stitched[1:, 0],
                    "sat_acc_c": self.acc_concentration,
                    "cumsum": self.frac_fb_stellar}
        return dictionary

    def write_out_massspec(self):

        dictionary = {"tree_index": self.tree_index, #this gets shuffled around because of the multiprocessing!
                    "host_mass": self.mass[0,0],
                    "host_Rvir": self.VirialRadius[0,0],
                    "host_Vcirc": self.host_Vmax[0],
                    "host_z10": self.host_z10,
                    "host_z50": self.host_z50,
                    "host_z90": self.host_z90,
                    "host_concentration": self.concentration[0,0],
                    "Nhalo": self.Nhalo - 1, #total number of subhalos accreted
                    "N_disrupted": self.N_disrupted, # Number of disrupted halos
                    "N_merged": self.N_merged, # number that merge onto the central
                    "N_surviving": self.N_surviving, # the number of surviving halos
                    "N_art88": self.N_art88,
                    "N_art92": self.N_art92,
                    "N_art96": self.N_art96,
                    "N_Rvir88": self.N_Rvir88,
                    "N_Rvir92": self.N_Rvir92,
                    "N_Rvir96": self.N_Rvir96,
                    "N_88": self.N_88,
                    "N_92": self.N_92,
                    "N_96": self.N_96}
        return dictionary
