import numpy as np
import warnings; warnings.simplefilter('ignore')
import sys
import h5py
import pandas as pd
import os
import json
import jsm_stats
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
import galhalo as gh
import evolve as ev
import astropy.units as u
import astropy.constants as const
import astropy.coordinates as crd
from treelib import Node, Tree


def FUNC_halo_mass_evo(tree, subhalo_ind):

    #based on Green et al 2019 fitting code using the transfer function
    rmax = tree.acc_rmax[subhalo_ind] * ev.GvdB_19(tree.fb[subhalo_ind], tree.acc_concentration[subhalo_ind], track_type="rad") #Green et al 2019 transfer function tidal track
    Vmax = tree.acc_Vmax[subhalo_ind] * ev.GvdB_19(tree.fb[subhalo_ind], tree.acc_concentration[subhalo_ind], track_type="vel") #Green et al 2019 transfer function tidal track

    return rmax, Vmax

def FUNC_stellar_mass_evo(tree, subhalo_ind):

    #based on Errani et al 2021 fitting code

    R50_by_rmax = tree.acc_R50[subhalo_ind]/tree.acc_rmax[subhalo_ind]
    R50_fb, stellarmass_fb = ev.g_EPW18(tree.fb[subhalo_ind], alpha=1.0, lefflmax=R50_by_rmax) #Errani 2018 tidal tracks for stellar evolution!
    R50 = tree.acc_R50[subhalo_ind]*R50_fb #scale the sizes!
    stellarmass = tree.acc_stellarmass[subhalo_ind]*stellarmass_fb #scale the masses!
    
    return R50, stellarmass
    
def FUNC_in_situ_SFR(tree):

    tree.zmask = ~np.isnan(tree.VirialRadius[0]) 
    tree.Rvir = tree.VirialRadius[0][tree.zmask][::-1] # need to flip so the integration works!
    tree.Mhalo = tree.mass[0][tree.zmask][::-1]
    tree.Vmaxhalo = tree.host_Vmax[tree.zmask][::-1]
    tree.zhalo = tree.redshift[tree.zmask][::-1]
    tree.thalo = tree.CosmicTime[tree.zmask][::-1]

    tree.t_dyn = gh.dynamical_time(tree.Rvir*u.kpc, tree.Mhalo*u.solMass).to(u.Gyr).value
    tree.SFR  = gh.SFR_B19(tree.Vmaxhalo, tree.zhalo)
    tree.Mstar, tree.f_lost = gh.integrate_SFH(tree.SFR, tree.thalo)

    padding = np.zeros(shape=(tree.CosmicTime.shape[0] - tree.Mstar.shape[0],))  # Create the padding array
    return np.concatenate((padding, tree.Mstar))[::-1]

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def acc_hierarchy(tree):

    tree_hierarchy = Tree()
    tree_hierarchy.create_node("Host Halo", "0", 
                    data={"acc_mass": tree.target_mass, "acc_redshift": 0, "acc_stellarmass": tree.target_stellarmass})  # The root node

    def add_node_with_parents(subhalo_ind):
        parent_id = str(tree.acc_ParentID[subhalo_ind])
        node_id = str(subhalo_ind)
        data_id = {"acc_mass": tree.acc_mass[subhalo_ind], "acc_redshift": tree.acc_redshift[subhalo_ind], "acc_stellarmass": tree.acc_stellarmass[subhalo_ind]}

        # If parent not yet in tree, add it (or recurse)
        if not tree_hierarchy.contains(parent_id):
            # Recursively ensure the parent is added first
            add_node_with_parents(int(parent_id))

        # Finally add the current node (if not already added)
        if not tree_hierarchy.contains(node_id):
            tree_hierarchy.create_node("subID:" + node_id, node_id, parent=parent_id, data=data_id)

    for subhalo_ind in range(1, tree.Nhalo):
        add_node_with_parents(subhalo_ind) 
    return tree_hierarchy


def find_late_events(tree):

    late_mergers = []
    late_disruptions = []

    for node in tree.final_tree.all_nodes():
        if node.is_root():
            continue  # skip root node

        parent = tree.final_tree.parent(node.identifier)

        child_z = node.data["final_redshift"]
        parent_z = parent.data["final_redshift"]

        # Check if child merged/disrupted after parent already merged
        if child_z < parent_z and parent.data["fate"] == "merged":
            child_fate = node.data["fate"]

            entry = {
                "child_id": node.identifier,
                "parent_id": parent.identifier,
                "child_z": child_z,
                "parent_z": parent_z,
                "child_fate": child_fate,
                "parent_fate": parent.data["fate"]
            }

            if child_fate == "merged":
                late_mergers.append(entry)
            elif child_fate == "disrupted":
                late_disruptions.append(entry)

    return late_disruptions, late_mergers

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def N90_cont(tree):

    mass_sorted = np.argsort(tree.contributed)[::-1] # sort the contributions to the stellar halo
    perc_sorted = tree.contributed[mass_sorted]/tree.total_ICL #measure the percentage
    perc_cm = np.cumsum(perc_sorted) #cumulaitve sum to find where the rank hits 90
    N90_rank = np.argmin(perc_cm < 0.9) #where does the rank hit 90
    if N90_rank == 0:
        N90_ids = mass_sorted[0:1] # the subhalos that contribute to that rank
    else:
        N90_ids = mass_sorted[0:N90_rank] # the subhalos that contribute to that rank
    fates = tree.subhalo_fates[N90_ids]
    return N90_ids, perc_cm, fates

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

def fb_surv_frac(tree, Nbins=30):

    DM_bins = np.linspace(-4, 0, Nbins+1)
    stellar_bins = np.linspace(-3, 0, Nbins+1)

    fb_DM = np.log10(tree.fb[tree.surviving_subhalos, 0])
    fb_stellar = np.log10(tree.fb_stellar[tree.surviving_subhalos, 0])

    fraction_DM = jsm_stats.cumulative_histogram(fb_DM, DM_bins)
    fraction_stellar = jsm_stats.cumulative_histogram(fb_stellar, stellar_bins)

    return fraction_DM, fraction_stellar

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

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

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def add_node_with_parents(tree, tree_hierarchy, subhalo_ind, z_ind):
    node_id = str(subhalo_ind)
    parent_id = str(tree.ParentID[subhalo_ind, z_ind])
    
    # Check if parent exists at this time step (not -99)
    if int(parent_id) != -1 and tree.ParentID[int(parent_id), z_ind] == -99:
        # Parent hasn't been born yet, skip adding this node
        return
    
    # If parent not yet in tree, add it (or recurse)
    if not tree_hierarchy.contains(parent_id):
        # Only recurse if parent is not the host halo (-1)
        if int(parent_id) != -1:
            # Recursively ensure the parent is added first
            add_node_with_parents(tree, tree_hierarchy, int(parent_id), z_ind)
            # If parent still not in tree after recursion, skip this node
            if not tree_hierarchy.contains(parent_id):
                return
    
    # Finally add the current node (if not already added)
    if not tree_hierarchy.contains(node_id):
        tree_hierarchy.create_node("subID:" + node_id, node_id, parent=parent_id)

def forest_generator(Tree_Vis):

    forest = []
    
    for z_ind in range(0, len(Tree_Vis.redshift)):
        
        tree_hierarchy_z = Tree()
        tree_hierarchy_z.create_node("Host Halo", "0", data={"redshift": Tree_Vis.redshift[z_ind]})
        forest.append(tree_hierarchy_z)
        
        parents = Tree_Vis.ParentID[:, z_ind]
        initialized_subhalos = np.where(parents != -99)[0]
        initialized_subhalos = initialized_subhalos[initialized_subhalos != -1]  # remove the host!
        
        if initialized_subhalos.shape[0] > 1:
            for subhalo_ind in initialized_subhalos:
                try:
                    add_node_with_parents(Tree_Vis, tree_hierarchy_z, subhalo_ind, z_ind)
                except RecursionError:
                    print(f"Recursion error at z_ind {z_ind}, subhalo {subhalo_ind}")
    
    return forest


def tree_walker(Tree_Vis, current_index):

    prev_index = current_index + 1

    current_tree = Tree_Vis.forest[current_index]
    prev_tree = Tree_Vis.forest[prev_index]

    order_jumps = 0
    births = 0
    host_halo_mergers = 0
    host_acc = 0

    for current_subhalo in current_tree.all_nodes_itr():
        current_subhalo_id = current_subhalo.identifier

        #skip the host!
        if current_subhalo_id == "0":
            continue
        
        #skip the subhalos that don't have a previous time-step counter part
        prev_subhalo = prev_tree.get_node(current_subhalo_id)
        if prev_subhalo == None:
            births += 1
            continue

        #just to count the order jumps
        current_parent = current_tree.parent(current_subhalo_id)
        current_parent_id = current_parent.identifier

        prev_parent = prev_tree.parent(current_subhalo_id)
        prev_parent_id = prev_parent.identifier

        if current_parent_id != prev_parent_id:
            order_jumps += 1

        #this should be fine since we have already skipped the subhalos that were just born this time_index
        current_subhalo_id_int = int(current_subhalo_id) # just grabbing as an integers
        current_parent_id_int = int(current_parent_id)

        #the subhalo hasn't been intialized with stellarmass
        if Tree_Vis.acc_index[current_subhalo_id_int] < current_index:
            continue

        current_mass = Tree_Vis.stellarmass[current_subhalo_id_int, current_index]
        prev_mass = Tree_Vis.stellarmass[current_subhalo_id_int, prev_index]
        mass_loss = prev_mass - current_mass

        if mass_loss > 0: # don't need to worry about the parent as much here since ICL is summed across all subhalos
            Tree_Vis.icl[current_parent_id_int, current_index] += mass_loss
            Tree_Vis.contributed[current_subhalo_id_int] += mass_loss

        #if the subhalo is found dead
        if Tree_Vis.final_index[current_subhalo_id_int] == current_index:

            fate = Tree_Vis.subhalo_fates[current_subhalo_id_int]
            is_late_event = Tree_Vis.final_index[current_subhalo_id_int]  < Tree_Vis.final_index[current_parent_id_int]

            if fate == "merged": #break up the mass
                icl_mass = Tree_Vis.fesc * current_mass
                merger_mass = current_mass - icl_mass

                # if the parent hasn't been born yet and is not the host!
                if Tree_Vis.acc_index[current_parent_id_int] != 0 and Tree_Vis.acc_index[current_parent_id_int] < current_index:
                    continue
                else:
                    if is_late_event: # the merger happens with the grandparent!
                        grandparent = current_tree.parent(current_parent_id)

                        if grandparent is None: #the host is the grandparent!
                            current_grandparent_id_int = 0
                            host_halo_mergers += 1
                            host_acc += merger_mass 
                            Tree_Vis.exsitu[current_subhalo_id_int, current_index] = merger_mass 

                        else: #the grandparent exists!
                            current_grandparent_id = grandparent.identifier
                            current_grandparent_id_int = int(current_grandparent_id)

                        #keep track of the ratio - need to measure this first!
                        merger_mass_ratio = merger_mass/Tree_Vis.stellarmass[current_grandparent_id_int, current_index]
                        merger_index = np.where(Tree_Vis.merged_subhalos == current_subhalo_id_int)[0][0]
                        Tree_Vis.merger_ratios[merger_index] = merger_mass_ratio

                        #disribute the mass
                        Tree_Vis.icl[current_grandparent_id_int, current_index] += icl_mass #this only happens at the time index
                        Tree_Vis.contributed[current_subhalo_id_int] += icl_mass

                        Tree_Vis.stellarmass[current_grandparent_id_int, :current_index] += merger_mass #this applies to everywhere after!

                    else: #the merger happens with the parent!!
                        if current_parent_id_int == 0: #the parent is the host!
                            host_halo_mergers += 1
                            host_acc += merger_mass 
                            Tree_Vis.exsitu[current_subhalo_id_int, current_index] = merger_mass 

                        #keep track of the ratio
                        merger_mass_ratio = merger_mass/Tree_Vis.stellarmass[current_parent_id_int, current_index]
                        merger_index = np.where(Tree_Vis.merged_subhalos == current_subhalo_id_int)[0][0]
                        Tree_Vis.merger_ratios[merger_index] = merger_mass_ratio

                        #disribute the mass
                        Tree_Vis.icl[current_parent_id_int, current_index] += icl_mass
                        Tree_Vis.contributed[current_subhalo_id_int] += icl_mass

                        Tree_Vis.stellarmass[current_parent_id_int, :current_index] += merger_mass

                # EXPLICITLY ZERO OUT THE DEAD SUBHALO'S MASS
                Tree_Vis.stellarmass[current_subhalo_id_int, :current_index+1] = 0.0

            if fate == "disrupted": #all of the mass goes to the ICL
                icl_mass = current_mass

                if is_late_event: # the disruption happens while in the grandparent!
                    grandparent = current_tree.parent(current_parent_id)

                    if grandparent is None: #the host is the grandparent!
                        current_grandparent_id_int = 0 
                    else: #the grandparent exists!
                        current_grandparent_id = grandparent.identifier
                        current_grandparent_id_int = int(current_grandparent_id)
                    #disribute the mass
                    Tree_Vis.icl[current_grandparent_id_int, current_index] += icl_mass
                    Tree_Vis.contributed[current_subhalo_id_int] += icl_mass


                else: #the merger happens with the parent!!
                    #disribute the mass
                    Tree_Vis.icl[current_parent_id_int, current_index] += icl_mass
                    Tree_Vis.contributed[current_subhalo_id_int] += icl_mass

                # EXPLICITLY ZERO OUT THE DEAD SUBHALO'S MASS
                Tree_Vis.stellarmass[current_subhalo_id_int, :current_index+1] = 0.0

            if fate == "surviving": #do nothing with it since it already lost mass!
                continue

# def find_associated_subhalos(tree, sub_ind, time_ind):
#     associated_set = []

#     # Checking to see if there are any direct children at this time step
#     direct_parent_merging = tree.ParentID[:, time_ind] == sub_ind 
#     if np.any(direct_parent_merging):
#         associated_subhalos = np.where(direct_parent_merging)[0]  # Any subhalos that have the same parent
#         disrupt_mask = tree.disrupt_index[associated_subhalos] < time_ind  # Disruption must happen after the merger
#         associated_subhalos = associated_subhalos[disrupt_mask]
#         associated_set.extend(associated_subhalos)
        
#         # Recursively collect descendants of each subhalo
#         for subhalo in associated_subhalos:
#             subhalo_descendants = find_associated_subhalos(tree, subhalo, time_ind)
#             if subhalo_descendants:  # Ensure no NoneType is returned
#                 associated_set.extend(subhalo_descendants)

#     return associated_set