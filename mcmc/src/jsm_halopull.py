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
    



################################
########## GRAVEYARD ###########
################################


        # Hnpy = np.load(self.metadir+"host_properties.npz")
        # Hkeys = ("lgMh", "z_50", "z_10", "Nsub_total")
        # Hdata = Table([Hnpy[:,0], Hnpy[:,1], Hnpy[:,2], Hnpy[:,3]], names=Hkeys)

        # print("writing out the host data")
        # Hdata.write(self.metadir+"host_prop.dat", format="ascii", overwrite=True)


    # def in_situ_SFR(self):

    #     zmask = ~np.isnan(self.VirialRadius[0])
    #     Rvir = self.VirialRadius[0][zmask][::-1]
    #     Mhalo = self.mass[0][zmask][::-1]
    #     zhalo = self.redshift[zmask][::-1]
    #     thalo = self.CosmicTime[zmask][::-1]

    #     #self.eps = gh.epsilon_M_z(Mhalo, zhalo, param_type="Oleary")
    #     self.eps = gh.lgMs_B18(lgMv=np.log10(Mhalo), z=zhalo, return_epsilon=True)
    #     self.t_dyn = gh.dynamical_time(Rvir*u.kpc, Mhalo*u.solMass).to(u.Gyr).value

    #     self.dMdt_ave = []
    #     for t, time in enumerate(thalo):
    #         tdyn = self.t_dyn[t]
    #         delta_t = time - tdyn
    #         delta_t_index = np.argmin(np.abs(thalo - delta_t))
    #         self.dMdt_ave.append((Mhalo[t] - Mhalo[delta_t_index])/tdyn)

    #     #fb = 0.15 need this if using Mosters 2018 model!
    #     self.SFR = self.eps*(self.dMdt_ave) #*0.15
    #     Mstar = np.nancumsum(self.SFR[:-1]*np.diff(thalo))
    #     padding = np.zeros(shape=(self.CosmicTime.shape[0] - Mstar.shape[0],))  # Create the padding array
    #     return np.concatenate((padding, Mstar))[::-1], zmask

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