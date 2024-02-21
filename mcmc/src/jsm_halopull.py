import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.table import Table
import os
import warnings; warnings.simplefilter('ignore')
import jsm_SHMR

##################################################
### FOR INTERFACING WITH THE "RAW" SATGEN DATA ###
##################################################

def anamass(file):
    tree = np.load(file) #open file and read
    mass = tree["mass"]
    redshift = tree["redshift"]
    coords = tree["coordinates"]
    orders = tree["order"]

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
    opentree = np.load(file) #open file and read
    z50 = opentree["redshift"][find_nearest1(opentree["mass"][0], opentree["mass"][0,0]/2)]
    z10 = opentree["redshift"][find_nearest1(opentree["mass"][0], opentree["mass"][0,0]/10)]
    return np.array([np.log10(opentree["mass"][0,0]), z50, z10, opentree["mass"].shape[0]])


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
            if filename.startswith('tree') and filename.endswith('evo.npz'): 
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

        for i,file in enumerate(files): # this part takes a while if you have a lot of trees
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
        tree = np.load(self.files[random_index])

        mass = tree["mass"]
        time = tree["CosmicTime"]

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

def differential(phi, phi_bins, phi_binsize): 
    N = np.histogram(phi, bins=phi_bins)[0]
    return N/phi_binsize

class MassMat:

    """
    An easy way of interacting with the condensed mass matricies.
    One instance of the Realizations class will create several SAGA-like samples.
    """
        
    def __init__(self, metadir, Nsamp=100, Mres=-4, phi_Nbins=45, phimin=-4, save=False, plot=False):

        self.metadir = metadir
        self.Mres = Mres
        self.Nsamp = Nsamp 
        self.Nbins = phi_Nbins
        self.phimin = phimin
        self.phi_bins = np.linspace(self.phimin, 0, phi_Nbins)
        self.phi_binsize = self.phi_bins[1] - self.phi_bins[0]
        self.save = save
        self.plot = plot

        self.prep_data()
        self.SHMF()
        #self.SAGA_break()
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
        #         final_coords = final_coords[mask]
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

        surv_mask = np.log10(self.final_mass/self.acc_mass) > self.Mres # now selecting only the survivers
        self.acc_surv_mass = np.ma.filled(np.ma.masked_array(self.acc_mass, mask=~surv_mask),fill_value=np.nan)

        self.lgMh_acc = np.log10(self.acc_mass) # accretion
        self.lgMh_final = np.log10(self.final_mass) # final mass
        self.lgMh_acc_surv = np.log10(self.acc_surv_mass) # the accretion mass of surviving halos

        self.hostprop = np.load(self.metadir+"host_properties.npy")
        self.Mhosts = self.hostprop[:,0]
        self.z_50 = self.hostprop[:,1]
        self.z_10 = self.hostprop[:,2]

        self.acc_phi = np.log10((self.acc_mass.T / self.Mhosts).T)  
        self.final_phi = np.log10((self.final_mass.T / self.Mhosts).T) 
        self.acc_surv_phi = np.log10((self.acc_surv_mass.T / self.Mhosts).T)
 
    def SHMF(self):
        self.acc_surv_phi_counts = np.apply_along_axis(differential, 1, self.acc_surv_phi, phi_bins=self.phi_bins, phi_binsize=self.phi_binsize) # the accretion mass of the surviving halos
        acc_surv_phi_SHMF_ave = np.average(self.acc_surv_phi_counts, axis=0)
        acc_surv_phi_SHMF_std = np.std(self.acc_surv_phi_counts, axis=0)
        self.acc_surv_SHMF_werr = np.array([acc_surv_phi_SHMF_ave, acc_surv_phi_SHMF_std])

        self.final_phi_counts = np.apply_along_axis(differential, 1, self.final_phi, phi_bins=self.phi_bins, phi_binsize=self.phi_binsize) # the final mass of all halos
        final_phi_SHMF_ave = np.average(self.final_phi_counts, axis=0)
        final_phi_SHMF_std = np.std(self.final_phi_counts, axis=0)
        self.final_SHMF_werr = np.array([final_phi_SHMF_ave, final_phi_SHMF_std])

        self.acc_phi_counts = np.apply_along_axis(differential, 1, self.acc_phi, phi_bins=self.phi_bins, phi_binsize=self.phi_binsize) # the accretion mass of all the halos
        acc_phi_SHMF_ave = np.average(self.acc_phi_counts, axis=0)
        acc_phi_SHMF_std = np.std(self.acc_phi_counts, axis=0)
        self.acc_SHMF_werr = np.array([acc_phi_SHMF_ave, acc_phi_SHMF_std])

        self.phi_bincenters = 0.5 * (self.phi_bins[1:] + self.phi_bins[:-1])
        if self.plot==True:
        
            fig, ax = plt.subplots(figsize=(8, 8))

            ax.plot(self.phi_bincenters, self.acc_SHMF_werr[0], label="Total population (z = z$_{\mathrm{acc}}$)", color="green", ls="-.")
            #plt.fill_between(self.phi_bincenters, y1=self.acc_SHMF_werr[0]-self.acc_SHMF_werr[1], y2=self.acc_SHMF_werr[0]+self.acc_SHMF_werr[1], alpha=0.1, color="grey")

            ax.plot(self.phi_bincenters, self.acc_surv_SHMF_werr[0], label="Surviving population (z = z$_{\mathrm{acc}}$)", color="cornflowerblue")
            ax.fill_between(self.phi_bincenters, y1=self.acc_surv_SHMF_werr[0]-self.acc_surv_SHMF_werr[1], y2=self.acc_surv_SHMF_werr[0]+self.acc_surv_SHMF_werr[1], alpha=0.2, color="cornflowerblue")

            ax.plot(self.phi_bincenters, self.final_SHMF_werr[0],  label="Surviving population (z = 0)", color="red", ls="-.")
            #plt.fill_between(self.phi_bincenters, y1=self.final_SHMF_werr[0]-self.final_SHMF_werr[1], y2=self.final_SHMF_werr[0]+self.final_SHMF_werr[1], alpha=0.1, color="grey")

            ax.axvline(self.Mres, ls="--", color="black")
            ax.text(self.Mres+0.05, 0.1, "resolution limit", rotation=90, color="black", fontsize=15)
            
            ax.set_xlabel("log (m/M$_{\mathrm{host}}$)", fontsize=15)
            ax.set_yscale("log")
            ax.set_ylabel("dN / dlog(m/M)", fontsize=15)

            if self.save==True:
                plt.savefig(self.metadir+"SHMF.pdf")

            plt.show()

    def SAGA_break(self):

        """_summary_
        only for realizations converted to stellar mass!
        """

        self.snip = self.lgMh_acc_surv.shape[0]%self.Nsamp
        if self.snip != 0.0:
            print("Cannot evenly divide your sample by the number of samples!")
            # lgMh_snip = np.delete(self.lgMh_acc_surv, np.arange(self.snip), axis=0)
            # self.Nsets = int(lgMh_snip.shape[0]/self.Nsamp) #dividing by the number of samples
            # print("dividing your sample into", self.Nsets, "sets.", self.snip, "trees were discarded")
            # self.lgMh_mat = np.array(np.split(lgMh_snip, self.Nsets, axis=0))
        else:
            self.Nsets = int(self.lgMh_acc_surv.shape[0]/self.Nsamp) #dividing by the number of samples
            self.Mhosts_mat = np.array(np.split(self.Mhosts, self.Nsets))
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

        if self.save==True:
            print("saving the accretion masses!")
            np.savez(self.metadir+"models.npz",
                    host_mass = self.Mhosts_mat,
                    mass = self.acc_surv_lgMh_mat,
                    redshift = self.acc_red_mat)

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
        tree_id = []
        sat_id = []

        for itree in range(self.Nsamp):
            Nsub_i = np.argwhere(~np.isnan(self.lgMh_acc_surv[itree]))[:,0]
            for j, isat in enumerate(Nsub_i):
                Nsub.append(len(Nsub_i))
                tree_id.append(itree+1)
                z_50.append(self.z_50[itree])
                z_10.append(self.z_10[itree])
                sat_id.append(j+1)
                M_acc.append(self.lgMh_acc_surv[itree][isat])
                z_acc.append(self.acc_red[itree][isat])
                M_star.append(jsm_SHMR.lgMs_RP17(self.lgMh_acc_surv[itree][isat], self.acc_red[itree][isat]))
                M_final.append(self.lgMh_final[itree][isat])
                x_final.append(self.fx[itree][isat])
                y_final.append(self.fy[itree][isat])
                z_final.append(self.fz[itree][isat])
                vx_final.append(self.fvx[itree][isat])
                vy_final.append(self.fvy[itree][isat])
                vz_final.append(self.fvz[itree][isat])
                acc_order.append(self.acc_order[itree][isat].astype("int"))
                final_order.append(self.final_order[itree][isat].astype("int"))

        keys = ("sat_id", "tree_id", "Nsub", "z_50", "z_10",
                "M_acc", "z_acc", "M_star", "M_final",
                "R(kpc)", "phi(rad)", "z(kpc)", "VR(kpc/Gyr)", "Vphi(kpc/Gyr)" ,"Vz(kpc/Gyr)",
                "k_acc", "k_final")
        
        data = Table([sat_id, tree_id, Nsub, z_50, z_10,
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


    def write_to_FORTRAN_SAGA(self):
        Nsub = []
        M_acc = []
        z_acc = []
        M_final = []
        x_final = []
        y_final = []
        z_final = []
        acc_order = []
        final_order = []
        vx_final = []
        vy_final = []
        vz_final = []
        SAGA_id = []
        tree_id = []
        sat_id = []

        for isaga in range(self.Nsets):
            for itree in range(self.Nsamp):
                Nsub_i = np.argwhere(~np.isnan(self.acc_surv_lgMh[itree]))[:,0]
                for j, isat in enumerate(Nsub_i):
                    Nsub.append(len(Nsub_i))
                    SAGA_id.append(isaga)
                    tree_id.append(itree+1)
                    sat_id.append(j+1)
                    M_acc.append(self.acc_surv_lgMh[itree][isat])
                    z_acc.append(self.acc_red[itree][isat])
                    M_final.append(self.final_lgMh[itree][isat])
                    x_final.append(self.fx[itree][isat])
                    y_final.append(self.fy[itree][isat])
                    z_final.append(self.fz[itree][isat])
                    vx_final.append(self.fvx[itree][isat])
                    vy_final.append(self.fvy[itree][isat])
                    vz_final.append(self.fvz[itree][isat])
                    acc_order.append(self.acc_order[itree][isat].astype("int"))
                    final_order.append(self.final_order[itree][isat].astype("int"))

        keys = ("sat_id", "tree_id", "SAGA_id", "Nsub",
                "M_acc", "z_acc", "M_final",
                "R(kpc)", "phi(rad)", "z(kpc)", "VR(kpc/Gyr)", "Vphi(kpc/Gyr)" ,"Vz(kpc/Gyr)",
                "k_acc", "k_final") # why are these units weird?
        
        data = Table([sat_id, tree_id, SAGA_id, Nsub,
                      M_acc, z_acc, M_final,
                      x_final, y_final, z_final, vx_final, vy_final, vz_final,
                      acc_order, final_order], names=keys)
        
        print("writing out the subhalo data")
        data.write(self.metadir+"FvdB_MCMC.dat", format="ascii", overwrite=True)
    
        Hnpy = np.load(self.metadir+"host_properties.npz")
        Hkeys = ("lgMh", "z_50", "z_10", "Nsub_total")
        Hdata = Table([Hnpy[:,0], Hnpy[:,1], Hnpy[:,2], Hnpy[:,3]], names=Hkeys)

        print("writing out the host data")
        Hdata.write(self.metadir+"FvdB_hostdata.dat", format="ascii", overwrite=True)
