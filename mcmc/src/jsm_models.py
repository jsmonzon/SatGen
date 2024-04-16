import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import jsm_stats
import warnings; warnings.simplefilter('ignore')
import jsm_halopull
import jsm_SHMR

class MOCK_DATA:

    def __init__(self, fid_theta:list, meta_path:str, savedir:str, SAGA_ind, Nsamples:int=1, redshift_depandance=False):
        self.fid_theta = fid_theta
        self.mfile = meta_path
        self.savedir = savedir
        self.Nsamples = Nsamples
        
        models = np.load(self.mfile+"models.npz")

        if type(SAGA_ind) == int:
            print("selecting a single SAGA sample")
            self.lgMh_data = models["mass"][SAGA_ind] # select the SAGA index
            self.zacc_data = models["redshift"][SAGA_ind]
            print(self.lgMh_data.shape)

        elif type(SAGA_ind) == list:
            print("selecting more than one SAGA sample")
            self.lgMh_data = np.vstack(models["mass"][SAGA_ind[0]:SAGA_ind[1]]) # select the SAGA indices
            self.zacc_data = np.vstack(models["redshift"][SAGA_ind[0]:SAGA_ind[1]])
            print(self.lgMh_data.shape)

        elif type(SAGA_ind) == float:
            print("selecting less than one SAGA sample")
            self.lgMh_data = np.vstack(models["mass"])[0:int(SAGA_ind)] 
            self.zacc_data = np.vstack(models["redshift"])[0:int(SAGA_ind)]
            print(self.lgMh_data.shape)

        if redshift_depandance == True:
            self.lgMs_data = jsm_SHMR.general(fid_theta, self.lgMh_data, self.zacc_data, self.Nsamples)
        else:
            self.lgMs_data = jsm_SHMR.general(fid_theta, self.lgMh_data, 0, self.Nsamples)

    def get_data_points(self, plot=True):
        self.lgMh_flat = self.lgMh_data.flatten()
        self.lgMs_flat = self.lgMs_data.flatten()
        if plot==True:
            plt.scatter(self.lgMh_flat, self.lgMs_flat, marker=".")
            plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
            plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
            plt.axhline(6.5, ls="--")
            plt.show()
    
    def save_data(self):
        np.save(self.savedir+"mock_data.npy", np.array([self.lgMh_data, self.lgMs_data, self.zacc_data]))

class INIT_DATA:

    def __init__(self, fid_theta:list, dfile:str):
        self.fid_theta = fid_theta
        self.lgMh = np.load(dfile)[0]
        self.lgMs = np.load(dfile)[1]

    def get_stats(self, min_mass):
        self.min_mass = min_mass
        self.stat = jsm_stats.SatStats_D(self.lgMs, self.min_mass)
        self.lgMs_flat = self.lgMs.flatten()[self.lgMs.flatten() > self.min_mass]
        self.lgMh_flat = self.lgMh.flatten()[self.lgMs.flatten() > self.min_mass]

    def get_nad_stats(self, min_mass, N_bin):
        self.min_mass = min_mass
        self.stat = jsm_stats.SatStats_NAD_D(self.lgMs, min_mass=min_mass, N_bin=N_bin)
        self.lgMs_flat = self.lgMs.flatten()[self.lgMs.flatten() > self.min_mass]
        self.lgMh_flat = self.lgMh.flatten()[self.lgMs.flatten() > self.min_mass]

    def plot_SHMR(self):
        plt.figure(figsize=(6, 6))
        plt.scatter(self.lgMh_flat, self.lgMs_flat, marker="*", color="black")
        plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
        plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
        plt.xlim(8.5, 12)
        plt.ylim(6.2, 10.5)
        plt.show()

    def plot_SMF(self):
        plt.figure(figsize=(6, 6))

        mass = []
        N_gtr = []
        for i in self.lgMs:
            example = np.sort(i)
            temp = example[~np.isnan(example)]
            clipped_mass = temp[temp > self.min_mass]
            clipped_N = np.arange(len(clipped_mass)-1, -1, -1)

            plt.plot(clipped_mass, clipped_N, color="black", alpha=0.4)
            mass.append(clipped_mass)
            N_gtr.append(clipped_N)
        plt.xlabel("log M$_{*}$ (M$_\odot$)", fontsize=15)
        plt.ylabel("N (> M$_{*}$)", fontsize=15)
        plt.ylim(-0.1, 30)
        plt.show() 

        return mass, N_gtr


class LOAD_MODELS:

    def __init__(self, mfile:str, Nsamples=1):    
        self.mfile = mfile
        models = np.load(mfile+"models.npz")
        self.lgMh_mat = models["mass"]
        self.zacc_mat = models["redshift"]
        self.lgMh_models = np.vstack(self.lgMh_mat)
        self.zacc_models = np.vstack(self.zacc_mat)
        self.Nsamples = Nsamples

    def get_stats(self, theta:list, min_mass):
        self.theta = theta
        self.min_mass = min_mass

        if theta[5] == 0:
            self.lgMs = jsm_SHMR.general(theta, self.lgMh_models, 0, self.Nsamples)
        else:
            self.lgMs = jsm_SHMR.general(theta, self.lgMh_models, self.zacc_models, self.Nsamples)

        self.stat = jsm_stats.SatStats_M(self.lgMs, self.min_mass)

    def get_nad_stats(self, theta:list, min_mass, N_bin):
        self.theta = theta
        self.min_mass = min_mass
        self.N_bin = N_bin

        if theta[5] == 0:
            self.lgMs = np.apply_along_axis(jsm_SHMR.general, 0, self.theta, self.lgMh_mat, 0, 1) # converting the data using the same value of theta fid!
        else:
            self.lgMs = np.apply_along_axis(jsm_SHMR.general, 0, self.theta, self.lgMh_mat, self.zacc_mat["redshift"], 1) # converting the data using the same value of theta fid!

        self.stat = jsm_stats.SatStats_NAD_M(self.lgMs, min_mass=6.5, N_bin=N_bin)
        # self.lgMs_split = np.array(np.split(self.lgMs, self.Ntree, axis=0))
        # self.correlations = np.array([jsm_stats.SatStats(i, self.min_mass).r for i in self.lgMs_split])




# class SAGA_sample:

#     def __init__(self, Nsamp, fid_theta:list, SAGA_ind:int, savedir:str, redshift_depandance=False):
#         self.fid_theta = fid_theta
#         self.savedir = savedir
#         sample = jsm_halopull.MassMat("../../../data/MW-analog/meta_data_psi4/", Nsamp=Nsamp, save=False, plot=False)
#         self.lgMh_data = sample.acc_surv_lgMh_mat[SAGA_ind] # select the SAGA index
#         self.zacc_data = sample.acc_red_mat[SAGA_ind]
#         if redshift_depandance == True:
#             self.lgMs_data = jsm_SHMR.general(fid_theta, self.lgMh_data, self.zacc_data)
#         else:
#             self.lgMs_data = jsm_SHMR.general(fid_theta, self.lgMh_data)

#     def get_data_points(self, plot=True):
#         self.lgMh_flat = self.lgMh_data.flatten()
#         self.lgMs_flat = self.lgMs_data.flatten()
#         if plot==True:
#             plt.scatter(self.lgMh_flat, self.lgMs_flat, marker=".")
#             plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
#             plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
    
#     def save_data(self):
#         np.save(self.savedir+"mock_data.npy", np.array([self.lgMh_data, self.lgMs_data, self.zacc_data]))