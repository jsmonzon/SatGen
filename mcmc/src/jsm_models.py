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


class SAGA_sample:

    def __init__(self, Nsamp, SHMR, truths:list, SAGA_ind:int, savedir:str, read_red=False):
        self.truths = truths
        self.savedir = savedir
        sample = jsm_halopull.MassMat("../../../data/MW-analog/meta_data_psi4/", Nsamp=Nsamp, save=False, plot=False)
        self.lgMh_data = sample.acc_surv_lgMh_mat[SAGA_ind] # select the SAGA index
        self.zacc_data = sample.acc_red_mat[SAGA_ind]
        if read_red == True:
            self.lgMs_data = SHMR(truths, self.lgMh_data, self.zacc_data)
        else:
            self.lgMs_data = SHMR(truths, self.lgMh_data)

    def get_data_points(self, plot=True):
        self.lgMh_flat = self.lgMh_data.flatten()
        self.lgMs_flat = self.lgMs_data.flatten()
        if plot==True:
            plt.scatter(self.lgMh_flat, self.lgMs_flat, marker=".")
            plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
            plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
    
    def save_data(self):
        np.save(self.savedir+"mock_data.npy", np.array([self.lgMh_data, self.lgMs_data, self.zacc_data]))

class mock_data:

    def __init__(self, SHMR, truths:list, SAGA_ind:int, meta_path:str, savedir:str, read_red=False):
        self.truths = truths
        self.mfile = meta_path
        self.savedir = savedir

        models = np.load(self.mfile+"models.npz")
        self.lgMh_data = models["mass"][SAGA_ind] # select the SAGA index
        self.zacc_data = models["redshift"][SAGA_ind]

        if read_red == True:
            self.lgMs_data = SHMR(truths, self.lgMh_data, self.zacc_data)
        else:
            self.lgMs_data = SHMR(truths, self.lgMh_data)

    def get_data_points(self, plot=True):
        self.lgMh_flat = self.lgMh_data.flatten()
        self.lgMs_flat = self.lgMs_data.flatten()
        if plot==True:
            plt.scatter(self.lgMh_flat, self.lgMs_flat, marker=".")
            plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
            plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
    
    def save_data(self):
        np.save(self.savedir+"mock_data.npy", np.array([self.lgMh_data, self.lgMs_data, self.zacc_data]))

class init_data:

    def __init__(self, truths:list, dfile:str):
        self.truths = truths
        self.lgMh = np.load(dfile)[0]
        self.lgMs = np.load(dfile)[1]

    def get_stats(self, min_mass):
        self.min_mass = min_mass
        self.stat = jsm_stats.SatStats(self.lgMs, self.min_mass)

    def get_data_points(self, plot=True):
        self.lgMs_flat = self.lgMs.flatten()[self.lgMs.flatten() > self.min_mass]
        self.lgMh_flat = self.lgMh.flatten()[self.lgMs.flatten() > self.min_mass]
        if plot==True:
            plt.scatter(self.lgMh_flat, self.lgMs_flat)
            plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
            plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
    

class load_models:

    def __init__(self, mfile:str):    
        self.mfile = mfile
        models = np.load(mfile+"models.npz")
        self.lgMh_models = np.vstack(models["mass"])
        self.zacc_models = np.vstack(models["redshift"])

    def push_theta(self, theta:list, SHMR, min_mass):
        self.theta = theta
        self.min_mass = min_mass
        #self.Ntree = Ntree

        if theta[5] == 0:
            self.lgMs = SHMR(theta, self.lgMh_models)
        else:
            self.lgMs = SHMR(theta, self.lgMh_models, self.zacc_models)

        self.stat = jsm_stats.SatStats(self.lgMs, self.min_mass)
        # self.lgMs_split = np.array(np.split(self.lgMs, self.Ntree, axis=0))
        # self.correlations = np.array([jsm_stats.SatStats(i, self.min_mass).r for i in self.lgMs_split])