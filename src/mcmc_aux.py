import numpy as np
import matplotlib.pyplot as plt
import corner
import jsm_halopull
import galhalo
from IPython.display import display, Math
import matplotlib as mpl


import warnings; warnings.simplefilter('ignore')

def fid_MODEL(lgMh_data, fid_theta, mass_list, return_counts=False):

    """_summary_
    the main model! goes from a SHMF mass to a CSMF and measures some statistics!

    Returns:
        np.ndarray: 1D model array populated with the statistics!
    """
    
    alpha, delta, sigma = fid_theta

    lgMs_2D = galhalo.SHMR(lgMh_data, alpha, delta, sigma) # will be a 3D array if sigma is non zero
    
    counts = np.apply_along_axis(jsm_halopull.cumulative, 1, lgMs_2D)
    quant = np.percentile(counts, np.array([5, 50, 95]), axis=0, method="closest_observation") # median and scatter

    mass_ind = jsm_halopull.find_nearest(mass_list)

    model = [] # counts at the mass indicies
    for i in mass_ind:
        model.append(quant[2, i])
        model.append(quant[1, i])
        model.append(quant[0, i])

    if return_counts == True:
        return np.array(model), quant
    else:
        return np.array(model)


class prep_run:

    """
    A class instance to steamline this same mcmc analysis to other data sets!
    """

    def __init__(self, datadir:str, Nsamp:int=100, fid_theta:list=[1.85, 0.2, 0.3], mass_bins:np.ndarray=np.linspace(4,11,45), mass_list:list=[6.5,7.,7.5]):

        self.datadir = datadir
        self.fid_theta = fid_theta
        self.Nsamp = Nsamp
        self.mass_bins = mass_bins
        self.mass_list = mass_list
        

    def create_SAGA_samples(self, pick=None):

        massmat = jsm_halopull.MassMat(self.datadir+"acc_surv_mass.npy")
        massmat.prep_data()
        Nsets = int(massmat.lgMh.shape[0]/self.Nsamp) #dividing by the number of samples
        print("dividing your sample into", Nsets-1, "sets")
        set_ind = np.arange(0,Nsets)*self.Nsamp

        D_mat = np.zeros(shape=(Nsets-1, len(self.mass_list*3)))
        count_mat = np.zeros(shape=(Nsets-1, 3, self.mass_bins.shape[0]))
        lgMh_mat = np.zeros(shape=(Nsets-1, self.Nsamp, massmat.lgMh.shape[1]))
        for i in range(Nsets-1):
            lgMh_i = massmat.lgMh[set_ind[i]:set_ind[i+1]]
            lgMh_mat[i] = lgMh_i
            D_mat[i], count_mat[i] = fid_MODEL(lgMh_i, self.fid_theta, self.mass_list, return_counts=True)

        self.D_mat = D_mat
        self.count_mat = count_mat
        self.sampave = np.average(self.D_mat,axis=0)
        self.covariance = np.cov(self.D_mat, rowvar=False)
        self.sampstd = np.sqrt(np.diag(self.covariance))
        self.inv_covar = np.linalg.inv(self.covariance)
        #print("determinant should equal", np.linalg.det(np.matmul(self.covariance, self.inv_covar)))

        if pick != None:
            self.pick_samp = pick
        else:
            self.pick_samp = np.random.randint(Nsets-1)
            
        print("chose ID "+str(self.pick_samp)+" as the random sample to use as the real data!")

        self.lgMh_real = lgMh_mat[self.pick_samp]
        self.lgMhs = np.delete(lgMh_mat, self.pick_samp, axis=0)
        self.D = self.D_mat[self.pick_samp]
        self.counts = self.count_mat[self.pick_samp]


    def initialize(self, SAGA_ID, chi_dim=20):

        self.lgMh = self.lgMhs[SAGA_ID]

        self.alpha_space = np.linspace(1,3,chi_dim)
        self.delta_space = np.linspace(-1,3,chi_dim)
        self.sigma_space = np.linspace(0,3,chi_dim)

        chi_mat = np.zeros(shape=(chi_dim,chi_dim,chi_dim))

        for i, aval in enumerate(self.alpha_space):
            for j, dval in enumerate(self.delta_space):
                for k, sval in enumerate(self.sigma_space):
                    model = fid_MODEL(self.lgMh, [aval, dval, sval], self.mass_list)
                    X = model - self.D
                    X_vec = np.expand_dims(X, axis=1)
                    chi_mat[i,j,k] = X_vec.transpose().dot(self.inv_covar).dot(X_vec)

        self.chi_mat = chi_mat
        ai, di, si = np.where(chi_mat == np.min(chi_mat))
        theta_0 = [self.alpha_space[ai][0], self.delta_space[di][0], self.sigma_space[si][0]]
        self.theta_0 = np.array(theta_0)

    def plot_real_data(self):

        plt.figure(figsize=(8, 8))
        plt.plot(self.mass_bins, self.counts[1], label="median", color="black")
        plt.fill_between(self.mass_bins, y1=self.counts[0], y2=self.counts[2], alpha=0.2, color="grey", label="5% - 95%")
        # plt.errorbar([self.mass_list[0]]*3, self.D[0:3], self.sampstd[0:3], fmt=".", color="red")
        # plt.errorbar([self.mass_list[1]]*3, self.D[3:6], self.sampstd[3:6], fmt=".", color="red")
        # plt.errorbar([self.mass_list[2]]*3, self.D[6:9], self.sampstd[6:9], fmt=".", color="red")
        plt.yscale("log")
        #plt.grid(alpha=0.4)
        plt.xlabel("log m$_{stellar}$ (M$_\odot$)", fontsize=15)
        plt.ylabel("log N (> m$_{stellar}$)", fontsize=15)
        plt.legend()
        plt.show()

class inspect_run:

    def __init__(self, sampler, fid_theta:list):
        self.samples = sampler.get_chain()
        self.flatchain = sampler.flatchain
        self.last_samp = sampler.get_last_sample().coords
        self.chisq = sampler.get_last_sample().log_prob*(-2)
        self.truths = fid_theta
        self.labels = ['$\\alpha$','$\\delta$','$\\sigma$']
        self.ndim = len(self.labels)
        self.priors = [(0.5, 3.5), (-0.5, 3.5), (0, 4)]


    def chain_plot(self):
        if self.samples.shape[1] > 500:
            a = 0.01
        else:
            a = 0.1

        fig, axes = plt.subplots(self.ndim, figsize=(10, 7), sharex=True)
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(self.samples[:, :, i], "k", alpha=a)
            ax.set_xlim(0, len(self.samples))
            ax.set_ylabel(self.labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.show()

    def best_fit_values(self):
        self.labels = ['alpha','delta','sigma']
        val = []
        for i in range(self.ndim):
            mcmc = np.percentile(self.last_samp[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], self.labels[i])
            display(Math(txt))
            val.append([mcmc[1], q[0], q[1]])
        return val


        
    def corner_plot(self, stack=False):
        
        if stack==True:
            fig = corner.corner(self.flatchain, show_titles=True, labels=self.labels, truths=self.truths,
                            range=self.priors, quantiles=[0.16, 0.5, 0.84], plot_datapoints=False)
            plt.show()

        else:
            fig = corner.corner(self.last_samp, show_titles=True, labels=self.labels, truths=self.truths,
                            range=self.priors, quantiles=[0.16, 0.5, 0.84], plot_datapoints=False)
            plt.show()
            

    def SHMR_plot(self):

        self.halo_masses = np.log10(np.logspace(6, 13, 100)) # just for the model

        SHMR_mat = np.zeros(shape=(self.last_samp.shape[0], self.halo_masses.shape[0]))
        for i,val in enumerate(self.last_samp):
            alpha_i, delta_i, sigma_i = val
            lgMs = galhalo.SHMR(self.halo_masses, alpha_i, delta_i, sigma_i)
            SHMR_mat[i] = lgMs

        self.ave_samp = np.average(SHMR_mat, axis=0)
        self.std_samp = np.std(SHMR_mat, axis=0)

        self.fid_Ms = galhalo.master_SHMR_1D(self.halo_masses, alpha=self.truths[0], delta=self.truths[1], sigma=self.truths[2], N_samples=10000)
        self.ave_fid = np.average(self.fid_Ms,axis=0)
        self.std_fid  = np.std(self.fid_Ms,axis=0)
        self.fid_label = "Fiducial: $\\alpha$="+str(self.truths[0])+", $\\delta$="+str(self.truths[1])+", $\\sigma$="+str(self.truths[2])

        plt.figure(figsize=(8, 8))
        plt.fill_between(self.halo_masses, y1=self.ave_samp + self.std_samp, y2=self.ave_samp - self.std_samp, alpha=0.3, color="grey")
        plt.fill_between(self.halo_masses, y1=self.ave_samp + 2*self.std_samp, y2=self.ave_samp - 2*self.std_samp, alpha=0.2, color="grey")
        plt.fill_between(self.halo_masses, y1=self.ave_samp + 3*self.std_samp, y2=self.ave_samp - 3*self.std_samp, alpha=0.1, color="grey")

        plt.plot(self.halo_masses, galhalo.lgMs_B13(self.halo_masses), color="red", label="Behroozi et al. 2013", ls="--")
        plt.plot(self.halo_masses, galhalo.lgMs_RP17(self.halo_masses), color="navy", label="Rodriguez-Puebla et al. 2017", ls="--")
        plt.plot(self.halo_masses, self.ave_fid, color="green", label=self.fid_label)
        plt.fill_between(self.halo_masses, y1=self.ave_fid + self.std_fid, y2=self.ave_fid - self.std_fid, color="green", alpha=0.5)
   
        plt.ylim(0,11)
        plt.ylabel("m$_{stellar}$ (M$_\odot$)", fontsize=15)
        plt.xlabel("m$_{halo}$ (M$_\odot$)", fontsize=15)
        plt.legend(fontsize=12)
        plt.xlim(6,12)
        plt.show()

    def chi_square_plot(self):

        mask = self.chisq > 0

        norm = plt.Normalize()
        colors = plt.cm.viridis_r(norm(self.chisq[mask]))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis_r)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True,figsize=(12,6))
        ax1.scatter(self.last_samp[:,0][mask], self.chisq[mask], marker=".")
        ax1.set_xlim(0.5, 3.5)
        ax1.set_xlabel("$\\alpha$", fontsize=12)

        ax2.scatter(self.last_samp[:,1][mask], self.chisq[mask], marker=".")
        ax2.set_xlim(-0.5, 3.5)
        ax2.set_xlabel("$\\delta$", fontsize=12)

        ax3.scatter(self.last_samp[:,2][mask], self.chisq[mask], marker=".")
        ax3.set_xlim(0,4)
        ax3.set_xlabel("$\\sigma$", fontsize=12)
        ax1.set_ylabel("$\\chi^2$", fontsize=12)
        plt.show()
