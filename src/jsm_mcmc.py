import numpy as np
import matplotlib.pyplot as plt
import corner
import galhalo
from IPython.display import display, Math
import matplotlib as mpl
import jsm_stats
from multiprocess import Pool
import emcee
import time
from scipy.stats import ks_2samp
import warnings; warnings.simplefilter('ignore')

##################################################
###    A SIMPLE FUNC TO MULTITHREAD THE MCMC   ###
##################################################

def RUN(theta_0, lnprob, nwalkers, niter, ndim, ncores=8, converge=False):

    p0 = [np.array(theta_0) + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]
    
    with Pool(ncores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        start = time.time()
        sampler.run_mcmc(p0, niter)
        end = time.time()
        multi_time = end - start
        print("Run took {0:.1f} seconds".format(multi_time))

    if converge==True:
        tau = sampler.get_autocorr_time()
        print('$\\alpha$ took', tau[0], 'steps')
        print('$\\delta$ took', tau[1], 'steps')
        print('$\\sigma$ took', tau[2], 'steps')

    return sampler

def lnL_Pnsat(model, data):
    lnL = np.sum(np.log(model[data]))
    if np.isnan(lnL):
        return -np.inf
    else:
        return lnL
    
def lnL_KS(model, data):
    return np.log(ks_2samp(model, data)[1])

# def forward(theta, lgMh, min_mass=6.5):
#     lgMs = galhalo.SHMR_2D_g(lgMh, alpha = theta[0], delta = theta[1], sigma = theta[2], gamma=theta[3])
#     stat = jsm_stats.SatStats(lgMs)
#     stat.Nsat(min_mass)
#     stat.Maxmass()
#     return stat.Pnsat, stat.Msmax, stat.ecdf_MsMax


##################################################
###    FOR TESTING RUNS WITH THE SAME INPUT   ###
##################################################

class test_data:

    def __init__(self, fid_theta:list, mfile:str):
        self.fid_theta = fid_theta
        self.lgMh_mat = np.load(mfile) # need to update this!
        self.lgMh = np.load("../../data/MCMC/test_lgMh.npy")
        self.lgMs = np.load("../../data/MCMC/test_lgMs.npy")
        self.lgMh_models = np.vstack(self.lgMh_mat)
        # temp = np.delete(self.lgMh_mat, SAGA_ind, axis=0) # delete the index used as the data
        # self.lgMh_models = np.vstack(temp) # no longer broken up into SAGA samples

    def get_stats(self, min_mass):
        self.min_mass = min_mass
        self.stat = jsm_stats.SatStats(self.lgMs)
        self.stat.Nsat(self.min_mass, plot=True)
        self.stat.Maxmass(plot=True)

    def get_data_points(self):
        lgMs = self.lgMs.flatten()[self.lgMs.flatten() > 6.5]
        lgMh = self.lgMh.flatten()[self.lgMs.flatten() > 6.5]
        return [lgMh, lgMs]
    

##################################################
###           FOR CREATING MOCK DATA           ###
##################################################

class mock_data:

    def __init__(self, fid_theta:list, SHMR, SAGA_ind:int, mfile:str, zfile:str=None):
        self.fid_theta = fid_theta
        if zfile != None: # if redshift data is provided!
            self.mfile = mfile
            self.zfile = zfile
            self.lgMh_mat = np.load(self.mfile) # load
            self.z_mat = np.load(self.zfile) 
            self.lgMh_data = self.lgMh_mat[SAGA_ind] # select the SAGA index
            self.z_data = self.z_mat[SAGA_ind]
            self.lgMs = SHMR(fid_theta, self.lgMh_data, self.z_data) #convert to Ms
            temp = np.delete(self.lgMh_mat, SAGA_ind, axis=0) # delete the index used as the data
            tempz = np.delete(self.z_mat, SAGA_ind, axis=0)
            self.lgMh_models = np.vstack(temp) # return the rest as models (no longer broken up into SAGA samples)
            self.z_models = np.vstack(tempz)

        else:
            self.mfile = mfile
            self.lgMh_mat = np.load(self.mfile)
            self.lgMh_data = self.lgMh_mat[SAGA_ind]
            self.lgMs = SHMR(fid_theta, self.lgMh_data)
            temp = np.delete(self.lgMh_mat, SAGA_ind, axis=0) 
            self.lgMh_models = np.vstack(temp) 

    def get_stats(self, min_mass):
        self.min_mass = min_mass
        self.stat = jsm_stats.SatStats(self.lgMs)
        self.stat.Nsat(self.min_mass, plot=True)
        self.stat.Maxmass(plot=True)

    def get_data_points(self):
        lgMs = self.lgMs.flatten()[self.lgMs.flatten() > 6.5]
        lgMh = self.lgMh_data.flatten()[self.lgMs.flatten() > 6.5]
        return np.array([lgMh, lgMs])

class models: 

    def __init__(self, theta:list, SHMR, lgMh_models): # do the same thing with z_acc
        self.theta = theta
        self.lgMh_models = lgMh_models
        self.lgMs = SHMR(theta, self.lgMh_models)

    def get_stats(self, min_mass):
        self.min_mass = min_mass
        self.stat = jsm_stats.SatStats(self.lgMs)
        self.stat.Nsat(self.min_mass)
        self.stat.Maxmass()


##################################################
###     TO INTERFACE WITH THE MCMC OUTPUT      ###
##################################################

class inspect_run:

    def __init__(self, sampler, fid_theta:list, labels:list, priors:list):
        self.truths = fid_theta
        self.labels = labels
        self.priors = priors
        self.ndim = len(self.priors)
        self.samples = sampler.get_chain()
        self.chisq = sampler.get_log_prob()
        self.flatchain = sampler.flatchain
        self.last_samp = sampler.get_last_sample().coords
        self.flatchisq = sampler.get_last_sample().log_prob*(-2)

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
        
    def corner_plot(self, burn=None, zoom=False):
        
        if burn!=None:
            nsteps = self.samples.shape[0]
            ssteps = nsteps - burn
            s = self.samples[ssteps:nsteps,:,:].shape
            self.burn = self.samples[ssteps:nsteps,:,:].reshape(s[0] * s[1], s[2]) 
            if zoom==True:
                fig = corner.corner(self.burn, show_titles=True, labels=self.labels, truths=self.truths, quantiles=[0.15, 0.5, 0.85], plot_datapoints=False)
            elif zoom==False:
                fig = corner.corner(self.burn, show_titles=True, labels=self.labels, truths=self.truths, range=self.priors , quantiles=[0.15, 0.5, 0.85], plot_datapoints=False)
        else:
            if zoom==True:
                fig = corner.corner(self.last_samp, show_titles=True, labels=self.labels, truths=self.truths, quantiles=[0.15, 0.5, 0.85], plot_datapoints=False)
            elif zoom==False:
                fig = corner.corner(self.last_samp, show_titles=True, labels=self.labels, truths=self.truths, range=self.priors , quantiles=[0.15, 0.5, 0.85], plot_datapoints=False)
        plt.show()
            

    def SHMR_plot(self, data, SHMR, show_scatter=False):

        self.halo_masses = np.log10(np.logspace(6, 13, 100)) # just for the model

        SHMR_mat = np.zeros(shape=(self.last_samp.shape[0], self.halo_masses.shape[0]))
        if show_scatter==True:
            self.fid_Ms = SHMR(self.truths, self.halo_masses)
            for i,val in enumerate(self.last_samp):
                lgMs = SHMR(val, self.halo_masses)
                SHMR_mat[i] = lgMs
        else:
            temp = self.truths
            #temp[2], temp[3] = 0, 0 # to not show the scatter!
            temp[2] =  0 # to not show the scatter!
            self.fid_Ms = SHMR(temp, self.halo_masses)
            for i,val in enumerate(self.last_samp):         
                val[2] =  0 # to not show the scatter!
                #val[2], val[3] = 0, 0 # to not show the scatter!
                lgMs = SHMR(val, self.halo_masses)
                SHMR_mat[i] = lgMs

        plt.figure(figsize=(10, 8))
        for i in SHMR_mat:
            plt.plot(self.halo_masses, i, alpha=0.1, color="grey")
        plt.plot(self.halo_masses, galhalo.lgMs_B13(self.halo_masses), color="red", label="Behroozi et al. 2013", ls="--", lw=2)
        plt.plot(self.halo_masses, galhalo.lgMs_RP17(self.halo_masses), color="navy", label="Rodriguez-Puebla et al. 2017", ls="--", lw=2)
        plt.plot(self.halo_masses, self.fid_Ms, color="cornflowerblue", label=str(self.truths), lw=2)
        plt.axhline(6.5, ls=":", color="green")

        dp = data.get_data_points()
        plt.scatter(dp[0], dp[1], marker="*", color="black")

        plt.ylim(4,11)
        plt.xlim(7.5,12)
        plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
        plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
        plt.legend(fontsize=12)
        plt.show()
    
    def stat_plot(self, data, forward):

        Ns, Ms, _ = forward(self.last_samp[0])
        Pnsat_mat = np.zeros(shape=(self.last_samp.shape[0], Ns.shape[0]))
        Msmax_mat = np.zeros(shape=(self.last_samp.shape[0], Ms.shape[0]))
        Msmaxe_mat = np.zeros(shape=(self.last_samp.shape[0], Ms.shape[0]))

        for i, theta in enumerate(self.last_samp):
            tPnsat, tMsmax, tecdf_MsMax = forward(theta)
            Pnsat_mat[i] = tPnsat
            Msmax_mat[i] = tMsmax      
            Msmaxe_mat[i] = tecdf_MsMax

        for i in Pnsat_mat:
            plt.plot(np.arange(i.shape[0]),i, color="grey", alpha=0.1)
        plt.plot(np.arange(data.stat.Pnsat.shape[0]),data.stat.Pnsat,marker="o", color="black")
        plt.xlabel("number of satellites > $10^{"+str(6.5)+"} \mathrm{M_{\odot}}$", fontsize=15)
        plt.ylabel("PDF", fontsize=15)
        plt.xlim(0,20)
        plt.show()

        for i, val in enumerate(Msmax_mat):
            plt.plot(val, Msmaxe_mat[i], color="grey", alpha=0.1)
        plt.plot(data.stat.Msmax, data.stat.ecdf_MsMax, color="black")
        plt.xlabel("stellar mass of most massive satellite ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
        plt.ylabel("CDF", fontsize=15)
        plt.show()

    def best_fit_values(self):
        val = []
        for i in range(self.ndim):
            mcmc = np.percentile(self.last_samp[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], self.labels[i])
            display(Math(txt))
            val.append([mcmc[1], q[0], q[1]])
        return val         

    def save_sample(self, path):
        np.save(path, self.samples)


##########################################################
### the following routines are for the old MCMC method ###
##########################################################

    # def chi_square_plot(self):

    #     mask = self.chisq > 0

    #     norm = plt.Normalize()
    #     colors = plt.cm.viridis_r(norm(self.chisq[mask]))
    #     cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis_r)

    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True,figsize=(12,6))
    #     ax1.scatter(self.last_samp[:,0][mask], self.chisq[mask], marker=".")
    #     ax1.set_xlim(0.5, 3.5)
    #     ax1.set_xlabel("$\\alpha$", fontsize=12)

    #     ax2.scatter(self.last_samp[:,1][mask], self.chisq[mask], marker=".")
    #     ax2.set_xlim(-2, 0.5)
    #     ax2.set_xlabel("$\\delta$", fontsize=12)

    #     ax3.scatter(self.last_samp[:,2][mask], self.chisq[mask], marker=".")
    #     ax3.set_xlim(0,4)
    #     ax3.set_xlabel("$\\sigma$", fontsize=12)
    #     ax1.set_ylabel("$\\chi^2$", fontsize=12)
    #     ax1.set_yscale("log")
    #     plt.show()

    # def measure_truth(self, mock_data_ind, plot=False):
    #     self.mock_data_ind = mock_data_ind
    #     #deleting the index from the matrix so we dont use the data in the mcmc
    #     self.maxsatmass_CDF_temp = np.delete(self.maxsatmass_CDF_mat, self.mock_data_ind, axis=0)
    #     self.satfreq_PDF_temp = np.delete(self.satfreq_PDF_mat, self.mock_data_ind, axis=0)
    #     #defining the truth to be the average
    #     self.data_CDF = np.average(self.maxsatmass_CDF_temp, axis=0)
    #     self.data_PDF = np.average(self.satfreq_PDF_temp, axis=0)
    #     self.error_CDF = np.std(self.maxsatmass_CDF_temp, axis=0)
    #     self.error_PDF = np.std(self.satfreq_PDF_temp, axis=0)

    #     if plot ==True:
    #         plt.errorbar(self.maxsatmass_bincenters, self.data_CDF, yerr=self.error_CDF)
    #         plt.xlabel("stellar mass of most massive satellite ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
    #         plt.ylabel("CDF", fontsize=15)
    #         plt.show()

    #         plt.errorbar(self.satfreq_bincenters, self.data_PDF, yerr=self.error_PDF)
    #         plt.xlabel("stellar mass of most massive satellite ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
    #         plt.xlabel("number of satellites > $10^{6.5} \mathrm{M_{\odot}}$", fontsize=15)
    #         plt.ylabel("PDF", fontsize=15)
    #         plt.show()

    #     self.model_lgMh = self.lgMh_mat[mock_data_ind]

# def fid_MODEL(lgMh_data, fid_theta, mass_list, return_counts=False):

#     """_summary_
#     the main model! goes from a SHMF mass to a CSMF and measures some statistics!

#     Returns:
#         np.ndarray: 1D model array populated with the statistics!
#     """
    
#     alpha, delta, sigma = fid_theta

#     lgMs_2D = galhalo.SHMR(lgMh_data, alpha, delta, sigma) # will be a 3D array if sigma is non zero
    
#     counts = np.apply_along_axis(jsm_halopull.cumulative, 1, lgMs_2D)
#     quant = np.percentile(counts, np.array([5, 50, 95]), axis=0, method="closest_observation") # median and scatter

#     mass_ind = jsm_halopull.find_nearest(mass_list)

#     model = [] # counts at the mass indicies
#     for i in mass_ind:
#         model.append(quant[2, i])
#         model.append(quant[1, i])
#         model.append(quant[0, i])

#     if return_counts == True:
#         return np.array(model), quant
#     else:
#         return np.array(model)


# class prep_self:

#     """
#     A class instance to steamline this same mcmc analysis to other data sets!
#     """

#     def __init__(self, datadir:str, Nsamp:int=100, fid_theta:list=[1.85, 0.2, 0.3], mass_bins:np.ndarray=np.linspace(4,11,45), mass_list:list=[6.5,7.,7.5]):

#         self.datadir = datadir
#         self.fid_theta = fid_theta
#         self.Nsamp = Nsamp
#         self.mass_bins = mass_bins
#         self.mass_list = mass_list
        

#     def create_SAGA_samples(self, pick=None):

#         massmat = jsm_halopull.MassMat(self.datadir+"acc_surv_mass.npy")
#         massmat.prep_data()
#         Nsets = int(massmat.lgMh.shape[0]/self.Nsamp) #dividing by the number of samples
#         print("dividing your sample into", Nsets-1, "sets")
#         set_ind = np.arange(0,Nsets)*self.Nsamp

#         D_mat = np.zeros(shape=(Nsets-1, len(self.mass_list*3)))
#         count_mat = np.zeros(shape=(Nsets-1, 3, self.mass_bins.shape[0]))
#         lgMh_mat = np.zeros(shape=(Nsets-1, self.Nsamp, massmat.lgMh.shape[1]))
#         for i in range(Nsets-1):
#             lgMh_i = massmat.lgMh[set_ind[i]:set_ind[i+1]]
#             lgMh_mat[i] = lgMh_i
#             D_mat[i], count_mat[i] = fid_MODEL(lgMh_i, self.fid_theta, self.mass_list, return_counts=True)

#         self.D_mat = D_mat
#         self.count_mat = count_mat
#         self.sampave = np.average(self.D_mat,axis=0)
#         self.covariance = np.cov(self.D_mat, rowvar=False)
#         self.sampstd = np.sqrt(np.diag(self.covariance))
#         self.inv_covar = np.linalg.inv(self.covariance)
#         #print("determinant should equal", np.linalg.det(np.matmul(self.covariance, self.inv_covar)))

#         if pick != None:
#             self.pick_samp = pick
#         else:
#             self.pick_samp = np.random.randint(Nsets-1)
            
#         print("chose ID "+str(self.pick_samp)+" as the random sample to use as the real data!")

#         self.lgMh_real = lgMh_mat[self.pick_samp]
#         self.lgMhs = np.delete(lgMh_mat, self.pick_samp, axis=0)
#         self.D = self.D_mat[self.pick_samp]
#         self.counts = self.count_mat[self.pick_samp]


#     def initialize(self, SAGA_ID, chi_dim=20):

#         self.lgMh = self.lgMhs[SAGA_ID]

#         self.alpha_space = np.linspace(1,3,chi_dim)
#         self.delta_space = np.linspace(-1,3,chi_dim)
#         self.sigma_space = np.linspace(0,3,chi_dim)

#         chi_mat = np.zeros(shape=(chi_dim,chi_dim,chi_dim))

#         for i, aval in enumerate(self.alpha_space):
#             for j, dval in enumerate(self.delta_space):
#                 for k, sval in enumerate(self.sigma_space):
#                     model = fid_MODEL(self.lgMh, [aval, dval, sval], self.mass_list)
#                     X = model - self.D
#                     X_vec = np.expand_dims(X, axis=1)
#                     chi_mat[i,j,k] = X_vec.transpose().dot(self.inv_covar).dot(X_vec)

#         self.chi_mat = chi_mat
#         ai, di, si = np.where(chi_mat == np.min(chi_mat))
#         theta_0 = [self.alpha_space[ai][0], self.delta_space[di][0], self.sigma_space[si][0]]
#         self.theta_0 = np.array(theta_0)

#     def plot_real_data(self):

#         plt.figure(figsize=(8, 8))
#         plt.plot(self.mass_bins, self.counts[1], label="median", color="black")
#         plt.fill_between(self.mass_bins, y1=self.counts[0], y2=self.counts[2], alpha=0.2, color="grey", label="5% - 95%")
#         # plt.errorbar([self.mass_list[0]]*3, self.D[0:3], self.sampstd[0:3], fmt=".", color="red")
#         # plt.errorbar([self.mass_list[1]]*3, self.D[3:6], self.sampstd[3:6], fmt=".", color="red")
#         # plt.errorbar([self.mass_list[2]]*3, self.D[6:9], self.sampstd[6:9], fmt=".", color="red")
#         plt.yscale("log")
#         #plt.grid(alpha=0.4)
#         plt.xlabel("log m$_{stellar}$ (M$_\odot$)", fontsize=15)
#         plt.ylabel("log N (> m$_{stellar}$)", fontsize=15)
#         plt.legend()
#         plt.show()