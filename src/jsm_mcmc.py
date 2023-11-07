import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import corner
#import galhalo
#from IPython.display import display, Math
import jsm_stats
import jsm_SHMR
from multiprocess import Pool
import emcee
import time
from scipy.stats import ks_2samp
import warnings; warnings.simplefilter('ignore')

##################################################
###    A SIMPLE FUNC TO MULTITHREAD THE MCMC   ###
##################################################

def RUN(theta_0, lnprob, nwalkers, niter, ndim, ncores=8, a_stretch=2.0):

    p0 = [np.array(theta_0) + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]
    
    with Pool(ncores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, moves=emcee.moves.StretchMove(a=a_stretch))
        start = time.time()
        sampler.run_mcmc(p0, niter)
        end = time.time()
        multi_time = end - start
        print("Run took {0:.1f} seconds".format(multi_time))

    return sampler

def lnL_Pnsat(model, data):
    lnL = np.sum(np.log(model[data]))
    if np.isnan(lnL):
        return -1e8
    else:
        return lnL
    
def lnL_KS(model, data):
    return np.log(ks_2samp(model, data)[1])

def good_guess(lnprob, priors, chidim=5):

    chi_mat = np.zeros(shape=(chidim, chidim, chidim, chidim))
    a1s = np.linspace(priors[0][0], priors[0][1], chidim)
    a2s = np.linspace(priors[1][0], priors[1][1], chidim)
    a3s = np.linspace(priors[2][0], priors[2][1], chidim)
    a4s = np.linspace(priors[3][0], priors[3][1], chidim)

    for i, a1val in enumerate(a1s):
        for j, a2val in enumerate(a2s):
            for k, a3val in enumerate(a3s):
                for l, a4val in enumerate(a4s):
                    chi_mat[i,j,k,l] = -2*lnprob([a1val, a2val, a3val, a4val])

    a1min, a2min, a3min, a4min = np.where(chi_mat == np.min(chi_mat))
    return [a1s[a1min][0], a2s[a2min][0], a3s[a3min][0], a4s[a4min][0]]



##################################################
###                MORE GENERALIZED            ###
##################################################

class init_data:

    def __init__(self, truths:list, dfile:str):
        self.truths = truths
        self.lgMh = np.load(dfile)[0]
        self.lgMs = np.load(dfile)[1]

    def get_stats(self, min_mass):
        self.min_mass = min_mass
        self.stat = jsm_stats.SatStats(self.lgMs)
        self.stat.Nsat(self.min_mass, plot=False)
        self.stat.Maxmass(plot=False)

    def get_data_points(self):
        lgMs = self.lgMs.flatten()[self.lgMs.flatten() > self.min_mass]
        lgMh = self.lgMh.flatten()[self.lgMs.flatten() > self.min_mass]
        return [lgMh, lgMs]
    

class load_models:

    def __init__(self, mfile:str, read_red=False):    
        self.mfile = mfile
        if read_red == True:
            models = np.load(mfile+"models.npz")
            self.lgMh_models = np.vstack(models["mass"])
            self.zacc_models = np.vstack(models["redshift"])
        elif read_red==False:
            models = np.load(mfile+"jsm_MCMC.npy")
            self.lgMh_models = np.vstack(models)

    def convert(self, theta:list, SHMR):
        self.theta = theta
        self.lgMs = SHMR(theta, self.lgMh_models)

    def convert_zacc(self, theta:list, SHMR):
        self.theta = theta
        self.lgMs = SHMR(theta, self.lgMh_models, self.zacc_models)

    def get_stats(self, min_mass):
        self.min_mass = min_mass
        self.stat = jsm_stats.SatStats(self.lgMs)
        self.stat.Nsat(self.min_mass)
        self.stat.Maxmass()


##################################################
###    FOR TESTING RUNS WITH THE SAME INPUT   ###
##################################################

class test_data:

    def __init__(self, truths:list, mfile:str, dfile:str):
        self.truths = truths
        self.lgMh_mat = np.load(mfile) # need to update this!
        self.lgMh = np.load(dfile)[0]
        self.lgMs = np.load(dfile)[1]
        self.lgMh_models = np.vstack(self.lgMh_mat)
        # temp = np.delete(self.lgMh_mat, SAGA_ind, axis=0) # delete the index used as the data
        # self.lgMh_models = np.vstack(temp) # no longer broken up into SAGA samples

    def get_stats(self, min_mass):
        self.min_mass = min_mass
        self.stat = jsm_stats.SatStats(self.lgMs)
        self.stat.Nsat(self.min_mass, plot=False)
        self.stat.Maxmass(plot=False)

    def get_data_points(self, min_mass):
        lgMs = self.lgMs.flatten()[self.lgMs.flatten() > min_mass]
        lgMh = self.lgMh.flatten()[self.lgMs.flatten() > min_mass]
        return [lgMh, lgMs]
    



##################################################
###           FOR CREATING MOCK DATA           ###
##################################################

class mock_data:

    def __init__(self, truths:list, SHMR, SAGA_ind:int, mfile:str, zfile:str=None):
        self.truths = truths
        if zfile != None: # if redshift data is provided!
            self.mfile = mfile
            self.zfile = zfile
            self.lgMh_mat = np.load(self.mfile) # load
            self.z_mat = np.load(self.zfile) 
            self.lgMh_data = self.lgMh_mat[SAGA_ind] # select the SAGA index
            self.z_data = self.z_mat[SAGA_ind]
            self.lgMs = SHMR(truths, self.lgMh_data, self.z_data) #convert to Ms
            temp = np.delete(self.lgMh_mat, SAGA_ind, axis=0) # delete the index used as the data
            tempz = np.delete(self.z_mat, SAGA_ind, axis=0)
            self.lgMh_models = np.vstack(temp) # return the rest as models (no longer broken up into SAGA samples)
            self.z_models = np.vstack(tempz)

        else:
            self.mfile = mfile
            self.lgMh_mat = np.load(self.mfile)
            self.lgMh_data = self.lgMh_mat[SAGA_ind]
            self.lgMs = SHMR(truths, self.lgMh_data)
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
    
    def save_data(self, path):
        np.save(path, np.array([self.lgMh_data, self.lgMs]))

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

    def __init__(self, sampler, truths:list, init_vals:list, labels:list, priors:list, savedir:str, data, SHMR, forward, min_mass, a_stretch):
        self.truths = truths
        self.init_vals = init_vals
        self.labels = labels
        self.priors = priors
        self.ndim = len(self.priors)
        self.samples = sampler.get_chain()
        self.chisq = sampler.get_log_prob()*(-2)
        self.flatchain = sampler.flatchain
        self.last_samp = sampler.get_last_sample().coords
        self.last_chisq = sampler.get_last_sample().log_prob*(-2)
        self.acceptance_frac = np.mean(sampler.acceptance_fraction)
        # try:
        #     self.tau = sampler.get_autocorr_time()
        # except AutocorrError:
        #     print("chain is too short")

        self.savedir = savedir
        self.min_mass = min_mass
        self.a_stretch = a_stretch
        
        # print("saving the chain!")
        # self.save_sample() #works

        # print("making some figures")
        # self.stat_plot(data, forward) #works
        # self.chain_plot() #works
        # self.chi_square_plot() #works
        # self.corner_last_sample(zoom=True) # works
        # #self.SHMR_plot(data, SHMR) # doesnt

    def save_sample(self):
        np.savez(self.savedir+"samples.npz", 
                 coords = self.samples,
                 chisq = self.chisq)
        values = []
        for i in range(self.ndim):
            post = np.percentile(self.last_samp[:, i], [16, 50, 84])
            q = np.diff(post)
            values.append([post[1], q[0], q[1]])
        self.constraints = values

        with open(self.savedir+"sampler_init.txt", 'w') as file: 
            
            write = ['This run was measured against data with truth values of '+str(self.truths)+'\n', 
            'It was initialized at '+str(self.init_vals)+'\n', 
            'The chain has '+str(self.samples.shape[1])+' walkers and '+str(self.samples.shape[0])+' steps\n', 
            'It was intialized with a_stretch = '+str(self.a_stretch)+'\n', 
            'The mean acceptance fraction is '+str(self.acceptance_frac)+'\n', 
            'The final step in the chain gives the following constraints\n', 
            'a1='+str(self.constraints[0])+'\n', 
            'a2='+str(self.constraints[1])+'\n', 
            'a3='+str(self.constraints[2])+'\n', 
            'a4='+str(self.constraints[3])+'\n']
            
            file.writelines("% s\n" % line for line in write) 
            file.close() 

    def chain_plot(self):
        if self.samples.shape[1] > 1000:
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
        plt.savefig(self.savedir+"chain.png")
        #plt.show()

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

        plt.figure(figsize=(8, 8))
        for i in Pnsat_mat:
            plt.plot(np.arange(i.shape[0]),i, color="grey", alpha=0.1)
        plt.plot(np.arange(data.stat.Pnsat.shape[0]),data.stat.Pnsat,marker="o", color="black")
        plt.xlabel("number of satellites > $10^{"+str(6.5)+"} \mathrm{M_{\odot}}$", fontsize=15)
        plt.ylabel("PDF", fontsize=15)
        plt.xlim(0,20)
        plt.savefig(self.savedir+"S1.png")
        #plt.show()

        plt.figure(figsize=(8, 8))
        for i, val in enumerate(Msmax_mat):
            plt.plot(val, Msmaxe_mat[i], color="grey", alpha=0.1)
        plt.plot(data.stat.Msmax, data.stat.ecdf_MsMax, color="black")
        plt.xlabel("stellar mass of most massive satellite ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
        plt.ylabel("CDF", fontsize=15)
        plt.savefig(self.savedir+"S2.png")
        #plt.show()

    def corner_last_sample(self, zoom=False):        
        if zoom==True:
            fig = corner.corner(self.last_samp, show_titles=True, labels=self.labels, truths=self.truths, quantiles=[0.15, 0.5, 0.85], plot_datapoints=False)
        elif zoom==False:
            fig = corner.corner(self.last_samp, show_titles=True, labels=self.labels, truths=self.truths, range=self.priors , quantiles=[0.15, 0.5, 0.85], plot_datapoints=False)
        plt.savefig(self.savedir+"corner.png")


    def SHMR_plot(self, data, SHMR):

        self.halo_masses = np.log10(np.logspace(6, 13, 100)) # just for the model

        SHMR_mat = np.zeros(shape=(self.last_samp.shape[0], self.halo_masses.shape[0]))
        norm = mpl.colors.Normalize(vmin=self.last_chisq.min(), vmax=self.last_chisq.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.magma_r)
        colors = mpl.cm.magma_r(np.linspace(0, 1, len(self.last_chisq)))

        a1, a2, a3, a4 = self.truths[0], self.truths[1], 0, self.truths[3] # just to define the fiducial model
        self.fid_Ms = SHMR([a1, a2, a3, a4], self.halo_masses)

        for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
            a1, a2, a3, a4 = val[0], val[1], 0, val[3]
            lgMs = SHMR([a1, a2, a3, a4], self.halo_masses)
            SHMR_mat[i] = lgMs

        plt.figure(figsize=(10, 8))
        for i,val in enumerate(SHMR_mat):
            plt.plot(self.halo_masses, val, color=colors[i], alpha=0.3, lw=1)
        #plt.plot(self.halo_masses, galhalo.lgMs_B13(self.halo_masses), color="red", label="Behroozi et al. 2013", ls="--", lw=2)
        #plt.plot(self.halo_masses, galhalo.lgMs_RP17(self.halo_masses), color="navy", label="Rodriguez-Puebla et al. 2017", ls="--", lw=2)
        plt.plot(self.halo_masses, self.fid_Ms, color="orange", label=str(self.truths), lw=3)
        plt.axhline(self.min_mass, label="mass limit", lw=1, ls=":", color="black")

        dp = data.get_data_points(min_mass=self.min_mass)
        plt.scatter(dp[0], dp[1], marker=".", color="black")

        plt.ylim(4,11)
        plt.xlim(7.5,12)
        plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
        plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
        plt.legend(fontsize=12)
        plt.colorbar(cmap, label="$\\chi^2$")
        plt.savefig(self.savedir+"SHMR.png")

    def chi_square_plot(self):
        fig, ax = plt.subplots(2, 2, sharey=True,figsize=(10,10))

        ax[0,0].scatter(self.last_samp[:,0], self.last_chisq, marker=".")
        ax[0,0].set_xlabel(self.labels[0], fontsize=12)
        ax[0,0].axvline(self.truths[0], ls=":", color="black")

        ax[1,0].scatter(self.last_samp[:,1], self.last_chisq, marker=".")
        ax[1,0].set_xlabel(self.labels[1], fontsize=12)
        ax[1,0].axvline(self.truths[1], ls=":", color="black")

        ax[0,1].scatter(self.last_samp[:,2], self.last_chisq, marker=".")
        ax[0,1].set_xlabel(self.labels[2], fontsize=12)
        ax[0,1].axvline(self.truths[2], ls=":", color="black")

        ax[1,1].scatter(self.last_samp[:,3], self.last_chisq, marker=".")
        ax[1,1].set_xlabel(self.labels[3], fontsize=12)
        ax[1,1].axvline(self.truths[3], ls=":", color="black")

        ax[0,0].set_ylabel("$\\chi^2$", fontsize=12)
        ax[1,0].set_ylabel("$\\chi^2$", fontsize=12)
        plt.savefig(self.savedir+"chi2_final.png")


# def new_SHMR_plot(self, data, theta_plot, convert=False):

#         self.halo_masses = np.log10(np.logspace(7, 13, 100)) # just for the figure

#         SHMR_mat = np.zeros(shape=(self.last_samp.shape[0], self.halo_masses.shape[0]))
#         norm = mpl.colors.Normalize(vmin=self.last_chisq.min(), vmax=self.last_chisq.max())
#         cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.magma_r)
#         colors = mpl.cm.magma_r(np.linspace(0, 1, len(self.last_chisq)))

#         if convert==True:
#             a1, a2, a3 = theta_plot[0], theta_plot[1], theta_plot[2]

#         a1, a2, a3, a4 = self.truths[0], self.truths[1], 0, self.truths[3] # just to define the fiducial model
#         self.fid_Ms = jsm_SHMR.fiducial([a1, a2, a3, a4], self.halo_masses)

#         for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
#             a1, a2, a3, a4 = val[0], val[1], 0, val[3]
#             lgMs = SHMR([a1, a2, a3, a4], self.halo_masses)
#             SHMR_mat[i] = lgMs

#         plt.figure(figsize=(10, 8))
#         for i,val in enumerate(SHMR_mat):
#             plt.plot(self.halo_masses, val, color=colors[i], alpha=0.3, lw=1)
#         #plt.plot(self.halo_masses, galhalo.lgMs_B13(self.halo_masses), color="red", label="Behroozi et al. 2013", ls="--", lw=2)
#         #plt.plot(self.halo_masses, galhalo.lgMs_RP17(self.halo_masses), color="navy", label="Rodriguez-Puebla et al. 2017", ls="--", lw=2)
#         plt.plot(self.halo_masses, self.fid_Ms, color="orange", label=str(self.truths), lw=3)
#         plt.axhline(self.min_mass, label="mass limit", lw=1, ls=":", color="black")

#         dp = data.get_data_points(min_mass=self.min_mass)
#         plt.scatter(dp[0], dp[1], marker=".", color="black")

#         plt.ylim(4,11)
#         plt.xlim(7.5,12)
#         plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
#         plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
#         plt.legend(fontsize=12)
#         plt.colorbar(cmap, label="$\\chi^2$")
#         plt.savefig(self.savedir+"SHMR.png")
    
    # def corner_stack_samples(self, stack, zoom=False, plot=False):    
    #     nsteps = self.samples.shape[0]
    #     ssteps = nsteps - stack
    #     s = self.samples[ssteps:nsteps,:,:].shape
    #     self.stack = self.samples[ssteps:nsteps,:,:].reshape(s[0] * s[1], s[2])
    #     if plot==True:
    #         if zoom==True:
    #             fig = corner.corner(self.burn, show_titles=True, labels=self.labels, truths=self.truths, quantiles=[0.15, 0.5, 0.85], plot_datapoints=False)
    #         elif zoom==False:
    #             fig = corner.corner(self.burn, show_titles=True, labels=self.labels, truths=self.truths, range=self.priors , quantiles=[0.15, 0.5, 0.85], plot_datapoints=False)
    #         plt.savefig(self.savedir+"corner.png")


# class cross_run:

#     def __init__(self, samples, truths:list, labels:list, priors:list, savedir:str):

#         self.truths = truths
#         self.labels = labels
#         self.priors = priors
#         self.ndim = len(self.priors)
#         self.Nsamp = samples.shape[0]
#         self.savedir = savedir

#     def best_fit_values(self):
#         list_val = []
#         for i in range(4):
#             mcmc = np.percentile(self.last_samp[:, i], [16, 50, 84])
#             q = np.diff(mcmc)
#             txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
#             txt = txt.format(mcmc[1], q[0], q[1], self.labels[i])
#             #display(Math(txt))
#             list_val.append([mcmc[1], q[0], q[1]])
#         return np.array(list_val)

#     def cross_sample(samples, xaxis, xlabel, labels):

#         Nsamples = samples.shape[0]
#         val_mat = np.zeros(shape=(Nsamples, 4, 3))

#         for i in range(Nsamples):
#             val_mat[i] = self.best_fit_values(samples[i], labels)

#         fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10,8))

#         axs[0,0].errorbar(xaxis, val_mat[:, 0, 0], yerr=[val_mat[:, 0, 1], val_mat[:, 0, 2]], fmt="o", color="black")
#         axs[0,0].axhline(1.8, ls=":")
#         axs[0,0].set_ylabel("a1")
#         axs[0,0].set_ylim(-1,5)

#         axs[1,0].errorbar(xaxis, val_mat[:, 1, 0], yerr=[val_mat[:, 1, 1], val_mat[:, 1, 2]], fmt="o", color="black")
#         axs[1,0].axhline(-0.2, ls=":")
#         axs[1,0].set_xlabel(xlabel)
#         axs[1,0].set_ylabel("a2")
#         axs[1,0].set_ylim(-2,1)

#         axs[0,1].errorbar(xaxis, val_mat[:, 2, 0], yerr=[val_mat[:, 2, 1], val_mat[:, 2, 2]], fmt="o", color="black")
#         axs[0,1].axhline(0.4, ls=":")
#         axs[0,1].set_ylabel("a3")
#         axs[0,1].set_ylim(0,4)


#         axs[1,1].errorbar(xaxis, val_mat[:, 3, 0], yerr=[val_mat[:, 3, 1], val_mat[:, 3, 2]], fmt="o", color="black")
#         axs[1,1].axhline(10.1, ls=":")
#         axs[1,1].set_xlabel(xlabel)
#         axs[1,1].set_ylabel("a4")
#         axs[1,1].set_ylim(9,11)
#         plt.show()    



##########################################################
### the following routines are for the old MCMC method ###
##########################################################

    # def chi_square_plot(self):

    #     mask = self.chisq > 0

    #     norm = plt.Normalize()
    #     colors = plt.cm.viridis_r(norm(self.last_chisq[mask]))
    #     cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis_r)

    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True,figsize=(12,6))
    #     ax1.scatter(self.last_samp[:,0][mask], self.last_chisq[mask], marker=".")
    #     ax1.set_xlim(0.5, 3.5)
    #     ax1.set_xlabel("$\\alpha$", fontsize=12)

    #     ax2.scatter(self.last_samp[:,1][mask], self.last_chisq[mask], marker=".")
    #     ax2.set_xlim(-2, 0.5)
    #     ax2.set_xlabel("$\\delta$", fontsize=12)

    #     ax3.scatter(self.last_samp[:,2][mask], self.last_chisq[mask], marker=".")
    #     ax3.set_xlim(0,4)
    #     ax3.set_xlabel("$\\sigma$", fontsize=12)
    #     ax1.set_ylabel("$\\chi^2$", fontsize=12)
    #     ax1.set_yscale("log")
    #     #plt.show()

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
    #         #plt.show()

    #         plt.errorbar(self.satfreq_bincenters, self.data_PDF, yerr=self.error_PDF)
    #         plt.xlabel("stellar mass of most massive satellite ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
    #         plt.xlabel("number of satellites > $10^{6.5} \mathrm{M_{\odot}}$", fontsize=15)
    #         plt.ylabel("PDF", fontsize=15)
    #         #plt.show()

    #     self.model_lgMh = self.lgMh_mat[mock_data_ind]

# def fid_MODEL(lgMh_data, truths, mass_list, return_counts=False):

#     """_summary_
#     the main model! goes from a SHMF mass to a CSMF and measures some statistics!

#     Returns:
#         np.ndarray: 1D model array populated with the statistics!
#     """
    
#     alpha, delta, sigma = truths

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

#     def __init__(self, datadir:str, Nsamp:int=100, truths:list=[1.85, 0.2, 0.3], mass_bins:np.ndarray=np.linspace(4,11,45), mass_list:list=[6.5,7.,7.5]):

#         self.datadir = datadir
#         self.truths = truths
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
#             D_mat[i], count_mat[i] = fid_MODEL(lgMh_i, self.truths, self.mass_list, return_counts=True)

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

#         a1s = np.linspace(1,3,chi_dim)
#         a1s = np.linspace(-1,3,chi_dim)
#         a1s = np.linspace(0,3,chi_dim)

#         chi_mat = np.zeros(shape=(chi_dim,chi_dim,chi_dim))

#         for i, aval in enumerate(a1s):
#             for j, dval in enumerate(a1s):
#                 for k, sval in enumerate(a1s):
#                     model = fid_MODEL(self.lgMh, [aval, dval, sval], self.mass_list)
#                     X = model - self.D
#                     X_vec = np.expand_dims(X, axis=1)
#                     chi_mat[i,j,k] = X_vec.transpose().dot(self.inv_covar).dot(X_vec)

#         self.chi_mat = chi_mat
#         ai, di, si = np.where(chi_mat == np.min(chi_mat))
#         theta_0 = [a1s[ai][0], a1s[di][0], a1s[si][0]]
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
#         #plt.show()