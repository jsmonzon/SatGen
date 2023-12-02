import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import corner
import jsm_stats
import jsm_SHMR
from multiprocess import Pool
import emcee
import time
from scipy.stats import ks_2samp
import warnings; warnings.simplefilter('ignore')

##################################################
###    A SIMPLE FUNC TO MULTICORE THE MCMC     ###
##################################################

# def RUN_old(theta_0, lnprob, nwalkers, niter, ndim, ncores, a_stretch, nfixed):

#     p0 = [np.array(theta_0) + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]
    
#     with Pool(ncores) as pool:
#         sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, moves=emcee.moves.StretchMove(a=a_stretch, nf=nfixed))
#         start = time.time()
#         sampler.run_mcmc(p0, niter)
#         end = time.time()
#         multi_time = end - start
#         print("Run took {0:.1f} seconds".format(multi_time))
#         print("Run took {0:.1f} minutes".format(multi_time/60))
#         print("Run took {0:.1f} hours".format(multi_time/3600))
#     return sampler

def lnL_Pnsat(model, data):
    lnL = np.sum(np.log(model[data]))
    if np.isnan(lnL):
        return -np.inf
    else:
        return lnL
    
def lnL_KS(model, data):
    return np.log(ks_2samp(model, data)[1])

# def good_guess(lnprob, priors, chidim=5):

#     chi_mat = np.zeros(shape=(chidim, chidim, chidim, chidim))
#     a1s = np.linspace(priors[0][0], priors[0][1], chidim)
#     a2s = np.linspace(priors[1][0], priors[1][1], chidim)
#     a3s = np.linspace(priors[2][0], priors[2][1], chidim)
#     a4s = np.linspace(priors[3][0], priors[3][1], chidim)

#     for i, a1val in enumerate(a1s):
#         for j, a2val in enumerate(a2s):
#             for k, a3val in enumerate(a3s):
#                 for l, a4val in enumerate(a4s):
#                     chi_mat[i,j,k,l] = -2*lnprob([a1val, a2val, a3val, a4val])

#     a1min, a2min, a3min, a4min = np.where(chi_mat == np.min(chi_mat))
#     return [a1s[a1min][0], a2s[a2min][0], a3s[a3min][0], a4s[a4min][0]]

##################################################
###                MORE GENERALIZED            ###
##################################################

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

    def get_stats(self, min_mass, plot=False):
        self.min_mass = min_mass
        self.stat = jsm_stats.SatStats(self.lgMs)
        self.stat.Nsat(self.min_mass, plot=plot)
        self.stat.Maxmass(plot=plot)

    def get_data_points(self, plot=True):
        self.lgMs_flat = self.lgMs.flatten()[self.lgMs.flatten() > self.min_mass]
        self.lgMh_flat = self.lgMh.flatten()[self.lgMs.flatten() > self.min_mass]
        if plot==True:
            plt.scatter(self.lgMh_flat, self.lgMs_flat)
            plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
            plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
    

class load_models:

    def __init__(self, mfile:str, read_red=False):    
        self.mfile = mfile
        models = np.load(mfile+"models.npz")
        self.lgMh_models = np.vstack(models["mass"])
        if read_red == True:
            self.zacc_models = np.vstack(models["redshift"])

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
        

class Hammer:

    def __init__(self, ftheta, gtheta, fixed, ndim, nwalk, nstep, ncores, a_stretch, min_mass, N_corr, p0_corr, **kwargs):

        self.ftheta = ftheta
        self.gtheta = gtheta
        self.fixed = fixed
        self.ndim = ndim
        self.nwalk = nwalk
        self.nstep = nstep
        self.ncores = ncores
        self.a_stretch = a_stretch
        self.min_mass = min_mass
        self.Ncore = N_corr
        self.p0_corr = p0_corr

        if N_corr==True:
            print("making the correction on Ndim in the stretch move algorithm!")
            self.nfixed = sum(self.fixed)
        elif N_corr==False:
            self.nfixed = 0

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.inital_guess()
        self.write_init()

    def inital_guess(self):
        p0 = [np.array(self.gtheta) + 1e-2 * np.random.randn(self.ndim) for i in range(self.nwalk)]

        if self.p0_corr==True:
            print("not allowing the fixed walkers to step!")
            p0_fixed = []
            for i in range(self.nwalk):
                p0_fixed.append(np.where(self.fixed, self.ftheta, p0[i]))
            self.p0 = p0_fixed

        elif self.p0_corr==False:
            print("allowing the fixed walkers to step! make sure the likelyhood evaluation is correct!")
            self.p0 = p0

    def write_init(self):
        with open(self.savedir+"chain_info.txt", 'w') as file: 
            
            write = ['This run was measured against data with truth values of '+str(self.ftheta)+'\n', 
            'It was initialized at '+str(self.gtheta)+'\n', 
            'The chain has '+str(self.nwalk)+' walkers and '+str(self.nstep)+' steps\n', 
            'It was initialized with a_stretch = '+str(self.a_stretch)+'\n']
            file.writelines("% s\n" % line for line in write) 
            file.close()

    def runit(self, lnprob):
        backend = emcee.backends.HDFBackend(self.savefile)
        backend.reset(self.nwalk, self.ndim)
        
        with Pool(self.ncores) as pool:
            sampler = emcee.EnsembleSampler(self.nwalk, self.ndim, lnprob, pool=pool, moves=emcee.moves.StretchMove(a=self.a_stretch, nf=self.nfixed), backend=backend)
            start = time.time()
            sampler.run_mcmc(self.p0, self.nstep, progress=True, skip_initial_state_check=self.p0_corr)
            end = time.time()
            multi_time = end - start
        
        print("Run took {0:.1f} hours".format(multi_time/3600))
        print("saving some information from the sampler class")

        self.runtime = multi_time/3600
        self.acceptance_frac = np.mean(sampler.acceptance_fraction)
        try:
            self.tau = sampler.get_autocorr_time()
        except:
            print("run a longer chain!")
            self.tau = 0

        self.samples = sampler.get_chain()
        self.chisq = sampler.get_log_prob()*(-2)
        self.last_samp = sampler.get_last_sample().coords
        self.last_chisq = sampler.get_last_sample().log_prob*(-2)

    def write_output(self):

        values = []
        for i in range(self.ndim):
            post = np.percentile(self.last_samp[:, i], [16, 50, 84])
            q = np.diff(post)
            values.append([post[1], q[0], q[1]])
        self.constraints = values

        with open(self.savedir+"chain_info.txt", 'a') as file:
            write1 = ['The run took {0:.1f} hours'.format(self.runtime)+'\n',
                      'The mean acceptance fraction turned out to be '+str(self.acceptance_frac)+'\n',
                      'The auto correlation time (Nsteps) was' + str(self.tau)+'\n',
                      'The final step in the chain gives the following constraints on theta\n']
            write2 = []
            for i,val in enumerate(self.constraints):
                write2.append(self.labels[i]+"="+str(val)+"\n")      

            file.writelines("% s\n" % line for line in write1) 
            file.writelines("% s\n" % line for line in write2) 
            file.close()

        self.plot_chain()
        self.plot_last_chisq()

    def plot_chain(self):
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
        if self.savefig == True:
            plt.savefig(self.savedir+"chain.png")


    def plot_last_chisq(self):
        fig, axs = plt.subplots(1, self.ndim, sharey=True, figsize=(14,8))
        fig.tight_layout(pad=2.0)

        for i in range(self.ndim):
            axs[i].scatter(self.last_samp[:,i], self.last_chisq, marker=".")
            axs[i].set_xlabel(self.labels[i], fontsize=12)
            axs[i].axvline(self.ftheta[i], ls=":", color="black")
        axs[0].set_ylabel("$\\chi^2$", fontsize=12)

        if self.savefig == True:
            plt.savefig(self.savedir+"chi2_final.png")


    def stat_fit(self):

        Ns, Ms, _ = self.forward(self.last_samp[0])
        Pnsat_mat = np.zeros(shape=(self.last_samp.shape[0], Ns.shape[0]))
        Msmax_mat = np.zeros(shape=(self.last_samp.shape[0], Ms.shape[0]))
        Msmaxe_mat = np.zeros(shape=(self.last_samp.shape[0], Ms.shape[0]))

        for i, theta in enumerate(self.last_samp):
            tPnsat, tMsmax, tecdf_MsMax = self.forward(theta)
            Pnsat_mat[i] = tPnsat
            Msmax_mat[i] = tMsmax      
            Msmaxe_mat[i] = tecdf_MsMax

        plt.figure(figsize=(8, 8))
        for i in Pnsat_mat:
            plt.plot(np.arange(i.shape[0]),i, color="grey", alpha=0.1)
        plt.plot(np.arange(self.data.stat.Pnsat.shape[0]),self.data.stat.Pnsat,marker="o", color="black")
        plt.xlabel("number of satellites > $10^{"+str(6.5)+"} \mathrm{M_{\odot}}$", fontsize=15)
        plt.ylabel("PDF", fontsize=15)
        plt.xlim(0,25)
        plt.savefig(self.savedir+"S1.png")

        plt.figure(figsize=(8, 8))
        for i, val in enumerate(Msmax_mat):
            plt.plot(val, Msmaxe_mat[i], color="grey", alpha=0.1)
        plt.plot(self.data.stat.Msmax, self.data.stat.ecdf_MsMax, color="black")
        plt.xlabel("stellar mass of most massive satellite ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
        plt.ylabel("CDF", fontsize=15)

        if self.savefig == True:
            plt.savefig(self.savedir+"S2.png")


    def SHMR_last_sample(self):

        self.halo_masses = np.linspace(6,12,50)
        SHMR_mat = np.zeros(shape=(self.last_samp.shape[0], self.halo_masses.shape[0]))

        if self.SHMR_model == "simple":
            self.fid_Ms = jsm_SHMR.simple([self.data.truths[0], 0], self.halo_masses)
            for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
                SHMR_mat[i] = jsm_SHMR.simple([val[0], 0], self.halo_masses)

        elif self.SHMR_model =="anchor":
            self.fid_Ms = jsm_SHMR.anchor([self.data.truths[0], 0, self.data.truths[2]], self.halo_masses)
            for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
                SHMR_mat[i] = jsm_SHMR.anchor([val[0], 0, val[2]], self.halo_masses)

        elif self.SHMR_model =="curve":
            self.fid_Ms = jsm_SHMR.curve([self.data.truths[0], 0, self.data.truths[2],  self.data.truths[3]], self.halo_masses)
            for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
                SHMR_mat[i] = jsm_SHMR.curve([val[0], 0, val[2], val[3]], self.halo_masses)

        elif self.SHMR_model =="sigma":
            self.fid_Ms = jsm_SHMR.sigma([self.data.truths[0], 0, self.data.truths[2],  self.data.truths[3], 0], self.halo_masses)
            for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
                SHMR_mat[i] = jsm_SHMR.sigma([val[0], 0, val[2], val[3], 0], self.halo_masses)

        elif self.SHMR_model =="redshift":
            self.fid_Ms = jsm_SHMR.redshift([self.data.truths[0], 0, self.data.truths[2],  self.data.truths[3], 0, self.data.truths[5]], self.halo_masses, np.zeros(shape=self.halo_masses.shape[0]))
            for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
                SHMR_mat[i] = jsm_SHMR.redshift([val[0], 0, val[2], val[3], 0, val[5]], self.halo_masses, np.zeros(shape=self.halo_masses.shape[0]))

        else:
            print("no model selected")
                
        plt.figure(figsize=(10, 8))
        for i,val in enumerate(SHMR_mat):
            plt.plot(self.halo_masses, val, alpha=0.3, lw=1, color='grey')
        plt.plot(self.halo_masses, self.fid_Ms, color="orange", label=str(self.data.truths), lw=3)

        plt.axhline(self.min_mass, label="mass limit", lw=1, ls=":", color="black")
        plt.scatter(self.data.lgMh_flat, self.data.lgMs_flat, marker=".", color="black")
        plt.ylim(4,11)
        plt.xlim(7.5,12)
        plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
        plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
        plt.legend(fontsize=12)

        if self.savefig == True:
            plt.savefig(self.savedir+"SHMR.png")



##################################################
###     TO INTERFACE WITH THE MCMC OUTPUT      ###
##################################################

# class inspect_run:

#     def __init__(self, sampler, **kwargs):
#         for key, value in kwargs.items():
#             setattr(self, key, value)
        
#         self.ndim = len(self.start_theta)
#         self.samples = sampler.get_chain()
#         self.chisq = sampler.get_log_prob()*(-2)
#         self.last_samp = sampler.get_last_sample().coords
#         self.last_chisq = sampler.get_last_sample().log_prob*(-2)
#         self.acceptance_frac = np.mean(sampler.acceptance_fraction)

#         values = []
#         for i in range(self.ndim):
#             post = np.percentile(self.last_samp[:, i], [16, 50, 84])
#             q = np.diff(post)
#             values.append([post[1], q[0], q[1]])
#         self.constraints = values

#         print("saving the chain!")
#         self.write_init()
#         self.save_sample()

#         print("making some figures")
#         self.full_chain()
#         self.chisq_last_sample()
#         self.corner_last_sample()
#         self.stat_fit()
#         self.SHMR_last_sample() # for the sigma value!

#     def write_init(self):
#         with open(self.savedir+"chain_initialization.txt", 'w') as file: 
            
#             write1 = ['This run was measured against data with truth values of '+str(self.data.truths)+'\n', 
#             'It was initialized at '+str(self.start_theta)+'\n', 
#             'The chain has '+str(self.samples.shape[1])+' walkers and '+str(self.samples.shape[0])+' steps\n', 
#             'It was initialized with a_stretch = '+str(self.a_stretch)+'\n', 
#             'The mean acceptance fraction is '+str(self.acceptance_frac)+'\n', 
#             'The final step in the chain gives the following constraints\n']

#             write2 = []
#             for i,val in enumerate(self.constraints):
#                 write2.append(self.labels[i]+"="+str(val)+"\n")      

#             file.writelines("% s\n" % line for line in write1) 
#             file.writelines("% s\n" % line for line in write2) 
#             file.close()

#     def save_sample(self):
#         np.savez(self.savedir+"samples.npz", 
#                  coords = self.samples,
#                  chisq = self.chisq)
        
#     def full_chain(self):
#         if self.samples.shape[1] > 1000:
#             a = 0.01
#         else:
#             a = 0.1

#         fig, axes = plt.subplots(self.ndim, figsize=(10, 7), sharex=True)
#         for i in range(self.ndim):
#             ax = axes[i]
#             ax.plot(self.samples[:, :, i], "k", alpha=a)
#             ax.set_xlim(0, len(self.samples))
#             ax.set_ylabel(self.labels[i])
#             ax.yaxis.set_label_coords(-0.1, 0.5)

#         axes[-1].set_xlabel("step number")
#         plt.savefig(self.savedir+"chain.png")

#     def corner_last_sample(self, zoom=True):        
#         if zoom==True:
#             fig = corner.corner(self.last_samp, show_titles=True, labels=self.labels, truths=self.data.truths, quantiles=[0.15, 0.5, 0.85], plot_datapoints=False)
#         elif zoom==False:
#             fig = corner.corner(self.last_samp, show_titles=True, labels=self.labels, truths=self.data.truths, range=self.priors , quantiles=[0.15, 0.5, 0.85], plot_datapoints=False)
#         plt.savefig(self.savedir+"corner.png")

#     def chisq_last_sample(self):
#         fig, axs = plt.subplots(1, self.ndim, sharey=True, figsize=(14,8))
#         fig.tight_layout(pad=2.0)

#         for i in range(self.ndim):
#             axs[i].scatter(self.last_samp[:,i], self.last_chisq, marker=".")
#             axs[i].set_xlabel(self.labels[i], fontsize=12)
#             axs[i].axvline(self.data.truths[i], ls=":", color="black")
#         axs[0].set_ylabel("$\\chi^2$", fontsize=12)

#         plt.savefig(self.savedir+"chi2_final.png")

#     def stat_fit(self):

#         Ns, Ms, _ = self.forward(self.last_samp[0])
#         Pnsat_mat = np.zeros(shape=(self.last_samp.shape[0], Ns.shape[0]))
#         Msmax_mat = np.zeros(shape=(self.last_samp.shape[0], Ms.shape[0]))
#         Msmaxe_mat = np.zeros(shape=(self.last_samp.shape[0], Ms.shape[0]))

#         for i, theta in enumerate(self.last_samp):
#             tPnsat, tMsmax, tecdf_MsMax = self.forward(theta)
#             Pnsat_mat[i] = tPnsat
#             Msmax_mat[i] = tMsmax      
#             Msmaxe_mat[i] = tecdf_MsMax

#         plt.figure(figsize=(8, 8))
#         for i in Pnsat_mat:
#             plt.plot(np.arange(i.shape[0]),i, color="grey", alpha=0.1)
#         plt.plot(np.arange(self.data.stat.Pnsat.shape[0]),self.data.stat.Pnsat,marker="o", color="black")
#         plt.xlabel("number of satellites > $10^{"+str(6.5)+"} \mathrm{M_{\odot}}$", fontsize=15)
#         plt.ylabel("PDF", fontsize=15)
#         plt.xlim(0,25)
#         plt.savefig(self.savedir+"S1.png")

#         plt.figure(figsize=(8, 8))
#         for i, val in enumerate(Msmax_mat):
#             plt.plot(val, Msmaxe_mat[i], color="grey", alpha=0.1)
#         plt.plot(self.data.stat.Msmax, self.data.stat.ecdf_MsMax, color="black")
#         plt.xlabel("stellar mass of most massive satellite ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
#         plt.ylabel("CDF", fontsize=15)
#         plt.savefig(self.savedir+"S2.png")

#     def SHMR_last_sample(self):

#         # plot_data = np.load("../plot_data.npy")
#         # self.halo_masses = plot_data[0]
#         # self.redshifts = plot_data[1]
#         self.halo_masses = np.linspace(6,12,50)
#         SHMR_mat = np.zeros(shape=(self.last_samp.shape[0], self.halo_masses.shape[0]))

#         if self.SHMR_model == "simple":
#             self.fid_Ms = jsm_SHMR.simple([self.data.truths[0], 0], self.halo_masses)
#             for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
#                 SHMR_mat[i] = jsm_SHMR.simple([val[0], 0], self.halo_masses)

#         elif self.SHMR_model =="anchor":
#             self.fid_Ms = jsm_SHMR.anchor([self.data.truths[0], 0, self.data.truths[2]], self.halo_masses)
#             for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
#                 SHMR_mat[i] = jsm_SHMR.anchor([val[0], 0, val[2]], self.halo_masses)

#         elif self.SHMR_model =="curve":
#             self.fid_Ms = jsm_SHMR.curve([self.data.truths[0], 0, self.data.truths[2],  self.data.truths[3]], self.halo_masses)
#             for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
#                 SHMR_mat[i] = jsm_SHMR.curve([val[0], 0, val[2], val[3]], self.halo_masses)

#         elif self.SHMR_model =="sigma":
#             self.fid_Ms = jsm_SHMR.sigma([self.data.truths[0], 0, self.data.truths[2],  self.data.truths[3], 0], self.halo_masses)
#             for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
#                 SHMR_mat[i] = jsm_SHMR.sigma([val[0], 0, val[2], val[3], 0], self.halo_masses)

#         elif self.SHMR_model =="redshift":
#             self.fid_Ms = jsm_SHMR.redshift([self.data.truths[0], 0, self.data.truths[2],  self.data.truths[3], 0, self.data.truths[5]], self.halo_masses, np.zeros(shape=self.halo_masses.shape[0]))
#             for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
#                 SHMR_mat[i] = jsm_SHMR.redshift([val[0], 0, val[2], val[3], 0, val[5]], self.halo_masses, np.zeros(shape=self.halo_masses.shape[0]))

#         else:
#             print("no model selected")
                
#         plt.figure(figsize=(10, 8))
#         for i,val in enumerate(SHMR_mat):
#             plt.plot(self.halo_masses, val, alpha=0.3, lw=1, color='grey')
#         plt.plot(self.halo_masses, self.fid_Ms, color="orange", label=str(self.data.truths), lw=3)

#         plt.axhline(self.min_mass, label="mass limit", lw=1, ls=":", color="black")
#         plt.scatter(self.data.lgMh_flat, self.data.lgMs_flat, marker=".", color="black")
#         plt.ylim(4,11)
#         plt.xlim(7.5,12)
#         plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
#         plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
#         plt.legend(fontsize=12)
#         plt.savefig(self.savedir+"SHMR.png")

    # def sigma_plot(self):
    #     sigma = np.zeros(shape=(self.last_samp.shape[0],self.data.lgMh_flat.shape[0]))

    #     for i, val in enumerate(self.last_samp):
    #         sigma[i] = 0.15 + val[2]*(self.data.lgMh_flat - 11.67)

    #     for i in sigma:
    #         plt.plot(self.data.lgMh_flat, i, alpha=0.1, color="grey")
    #     plt.ylabel("$\sigma_{M_*}$", fontsize=15)
    #     plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
    #     plt.savefig(self.savedir+"sigma.png")

        # if color_ind!=None:
        #     norm = mpl.colors.Normalize(vmin=self.last_samp[:,color_ind].min(), vmax=self.last_samp[:,color_ind].max())
        #     cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.magma_r)
        #     colors = mpl.cm.magma_r(np.linspace(0, 1, len(self.last_samp[:,color_ind])))
        #     clabel = self.labels[color_ind]

        # else:
        #     norm = mpl.colors.Normalize(vmin=self.last_chisq.min(), vmax=self.last_chisq.max())
        #     cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.magma_r)
        #     colors = mpl.cm.magma_r(np.linspace(0, 1, len(self.last_chisq)))
        #     clabel = "$\\chi^2$"



class analyze_chains:


    def __init__(self, samplez, **kwargs):
        self.samplez = samplez
        for key, value in kwargs.items():
            setattr(self, key, value)

    def grab(self, stack):
        max_ind = self.samplez[0].shape[0]
        for idx, i in enumerate(self.samplez):
            nsteps = i.shape[0]
            ssteps = nsteps - stack
            s = i[ssteps:nsteps, :, :].shape
            setattr(self, f"stack_{idx}", i[ssteps:nsteps, :, :].reshape(s[0] * s[1], s[2]))
            setattr(self, f"last_samp{idx}", i[max_ind-1, :, :])

    def violin(self, plabels, priors, truths, labels, xlabel):

        fig, ax = plt.subplots(2, 2, sharex=True,figsize=(10,10))

        a1_data = [self.samplez[0][:,0], self.samplez[1][:,0], self.samplez[2][:,0]]
        a2_data = [self.samplez[0][:,1], self.samplez[1][:,1], self.samplez[2][:,1]]
        a3_data = [self.samplez[0][:,2], self.samplez[1][:,2], self.samplez[2][:,2]]
        a4_data = [self.samplez[0][:,3], self.samplez[1][:,3], self.samplez[2][:,3]]

        ax[0,0].violinplot(a1_data, vert=True, showextrema=True, widths=0.5, showmedians=False)
        ax[0,0].axhline(truths[0], ls="--", lw=1, color="black")
        ax[0,0].set_ylabel(plabels[0])
        ax[0,0].set_ylim(priors[0][0], priors[0][1])

        ax[0,1].violinplot(a2_data, vert=True, showextrema=True, widths=0.5, showmedians=False)
        ax[0,1].axhline(truths[1], ls="--", lw=1, color="black")
        ax[0,1].set_ylabel(plabels[1])
        ax[0,1].set_ylim(priors[1][0], priors[1][1])

        ax[1,0].violinplot(a3_data, vert=True, showextrema=True, widths=0.5, showmedians=False)
        ax[1,0].axhline(truths[2], ls="--", lw=1, color="black")
        ax[1,0].set_ylabel(plabels[2])
        ax[1,0].set_ylim(priors[2][0], priors[2][1])
        ax[1,0].set_xlabel(xlabel)

        ax[1,1].violinplot(a4_data, vert=True, showextrema=True, widths=0.5, showmedians=False)
        ax[1,1].axhline(truths[3], ls="--", lw=1, color="black")
        ax[1,1].set_ylabel(plabels[3])
        ax[1,1].set_xticks([1,2,3], labels=labels)
        ax[1,1].set_xlabel(xlabel)
        plt.show() 

    def SHMR_colored(sample, SHMR_model, labels, color_ind, plot_data=True):
        halo_masses = np.log10(np.logspace(6, 13, 100))  # just for the model

        SHMR_mat = np.zeros(shape=(sample.shape[0], halo_masses.shape[0]))

        # Extract the color values for each data point
        colors = sample[:, color_ind]

        norm = mpl.colors.Normalize(vmin=colors.min(), vmax=colors.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.magma_r)

        if SHMR_model == "simple":
            for i,val in enumerate(sample):  # now pushing all thetas through!
                SHMR_mat[i] = jsm_SHMR.simple([val[0], 0], halo_masses)

        elif SHMR_model =="anchor":
            for i,val in enumerate(sample):  # now pushing all thetas through!
                SHMR_mat[i] = jsm_SHMR.anchor([val[0], 0, val[2]], halo_masses)

        elif SHMR_model =="curve":
            for i,val in enumerate(sample):  # now pushing all thetas through!
                SHMR_mat[i] = jsm_SHMR.curve([val[0], 0, val[2], val[3]], halo_masses)

        elif SHMR_model =="sigma":
            for i,val in enumerate(sample):  # now pushing all thetas through!
                SHMR_mat[i] = jsm_SHMR.sigma([val[0], 0, val[2], val[3], 0], halo_masses)

        elif SHMR_model =="redshift":
            for i,val in enumerate(sample):  # now pushing all thetas through!
                SHMR_mat[i] = jsm_SHMR.redshift([val[0], 0, val[2], val[3], 0, val[5]], halo_masses, np.zeros(shape=halo_masses.shape[0]))

        plt.figure(figsize=(10, 8))
        for i, val in enumerate(SHMR_mat):
            plt.plot(halo_masses, val, color=cmap.to_rgba(colors[i]), alpha=0.3, lw=1)


        if plot_data==True:
            hmm = np.load("../analysis/model_test/mock_data.npy")
            plt.scatter(hmm[0], hmm[1], marker=".", color="grey")
            plt.axhline(6.5, label="mass limit", lw=3, ls=":", color="black")

            
        plt.ylim(4, 11)
        plt.xlim(7.5, 12)
        plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
        plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)

        # Create a colorbar using the ScalarMappable
        cbar = plt.colorbar(cmap, label=labels[color_ind])
        cbar.set_label(labels[color_ind])

        plt.show()
