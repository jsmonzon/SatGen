import numpy as np
import matplotlib.pyplot as plt
from multiprocess import Pool
import emcee
import time
import warnings; warnings.simplefilter('ignore')
import pygtc
import logging


class Hammer:

    """
    The machine I use to run an MCMC. It is a built to interface with the emcee package.
    """

    def __init__(self, fid_theta, fixed, nwalk, nstep, ncores, **kwargs):

        self.fid_theta = fid_theta
        self.fixed = fixed
        self.ndim = len(fid_theta) #sum(self.fixed)
        self.nwalk = nwalk
        self.nstep = nstep
        self.ncores = ncores
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.N_corr==True:
            self.nfixed = sum(self.fixed)
        elif self.N_corr==False:
            print("not making the correction on Ndim in the stretch move algorithm!")
            self.nfixed = 0

        self.inital_guess()
        logging.basicConfig(filename=self.savedir+"chain_info.log", level=logging.INFO, filemode='w')
        self.write_init()

    def inital_guess(self):
        p0 = [np.array(self.fid_theta) + self.init_gauss * np.random.randn(self.ndim) for i in range(self.nwalk)]
        if self.p0_corr==True:
            p0_fixed = []
            for i in range(self.nwalk):
                p0_fixed.append(np.where(self.fixed, self.fid_theta, p0[i]))
            self.p0 = p0_fixed

        elif self.p0_corr==False:
            print("allowing all walkers to step! make sure the likelyhood evaluation is correct!")
            self.p0 = p0

    def write_init(self):
        logging.info('This run was measured against data with truth values of %s', self.fid_theta)
        logging.info('It was initialized at %s with a gaussian width of %s', self.fid_theta, self.init_gauss)
        logging.info('The chain has %s walkers and %s steps', self.nwalk, self.nstep)
        logging.info('It was initialized with a_stretch = %s', self.a_stretch)

    def runit(self, lnprob, dtype):

        backend = emcee.backends.HDFBackend(self.savefile)
        if self.reset == True:
            backend.reset(self.nwalk, self.ndim)
            with Pool(self.ncores) as pool:
                sampler = emcee.EnsembleSampler(self.nwalk, self.ndim, lnprob, blobs_dtype=dtype, pool=pool, moves=emcee.moves.StretchMove(a=self.a_stretch, nf=self.nfixed), backend=backend)
                start = time.time()
                sampler.run_mcmc(self.p0, self.nstep, progress=True, skip_initial_state_check=self.p0_corr)
                end = time.time()
                multi_time = end - start

        elif self.reset == False:
            with Pool(self.ncores) as pool:
                sampler = emcee.EnsembleSampler(self.nwalk, self.ndim, lnprob, blobs_dtype=dtype, pool=pool, moves=emcee.moves.StretchMove(a=self.a_stretch, nf=self.nfixed), backend=backend)
                start = time.time()
                sampler.run_mcmc(None, self.nstep, progress=True, skip_initial_state_check=self.p0_corr)
                end = time.time()
                multi_time = end - start
        
        # print("Run took {0:.1f} hours".format(multi_time/3600))
        # print("saving some information from the sampler class")

        self.runtime = multi_time/3600
        self.acceptance_frac = np.mean(sampler.acceptance_fraction)
        try:
            self.tau = sampler.get_autocorr_time(quiet=True)
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

        logging.info('The run took %.1f hours', self.runtime)
        logging.info('The mean acceptance fraction turned out to be %s', self.acceptance_frac)
        logging.info('The auto correlation time (Nsteps) was %s', self.tau)
        logging.info('The final step in the chain gives the following constraints on theta')
        for i, val in enumerate(self.constraints):
            logging.info('%s=%s', self.labels[i], val)

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
            axs[i].axvline(self.fid_theta[i], ls=":", color="black")
        axs[0].set_ylabel("$\\chi^2$", fontsize=12)

        if self.savefig == True:
            plt.savefig(self.savedir+"chi2_final.png")

    # def plot_last_correlation(self, forward, data):
            
                    # plt.figure(figsize=(8, 8))
        # for i in Pnsat_mat:
        #     plt.plot(np.arange(i.shape[0]),i, color="grey", alpha=0.1)
        # plt.plot(np.arange(self.data.stat.Pnsat.shape[0]),self.data.stat.Pnsat, color="black")
        # plt.xlabel("Nsat", fontsize=15)
        # plt.ylabel("PDF", fontsize=15)
        # plt.xlim(0,30)
        # plt.savefig(self.savedir+"S1.png")

    #     self.forward = forward
    #     self.data = data
    #     _, _, _, rs = self.forward(self.last_samp[0])

    #     plt.figure(figsize=(8, 8))
    #     plt.hist(rs, alpha=0.3, color="grey")
    #     plt.axvline(np.average(rs), color="grey")
    #     plt.axvline(np.average(rs) + np.std(rs), ls="--", color="grey")
    #     plt.axvline(np.average(rs) - np.std(rs), ls="--", color="grey")
    #     plt.axvline(data.stat.r, color="black", ls=":")
    #     plt.xlabel("r (Nsat | max(Ms))")

    #     if self.savefig == True:
    #         plt.savefig(self.savedir+"S3.png")


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
            

class Chain:

    def __init__(self, h5_dir, fixed, Nburn, Nthin, Ncut=None, Nstack=None, **kwargs):
        self.dir = h5_dir
        self.Nstack = Nstack
        self.Nburn = Nburn
        self.Nthin = Nthin
        self.Ncut = Ncut
        self.fixed = fixed

        self.labels = np.array(["$M_{*}$", "$\\alpha$", "$\\beta$"," $\\gamma$", "$\\sigma$", "$\\nu$"])

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.read_chain()
        self.stack_thin()
        self.constrain()

    #     self.stack_end()
    #     self.cut_end()

    # def cut_end(self):
    #     if self.Ncut == None:
    #         self.Ncut = self.samples
    #     else:
    #         self.Ncut = self.samples[0:self.Ncut, :, :]

    # def stack_end(self):
    #     self.end = self.samples[-self.Nstack:, :, :].reshape(-1, self.samples.shape[2])

    def read_chain(self):
        reader = emcee.backends.HDFBackend(self.dir) 
        self.samples = reader.get_chain()
        self.chisq = reader.get_log_prob()*(-2)
        self.blobs = reader.get_blobs(flat=True)

    def stack_thin(self):
        self.thin = self.samples[self.Nburn::self.Nthin, :, :].reshape(-1, self.samples.shape[2])
        self.clean = self.thin.T[self.fixed].T

        self.chisq_thin = self.chisq[self.Nburn::self.Nthin, :].reshape(-1)

    def constrain(self):
        self.constraints = []
        for param in self.clean.T:
            post = np.percentile(param, [5, 50, 95])
            q = np.diff(post)
            self.constraints.append(f"${post[1]:.2f}_{{-{q[0]:.3f}}}^{{+{q[1]:.3f}}}$")

    def plot_posteriors(self, paper=False, **kwargs):
        self.Ndim = sum(self.fixed)
        if paper:
            figsize=7.0
        else:
            figsize=5*self.Ndim
        GTC = pygtc.plotGTC(chains=self.clean,
                        paramNames = self.labels[self.fixed],
                        figureSize= figsize,
                        customTickFont={'family':'Times', 'size':12},
                        customLegendFont={'family':'Times', 'size':12},
                        customLabelFont={'family':'Times', 'size':12},
                        mathTextFontSet=None,
                        panelSpacing='loose',
                        labelRotation=(False, False),
                        **kwargs)


class MulitChain:

    """
    A cleaner way to analyse production chains
    """

    def __init__(self, chains, chain_labels, fixed, **kwargs):
        self.chains = chains
        self.Nchain = len(self.chains)
        self.chain_labels = chain_labels
        self.fixed = fixed
        self.Ndim = sum(self.fixed)
        self.labels = np.array(["$M_{*}$", "$\\alpha$", "$\\beta$"," $\\gamma$", "$\\sigma$", "$\\nu$"])

        for key, value in kwargs.items():
            setattr(self, key, value)

    def plot_posteriors(self, paper=False, **kwargs):
        if paper:
            figsize=3.5
        else:
            figsize=3*self.Ndim
        GTC = pygtc.plotGTC(chains=self.chains,
                        paramNames = self.labels[self.fixed],
                        chainLabels = self.chain_labels,
                        figureSize= figsize,
                        customTickFont={'family':'Times', 'size':12},
                        customLegendFont={'family':'Times', 'size':12},
                        customLabelFont={'family':'Times', 'size':12},
                        mathTextFontSet=None,
                        panelSpacing='tight',
                        labelRotation=(False, False),
                        **kwargs)

    def violin(self, truths, model_labels, title, save_file=None):
        self.truths = truths
        self.model_labels = model_labels
        self.title = title

        fig, axes = plt.subplots(nrows=self.Ndim, ncols=1, figsize=(8, 10), sharex=True)
        axes[0].set_title(self.title, fontsize=15)

        # Loop through posteriors and create violin plots
        for j in range(self.Ndim):
            parts = axes[j].violinplot([self.chains[i][:, j] for i in range(self.Nchain)], showmeans=True, showextrema=False)
            axes[j].set_ylabel(self.labels[self.fixed][j], fontsize=15)
            axes[j].axhline(self.truths[j], ls=":", zorder=0, alpha=0.5, color="grey")
            #axes[j].set_ylim(self.priors[j][0], self.priors[j][1])

            for pc in parts['bodies']:
                pc.set_facecolor('grey')
                #pc.set_edgecolor('black')
                pc.set_alpha(0.6)

            parts['cmeans'].set_color('black')

        axes[-1].set_xticks(range(1,self.Nchain+1), labels=self.model_labels, fontsize=15)
        if save_file!=None:
            plt.savefig(save_file, bbox_inches="tight")
        plt.show()




#     def corner_last_sample(self, zoom=True):        
#         if zoom==True:
#             fig = corner.corner(self.last_samp, show_titles=True, labels=self.labels, fid_theta=self.data.fid_theta, quantiles=[0.15, 0.5, 0.85], plot_datapoints=False)
#         elif zoom==False:
#             fig = corner.corner(self.last_samp, show_titles=True, labels=self.labels, fid_theta=self.data.fid_theta, range=self.priors , quantiles=[0.15, 0.5, 0.85], plot_datapoints=False)
#         plt.savefig(self.savedir+"corner.png")

#     def chisq_last_sample(self):
#         fig, axs = plt.subplots(1, self.ndim, sharey=True, figsize=(14,8))
#         fig.tight_layout(pad=2.0)

#         for i in range(self.ndim):
#             axs[i].scatter(self.last_samp[:,i], self.last_chisq, marker=".")
#             axs[i].set_xlabel(self.labels[i], fontsize=12)
#             axs[i].axvline(self.data.fid_theta[i], ls=":", color="black")
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
#             self.fid_Ms = jsm_SHMR.simple([self.data.fid_theta[0], 0], self.halo_masses)
#             for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
#                 SHMR_mat[i] = jsm_SHMR.simple([val[0], 0], self.halo_masses)

#         elif self.SHMR_model =="anchor":
#             self.fid_Ms = jsm_SHMR.anchor([self.data.fid_theta[0], 0, self.data.fid_theta[2]], self.halo_masses)
#             for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
#                 SHMR_mat[i] = jsm_SHMR.anchor([val[0], 0, val[2]], self.halo_masses)

#         elif self.SHMR_model =="curve":
#             self.fid_Ms = jsm_SHMR.curve([self.data.fid_theta[0], 0, self.data.fid_theta[2],  self.data.fid_theta[3]], self.halo_masses)
#             for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
#                 SHMR_mat[i] = jsm_SHMR.curve([val[0], 0, val[2], val[3]], self.halo_masses)

#         elif self.SHMR_model =="sigma":
#             self.fid_Ms = jsm_SHMR.sigma([self.data.fid_theta[0], 0, self.data.fid_theta[2],  self.data.fid_theta[3], 0], self.halo_masses)
#             for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
#                 SHMR_mat[i] = jsm_SHMR.sigma([val[0], 0, val[2], val[3], 0], self.halo_masses)

#         elif self.SHMR_model =="redshift":
#             self.fid_Ms = jsm_SHMR.redshift([self.data.fid_theta[0], 0, self.data.fid_theta[2],  self.data.fid_theta[3], 0, self.data.fid_theta[5]], self.halo_masses, np.zeros(shape=self.halo_masses.shape[0]))
#             for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
#                 SHMR_mat[i] = jsm_SHMR.redshift([val[0], 0, val[2], val[3], 0, val[5]], self.halo_masses, np.zeros(shape=self.halo_masses.shape[0]))

#         else:
#             print("no model selected")
                
#         plt.figure(figsize=(10, 8))
#         for i,val in enumerate(SHMR_mat):
#             plt.plot(self.halo_masses, val, alpha=0.3, lw=1, color='grey')
#         plt.plot(self.halo_masses, self.fid_Ms, color="orange", label=str(self.data.fid_theta), lw=3)

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
