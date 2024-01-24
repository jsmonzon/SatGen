import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import jsm_SHMR
from multiprocess import Pool
import emcee
import time
import warnings; warnings.simplefilter('ignore')
import pygtc


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

class Hammer:

    """
    The machine I use to run an MCMC. It is a built from the emcee package.
    """

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
            print("allowing all walkers to step! make sure the likelyhood evaluation is correct!")
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
        if self.reset == True:
            backend.reset(self.nwalk, self.ndim)
            with Pool(self.ncores) as pool:
                sampler = emcee.EnsembleSampler(self.nwalk, self.ndim, lnprob, pool=pool, moves=emcee.moves.StretchMove(a=self.a_stretch, nf=self.nfixed), backend=backend)
                start = time.time()
                sampler.run_mcmc(self.p0, self.nstep, progress=True, skip_initial_state_check=self.p0_corr)
                end = time.time()
                multi_time = end - start

        elif self.reset == False:
            with Pool(self.ncores) as pool:
                sampler = emcee.EnsembleSampler(self.nwalk, self.ndim, lnprob, pool=pool, moves=emcee.moves.StretchMove(a=self.a_stretch, nf=self.nfixed), backend=backend)
                start = time.time()
                sampler.run_mcmc(None, self.nstep, progress=True, skip_initial_state_check=self.p0_corr)
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


    def plot_last_statfit(self, forward, data):

        self.forward = forward
        self.data = data
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
        plt.xlim(0,30)
        plt.savefig(self.savedir+"S1.png")

        plt.figure(figsize=(8, 8))
        for i, val in enumerate(Msmax_mat):
            plt.plot(val, Msmaxe_mat[i], color="grey", alpha=0.1)
        plt.plot(self.data.stat.Msmax, self.data.stat.ecdf_MsMax, color="black")
        plt.xlabel("stellar mass of most massive satellite ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
        plt.ylabel("CDF", fontsize=15)

        if self.savefig == True:
            plt.savefig(self.savedir+"S2.png")


    def plot_last_SHMR(self):

        self.halo_masses = np.linspace(6,12,50)
        SHMR_mat = np.zeros(shape=(self.last_samp.shape[0], self.halo_masses.shape[0]))

        det_z0 = self.ftheta
        det_z0[2], det_z0[3], det_z0[5] = 0,0,0
        self.fid_Ms = jsm_SHMR.general(self.ftheta, self.halo_masses, 0)

        for i,val in enumerate(self.last_samp):  # now pushing all thetas through!
            temp = val
            temp[2], temp[3], temp[5] = 0,0,0
            SHMR_mat[i] = jsm_SHMR.general(temp, self.halo_masses, 0)
                
        plt.figure(figsize=(10, 8))
        for i,val in enumerate(SHMR_mat):
            plt.plot(self.halo_masses, val, alpha=0.3, lw=1, color='grey')
        plt.plot(self.halo_masses, self.fid_Ms, color="orange", lw=3)

        plt.axhline(self.min_mass, label="mass limit", lw=1, ls=":", color="black")
        plt.scatter(self.data.lgMh_flat, self.data.lgMs_flat, marker=".", color="black")
        plt.ylim(self.min_mass-0.5,11)
        plt.xlim(8.5,12)
        plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
        plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
        plt.legend(fontsize=12)

        if self.savefig == True:
            plt.savefig(self.savedir+"SHMR.png")

class single_chain:

    def __init__(self, h5_dir, Nstack, Nburn, Nthin, truths=None, labels=None, plotfig=False):
        self.dir = h5_dir
        self.Nstack = Nstack
        self.Nburn = Nburn
        self.Nthin = Nthin
        self.truths = truths
        self.labels = labels

        self.read_chain()
        self.stack_thin()
        self.stack_end()

        if plotfig==True:
            self.plot_posteriors()

    def read_chain(self):
        reader = emcee.backends.HDFBackend(self.dir) 
        self.samples = reader.get_chain()

    def stack_thin(self):
        self.thin = self.samples[self.Nburn::self.Nthin, :, :].reshape(-1, self.samples.shape[2])

    def stack_end(self):
        self.end = self.samples[-self.Nstack:, :, :].reshape(-1, self.samples.shape[2])


    def plot_posteriors(self):
        GTC = pygtc.plotGTC(chains=self.end[:,0:self.Ndim],
                        paramNames = self.labels[0:self.Ndim],
                        truths = self.truths[0:self.Ndim],
                        nContourLevels=3,
                        figureSize=int(8*self.Ndim/3),
                        smoothingKernel=1.1,
                        filledPlots=True,
                        customTickFont={'family':'Arial', 'size':12},
                        customLegendFont={'family':'Arial', 'size':15},
                        customLabelFont={'family':'Arial', 'size':12})


class multi_chain:

    """
    A cleaner way to analyse production chains
    """

    def __init__(self, samplez, Ndim, truths, priors, plabels, mlabels, **kwargs):
        self.samplez = samplez
        self.Nchain = self.samplez.shape[0]
        self.Ndim = Ndim
        self.truths = truths
        self.priors = priors
        self.plabels = plabels
        self.mlabels = mlabels

        for key, value in kwargs.items():
            setattr(self, key, value)

    def trim(self):
        self.T_samplez = self.samplez[:,:,0:self.Ndim]
        self.T_truths = self.truths[0:self.Ndim]
        self.T_priors = self.priors[0:self.Ndim]
        self.T_plabels = self.plabels[0:self.Ndim]

    def plot_posteriors(self, save_file=None):
        GTC = pygtc.plotGTC(chains=self.T_samplez,
                        paramNames = self.T_plabels,
                        truths = self.T_truths,
                        paramRanges=self.T_priors,
                        chainLabels = self.mlabels,
                        figureSize=int(8*self.Ndim/3),
                        nContourLevels=self.nsigma,
                        smoothingKernel=self.smooth,
                        filledPlots=self.fill,
                        customTickFont={'family':'Arial', 'size':10},
                        customLegendFont={'family':'Arial', 'size':15},
                        customLabelFont={'family':'Arial', 'size':15},
                        plotName = save_file)
        

    def violin(self, N_param, save_file=None):

        fig, axes = plt.subplots(nrows=N_param, ncols=1, figsize=(6, 12), sharex=True)
        axes[0].set_title(self.title, fontsize=15)

        # Loop through posteriors and create violin plots
        for j in range(N_param):
            parts = axes[j].violinplot([self.samplez[i, :, j] for i in range(self.Nchain)], showmeans=True, showextrema=False)
            axes[j].set_ylabel(self.plabels[j])
            axes[j].axhline(self.truths[j], ls="--", lw=1, color="red")
            axes[j].set_ylim(self.priors[j][0], self.priors[j][1])

            for pc in parts['bodies']:
                pc.set_facecolor('cornflowerblue')
                pc.set_edgecolor('navy')
                pc.set_alpha(0.3)

        axes[-1].set_xticks(range(1,self.Nchain+1), labels=self.mlabels)
        if save_file!=None:
            plt.savefig(save_file, bbox_inches="tight")
        plt.show()




##################################################
###     TO INTERFACE WITH THE MCMC OUTPUT      ###
##################################################


    # def SHMR_colored(sample, SHMR_model, labels, color_ind, plot_data=True):
    #     halo_masses = np.log10(np.logspace(6, 13, 100))  # just for the model

    #     SHMR_mat = np.zeros(shape=(sample.shape[0], halo_masses.shape[0]))

    #     # Extract the color values for each data point
    #     colors = sample[:, color_ind]

    #     norm = mpl.colors.Normalize(vmin=colors.min(), vmax=colors.max())
    #     cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.magma_r)

    #     if SHMR_model == "simple":
    #         for i,val in enumerate(sample):  # now pushing all thetas through!
    #             SHMR_mat[i] = jsm_SHMR.simple([val[0], 0], halo_masses)

    #     elif SHMR_model =="anchor":
    #         for i,val in enumerate(sample):  # now pushing all thetas through!
    #             SHMR_mat[i] = jsm_SHMR.anchor([val[0], 0, val[2]], halo_masses)

    #     elif SHMR_model =="curve":
    #         for i,val in enumerate(sample):  # now pushing all thetas through!
    #             SHMR_mat[i] = jsm_SHMR.curve([val[0], 0, val[2], val[3]], halo_masses)

    #     elif SHMR_model =="sigma":
    #         for i,val in enumerate(sample):  # now pushing all thetas through!
    #             SHMR_mat[i] = jsm_SHMR.sigma([val[0], 0, val[2], val[3], 0], halo_masses)

    #     elif SHMR_model =="redshift":
    #         for i,val in enumerate(sample):  # now pushing all thetas through!
    #             SHMR_mat[i] = jsm_SHMR.redshift([val[0], 0, val[2], val[3], 0, val[5]], halo_masses, np.zeros(shape=halo_masses.shape[0]))

    #     plt.figure(figsize=(10, 8))
    #     for i, val in enumerate(SHMR_mat):
    #         plt.plot(halo_masses, val, color=cmap.to_rgba(colors[i]), alpha=0.3, lw=1)


    #     if plot_data==True:
    #         hmm = np.load("../analysis/model_test/mock_data.npy")
    #         plt.scatter(hmm[0], hmm[1], marker=".", color="grey")
    #         plt.axhline(6.5, label="mass limit", lw=3, ls=":", color="black")

            
    #     plt.ylim(4, 11)
    #     plt.xlim(7.5, 12)
    #     plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
    #     plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)

    #     # Create a colorbar using the ScalarMappable
    #     cbar = plt.colorbar(cmap, label=labels[color_ind])
    #     cbar.set_label(labels[color_ind])

    #     plt.show()




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
