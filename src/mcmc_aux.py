import numpy as np
import matplotlib.pyplot as plt
import corner
import anaclass
import galhalo

import warnings; warnings.simplefilter('ignore')

def fid_MODEL(lgMh_data, fid_theta):

    """_summary_
    the main model! goes from a SHMF mass to a CSMF and measures some statistics!

    Returns:
        np.ndarray: 1D model array populated with the statistics!
    """
    
    alpha, delta, sigma = fid_theta

    lgMs_2D = galhalo.SHMR(lgMh_data, alpha, delta, sigma) # will be a 3D array if sigma is non zero
    
    counts = np.apply_along_axis(anaclass.cumulative, 1, lgMs_2D)
    quant = np.percentile(counts, np.array([5, 50, 95]), axis=0, method="closest_observation") # median and scatter

    mass_ind = anaclass.find_nearest([6.5, 7, 7.5])

    model = [] # counts at the mass indicies
    for i in mass_ind:
        model.append(quant[2, i])
        model.append(quant[1, i])
        model.append(quant[0, i])

    return np.array(model)

class prep_run:

    """
    A class instance to steamline this same mcmc analysis to other data sets!
    """

    def __init__(self, datadir:str, Nsamp:int=100, fid_theta:list=[1.85, 0.2, 0.3], mass_bins:np.ndarray=np.linspace(4,11,45), mass_ind:np.ndarray=np.array([6.5,7.,7.5])):

        self.datadir = datadir
        self.fid_theta = fid_theta
        self.Nsamp = Nsamp
        self.mass_bins = mass_bins
        self.mass_ind = mass_ind
        

    def select_chunks(self, pick_samp=None):

        massmat = anaclass.MassMat(self.datadir+"acc_surv_mass.npy")
        massmat.prep_data()
        Nsets = int(massmat.lgMh.shape[0]/self.Nsamp) #dividing by the number of samples
        print("dividing your sample into", Nsets-1, "sets")
        set_ind = np.arange(0,Nsets)*self.Nsamp

        mat = np.zeros(shape=(Nsets-1, self.mass_ind.shape[0]*3))
        for i in range(Nsets-1):
            mat[i] = fid_MODEL(massmat.lgMh[set_ind[i]:set_ind[i+1]], self.fid_theta)

        self.D_mat = mat
        self.sampave = np.average(self.D_mat,axis=0)
        self.covariance = np.cov(self.D_mat, rowvar=False)
        self.sampstd = np.sqrt(np.diag(self.covariance))
        self.inv_covar = np.linalg.inv(self.covariance)

        if pick_samp!=None:
            self.lgMh = massmat.lgMh[pick_samp*self.Nsamp:pick_samp*self.Nsamp+self.Nsamp]
            self.D = self.D_mat[pick_samp]
        else:
            pick_samp = np.random.randint(mat.shape[0])
            print("chose a random sample to use as the real data!")
            self.lgMh = massmat.lgMh[pick_samp*self.Nsamp:pick_samp*self.Nsamp+self.Nsamp]
            self.D = self.D_mat[pick_samp]


    def pick_start(self, chi_dim=20):

        alpha_space = np.linspace(1,3,chi_dim)
        delta_space = np.linspace(-1,3,chi_dim)
        sigma_space = np.linspace(0,3,chi_dim)

        chi_mat = np.zeros(shape=(chi_dim,chi_dim,chi_dim))

        for i, aval in enumerate(alpha_space):
            for j, dval in enumerate(delta_space):
                for k, sval in enumerate(sigma_space):
                    model = fid_MODEL(self.lgMh, [aval, dval, sval])
                    chi_mat[i,j,k] = np.sum((model - self.D)**2/self.sampstd**2)

        ai, di, si = np.where(chi_mat == np.min(chi_mat))
        theta_0 = [alpha_space[ai][0], delta_space[di][0], sigma_space[si][0]]
        self.theta_0 = np.array(theta_0)

class inspect_run:

    def __init__(self, datadir:str, file:str):
        self.datadir = datadir
        self.samples = np.load(datadir+file)
        self.labels = ['$\\alpha$','$\\delta$','$\\sigma$']
        self.ndim = len(self.labels)


    def chain_plot(self):
        if self.samples.shape[0] > 500:
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

        
    def corner_plot(self, stack=False):
        
        if stack==True:
            fig = corner.corner(self.samples, show_titles=True, labels=self.labels, truths=[1.85, 0.2, 0.3],
                            range=[(1, 3), (-1, 3), (0, 3)], quantiles=[0.16, 0.5, 0.84], plot_datapoints=False)

        else:
            ind = self.samples.shape[0]-1
            fig = corner.corner(self.samples[ind], show_titles=True, labels=self.labels, truths=[1.85, 0.2, 0.3],
                            range=[(1, 3), (-1, 3), (0, 3)], quantiles=[0.16, 0.5, 0.84], plot_datapoints=False)
            
    def chisquare_plot(self):
        pass

    def SHMR_plot():
        pass

    def CSMF_plot():
        pass
