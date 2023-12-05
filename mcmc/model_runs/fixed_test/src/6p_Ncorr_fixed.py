import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
    
import sys 
#sys.path.insert(0, '/Users/jsmonzon/Research/SatGen/mcmc/src/')
sys.path.insert(0, '/home/jsm99/SatGen/mcmc/src')
import jsm_SHMR
import jsm_mcmc

print("Setting up the run")

chain_name = "6p_Ncorr_fixed/"

fid_theta = [1.9, 0.3, 10.5, 0, 0, 0]
priors = [[-1,7], [0,5], [10,11], [-3,2], [-2,2], [-2,2]]
labels = ["$\\alpha$", "$\\sigma_0$", "$M_{*}$", "$\\delta$", "$\\beta$", "$\\gamma$"]
fixed = [False, False, True, True, True, True]

ndim = len(fid_theta)
nfixed = sum(fixed)
N_corr = True
p0_corr = True

a_stretch = 2.0
nwalk = 100
nstep = 1000
ncores = 16
min_mass = 6.5

#savedir = "/Users/jsmonzon/Research/SatGen/mcmc/model_runs/fixed_test/"+chain_name
savedir = "/home/jsm99/SatGen/mcmc/model_runs/fixed_test/"+chain_name
savefile = savedir+"chain.h5"

hammer = jsm_mcmc.Hammer(ftheta=fid_theta, gtheta=fid_theta, fixed=fixed, ndim=ndim, nwalk=nwalk, nstep=nstep, ncores=ncores,
                         a_stretch=a_stretch, min_mass=min_mass, N_corr=N_corr, p0_corr=p0_corr, savedir=savedir, savefile=savefile, labels=labels, savefig=True)

print("reading in the data")
#massdir = "/Users/jsmonzon/Research/data/MW-analog/meta_data_psi3/"
massdir = "/home/jsm99/data/meta_data_psi3/"

#data = jsm_mcmc.init_data(fid_theta, "/Users/jsmonzon/Research/SatGen/mcmc/model_runs/fixed_test/mock_data.npy")
data = jsm_mcmc.init_data(fid_theta, "/home/jsm99/SatGen/mcmc/model_runs/fixed_test/mock_data.npy")

data.get_stats(min_mass=min_mass, plot=False)
data.get_data_points(plot=False)

print("defining the forward model")
models = jsm_mcmc.load_models(massdir, read_red=True) # need to change this for every run!

def lnprior(theta):
    if priors[0][0] < theta[0] < priors[0][1] and\
        priors[1][0] < theta[1] < priors[1][1] and\
         priors[2][0] < theta[2] < priors[2][1] and\
          priors[3][0] < theta[3] < priors[3][1] and\
           priors[4][0] < theta[4] < priors[4][1] and\
            priors[5][0] < theta[5] < priors[5][1]:
        lp = 0.0
    else:
        lp = -np.inf
    return lp

def forward(theta):
    models.convert_zacc(theta, jsm_SHMR.general)
    models.get_stats(min_mass=min_mass)
    return models.stat.Pnsat, models.stat.Msmax, models.stat.ecdf_MsMax

def lnlike(theta):
    model_Pnsat, models_Msmax, _ = forward(theta)
    lnL_sat = jsm_mcmc.lnL_Pnsat(model_Pnsat, data.stat.satfreq)
    lnL_max = jsm_mcmc.lnL_KS(models_Msmax, data.stat.Msmax)
    return lnL_sat + lnL_max

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike(theta)
    
print("running the mcmc!")

hammer.runit(lnprob)
hammer.write_output()

    
