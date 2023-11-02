import numpy as np
import sys 
sys.path.insert(0, '../')
import jsm_mcmc
import jsm_SHMR
import warnings; warnings.simplefilter('ignore')

print("intializing the run")

fid_theta = [1.95, -0.05, 0.25, 10]
priors = [[-2, 5],[-3, 2],[0, 5], [9,12]]
params = ["a_1", "a_2", "a_3", "a_4"]
ndim = len(fid_theta)

start_theta = [2, 0, 1, 10] 
nwalk = 300
nsteps = 1000
ncores = 12

massdir = "/home/jsm99/data/meta_data_psi3/"
datadir = "/home/jsm99/data/Nhosts/start5/"

min_mass = 6.5
a_stretch = 2.5

print("reading in the data")

data = jsm_mcmc.test_data(fid_theta, massdir+"jsm_MCMC.npy", datadir+"data.npy")
data.get_stats(min_mass=min_mass)

print("defining the forward model")

def forward(theta):
    models = jsm_mcmc.models(theta, jsm_SHMR.anchor, data.lgMh_models)
    models.get_stats(min_mass=min_mass)
    return models.stat.Pnsat, models.stat.Msmax, models.stat.ecdf_MsMax

def lnlike(theta):
    model_Pnsat, models_Msmax, _ = forward(theta)
    lnL_sat = jsm_mcmc.lnL_Pnsat(model_Pnsat, data.stat.satfreq)
    lnL_max = jsm_mcmc.lnL_KS(models_Msmax, data.stat.Msmax)
    return lnL_sat + lnL_max

def lnprior(theta):
    if priors[0][0] < theta[0] < priors[0][1] and priors[1][0] < theta[1] < priors[1][1] and priors[2][0] < theta[2] < priors[2][1]:
        lp = 0
    else:
        lp = -np.inf
    chi2_pr = ((theta[3] - 10.0) / 0.2) ** 2
    lnLPR = -chi2_pr / 2.0
    return lnLPR + lp

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike(theta)
    
print("running the MCMC!")

mcmc_out = jsm_mcmc.RUN(start_theta, lnprob, nwalkers=nwalk, niter=nsteps, ndim=ndim, ncores=ncores, a_stretch=a_stretch)
run = jsm_mcmc.inspect_run(mcmc_out, truths=data.truths, init_vals=start_theta, labels=params, priors=priors, savedir=datadir, data=data, SHMR=jsm_SHMR.anchor, forward=forward, min_mass=min_mass, a_stretch=a_stretch)
