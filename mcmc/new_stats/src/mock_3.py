print("Importing!")

import sys
import numpy as np
import json

# Read configuration from the JSON file
with open("config.json", "r") as f:
    config = json.load(f)

# Access global variables from the configuration
location = config["location"]
a_stretch = config["a_stretch"]
nwalk = config["nwalk"]
nstep = config["nstep"]
ncores = config["ncores"]
min_mass = config["min_mass"]
Nsamp = config["Namp"]
N_corr = config["N_corr"]
p0_corr = config["p0_corr"]
savefig = config["savefig"]
reset = config["reset"]

if location=="server":
    massdir = "/home/jsm99/data/meta_data_psi3/"
    parentdir = "/home/jsm99/SatGen/mcmc/"

elif location=="local":
    parentdir = "/Users/jsmonzon/Research/SatGen/mcmc/"
    massdir = "/Users/jsmonzon/Research/data/MW-analog/meta_data_psi3/"

sys.path.insert(0, parentdir+"/src/")
import jsm_SHMR
import jsm_mcmc
import jsm_models
import jsm_stats

print("Setting up walkers")

chain_name = "mock_3/"
savedir = "../"+chain_name
savefile = savedir+"chain.h5"

# theta_0: the stellar mass anchor point (M_star_a)
# theta_1: power law slope (alpha)
# theta_2: log normal scatter (sigma)
# theta_3: slope of scatter as function of log halo mass (gamma)
# theta_4: quadratic term to curve the relation (beta)
# theta_5: redshift dependance on the quadratic term (tau)

fid_theta = [10.5, 1.75, 0.2, 0, 0, 0]
priors = [[10,11], [-1,7], [0,5], [-2,3], [-3,2], [-3,2]]
labels = ["$M_{*}$", "$\\alpha$", "$\\sigma$"," $\\gamma$", "$\\beta$", "$\\tau$"]
fixed = [True, False, False, True, True, True]

ndim = len(fid_theta)
nfixed = sum(fixed)

hammer = jsm_mcmc.Hammer(ftheta=fid_theta, gtheta=fid_theta, fixed=fixed, ndim=ndim, nwalk=nwalk, nstep=nstep, ncores=ncores,
                         a_stretch=a_stretch, min_mass=min_mass, N_corr=N_corr, p0_corr=p0_corr, savedir=savedir, savefile=savefile,
                        labels=labels, savefig=savefig, reset=reset)

print("reading in the data")

data = jsm_models.init_data(fid_theta, savedir+"/mock_data.npy")
data.get_stats(min_mass=min_mass)

print("defining the forward model")
models = jsm_models.load_models(massdir, Nsamples=Nsamp)

def lnprior(theta):
    chi2_pr = ((theta[0] - 10.5) / 0.2) ** 2
    if priors[1][0] < theta[1] < priors[1][1] and\
        priors[2][0] < theta[2] < priors[2][1] and\
         priors[3][0] < theta[3] < priors[3][1] and\
          priors[4][0] < theta[4] < priors[4][1] and\
           priors[5][0] < theta[5] < priors[5][1]:
        lp = 0.0
    else:
        lp = -np.inf
    return lp + (-chi2_pr / 2.0)

def lnlike(theta):
    models.get_stats(theta, min_mass, jsm_SHMR.general)
    lnL_Pnsat = jsm_stats.lnL_PNsat(models, data)
    lnL_KS = np.nansum(jsm_stats.lnL_KS(models, data))
    lnL = lnL_Pnsat + lnL_KS
    return lnL, lnL_Pnsat, lnL_KS

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf, -np.inf, -np.inf
    ll, lnL_N, lnL_M = lnlike(theta)
    if not np.isfinite(ll):
        return lp, -np.inf, -np.inf
    return lp + ll, lnL_N, lnL_M
    
print("running the mcmc!")
hammer.runit(lnprob)

print("making some figures")
hammer.write_output()
hammer.plot_chain()
hammer.plot_last_chisq()
