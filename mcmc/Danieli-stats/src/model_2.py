print("Importing!")

import sys
import numpy as np
import json

# Read configuration from the JSON file
with open("config.json", "r") as f:
    config = json.load(f)

# Access global variables from the configuration
location = config["location"]
if location=="server":
    parentdir = "/home/jsm99/SatGen/mcmc/"
    massdir = "/home/jsm99/data/meta_data_psi3/Danieli-stats/model_2/"

elif location=="local":
    parentdir = "/Users/jsmonzon/Research/SatGen/mcmc/"
    massdir = "/Users/jsmonzon/Research/data/MW-analog/meta_data_psi3/"

sys.path.insert(0, parentdir+"/src/")
import jsm_SHMR
import jsm_mcmc
import jsm_models
import jsm_stats

print("Setting up walkers")

chain_name = "model_2/"
savedir = "../"+chain_name
savefile = savedir+"chain.h5"

# theta_0: the stellar mass anchor point (M_star_a)
# theta_1: power law slope (alpha)
# theta_2: log normal scatter (sigma)
# theta_3: slope of scatter as function of log halo mass (gamma)
# theta_4: quadratic term to curve the relation (beta)
# theta_5: redshift dependance on the quadratic term (tau)

fid_theta = [10.5, 2.0, 0.2, 0, 0, 0]
priors = [[10,11], [-1,7], [0,5], [-2,3], [-3,2], [-3,2]]
labels = ["$M_{*}$", "$\\alpha$", "$\\sigma$"," $\\gamma$", "$\\beta$", "$\\tau$"]
fixed = [True, False, False, True, True, True]
N_bin=31

hammer = jsm_mcmc.Hammer(fid_theta=fid_theta, fixed=fixed, nwalk=config["nwalk"], nstep=config["nstep"], ncores=config["ncores"],
                         a_stretch=config["a_stretch"], N_corr=config["N_corr"], p0_corr=config["p0_corr"], init_gauss=config["init_gauss"],
                         reset=config["reset"], savefig=config["savefig"], savedir=savedir, savefile=savefile, labels=labels)

print("reading in the data")

data = jsm_models.INIT_DATA(fid_theta, savedir+"/mock_data.npy")
data.get_nad_stats(min_mass=config["min_mass"], N_bin=N_bin)

print("defining the forward model")
models = jsm_models.LOAD_MODELS(massdir, Nsamples=config["Nsamp"])

def lnprior(theta):
    chi2_pr = ((theta[0] - 10.5) / 0.1) ** 2
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
    models.get_nad_stats(theta, config["min_mass"], N_bin=N_bin)
    lnL = jsm_stats.lnL_Nadler(data, models)
    return lnL

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)

print("running the mcmc!")
hammer.runit(lnprob)

print("making some figures")
hammer.write_output()
hammer.plot_chain()
hammer.plot_last_chisq()
