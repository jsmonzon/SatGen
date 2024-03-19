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
    massdir = "/home/jsm99/data/LN_meta_data_psi3/"
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

chain_name = "mock_min_mass/"
savedir = "../"+chain_name
savefile = savedir+"chain.h5"

# theta_0: the stellar mass anchor point (M_star_a)
# theta_1: power law slope (alpha)
# theta_2: log normal scatter (sigma)
# theta_3: slope of scatter as function of log halo mass (gamma)
# theta_4: quadratic term to curve the relation (beta)
# theta_5: redshift dependance on the quadratic term (tau)

fid_theta = [10.5, 2.3, 0.15, -0.1, 0.1, 0.6]
priors = [[10,11], [-1,7], [0,5], [-2,3], [-3,2], [0,7]]
labels = ["$M_{*}$", "$\\alpha$", "$\\sigma$"," $\\gamma$", "$\\beta$", "$\\tau$"]
fixed = [True, False, False, False, False, False]

hammer = jsm_mcmc.Hammer(fid_theta=fid_theta, fixed=fixed, nwalk=config["nwalk"], nstep=config["nstep"], ncores=config["ncores"],
                         a_stretch=config["a_stretch"], N_corr=config["N_corr"], p0_corr=config["p0_corr"], init_gauss=config["init_gauss"],
                         reset=config["reset"], savefig=config["savefig"], savedir=savedir, savefile=savefile, labels=labels)

print("reading in the data")

min_mass_HARD = 5.5

data = jsm_models.INIT_DATA(fid_theta, savedir+"/mock_data.npy")
data.get_stats(min_mass=min_mass_HARD)

print("defining the forward model")
models = jsm_models.LOAD_MODELS(massdir, Nsamples=config["Nsamp"])

def lnprior(theta):
    lp = 0.0
    for i, param in enumerate(theta):
        if not fixed[i] and not (priors[i][0] < param < priors[i][1]): # the flat priors
            return -np.inf
        if i == 0:
            lp += -(((param - 10.5) / 0.1) ** 2) / 2.0 # the gaussian prior on the anchor point
        elif i == 3 and param > 0 and not ((-theta[2] / param) + 12) < 9: # the positive mass dependance of sigma
            return -np.inf
        elif i == 4 and param > 0 and not ((-theta[1] / (2 * param)) + 12) < 9: # the curvature issue
            return -np.inf
    return lp

def lnlike(theta):
    models.get_stats(theta, min_mass_HARD, jsm_SHMR.general)
    lnL_Pnsat = jsm_stats.lnL_PNsat(data, models)
    lnL_KS_max = jsm_stats.lnL_KS_max(data, models)
    lnL = lnL_Pnsat + lnL_KS_max 
    return lnL, lnL_Pnsat, lnL_KS_max

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp): # if infinity
        return -np.inf, -np.inf, -np.inf
    ll, lnL_N, lnL_Mx = lnlike(theta)
    return lp + ll, lnL_N, lnL_Mx

dtype = [("lnL_N", float), ("lnL_Msmax", float)]

print("running the mcmc!")
hammer.runit(lnprob, dtype)

print("making some figures")
hammer.write_output()
hammer.plot_chain()
hammer.plot_last_chisq()
