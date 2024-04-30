print("Importing packages and loading config file")

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

elif location=="local":
    parentdir = "/Users/jsmonzon/Research/SatGen/mcmc/"

sys.path.insert(0, parentdir+"/src/")
import jsm_SHMR
import jsm_mcmc
import jsm_models
import jsm_stats

print("reading in the data")

chain_name = "model_2/"
savedir = "../"+chain_name
savefile = savedir+"chain.h5"

data = jsm_models.LOAD_DATA(savedir+"mock_data.npz")
data.get_stats(min_mass=config["min_mass"], max_N=config["max_N"])

print("Setting up the chain")

#     lgMh_2D (np.ndarray): 2D halo mass array
#     theta_0: the stellar mass anchor point (M_star_a)
#     theta_1: power law slope (alpha)
#     theta_2: quadratic term to curve the relation (beta)
#     theta_3: redshift dependance on the quadratic term (tau)
#     theta_4: log normal scatter (sigma)
#     theta_5: slope of scatter as function of log halo mass (gamma)

labels = ["$M_{*}$", "$\\alpha$", "$\\beta$"," $\\gamma$", "$\\sigma$", "$\\nu$"]
fixed = [True, False, True, True, False, False]

hammer = jsm_mcmc.Hammer(fid_theta=data.fid_theta, fixed=fixed, nwalk=config["nwalk"], nstep=config["nstep"], ncores=config["ncores"],
                         a_stretch=config["a_stretch"], N_corr=config["N_corr"], p0_corr=config["p0_corr"], init_gauss=config["init_gauss"],
                         reset=config["reset"], savefig=config["savefig"], savedir=savedir, savefile=savefile, labels=labels)

print("defining the forward model")
models = jsm_models.LOAD_MODELS(savedir+"remaining_models.npz")

def lnlike(theta):
    models.get_stats(theta=theta, min_mass=config["min_mass"], max_N=config["max_N"], Nsigma_samples=config["Nsamp"])
    lnL_Pnsat = jsm_stats.lnL_PNsat(data, models)
    lnL_KS_max = jsm_stats.lnL_KS_max(data, models)
    lnL = lnL_Pnsat + lnL_KS_max 
    return lnL, lnL_Pnsat, lnL_KS_max

def lnprob(theta):
    lp = jsm_SHMR.lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf, -np.inf, -np.inf
    ll, lnL_N, lnL_Mx = lnlike(theta)
    if not np.isfinite(ll):
        return -np.inf, -np.inf, -np.inf
    return lp + ll, lnL_N, lnL_Mx

dtype = [("lnL_N", float), ("lnL_Msmax", float)]

print("running the mcmc!")
hammer.runit(lnprob, dtype)

print("making some figures")
hammer.write_output()
hammer.plot_chain()
hammer.plot_last_chisq()