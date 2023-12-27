import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
    
#parentdir = "/home/jsm99/SatGen/mcmc/"
parentdir = "/Users/jsmonzon/Research/SatGen/mcmc/"

import sys 
sys.path.insert(0, parentdir+"/src/")
import jsm_SHMR
import jsm_mcmc
import jsm_models
import jsm_stats

print("Setting up the run")

chain_name = "model_6-5/"
savedir = parentdir+"model_runs/survey/MinMass/"+chain_name
savefile = savedir+"chain.h5"

# theta_0: the stellar mass anchor point (M_star_a)
# theta_1: power law slope (alpha)
# theta_2: log normal scatter (sigma)
# theta_3: slope of scatter as function of log halo mass (gamma)
# theta_4: quadratic term to curve the relation (beta)
# theta_5: redshift dependance on the quadratic term (tau)


fid_theta = [10.5, 1.9, 0.2, 0, 0, 0]
priors = [[10,11], [-1,7], [0,5], [-2,2], [-3,2], [-2,2]]
labels = ["$M_{*}$", "$\\alpha$", "$\\sigma$"," $\\gamma$", "$\\beta$", "$\\tau$"]
fixed = [False, False, False, True, True, True]

ndim = len(fid_theta)
N_corr = True
p0_corr = True
nfixed = sum(fixed)
if nfixed == 0:
    print("not holding any parameters fixed!")
    N_corr = False
    p0_corr = False

a_stretch = 2.3
nwalk = 100
nstep = 2000
ncores = 16
min_mass = 6.5


hammer = jsm_mcmc.Hammer(ftheta=fid_theta, gtheta=fid_theta, fixed=fixed, ndim=ndim, nwalk=nwalk, nstep=nstep, ncores=ncores,
                         a_stretch=a_stretch, min_mass=min_mass, N_corr=N_corr, p0_corr=p0_corr, savedir=savedir, savefile=savefile,
                        labels=labels, savefig=True, reset=True)

print("reading in the data")
massdir = "/Users/jsmonzon/Research/data/MW-analog/meta_data_psi4/"
#massdir = "/home/jsm99/data/meta_data_psi4/"

data = jsm_models.init_data(fid_theta, parentdir+"model_runs/survey/MinMass/mock_data.npy")

data.get_stats(min_mass=min_mass, plot=False)
data.get_data_points(plot=False)

print("defining the forward model")
models = jsm_models.load_models(massdir, read_red=True) # need to change this for every run!

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

def forward(theta):
    models.convert_zacc(theta, jsm_SHMR.general)
    models.get_stats(min_mass=min_mass)
    return models.stat.Pnsat, models.stat.Msmax, models.stat.ecdf_MsMax

def lnlike(theta):
    model_Pnsat, models_Msmax, _ = forward(theta)
    lnL_sat = jsm_stats.lnL_Pnsat(model_Pnsat, data.stat.satfreq)
    lnL_max = jsm_stats.lnL_KS(models_Msmax, data.stat.Msmax)
    return lnL_sat + lnL_max

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike(theta)
    
print("running the mcmc!")
hammer.runit(lnprob)

print("making some figures")
hammer.write_output()
hammer.plot_chain()
hammer.plot_last_chisq()
hammer.plot_last_statfit(forward, data)
hammer.plot_last_SHMR()