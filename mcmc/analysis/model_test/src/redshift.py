import numpy as np
import jsm_mcmc
import jsm_SHMR
import warnings; warnings.simplefilter('ignore')

print("intializing the redshift run")
ndim = 6 # need to change this for every run!

fid_theta_total = [2, 0.2, 10.5, 0, 0, 0]
priors_total = [[-1,7], [0,5], [9,11], [-3,2], [-2,2], [-2,2]]
params_total = ["slope", "sigma_0", "anchor", "curvature", "sigma", "redshift"]
start_theta_total = [1.8, 0.3, 10.5, -0.1, -0.01, 0]

fid_theta = fid_theta_total[0:ndim]
priors = priors_total[0:ndim]
params = params_total[0:ndim]
start_theta = start_theta_total[0:ndim]

nwalk = 30
nsteps = 50
ncores = 8

# massdir = "/home/jsm99/data/meta_data_psi3/"
# datadir = "/home/jsm99/data/model_complexity/fiducial/" # need to change this for every run!

massdir = "/Users/jsmonzon/Research/data/MW-analog/meta_data_psi3/"
datadir = "/Users/jsmonzon/Research/data/paper/model_test/redshift/"

min_mass = 6.5
a_stretch = 2.3

print("reading in the data")

data = jsm_mcmc.init_data(fid_theta, "/Users/jsmonzon/Research/data/paper/model_test/mock_data.npy")
data.get_stats(min_mass=min_mass, plot=False)
data.get_data_points(plot=False)

print("defining the forward model")

models = jsm_mcmc.load_models(massdir, read_red=True) # need to change this for every run!

def forward(theta):
    models.convert_zacc(theta, jsm_SHMR.redshift) # need to change this for every run!
    models.get_stats(min_mass=min_mass)
    return models.stat.Pnsat, models.stat.Msmax, models.stat.ecdf_MsMax

def lnlike(theta):
    model_Pnsat, models_Msmax, _ = forward(theta)
    lnL_sat = jsm_mcmc.lnL_Pnsat(model_Pnsat, data.stat.satfreq)
    lnL_max = jsm_mcmc.lnL_KS(models_Msmax, data.stat.Msmax)
    return lnL_sat + lnL_max

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

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike(theta)
    
print("running the MCMC!")

assert lnprior(start_theta) == 0.0

mcmc_out = jsm_mcmc.RUN(start_theta, lnprob, nwalkers=nwalk, niter=nsteps, ndim=ndim, ncores=ncores, a_stretch=a_stretch)
test = jsm_mcmc.inspect_run(mcmc_out, data=data, forward=forward, start_theta=start_theta,
                            labels=params, priors=priors, savedir=datadir, min_mass=min_mass, a_stretch=a_stretch, SHMR_model="redshift")