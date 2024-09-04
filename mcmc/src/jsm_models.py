import numpy as np
import matplotlib.pyplot as plt
import jsm_stats
import warnings; warnings.simplefilter('ignore')
import jsm_SHMR


class SAMPLE_SAGA_NEW:

    def __init__(self, fid_theta:list, meta_path:str, SAGA_ind:int, Nhost_M=None, savedir=None, savemodel=None, plot=False):
        self.fid_theta = fid_theta
        self.meta_path = meta_path
        self.SAGA_ind = SAGA_ind
        self.Nhost_M = Nhost_M
        self.savedir = savedir
        self.savemodel = savemodel
        self.plot = plot
        
        models = np.load(self.meta_path+"models.npz") # already broken up into SAGA samples!

        print("selecting the", str(self.SAGA_ind), " SAGA sample")
        self.lgMh_data = models["mass"][self.SAGA_ind] 

        print("converting the subhalos to satellites and creating the mock data instance")
        self.lgMs_data = jsm_SHMR.general_new(fid_theta, self.lgMh_data, 0, 1)
        if self.savedir != None:
            print("saving the mock data")
            np.savez(self.savedir+"mock_data.npz",
                    halo_mass = self.lgMh_data,
                    stellar_mass = self.lgMs_data,
                    fid_theta = self.fid_theta,
                    SAGA_ind = self.SAGA_ind)
            
        print("breaking off the remaining samples and creating the model instance")
        if self.Nhost_M != None:
            print("Only using the first", str(self.Nhost_M), "trees for the model")
            self.lgMh_models = np.vstack(np.delete(models["mass"], self.SAGA_ind, axis=0))[0:self.Nhost_M]
            if self.savemodel != None:
                print("saving the models")
                np.savez_compressed(self.savemodel+"remaining_models.npz",
                        halo_mass = self.lgMh_models,
                        Nhost_M = self.Nhost_M)    

        else:
            self.lgMh_models = np.vstack(np.delete(models["mass"], self.SAGA_ind, axis=0))
            if self.savemodel != None:
                print("saving the models")
                np.savez_compressed(self.savemodel+"remaining_models.npz",
                        halo_mass = self.lgMh_models,
                        Nhost_M = 10000)    
                    
        if self.plot:
            self.lgMh_data_flat = self.lgMh_data.flatten()
            self.lgMs_data_flat = self.lgMs_data.flatten()
            plt.figure(figsize=(8, 8))
            plt.title("$\\theta_{\mathrm{fid}}$ = "+f"{self.fid_theta}")
            plt.scatter(self.lgMh_data_flat, self.lgMs_data_flat, marker="*", color="black")
            plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
            plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
            plt.xlim(8.5, 12)
            plt.ylim(6.0, 10.5)
            if self.savedir != None:
                plt.savefig(self.savedir+"mock_SHMR.pdf")
            plt.show()


class SAMPLE_SAGA_MODELS:

    def __init__(self, fid_theta: list, meta_path: str, extra_path: str, SAGA_ind: int, reshape=None, verbose=True):
        self.fid_theta = fid_theta
        self.meta_path = meta_path
        self.extra_path = extra_path
        self.verbose = verbose
        
        if extra_path is not None:
            models = np.load(self.meta_path+"models.npz")  # already broken up into SAGA samples!
            models_extra = np.load(self.extra_path+"models.npz")  # already broken up into SAGA samples!

        if reshape is not None:
            if self.verbose:
                print("reshaping the array, it should still be 3D!!")
            self.lgMh = models["mass"].reshape([reshape[0], reshape[1], models["mass"].shape[2]])
            self.lgMh_extra = models_extra["mass"].reshape([reshape[0], reshape[1], models_extra["mass"].shape[2]]) 
        else:
            self.lgMh = models["mass"]
            self.lgMh_extra = models_extra["mass"]

        if self.verbose:
            print("selecting the", str(SAGA_ind), " SAGA sample")
        self.lgMh_data = self.lgMh[SAGA_ind]

        if self.verbose:
            print("Grabbing the extra trees to make the model and collapsing to 2D!")
        self.lgMh_model = np.vstack(self.lgMh_extra)

        if self.verbose:
            print("converting the subhalos to satellites")
        self.lgMs_data = jsm_SHMR.general_new(fid_theta, self.lgMh_data, 0, 1)
        self.lgMs_model = jsm_SHMR.general_new(fid_theta, self.lgMh_model, 0, 1)


class SAMPLE_CROSS:

    def __init__(self, fid_theta: list, meta_path: str, SAGA_ind: int, reshape=None, verbose=True):
        self.fid_theta = fid_theta
        self.meta_path = meta_path
        self.verbose = verbose
        
        models = np.load(self.meta_path+"models.npz")  # already broken up into SAGA samples!

        if reshape is not None:
            if self.verbose:
                print("reshaping the array, it should still be 3D!!")
            self.lgMh = models["mass"].reshape([reshape[0], reshape[1], models["mass"].shape[2]])
        else:
            self.lgMh = models["mass"]

        if self.verbose:
            print("selecting the", str(SAGA_ind), " SAGA sample")
        self.lgMh_data = self.lgMh[SAGA_ind]

        if self.verbose:
            print("Grabbing the remaining trees to make the model and collapsing to 2D!")
        self.lgMh_model = np.vstack(np.delete(self.lgMh, SAGA_ind, axis=0))

        if self.verbose:
            print("converting the subhalos to satellites")
        self.lgMs_data = jsm_SHMR.general_new(fid_theta, self.lgMh_data, 0, 1)
        self.lgMs_model = jsm_SHMR.general_new(fid_theta, self.lgMh_model, 0, 1)

        # self.Nreal_per_model = Nreal_per_model

        # #print(f"there are {self.N_model_realizations} unique model realizations, each made up of {self.Nreal_per_model} merger trees")
        # self.Nreal_extra = self.lgMh_models.shape[0] % self.Nreal_per_model
        #     #print(f"there are {self.Nreal_extra} extra merger trees, deleting these unused trees")
        #     if self.Nreal_extra == 0:
        #         self.lgMh_models = self.lgMh_models.reshape([self.N_model_realizations, self.Nreal_per_model, self.lgMh_models.shape[1]])
        #     else:
        #         self.lgMh_models = np.delete(self.lgMh_models, np.arange(self.Nreal_extra), axis=0) 
        #         self.lgMh_models = self.lgMh_models.reshape([self.N_model_realizations, self.Nreal_per_model, self.lgMh_models.shape[1]])

        #     else:
        #         print("breaking off the remaining samples and creating the model instance")    
        #         self.lgMh_models = np.vstack(np.delete(models["mass"], SAGA_ind, axis=0))
        #         self.zacc_models = np.vstack(np.delete(models["redshift"], SAGA_ind, axis=0))

        #     if savedir != None:
        #         print("saving the models")
        #         np.savez_compressed(self.savedir+"remaining_models.npz",
        #                 halo_mass = self.lgMh_models,
        #                 zacc = self.zacc_models)
            
        # elif SAGA_ind > 100:
        #     N_SAGA = int(SAGA_ind/100)
        #     print("selecting the first", N_SAGA, "SAGA samples")
        #     self.lgMh_data = np.vstack(models["mass"][0:N_SAGA]) 
        #     self.zacc_data = np.vstack(models["redshift"][0:N_SAGA])

        #     print("converting the subhalos to satellites and creating the mock data instance")
        #     if redshift_depandance == True:
        #         self.lgMs_data = jsm_SHMR.general_new(fid_theta, self.lgMh_data, self.zacc_data, self.Nsigma_samples)
        #     else:
        #         self.lgMs_data = jsm_SHMR.general_new(fid_theta, self.lgMh_data, 0, self.Nsigma_samples)

        #     if savedir != None:
        #         print("saving the mock data")
        #         np.savez(self.savedir+"mock_data.npz",
        #                 halo_mass = self.lgMh_data,
        #                 stellar_mass = self.lgMs_data,
        #                 zacc = self.zacc_data,
        #                 fid_theta = self.fid_theta)
        
        #     self.lgMh_data_flat = self.lgMh_data.flatten()
        #     self.lgMs_data_flat = self.lgMs_data.flatten()
        #     # plt.figure(figsize=(8, 8))
        #     # plt.title("$\\theta_{\mathrm{fid}}$ = "+f"{self.fid_theta}")
        #     # plt.scatter(self.lgMh_data_flat, self.lgMs_data_flat, marker="*", color="black")
        #     # plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
        #     # plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
        #     # plt.xlim(8.5, 12)
        #     # plt.ylim(6.0, 10.5)
        #     # if savedir != None:
        #     #     plt.savefig(self.savedir+"mock_SHMR.pdf")
        #     # plt.show()

        #     print("breaking off the remaining samples and creating the model instance")    
        #     self.lgMh_models = np.vstack(np.delete(models["mass"], np.arange(N_SAGA), axis=0))
        #     self.zacc_models = np.vstack(np.delete(models["redshift"], np.arange(N_SAGA), axis=0))

        #     if savedir != None:
        #         print("saving the models")
        #         np.savez_compressed(self.savedir+"remaining_models.npz",
        #                 halo_mass = self.lgMh_models,
        #                 zacc = self.zacc_models)

class LOAD_DATA:

    def __init__(self, dfile:str):
        self.load_file = np.load(dfile)
        self.lgMh_data = self.load_file["halo_mass"]
        self.lgMs_data = self.load_file["stellar_mass"]
        self.fid_theta = self.load_file["fid_theta"]

    def get_stats(self, min_mass:float, max_N:float):
        self.min_mass = min_mass
        self.max_N = max_N
        self.stat = jsm_stats.SatStats_D(self.lgMs_data, self.min_mass, self.max_N)

    def get_NADLER_stats(self, min_mass:float, max_mass:float, N_bin:int):
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.N_bin = N_bin
        self.stat = jsm_stats.SatStats_D_NADLER(self.lgMs_data, min_mass=min_mass, max_mass=self.max_mass, N_bin=N_bin)

class LOAD_MODELS:

    def __init__(self, mfile:str):    
        self.mfile = mfile
        models = np.load(mfile)
        self.lgMh_models = models["halo_mass"]
        try:
            self.zacc_models = models["zacc"]
        except KeyError:
            print("no redshift info supplied")

    def get_stats(self, theta:list, min_mass:float, max_N:float, Nsigma_samples=1):
        self.theta = theta
        self.min_mass = min_mass
        self.max_N = max_N
        self.Nsigma_samples = Nsigma_samples

        if theta[3] == 0:
            self.lgMs = jsm_SHMR.general_new(theta, self.lgMh_models, 0, self.Nsigma_samples)
        else:
            self.lgMs = jsm_SHMR.general_new(theta, self.lgMh_models, self.zacc_models, self.Nsigma_samples)

        self.stat = jsm_stats.SatStats_M(self.lgMs, self.min_mass, self.max_N)

    def get_NADLER_stats(self, theta:list, min_mass:float, max_mass:float, N_bin:int, Nsigma_samples=1):
        self.theta = theta
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.N_bin = N_bin
        self.Nsigma_samples = Nsigma_samples

        if theta[3] == 0:
            self.lgMs = jsm_SHMR.general_new(theta, self.lgMh_models, 0, self.Nsigma_samples)
        else:
            self.lgMs = jsm_SHMR.general_new(theta, self.lgMh_models, self.zacc_models, self.Nsigma_samples)

        self.Nreal = self.lgMs.shape[0]/100
        self.lgMs_mat = np.array(np.split(self.lgMs, self.Nreal, axis=0))
        self.stat = jsm_stats.SatStats_M_NADLER(self.lgMs_mat, min_mass=self.min_mass, max_mass=self.max_mass, N_bin=self.N_bin)



        # def plot_data(self, save=True):
        #     self.lgMh_data_flat = self.lgMh_data.flatten()
        #     self.lgMs_data_flat = self.lgMs_data.flatten()
        #     plt.scatter(self.lgMh_data_flat, self.lgMs_data_flat, marker="*")
        #     plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
        #     plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
        #     plt.axhline(6.5, ls="--")
        #     plt.show()

        # elif type(SAGA_ind) == list:
        #     print("selecting more than one SAGA sample, not creating a new modelz instance")
        #     self.lgMh_data = np.vstack(models["mass"][SAGA_ind[0]:SAGA_ind[1]]) # select the SAGA indices
        #     self.zacc_data = np.vstack(models["redshift"][SAGA_ind[0]:SAGA_ind[1]])
        #     print("you have", self.lgMh_data.shape, "merger tree realizations in the data")
        #     if redshift_depandance == True:
        #         self.lgMs_data = jsm_SHMR.general_new(fid_theta, self.lgMh_data, self.zacc_data, self.Nsigma_samples)
        #     else:
        #         self.lgMs_data = jsm_SHMR.general_new(fid_theta, self.lgMh_data, 0, self.Nsigma_samples)

        # elif type(SAGA_ind) == float:
        #     print("selecting less than one SAGA sample, not creating a new modelz instance")
        #     self.lgMh_data = np.vstack(models["mass"])[0:int(SAGA_ind)] 
        #     self.zacc_data = np.vstack(models["redshift"])[0:int(SAGA_ind)]
        #     print("you have", self.lgMh_data.shape, "merger tree realizations in the data")
        #     if redshift_depandance == True:
        #         self.lgMs_data = jsm_SHMR.general_new(fid_theta, self.lgMh_data, self.zacc_data, self.Nsigma_samples)
        #     else:
        #         self.lgMs_data = jsm_SHMR.general_new(fid_theta, self.lgMh_data, 0, self.Nsigma_samples)



# class SAGA_sample:

#     def __init__(self, Nsamp, fid_theta:list, SAGA_ind:int, savedir:str, redshift_depandance=False):
#         self.fid_theta = fid_theta
#         self.savedir = savedir
#         sample = jsm_halopull.MassMat("../../../data/MW-analog/meta_data_psi4/", Nsamp=Nsamp, save=False, plot=False)
#         self.lgMh_data = sample.acc_surv_lgMh_mat[SAGA_ind] # select the SAGA index
#         self.zacc_data = sample.acc_red_mat[SAGA_ind]
#         if redshift_depandance == True:
#             self.lgMs_data = jsm_SHMR.general_new(fid_theta, self.lgMh_data, self.zacc_data)
#         else:
#             self.lgMs_data = jsm_SHMR.general_new(fid_theta, self.lgMh_data)

#     def get_data_points(self, plot=True):
#         self.lgMh_flat = self.lgMh_data.flatten()
#         self.lgMs_flat = self.lgMs_data.flatten()
#         if plot==True:
#             plt.scatter(self.lgMh_flat, self.lgMs_flat, marker=".")
#             plt.ylabel("M$_{*}$ (M$_\odot$)", fontsize=15)
#             plt.xlabel("M$_{\mathrm{vir}}$ (M$_\odot$)", fontsize=15)
    
#     def save_data(self):
#         np.save(self.savedir+"mock_data.npy", np.array([self.lgMh_data, self.lgMs_data, self.zacc_data]))