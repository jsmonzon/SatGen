################################ TreeGen_Sub ################################

# Generate halo merger trees using the Parkinson et al. (2008) algorithm.
# Slightly modified from original TreeGen to only produce quantities necessary
# for dark matter subhalo evolution. Additionally, this version introduces
# Zhao-Zhou Li's infall orbital parameter distribution.

# Arthur Fangzhou Jiang 2015 Yale University
# Arthur Fangzhou Jiang 2016 Hebrew University
# Arthur Fangzhou Jiang 2019 Hebrew University
# Sheridan Beckwith Green 2020 Yale University
# Sebastian Monzon 2022 Yale University

#---user modules
import config as cfg
import cosmo as co
import init
from profiles import NFW
import aux

#---python modules 
import numpy as np
import time 
from multiprocessing import Pool
import sys
import os

#---parameters! this user input is hared coded in!

target_mass = 12
mass_res = 5
zevo=0
Ntree=3
stree=0
final_path="/Users/jsmonzon/Research/data/MAHs/"

#---target halo and desired resolution 

lgM0 =  target_mass #- np.log10(cfg.h) # log10(Msun), corresponds to 10^12 Msun/h
cfg.psi_res = 10**(-mass_res)
z0 = zevo
lgMres = lgM0 + np.log10(cfg.psi_res) 

#---orbital parameter sampler preference
optype =  'zzli' # 'zzli' or 'zentner' or 'jiang'
#---concentration model preference
conctype = 'zhao' # 'zhao' or 'vdb'

# #---creating the data directory
# datadir_name = str(target_mass)+'_'+str(mass_res)+'_'+str(zevo)+'/'
# final_path = os.path.join(datadir, datadir_name)
# if not os.path.isdir(final_path):
#     os.makedirs(final_path)
# else:
#     print("already a directory")

#---now the iterable where all the calculations are made
def loop(itree): 

    time_start = time.time()

    name = "tree_" + str(itree) + ".npz"

    # check if this one has already been ran
    if os.path.exists(final_path+name):
        return
        
    np.random.seed() # [important!] reseed the random number generator

    cfg.M0 = 10.**lgM0
    cfg.z0 = z0
    cfg.Mres = 10.**lgMres 
    cfg.Mmin = 0.04*cfg.Mres

    k = 0               # the level, k, of the branch being considered
    ik = 0              # how many level-k branches have been finished
    Nk = 1              # total number of level-k branches
    Nbranch = 1         # total number of branches in the current tree

    Mak = [cfg.M0]      # accretion masses of level-k branches
    zak = [cfg.z0]
    idk = [0]           # branch ids of level-k branches
    ipk = [-1]          # parent ids of level-k branches (-1: no parent) 

    Mak_tmp = []
    zak_tmp = []
    idk_tmp = []
    ipk_tmp = []

    mass = np.zeros((cfg.Nmax,cfg.Nz)) - 99.
    order = np.zeros((cfg.Nmax,cfg.Nz),np.int8) - 99
    ParentID = np.zeros((cfg.Nmax,cfg.Nz),np.int16) - 99

    VirialRadius = np.zeros((cfg.Nmax,cfg.Nz),np.float32) - 99.
    concentration = np.zeros((cfg.Nmax,cfg.Nz),np.float32) - 99.
                                                        
    coordinates = np.zeros((cfg.Nmax,cfg.Nz,6),np.float32)

    while True: # loop over branches, until the full tree is completed.
    # Starting from the main branch, draw progenitor(s) using the 
    # Parkinson+08 algorithm. When there are two progenitors, the less 
    # massive one is the root of a new branch. We draw branches level by
    # level, i.e., When a new branch occurs, we record its root, but keep 
    # finishing the current branch and all the branches of the same level
    # as the current branch, before moving on to the next-level branches.

        M = [Mak[ik]]   # mass history of current branch in fine timestep
        z = [zak[ik]]   # the redshifts of the mass history
        cfg.M0 = Mak[ik]# descendent mass
        cfg.z0 = zak[ik]# descendent redshift
        id = idk[ik]    # branch id
        ip = ipk[ik]    # parent id
        
        while cfg.M0>cfg.Mmin:
        
            if cfg.M0>cfg.Mres: zleaf = cfg.z0 # update leaf redshift
        
            co.UpdateGlobalVariables(**cfg.cosmo)
            M1,M2,Np = co.DrawProgenitors(**cfg.cosmo)
            
            # update descendent halo mass and descendent redshift 
            cfg.M0 = M1
            cfg.z0 = cfg.zW_interp(cfg.W0+cfg.dW)
            if cfg.z0>cfg.zmax: break
            
            if Np>1 and cfg.M0>cfg.Mres: # register next-level branches
            
                Nbranch += 1
                Mak_tmp.append(M2)
                zak_tmp.append(cfg.z0)
                idk_tmp.append(Nbranch)
                ipk_tmp.append(id)
            
            # record the mass history at the original time resolution
            M.append(cfg.M0)
            z.append(cfg.z0)
        
        # Now that a branch is fully grown, do some book-keeping
        
        # convert mass-history list to array 
        M = np.array(M)
        z = np.array(z)
        
        # downsample the fine-step mass history, M(z), onto the
        # coarser output timesteps, cfg.zsample     
        Msample,zsample = aux.downsample(M,z,cfg.zsample)
        iz = aux.FindClosestIndices(cfg.zsample,zsample)
        if(isinstance(iz,np.int64)):
            iz = np.array([iz]) # avoids error in loop below
            zsample = np.array([zsample])
            Msample = np.array([Msample])
        izleaf = aux.FindNearestIndex(cfg.zsample,zleaf)
        # Note: zsample[j] is same as cfg.zsample[iz[j]]
        
        # compute halo structure throughout time on the coarse grid, up
        # to the leaf point
        t = co.t(z,cfg.h,cfg.Om,cfg.OL)
        c2,Rv = [],[]
        for i in iz:
            if i > (izleaf+1): break # only compute structure below leaf
            msk = z>=cfg.zsample[i]
            if True not in msk: break # safety
            c2i=init.c2_fromMAH(M[msk],t[msk],conctype)
            Rvi = init.Rvir(M[msk][0],Delta=cfg.Dvsample[i], z=cfg.zsample[i])
            c2.append(c2i)
            Rv.append(Rvi)
            #print('    i=%6i,ci=%8.2f,ai=%8.2f,log(Msi)=%8.2f,c2i=%8.2f'%\
            #    (i,ci,ai,np.log10(Msi),c2i)) # <<< for test
        c2 = np.array(c2) 
        Rv = np.array(Rv)
        Nc = len(c2) # length of a branch over which c2 is computed 
        
        
        # use the redshift id and parent-branch id to access the parent
        # branch's information at our current branch's accretion epoch,
        # in order to initialize the orbit
        if ip==-1: # i.e., if the branch is the main branch
            xv = np.zeros(6)
        else:
            Mp  = mass[ip,iz[0]] #mass of the host at z=zacc (iz[0] is the accretion index!!!)
            c2p = concentration[ip,iz[0]] #concentration of the host at z=zacc
            hp  = NFW(Mp,c2p,Delta=cfg.Dvsample[iz[0]],z=zsample[0])
            if(optype == 'zentner'):
                eps = 1./np.pi*np.arccos(1.-2.*np.random.random())
                xv  = init.orbit(hp,xc=1.,eps=eps)
            elif(optype == 'zzli'):
                vel_ratio, gamma = init.ZZLi2020(hp, Msample[0], zsample[0])
                xv = init.orbit_from_Li2020(hp, vel_ratio, gamma)
            elif(optype == 'jiang'):
                sp = NFW(Msample[0],c2[0],Delta=cfg.Dvsample[iz[0]],z=zsample[0])
                xv = init.orbit_from_Jiang2015(hp,sp,zsample[0])
        
        # <<< test
        #print('    id=%6i,k=%2i,z[0]=%7.2f,log(M[0])=%7.2f,c=%7.2f,a=%7.2f,c2=%7.2f,log(Ms)=%7.2f,Re=%7.2f,xv=%7.2f,%7.2f,%7.2f,%7.2f,%7.2f,%7.2f'%\
        #    (id,k,z[0],np.log10(M[0]),c[0],a[0],c2[0],np.log10(Ms),Re, xv[0],xv[1],xv[2],xv[3],xv[4],xv[5]))
        
        # update the arrays for output
        mass[id,iz] = Msample
        order[id,iz] = k
        ParentID[id,iz] = ip
        
        VirialRadius[id,iz[0]:iz[0]+Nc] = Rv
        concentration[id,iz[0]:iz[0]+Nc] = c2
        
        coordinates[id,iz[0],:] = xv
                
        # Check if all the level-k branches have been dealt with: if so, 
        # i.e., if ik==Nk, proceed to the next level.
        ik += 1
        if ik==Nk: # all level-k branches are done!
            Mak = Mak_tmp
            zak = zak_tmp
            idk = idk_tmp
            ipk = ipk_tmp
            Nk = len(Mak)
            ik = 0
            Mak_tmp = []
            zak_tmp = []
            idk_tmp = []
            ipk_tmp = []
            if Nk==0: 
                break # jump out of "while True" if no next-level branch 
            k += 1 # update level

    # trim and output 
    mass = mass[:id+1,:]
    order = order[:id+1,:]
    ParentID = ParentID[:id+1,:]
    VirialRadius = VirialRadius[:id+1,:]
    concentration = concentration[:id+1,:]
    coordinates = coordinates[:id+1,:,:]


    np.savez(final_path+name, 
        redshift = cfg.zsample,
        CosmicTime = cfg.tsample,
        mass = mass,
        order = order,
        ParentID = ParentID,
        VirialRadius = VirialRadius,
        concentration = concentration,
        coordinates = coordinates,
        )
    
    time_end = time.time()

    print('time elapsed for', name,':', ((time_end - time_start) / 60.), 'minutes')
    print('with', Nbranch, 'branches and ', k, 'orders')


if __name__ == "__main__":
    pool = Pool(6) # use as many as requested
    pool.map(loop, range(stree, stree+Ntree))#, chunksize=1)
