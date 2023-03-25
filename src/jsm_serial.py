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

def generate(target_mass:float, mass_res:float, zevo:float, Ntree:int, stree:int, datadir:str):
    """_summary_

    Args:
        target_mass (float): _description_
        mass_res (float): _description_
        zevo (float): _description_
        Ntree (int): _description_
        stree (int): _description_
        datadir (str): _description_
    """

    #---user modules
    import config as cfg
    import cosmo as co
    import init
    from profiles import NFW
    import aux

    #---python modules
    import numpy as np
    import time 
    #from multiprocessing import Pool, cpu_count
    import sys
    import os

    #---target halo and desired resolution 

    lgM0 =  target_mass - np.log10(cfg.h) # log10(Msun), corresponds to 10^12 Msun/h
    cfg.psi_res = 10**(-mass_res)
    z0 = zevo
    lgMres = lgM0 + np.log10(cfg.psi_res) 

    #---orbital parameter sampler preference
    optype =  'zzli' # 'zzli' or 'zentner' or 'jiang'

    #---concentration model preference
    conctype = 'zhao' # 'zhao' or 'vdb'

    #---creating the data directory
    datadir_name = str(target_mass)+'_'+str(mass_res)+'_'+str(zevo)+'/'
    final_path = os.path.join(datadir, datadir_name)
    if not os.path.isdir(final_path):
        os.makedirs (final_path)
    else:
        print("already a directory")

    #---now to run the loop!
    for itree in range(stree, stree+Ntree): 

        time_start = time.time()

        name = "tree_" + str(itree) + ".npz"

        print("now seeding tree", itree)
        np.random.seed() # [important!] reseed the random number generator
        
        cfg.M0 = 10.**lgM0
        cfg.z0 = z0
        cfg.Mres = 10.**lgMres 
        cfg.Mmin = 0.04*cfg.Mres #this is the leaf mass
        
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
                Mp  = mass[ip,iz[0]]
                c2p = concentration[ip,iz[0]]
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

        # now to save!
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



################################ SubEvo #################################

# Program that evolves the subhaloes intialized by TreeGen_Sub.py
# This version of the code is meant to work with the Green model of
# stripped subhalo density profiles.

# Arthur Fangzhou Jiang 2015 Yale University
# Arthur Fangzhou Jiang 2016-2017 Hebrew University
# Arthur Fangzhou Jiang 2020 Caltech
# Sheridan Beckwith Green 2020 Yale University
# Sebastian Monzon 2022 Yale University
# -- Changed loop order so that redshift is the outermost loop,
#    which enables mass of ejected subhaloes to be removed from
#    the corresponding host; necessary for mass conservation

def evolve(datadir:str, mass_res:float, evo_mode:str="arbres", rad_res:float=3, fric_frac:float=0.75):

    """
    mass_res: the mass resolution that the halos will be evolved down to! (float)
    datadir: path to the unevovled merger trees (str)
    rad_res: radial resolution, set by default value (float)
    fric_frac: dynamical friction stength, set by default value (float)
    """


    import config as cfg
    import cosmo as co
    import evolve as ev
    from profiles import NFW,Green
    from orbit import orbit
    import aux

    import numpy as np
    import sys
    import os 
    import time
    from multiprocessing import Pool, cpu_count


    # <<< for clean on-screen prints, use with caution, make sure that 
    # the warning is not prevalent or essential for the result
    import warnings
    #warnings.simplefilter('always', UserWarning)
    warnings.simplefilter("ignore", UserWarning)


    Rres_factor = 10**-(rad_res) # (Defunct)

    #---stripping efficiency type
    alpha_type = 'conc' # 'fixed' or 'conc'

    #---dynamical friction strength
    cfg.lnL_pref = fric_frac # Fiducial, but can also use 1.0

    #---evolution mode (resolution limit in m/m_{acc} or m/M_0)
    cfg.evo_mode = evo_mode # 'arbres' or 'withering'
    cfg.phi_res = 10**(-mass_res) # when cfg.evo_mode == 'arbres',
    #                        cfg.phi_res sets the lower limit in m/m_{acc}
    #                        that subhaloes evolve down until

    ########################### evolve satellites ###########################

    #---get the list of data files
    files = []    
    for filename in os.listdir(datadir):
        if filename.startswith('tree') and filename.endswith('.npz'): 
            files.append(os.path.join(datadir, filename))
    files.sort()

    print("evolving", len(files), "tree realizations")
    print(cfg.evo_mode)

    #---
    for file in files: # <<< serial run, only for testing
        """
        Replaces the loop "for file in files:", for parallelization.
        """

        time_start = time.time() 
        name = file[0:-4]+"evo" 
        print("evolving", file)
        
        #---load trees
        f = np.load(datadir+file)
        
        #---load trees
        f = np.load(file)
        redshift = f['redshift']
        CosmicTime = f['CosmicTime']
        mass = f['mass']
        order = f['order']
        ParentID = f['ParentID']
        VirialRadius = f['VirialRadius']
        concentration = f['concentration']
        coordinates = f['coordinates']

        # compute the virial overdensities for all redshifts
        VirialOverdensity = co.DeltaBN(redshift, cfg.Om, cfg.OL) # same as Dvsample
        GreenRte = np.zeros(VirialRadius.shape) - 99. # contains r_{te} values
        alphas = np.zeros(VirialRadius.shape) - 99.
        tdyns  = np.zeros(VirialRadius.shape) - 99.

        #---identify the roots of the branches
        izroot = mass.argmax(axis=1) # root-redshift ids of all the branches
        idx = np.arange(mass.shape[0]) # branch ids of all the branches
        levels = np.unique(order[order>=0]) # all >0 levels in the tree
        izmax = mass.shape[1] - 1 # highest redshift index

        #---get smallest host rvir from tree
        #   Defunct, we no longer use an Rres; all subhaloes are evolved
        #   until their mass falls below resolution limit
        min_rvir = VirialRadius[0, np.argwhere(VirialRadius[0,:] > 0)[-1][0]]
        cfg.Rres = min(0.1, min_rvir * Rres_factor) # Never larger than 100 pc

        #---list of potentials and orbits for each branch
        #   additional, mass of ejected subhaloes stored in ejected_mass
        #   to be removed from corresponding host at next timestep
        potentials = [0] * mass.shape[0]
        orbits = [0] * mass.shape[0]
        trelease = np.zeros(mass.shape[0])
        ejected_mass = np.zeros(mass.shape[0])

        #---list of minimum masses, below which we stop evolving the halo
        M0 = mass[0,0]
        min_mass = np.zeros(mass.shape[0])

        #---evolve
        for iz in np.arange(izmax, 0, -1): # loop over time to evolve
            iznext = iz - 1                
            z = redshift[iz]
            tcurrent = CosmicTime[iz]
            tnext = CosmicTime[iznext]
            dt = tnext - tcurrent
            Dv = VirialOverdensity[iz]

            for level in levels: #loop from low-order to high-order systems
                for id in idx: # loop over branches
                    if order[id,iz]!=level: continue # level by level
                    if(iz <= izroot[id]):
                        if(iz == izroot[id]): # accretion happens at this timestep
                            # initialize Green profile and orbit

                            za = z
                            ta = tcurrent
                            Dva = Dv
                            ma = mass[id,iz] # initial mass that we will use for f_b
                            c2a = concentration[id,iz]
                            xva = coordinates[id,iz,:]

                            # some edge case produces nan in velocities in TreeGen
                            # if so, print warning and mass fraction lost
                            if(np.any(np.isnan(xva))):
                                print('    WARNING: NaNs detected in init xv of id %d'\
                                    % id)
                                print('    Mass fraction of tree lost: %.1e'\
                                    % (ma/mass[0,0]))
                                mass[id,:] = -99.
                                coordinates[id,:,:] = 0.
                                idx = np.delete(idx, np.argwhere(idx == id)[0])
                                # this is an extremely uncommon event, but should
                                # eventually be fixed
                                continue

                            potentials[id] = Green(ma,c2a,Delta=Dva,z=za)
                            orbits[id] = orbit(xva)
                            trelease[id] = ta

                            if cfg.evo_mode == 'arbres':
                                min_mass[id] = cfg.phi_res * ma
                            elif cfg.evo_mode == 'withering':
                                min_mass[id] = cfg.psi_res * M0

                        #---main loop for evolution

                        # the p,s,o objects are updated in-place in their arrays
                        # unless the orbit is replaced with a new object when released
                        ip = ParentID[id,iz]
                        p = potentials[ip]
                        s = potentials[id]

                        # update mass of subhalo object based on mass-loss in previous snapshot
                        # we wait to do it until now so that the pre-stripped subhalo can be used
                        # in the evolution of any higher-order subhaloes
                        # We also strip off the mass of any ejected systems
                        # the update_mass function handles cases where we fall below resolution limit
                        if(s.Mh > min_mass[id]):
                            if(ejected_mass[id] > 0):
                                mass[id,iz] -= ejected_mass[id]
                                ejected_mass[id] = 0
                                mass[id,iz] = max(mass[id,iz], cfg.phi_res*s.Minit)

                            s.update_mass(mass[id,iz])
                            rte = s.rte()

                        o = orbits[id]
                        xv = orbits[id].xv
                        m = s.Mh
                        m_old = m
                        r = np.sqrt(xv[0]**2+xv[2]**2)

                        #---time since in current host
                        t = tnext - trelease[id]

                        # Order should always be one higher than parent unless 
                        # ejected,in which case it should be the same as parent
                        k = order[ip,iznext] + 1

                        # alpha: stripping efficiency
                        if(alpha_type == 'fixed'):
                            alpha = 0.55
                        elif(alpha_type == 'conc'):
                            alpha = ev.alpha_from_c2(p.ch, s.ch)

                        #---evolve satellite
                        # as long as the mass is larger than resolution limit
                        if m > min_mass[id]:

                            # evolve subhalo properties
                            m,lt = ev.msub(s,p,xv,dt,choice='King62',
                                alpha=alpha)

                        else: # we do nothing about disrupted satellite, s.t.,
                            # its properties right before disruption would be 
                            # stored in the output arrays
                            pass

                        #---evolve orbit
                        if m > min_mass[id]:
                            # NOTE: We previously had an additional check on r>Rres
                            # here, where Rres = 10^-3 Rvir(z), but I removed it
                            # All subhalo orbits are evolved until their mass falls
                            # below the resolution limit.
                            # NOTE: No use integrating orbit any longer once the halo
                            # is disrupted, this just slows it down
                        
                            tdyn = p.tdyn(r)
                            o.integrate(t,p,m_old)
                            xv = o.xv # note that the coordinates are updated 
                            # internally in the orbit instance "o" when calling
                            # the ".integrate" method, here we assign them to 
                            # a new variable "xv" only for bookkeeping
                            
                        else: # i.e., the satellite has merged to its host, so
                            # no need for orbit integration; to avoid potential 
                            # numerical issues, we assign a dummy coordinate that 
                            # is almost zero but not exactly zero
                            tdyn = p.tdyn(cfg.Rres)
                            xv = np.array([cfg.Rres,0.,0.,0.,0.,0.])

                        r = np.sqrt(xv[0]**2+xv[2]**2)
                        m_old = m


                        #---if order>1, determine if releasing this high-order 
                        #   subhalo to its grandparent-host, and if releasing,
                        #   update the orbit instance
                        if k>1:
                        
                            if (r > VirialRadius[ip,iz]) & (iz <= izroot[ip]): 
                                # <<< Release condition:
                                # 1. Host halo is already within a grandparent-host
                                # 2. Instant orbital radius is larger than the host
                                # TIDAL radius (note that VirialRadius also contains
                                # the tidal radii for the host haloes once they fall
                                # into a grandparent-host)
                                # 3. (below) We compute the fraction of:
                                #             dynamical time / alpha
                                # corresponding to this dt, and release with
                                # probability dt / (dynamical time / alpha)

                                # Compute probability of being ejected
                                odds = np.random.rand()
                                dyntime_frac = alphas[ip,iz] * dt / tdyns[ip,iz]
                                if(odds < dyntime_frac):
                                    if(ParentID[ip,iz] == ParentID[ip,iznext]):
                                        # host wasn't also released at same time
                                        # New coordinates at next time are the
                                        # updated subhalo coordinates plus the updated
                                        # host coordinates inside of grandparent
                                        xv = aux.add_cyl_vecs(xv,coordinates[ip,iznext,:])
                                    else:
                                        xv = aux.add_cyl_vecs(xv,coordinates[ip,iz,:])
                                        # This will be extraordinarily rare, but just
                                        # a check in case so that the released order-k
                                        # subhalo isn't accidentally double-released
                                        # in terms of updated coordinates, but not
                                        # in terms of new host ID.
                                    orbits[id] = orbit(xv) # update orbit object
                                    k = order[ip,iz] # update instant order to the same as the parent
                                    ejected_mass[ip] += m 
                                    # add updated subhalo mass to a bucket to be removed from host
                                    # at start of next timestep
                                    ip = ParentID[ip,iz] # update parent id
                                    trelease[id] = tnext # update release time

                        #---update the arrays for output
                        mass[id,iznext] = m
                        order[id,iznext] = k
                        ParentID[id,iznext] = ip
                        try:
                            VirialRadius[id,iznext] = lt # storing tidal radius
                        except UnboundLocalError:
                            # TreeGen gives a few subhaloes with root mass below the
                            # given resolution limit so some subhaloes will never get
                            # an lt assigned if they aren't evolved one step. This can
                            # be fixed by lowering the resolution limit of SubEvo
                            # relative to TreeGen by some tiny epsilon, say 0.05 dex
                            print("No lt for id ", id, "iz ", iz, "masses ",
                                np.log10(mass[id,iz]), np.log10(mass[id,iznext]), file)
                            return

                        # NOTE: We store tidal radius in lieu of virial radius
                        # for haloes after they start getting stripped
                        GreenRte[id,iz] = rte 
                        coordinates[id,iznext,:] = xv

                        # NOTE: the below two are quantities at current timestep
                        # instead, since only used for host release criteria
                        # This won't be output since only used internally
                        alphas[id,iz] = alpha
                        tdyns[id,iz] = tdyn

                    else: # before accretion, halo is an NFW profile
                        if(concentration[id,iz] > 0): 
                            # the halo has gone above tree mass resolution
                            # different than SatEvo mass resolution by small delta
                            potentials[id] = NFW(mass[id,iz],concentration[id,iz],
                                                Delta=VirialOverdensity[iz],z=redshift[iz])

        #---output
        np.savez(datadir+name, 
            redshift = redshift,
            CosmicTime = CosmicTime,
            mass = mass,
            order = order,
            #ParentID = ParentID,
            VirialRadius = VirialRadius,
            #GreenRte = GreenRte,
            # this contains values during stripping, -99 prior to stripping and
            # once the halo falls below the resolution limit
            #concentration = concentration, # this is unchanged from TreeGen output
            #coordinates = coordinates,
            )
        
        time_end = time.time()
        print('time elapsed for', name,':', ((time_end - time_start) / 60.), 'minutes')
