########################### master_cluster_2 ############################
#
# cluster_lens, cluster 2: 1.8e15 Msun host at z_obs = 0.439, resolution
# scaled with host mass (Mres = M_floor = 1.8e10 = 1e-5 M0). Otherwise
# identical to master_cluster_1.py (cluster 1: 5e14 Msun at z = 0.436).
#
# Grow AND evolve N SatGen trees for a single host mass observed at
# z_obs > 0, in one script. Each worker grows one tree (Parkinson+08,
# Benson+21 params) and then immediately evolves it (Green+19 profiles,
# Jiang/Green stripping, Chandrasekhar DF) down to z_obs.
#
# Physics is copied from src/jsm_SubGen_bolshoi_rep.py (tree growth) and
# src/jsm_SubEvo_masscut.py (evolution). Differences, all deliberate:
#   1. The output redshift grid is rebuilt to START at z_obs (config.py
#      builds it from z=0 at import). Without this, the host row is -99
#      below z_obs and subhaloes keep evolving in a frozen host to z=0.
#   2. Cosmology is taken unchanged from config.py (BolshoiP).
#   3. Evolution floor is an ABSOLUTE mass, M_floor, not phi_res*m_acc.
#   4. ParentID stored as int32 (int16 overflows at 32767 branches;
#      psi_res = 1e-5 trees get close).
#   5. lt / rte are reset per branch-step, so branches that are not
#      evolved get -99 instead of silently inheriting the previous
#      branch's tidal radius (and no UnboundLocalError tree abort).
#
# Sebastian Monzon 2026 Yale University
#########################################################################

import os, sys, time, warnings
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))   # SatGen/src

import numpy as np
from multiprocessing import Pool

import config as cfg
import cosmo as co
import init
import aux
import evolve as ev
from profiles import NFW, Green
from orbit import orbit

warnings.simplefilter("ignore", UserWarning)

########################### user control ################################

datadir   = "/netb/vdbosch/jsm99/data/cluster_lens/cluster_2/"
Ntree     = 5
ncores    = 5          # one tree per core
seed_base = None       # int -> tree i uses seed_base+i (reproducible)

#---host
M0    = 1.8e15          # [Msun] host Mvir (Bryan & Norman 98) at z_obs
z_obs = 0.439

#---tree resolution
Mres      = 1.8e10       # smallest branch registered as a subhalo [Msun]
Mmin_frac = 0.04       # main-branch MAHs followed to Mmin = 0.04*Mres
                       # (needed for the Zhao+09 c(MAH) integral)

#---evolution resolution
M_floor = 1.8e10         # stop evolving once m < M_floor, any m_acc [Msun]

#---model choices
conctype     = 'zhao'    # 'zhao' or 'vdb' (Zhao+09 c from MAH)
optype       = 'zzli'    # 'zzli', 'zentner', 'jiang' infall orbits
alpha_type   = 'conc'    # stripping efficiency: 'fixed' (0.55) or 'conc'
tidal_choice = 'King62'  # 'King62' or 'Tormen98'
lnL_pref     = 1.0       # DF Coulomb-log multiplier (0.75 = Green+20)
lnL_type     = 0         # 0: lnL = ln(M_host/m), instantaneous
Rres_factor  = 1e-4      # defunct spatial floor, kept for consistency

########################## output z-grid ###############################

def set_output_grid(z0):
    """Rebuild config.py's output redshift grid so it starts at z0
    instead of z=0 (same step rule: dt = min(0.06 Gyr, 0.1 t_dyn)).
    Cosmology itself (BolshoiP in config.py) is untouched."""
    zs, dts, z = [z0], [], z0
    while z <= cfg.zmax:
        tlkbk = co.tlkbk(z, cfg.h, cfg.Om, cfg.OL)
        dt = min(0.06, 0.1*co.tdyn(z, cfg.h, cfg.Om, cfg.OL))
        z = cfg.ztlkbk_interp(tlkbk + dt)
        zs.append(z); dts.append(dt)
    dts.append(0.)
    cfg.zsample     = np.array(zs)
    cfg.dtsample    = np.array(dts)
    cfg.Wsample     = co.deltac(cfg.zsample, cfg.Om)
    cfg.tlkbksample = co.tlkbk(cfg.zsample, cfg.h, cfg.Om, cfg.OL)
    cfg.tsample     = co.t(cfg.zsample, cfg.h, cfg.Om, cfg.OL)
    cfg.Dvsample    = co.DeltaBN(cfg.zsample, cfg.Om, cfg.OL)
    cfg.Omsample    = co.Omega(cfg.zsample, cfg.Om, cfg.OL)
    cfg.Nz          = len(cfg.zsample)

# module level on purpose: workers inherit it under fork AND spawn
set_output_grid(z_obs)
cfg.lnL_pref = lnL_pref
cfg.lnL_type = lnL_type
# Green+19 tables stop at m/m_acc = 1e-5 (cfg.fbv_min); profiles.py and
# evolve.py clamp the bound fraction at cfg.phi_res, so it must not go
# below the table edge. Subhaloes with m_acc > M_floor/1e-5 would hit
# that edge first (impossible here: M_floor/1e-5 = 1.8e15 = M0).
cfg.phi_res  = cfg.fbv_min


############################ grow a tree ################################

def grow_tree(itree):
    name = os.path.join(datadir, f"tree_{itree}")
    if os.path.exists(name + ".npz"):
        print(f"[grow {itree}] exists, skipping", flush=True)
        return name + ".npz"

    t0 = time.time()
    np.random.seed(None if seed_base is None else seed_base + itree)

    cfg.M0, cfg.z0 = M0, z_obs
    cfg.Mres = Mres
    cfg.Mmin = Mmin_frac*Mres

    k, ik, Nk, Nbranch = 0, 0, 1, 1
    Mak, zak, idk, ipk = [cfg.M0], [cfg.z0], [0], [-1]
    Mak_tmp, zak_tmp, idk_tmp, ipk_tmp = [], [], [], []

    mass          = np.zeros((cfg.Nmax, cfg.Nz)) - 99.
    order         = np.zeros((cfg.Nmax, cfg.Nz), np.int8) - 99
    ParentID      = np.zeros((cfg.Nmax, cfg.Nz), np.int32) - 99
    VirialRadius  = np.zeros((cfg.Nmax, cfg.Nz), np.float32) - 99.
    concentration = np.zeros((cfg.Nmax, cfg.Nz), np.float32) - 99.
    coordinates   = np.zeros((cfg.Nmax, cfg.Nz, 6), np.float32)

    while True:
        M, z = [Mak[ik]], [zak[ik]]
        cfg.M0, cfg.z0 = Mak[ik], zak[ik]
        id, ip = idk[ik], ipk[ik]

        while cfg.M0 > cfg.Mmin:
            if cfg.M0 > cfg.Mres: zleaf = cfg.z0
            co.UpdateGlobalVariables(**cfg.cosmo)
            M1, M2, Np = co.DrawProgenitors(**cfg.cosmo)
            cfg.M0 = M1
            cfg.z0 = cfg.zW_interp(cfg.W0 + cfg.dW)
            if cfg.z0 > cfg.zmax: break
            if Np > 1 and cfg.M0 > cfg.Mres:
                Nbranch += 1
                Mak_tmp.append(M2); zak_tmp.append(cfg.z0)
                idk_tmp.append(Nbranch); ipk_tmp.append(id)
            M.append(cfg.M0); z.append(cfg.z0)

        M, z = np.array(M), np.array(z)
        Msample, zsample = aux.downsample(M, z, cfg.zsample)
        iz = aux.FindClosestIndices(cfg.zsample, zsample)
        if isinstance(iz, np.int64):
            iz = np.array([iz])
            zsample = np.array([zsample]); Msample = np.array([Msample])
        izleaf = aux.FindNearestIndex(cfg.zsample, zleaf)

        t = co.t(z, cfg.h, cfg.Om, cfg.OL)
        c2, Rv = [], []
        for i in iz:
            if i > (izleaf + 1): break
            msk = z >= cfg.zsample[i]
            if True not in msk: break
            c2.append(init.c2_fromMAH(M[msk], t[msk], conctype))
            Rv.append(init.Rvir(M[msk][0], Delta=cfg.Dvsample[i],
                                z=cfg.zsample[i]))
        c2, Rv = np.array(c2), np.array(Rv)
        Nc = len(c2)

        if ip == -1:
            xv = np.zeros(6)
        else:
            Mp  = mass[ip, iz[0]]
            c2p = concentration[ip, iz[0]]
            hp  = NFW(Mp, c2p, Delta=cfg.Dvsample[iz[0]], z=zsample[0])
            if optype == 'zentner':
                eps = 1./np.pi*np.arccos(1. - 2.*np.random.random())
                xv  = init.orbit(hp, xc=1., eps=eps)
            elif optype == 'zzli':
                vel_ratio, gamma = init.ZZLi2020(hp, Msample[0], zsample[0])
                xv = init.orbit_from_Li2020(hp, vel_ratio, gamma)
            elif optype == 'jiang':
                sp = NFW(Msample[0], c2[0], Delta=cfg.Dvsample[iz[0]],
                         z=zsample[0])
                xv = init.orbit_from_Jiang2015(hp, sp, zsample[0])

        mass[id, iz] = Msample
        order[id, iz] = k
        ParentID[id, iz] = ip
        VirialRadius[id, iz[0]:iz[0]+Nc] = Rv
        concentration[id, iz[0]:iz[0]+Nc] = c2
        coordinates[id, iz[0], :] = xv

        ik += 1
        if ik == Nk:
            Mak, zak, idk, ipk = Mak_tmp, zak_tmp, idk_tmp, ipk_tmp
            Nk, ik = len(Mak), 0
            Mak_tmp, zak_tmp, idk_tmp, ipk_tmp = [], [], [], []
            if Nk == 0: break
            k += 1

    n = id + 1
    np.savez(name,
        redshift=cfg.zsample, CosmicTime=cfg.tsample,
        mass=mass[:n], order=order[:n], ParentID=ParentID[:n],
        VirialRadius=VirialRadius[:n], concentration=concentration[:n],
        coordinates=coordinates[:n],
        # run metadata
        M0=M0, z_obs=z_obs, Mres=Mres, Mmin=cfg.Mmin, M_floor=M_floor,
        cosmo=np.array([cfg.h, cfg.Om, cfg.Ob, cfg.OL, cfg.s8, cfg.ns]),
        conctype=conctype, optype=optype)
    print(f"[grow {itree}] {Nbranch} branches, {k} orders, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    return name + ".npz"

########################### evolve a tree ###############################

def evolve_tree(file):
    name = file[:-4] + "_evo"
    if os.path.exists(name + ".npz"):
        print(f"[evolve] {os.path.basename(name)} exists, skipping",
              flush=True)
        return
    t0 = time.time()

    f = np.load(file)
    redshift      = f['redshift']
    CosmicTime    = f['CosmicTime']
    mass          = f['mass']
    order         = f['order']
    ParentID      = f['ParentID']
    VirialRadius  = f['VirialRadius']
    concentration = f['concentration']
    coordinates   = f['coordinates']

    VirialOverdensity = co.DeltaBN(redshift, cfg.Om, cfg.OL)
    GreenRte = np.zeros(VirialRadius.shape) - 99.
    alphas   = np.zeros(VirialRadius.shape) - 99.
    tdyns    = np.zeros(VirialRadius.shape) - 99.

    izroot = mass.argmax(axis=1)
    idx    = np.arange(mass.shape[0])
    levels = np.unique(order[order >= 0])
    izmax  = mass.shape[1] - 1

    min_rvir = VirialRadius[0, np.argwhere(VirialRadius[0, :] > 0)[-1][0]]
    cfg.Rres = min(0.1, min_rvir*Rres_factor)

    potentials    = [0]*mass.shape[0]
    orbits        = [0]*mass.shape[0]
    trelease      = np.zeros(mass.shape[0])
    ejected_mass  = np.zeros(mass.shape[0])
    min_mass      = np.full(mass.shape[0], M_floor)
    n_born_below  = 0

    for iz in np.arange(izmax, 0, -1):
        iznext = iz - 1
        z = redshift[iz]
        tcurrent, tnext = CosmicTime[iz], CosmicTime[iznext]
        dt = tnext - tcurrent
        Dv = VirialOverdensity[iz]

        for level in levels:
            for id in idx:
                if order[id, iz] != level: continue
                if iz <= izroot[id]:
                    if iz == izroot[id]:
                        za, ta, Dva = z, tcurrent, Dv
                        ma  = mass[id, iz]
                        c2a = concentration[id, iz]
                        xva = coordinates[id, iz, :]
                        if np.any(np.isnan(xva)):
                            print(f"    WARNING: NaN init xv, id {id}, "
                                  f"m/M0 = {ma/mass[0,0]:.1e}", flush=True)
                            mass[id, :] = -99.
                            coordinates[id, :, :] = 0.
                            idx = np.delete(idx, np.argwhere(idx == id)[0])
                            continue
                        potentials[id] = Green(ma, c2a, Delta=Dva, z=za)
                        orbits[id] = orbit(xva)
                        trelease[id] = ta
                        min_mass[id] = max(M_floor, cfg.phi_res*ma)
                        if ma <= M_floor: n_born_below += 1

                    lt, rte = -99., -99.   # reset (see header, item 5)

                    ip = ParentID[id, iz]
                    p  = potentials[ip]
                    s  = potentials[id]

                    if s.Mh > min_mass[id]:
                        if ejected_mass[id] > 0:
                            mass[id, iz] -= ejected_mass[id]
                            ejected_mass[id] = 0
                            mass[id, iz] = max(mass[id, iz], M_floor)
                        s.update_mass(mass[id, iz])
                        rte = s.rte()

                    o  = orbits[id]
                    xv = orbits[id].xv
                    m  = s.Mh
                    m_old = m
                    r  = np.sqrt(xv[0]**2 + xv[2]**2)
                    t  = tnext - trelease[id]
                    k  = order[ip, iznext] + 1

                    if alpha_type == 'fixed':
                        alpha = 0.55
                    elif alpha_type == 'conc':
                        alpha = ev.alpha_from_c2(p.ch, s.ch)

                    if m > min_mass[id]:
                        m, lt = ev.msub(s, p, xv, dt, choice=tidal_choice,
                                        alpha=alpha)

                    if m > min_mass[id]:
                        tdyn = p.tdyn(r)
                        o.integrate(t, p, m_old)
                        xv = o.xv
                    else:
                        tdyn = p.tdyn(cfg.Rres)
                        xv = np.array([cfg.Rres, 0., 0., 0., 0., 0.])

                    r = np.sqrt(xv[0]**2 + xv[2]**2)

                    if k > 1:
                        if (r > VirialRadius[ip, iz]) & (iz <= izroot[ip]):
                            odds = np.random.rand()
                            dyntime_frac = alphas[ip, iz]*dt/tdyns[ip, iz]
                            if odds < dyntime_frac:
                                if ParentID[ip, iz] == ParentID[ip, iznext]:
                                    xv = aux.add_cyl_vecs(
                                        xv, coordinates[ip, iznext, :])
                                else:
                                    xv = aux.add_cyl_vecs(
                                        xv, coordinates[ip, iz, :])
                                orbits[id] = orbit(xv)
                                k = order[ip, iz]
                                ejected_mass[ip] += m
                                ip = ParentID[ip, iz]
                                trelease[id] = tnext

                    mass[id, iznext]     = m
                    order[id, iznext]    = k
                    ParentID[id, iznext] = ip
                    VirialRadius[id, iznext] = lt
                    GreenRte[id, iz] = rte
                    coordinates[id, iznext, :] = xv
                    alphas[id, iz] = alpha
                    tdyns[id, iz]  = tdyn

                else:
                    if concentration[id, iz] > 0:
                        potentials[id] = NFW(mass[id, iz],
                            concentration[id, iz],
                            Delta=VirialOverdensity[iz], z=redshift[iz])

    np.savez(name,
        redshift=redshift, CosmicTime=CosmicTime, mass=mass, order=order,
        ParentID=ParentID, VirialRadius=VirialRadius, GreenRte=GreenRte,
        concentration=concentration, coordinates=coordinates,
        M_floor=M_floor, lnL_pref=lnL_pref, lnL_type=lnL_type,
        alpha_type=alpha_type, tidal_choice=tidal_choice)
    print(f"[evolve] {os.path.basename(name)}: "
          f"{(time.time()-t0)/60:.1f} min; {n_born_below} branches had "
          f"m_acc <= M_floor (never evolved)", flush=True)

############################## driver ###################################

def run_one(itree):
    try:
        evolve_tree(grow_tree(itree))
    except Exception as e:
        print(f"[tree {itree}] FAILED: {type(e).__name__}: {e}", flush=True)

if __name__ == "__main__":
    os.makedirs(datadir, exist_ok=True)
    print(f"host M0 = {M0:.2e} at z_obs = {z_obs}; Mres = {Mres:.1e}, "
          f"M_floor = {M_floor:.1e}; Nz = {cfg.Nz}, "
          f"z grid {cfg.zsample[0]:.3f} -> {cfg.zsample[-1]:.1f}",
          flush=True)
    print(f"cosmology: h={cfg.h}, Om={cfg.Om}, Ob={cfg.Ob}, OL={cfg.OL}, "
          f"s8={cfg.s8}, ns={cfg.ns}; lnL_pref={cfg.lnL_pref}; "
          f"{ncores} workers", flush=True)
    with Pool(ncores) as pool:
        pool.map(run_one, range(Ntree), chunksize=1)
    print("done", flush=True)
