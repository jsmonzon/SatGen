####################### Ludlow et al. (2016) concentration ##############
#
# Physically-motivated halo-concentration model of Ludlow, Bose, Angulo,
# Wang, Hellwing, Navarro, Cole & Frenk (2016, MNRAS 460, 1214), Section
# 4.3, evaluated using the *numerically tabulated* collapsed mass history
# (CMH) of a halo, read directly off a SatGen merger tree -- as opposed
# to the smooth EPS/erfc approximation to the CMH given in their eq.(6),
# which is only needed when no explicit tree is available.
#
# Model summary (L16 sec.4.3, App.A):
#
#   <rho_-2> == 3 M_-2 / (4 pi r_-2^3) = A * rhoc(z_-2)          (A1)
#
# where r_-2 is the NFW scale radius (i.e. Rvir/c), M_-2 = M(<r_-2) is
# the host's own mass presently enclosed within r_-2, and z_-2 is the
# "formation redshift": the redshift at which the collapsed mass
# history of the host -- the summed mass of all progenitor branches
# more massive than f*M0 -- first reaches M_-2. A is an empirical
# proportionality constant. L16 state explicitly (Sec.4.3): "Merger
# trees generated using Monte Carlo methods tailored to reproduce
# simulation results (e.g. Parkinson et al. 2008) should adopt
# C=400 and f=0.02." SatGen's own trees are themselves Parkinson+08
# Monte Carlo trees, so A=400 (their "C") -- not the ~900 range L16
# quote elsewhere for concentrations calibrated directly against
# resolved N-body trees -- is the value called for here.
#
# Since M_-2 = M0 * f(1)/f(c) for an NFW profile (f(x)=ln(1+x)-x/(1+x)),
# and <rho_-2> = Delta*rhoc(z0)*c^3*f(1)/f(c), eq.(A1) is an implicit,
# self-consistent equation for c that is solved here by fixed-point
# iteration, following the standard practical recipe (e.g. Ludlow+2016;
# Johnson, Benson & Grin 2019; Correa+2015): guess c -> get M_-2 -> read
# off z_-2 from the CMH -> get a new target <rho_-2> -> invert for a new
# c -> repeat.
#
# IMPORTANT: the collapsed mass history must be built by summing the
# mass of ALL branches in the tree above the f*M0 threshold at each
# snapshot -- not just the main branch (id=0). Because a SatGen tree is
# rooted at the z0 host, every branch that exists (mass>0) at a given
# snapshot is, by construction, a progenitor of that host, so the CMH is
# simply the threshold-clipped sum over branches. Using the main-branch
# mass history alone under-counts the collapsed mass at any epoch before
# the last major merger, biases z_-2 low, and hence biases c low too.
#
# Sheridan Green's SatGen conventions (config.py/cosmo.py/profiles.py)
# are used throughout: cfg.zsample/cfg.tsample are the tree's snapshot
# redshifts/cosmic times (zsample[0]=z0, increasing thereafter), and
# co.rhoc(z,h,Om,OL) is the critical density [Msun/kpc^3].
#
#########################################################################

import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import InterpolatedUnivariateSpline

import config as cfg
import cosmo as co

#########################################################################

#---model parameters (Ludlow+2016, Sec.4.3 / Appendix A)

#f_CMH = 0.02   # progenitor-mass threshold defining the CMH, in units of
               # M0 (the host's own z0 mass); fixed at 0.02 in L16
#A_L16 = 400.   # <<< calibration constant in <rho_-2> = A*rhoc(z_-2).
               # Per L16 Sec.4.3, verbatim: "Merger trees generated
               # using Monte Carlo methods tailored to reproduce
               # simulation results (e.g. Parkinson et al. 2008) should
               # adopt C=400 and f=0.02." SatGen's trees are themselves
               # Parkinson+08-based Monte Carlo trees, so A=400 (L16's
               # "C") is the appropriate value here -- not the ~900
               # figure quoted elsewhere in L16 for concentrations
               # measured directly from resolved N-body merger trees.
               # Recalibrate only if you have specific reason to think
               # SatGen's branching/resolution departs from Parkinson+08
               # enough to matter for your science case.

#---auxiliary NFW function

def f_NFW(x):
    """
    NFW auxiliary function

        f(x) = ln(1+x) - x/(1+x)

    reproduced here in stand-alone (root-finder-friendly) form -- it is
    identical to the .f(x) method of profiles.NFW, but that method lives
    on an already-instantiated halo object, which we don't have yet
    while solving for c.

    Syntax:

        f_NFW(x)

    where

        x: dimensionless radius, r/rs (float or array)
    """
    return np.log(1.+x) - x/(1.+x)

def F_of_c(c):
    """
    Fraction of a halo's own mass, M0, that is enclosed within its own
    NFW scale radius r_-2 = Rvir/c, i.e.,

        F(c) = M(<r_-2)/M0 = f_NFW(1)/f_NFW(c)

    (L16 eq. analogous to their App.A; f_NFW(1)=ln(2)-1/2).

    Syntax:

        F_of_c(c)

    where

        c: NFW concentration, c = Rvir/r_-2 (float or array)
    """
    return f_NFW(1.) / f_NFW(c)

#---collapsed mass history, directly from a SatGen tree

def collapsed_mass_history(mass,order,ParentID,f):
    """
    Collapsed mass history (CMH), computed directly from a SatGen merger
    tree's mass/order/ParentID arrays (as saved by TreeGen.py, or as
    tracked by an evolved-tree pipeline), WITHOUT double-counting nested
    substructure.

    Syntax:

        collapsed_mass_history(mass,order,ParentID,f=0.02)

    where

        mass: merger-tree (or evolved-tree) mass array, shape
            (Nbranch,Nz), with mass[id,iz] = mass [Msun] of branch "id"
            at snapshot "iz", or <=0 (SatGen convention: -99.) if branch
            "id" does not exist at snapshot "iz" (2d array). mass[id,iz]
            is taken to be INCLUSIVE: e.g. a 1st-order subhalo's mass is
            its own smooth/phase-mixed mass plus the mass of any 2nd+
            order sub-subhalos it hosts, and so on recursively -- as is
            the case for both raw TreeGen.py output and post-evolution
            (SatEvo-type) mass arrays.
        order: instantaneous nesting order at each snapshot (0=host,
            1=1st-order subhalo, 2=2nd-order, ...), same shape as mass
            (2d int array)
        ParentID: id of each branch's IMMEDIATE parent/host branch at
            each snapshot (i.e. order[ParentID[id,iz],iz] ==
            order[id,iz]-1), same shape as mass; the root branch (id=0)
            has ParentID<0 (2d int array)
        f: progenitor-mass threshold below which a branch does NOT
            contribute to the collapsed mass, in units of M0=mass[0,0]
            (float, default=0.02, following Ludlow+2016)

    Because masses are inclusive, a branch above the f*M0 threshold must
    NOT be counted if any of its ancestors (parent, grandparent, ...) is
    itself above threshold: that ancestor's own mass value already
    contains this branch's mass as nested substructure. So at each
    snapshot we only sum the OUTERMOST above-threshold branch in every
    nested chain -- i.e. a branch is counted iff it is above threshold
    AND no ancestor of it (walking ParentID all the way to the root) is
    also above threshold. This is resolved order-level by order-level
    (vectorized over branches and snapshots within each level), since a
    branch's ancestor-chain status is fully determined once its
    immediate parent's status is known.

    Return:

        CMH: collapsed mass history [Msun], array of length Nz aligned
            with the tree's own snapshots (float array)
        M0: host mass at z0, i.e. mass[0,0] [Msun] (float)
    """
    Nbranch,Nz = mass.shape
    M0 = mass[0,0]
    Mthresh = f * M0

    alive = mass > 0.
    above = alive & (mass >= Mthresh)

    # subsumed[id,iz]: True once branch id's mass at snapshot iz is
    # already folded into some ancestor that either is itself counted
    # (above threshold) or is itself subsumed by a still-higher counted
    # ancestor. Initialized to the order-0 (host) case, which has no
    # parent and is therefore only ever "subsumed" by itself.
    subsumed = above.copy()
    excluded = np.zeros((Nbranch,Nz),dtype=bool)

    max_order = int(order[alive].max()) if np.any(alive) else 0
    for k in range(1,max_order+1):
        branch_idx,time_idx = np.where(alive & (order==k))
        if branch_idx.size==0:
            continue
        parent_ids = ParentID[branch_idx,time_idx]
        valid = parent_ids>=0
        par_subsumed = np.zeros(branch_idx.size,dtype=bool)
        par_subsumed[valid] = subsumed[parent_ids[valid],time_idx[valid]]
        excluded[branch_idx,time_idx] = par_subsumed
        subsumed[branch_idx,time_idx] = above[branch_idx,time_idx] | \
            par_subsumed

    counted = above & ~excluded
    CMH = np.sum(np.where(counted,mass,0.),axis=0)
    return CMH,M0

def z_of_CMH(F,CMH,M0,zsample):
    """
    Invert a tabulated collapsed mass history to find the redshift,
    z_-2, at which a given fraction, F, of the host's final mass had
    first assembled into progenitors above the f*M0 threshold.

    Syntax:

        z_of_CMH(F,CMH,M0,zsample)

    where

        F: target mass fraction, M_-2/M0 (float)
        CMH: collapsed mass history [Msun], array (as returned by
            collapsed_mass_history)
        M0: host mass at z0 [Msun] (float)
        zsample: redshift of each CMH snapshot, in INCREASING order
            (zsample[0]=z0), e.g. cfg.zsample[:len(CMH)] (float array)

    CMH/M0 must be a non-increasing function of z, up to Poisson/
    stochastic noise from the tree's finite branching -- we therefore
    first take the running minimum going backward in time, to make the
    root-finding robust against small, unphysical up-down fluctuations
    in any single realization of a merger tree.

    Return:

        z_-2 (float) -- clipped to the sampled redshift range if F falls
        outside the range spanned by the (monotonized) CMH
    """
    CMHfrac = np.minimum.accumulate(CMH/M0) # enforce monotonicity
    if F >= CMHfrac[0]:
        return zsample[0]
    if F <= CMHfrac[-1]:
        warnings.warn("z_of_CMH: target fraction F=%.3g falls below the "
            "CMH sampled over this tree's redshift range (min=%.3g); "
            "returning the highest sampled redshift. Consider a deeper/"
            "higher-resolution tree if this happens often."%
            (F,CMHfrac[-1]))
        return zsample[-1]

    spl = InterpolatedUnivariateSpline(zsample, CMHfrac-F, k=1)

    f1 = spl(zsample[0])
    f2 = spl(zsample[-1])

    if not (np.isfinite(f1) and np.isfinite(f2)) or f1 * f2 >= 0:
        warnings.warn(
            "z_of_CMH: brentq bracket failure; returning closest sampled redshift"
        )
        return zsample[np.argmin(np.abs(CMHfrac - F))]

    return brentq(spl, zsample[0], zsample[-1])
    # return brentq(spl,zsample[0],zsample[-1])

#---the self-consistent concentration solver

def rho_m2_of_c(c,Delta,rhoc0):
    """
    Mean density [Msun/kpc^3] enclosed within the NFW scale radius,

        <rho_-2>(c) = Delta * rhoc0 * c^3 * f_NFW(1)/f_NFW(c)

    i.e. eq.(A1)'s LHS expressed purely in terms of c, given the host's
    own (Delta,z0) mass definition. Monotonically increasing in c.

    Syntax:

        rho_m2_of_c(c,Delta,rhoc0)

    where

        c: NFW concentration (float or array)
        Delta: spherical overdensity defining M0/Rvir (float)
        rhoc0: critical density at z0 [Msun/kpc^3] (float)
    """
    return Delta * rhoc0 * c**3 * F_of_c(c)

def c_of_rho_m2(rho_target,Delta,rhoc0,c_lo=0.5,c_hi=200.):
    """
    Invert rho_m2_of_c for c, given a target <rho_-2> -- safe to do with
    a simple bisection-type root-finder since rho_m2_of_c(c) is
    monotonically increasing in c.

    Syntax:

        c_of_rho_m2(rho_target,Delta,rhoc0,c_lo=0.5,c_hi=200.)

    where

        rho_target: target <rho_-2> [Msun/kpc^3] (float)
        Delta,rhoc0: as in rho_m2_of_c
        c_lo,c_hi: root-finding bracket (floats, default=0.5,200., wide
            enough for essentially any halo of interest)
    """
    resid = lambda c: rho_m2_of_c(c,Delta,rhoc0) - rho_target

    f_lo = resid(c_lo)
    f_hi = resid(c_hi)

    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi >= 0:
        warnings.warn(
            "c_of_rho_m2: target density outside concentration bracket; "
            "returning NaN"
        )
        return np.nan

    return brentq(resid, c_lo, c_hi, xtol=1e-5)


def concentration_Ludlow2016(mass,order,ParentID,z0=0.,Delta=200.,
    f=0.02,A=400.,c0=10.,tol=1e-3,maxiter=50):
    """
    Halo concentration from the Ludlow+2016 (Sec.4.3) physically-
    motivated model, using the collapsed mass history read directly off
    a SatGen merger tree.

    Syntax:

        concentration_Ludlow2016(mass,order,ParentID,z0=0.,Delta=200.,
            f=0.02,A=400.,c0=10.,tol=1e-3,maxiter=50)

    where

        mass,order,ParentID: merger-tree arrays, shape (Nbranch,Nz), as
            saved by TreeGen.py (mass[0,0]=M0, the z0 host mass); see
            collapsed_mass_history for the exact convention expected of
            each (2d arrays)
        z0: redshift at which the host is identified (float, default=0.;
            should match cfg.zsample[0]/cfg.z0)
        Delta: spherical overdensity defining M0 (float, default=200.,
            i.e. M0=M_200c; use cfg.Dvsample[0] instead if the tree's
            host mass is defined via Bryan & Norman virial overdensity)
        f: CMH progenitor-mass threshold, in units of M0 (float,
            default=0.02, per Ludlow+2016)
        A: calibration constant in <rho_-2>=A*rhoc(z_-2) (float,
            default=400., i.e. L16's own recommended value -- their
            "C" -- for Parkinson+08-style Monte Carlo trees, which is
            what SatGen itself generates; see the A_L16 module-level
            comment)
        c0: initial guess for c, to seed the fixed-point iteration
            (float, default=10.). Also serves as the fallback return
            value for c if the iteration fails to converge.
        tol: convergence tolerance on c between iterations (float,
            default=1e-3)
        maxiter: maximum number of fixed-point iterations (int,
            default=50)

    If the fixed-point iteration does NOT converge within maxiter steps,
    a warning is raised and the function backs off to its own INPUT
    values, c0 and z0 -- rather than returning the last (untrusted,
    non-converged) iterate -- so that a caller sweeping over many halos
    can reliably detect failures downstream (e.g. by checking c==c0)
    without accidentally ingesting a bogus concentration.

    Return:

        c: NFW concentration, c=Rvir/r_-2 (float; equal to the input c0
            if the iteration failed to converge)
        z_2: the inferred formation redshift z_-2 (float; equal to the
            input z0 if the iteration failed to converge)
        CMH: the collapsed mass history used [Msun] (array), in case you
            want to inspect/plot it as a diagnostic -- returned
            regardless of convergence, since it doesn't depend on the
            iteration itself
    """

    CMH,M0 = collapsed_mass_history(mass,order,ParentID,f=f)
    zsample = cfg.zsample[:len(CMH)]
    rhoc0 = co.rhoc(z0,cfg.h,cfg.Om,cfg.OL)

    c = c0
    for i in range(maxiter):
        Mm2 = M0 * F_of_c(c)
        z_2 = z_of_CMH(Mm2/M0,CMH,M0,zsample)
        rho_target = A * co.rhoc(z_2,cfg.h,cfg.Om,cfg.OL)
        c_new = c_of_rho_m2(rho_target,Delta,rhoc0)
        if abs(c_new-c) < tol:
            c = c_new
            break
        c = c_new
    else:
        warnings.warn("concentration_Ludlow2016: fixed-point iteration "
            "did not converge after %d iterations (last |delta c|=%.3g)"
            " -- falling back to the input c0=%.3g, z0=%.3g. Inspect "
            "the returned CMH for pathologies (e.g. a tree that is too "
            "shallow/under-resolved)."%
            (maxiter,abs(c_new-c),c0,z0))
        return c0,z0,CMH

    return c,z_2,CMH

#########################################################################
#
# Per-snapshot Ludlow+2016 concentration HISTORY of the host, from a
# SatGen tree's full collapsed mass history (added 2026-11-03).
#
# Why a separate implementation: collapsed_mass_history() above decides
# whether a branch is "already inside" a counted ancestor from the
# order/ParentID arrays. In SatGen trees those arrays carry a branch's
# ACCRETION-TIME order/parent at every snapshot it exists, including the
# snapshots BEFORE accretion, when the branch is still a distinct halo
# whose mass is NOT contained in its eventual parent. Since the host is
# always above f*M0, every 1st-order progenitor (and, recursively, all
# deeper ones) is excluded, and the CMH collapses to the main-branch MAH
# -- i.e. it under-counts collapsed mass, biasing z_-2 and c low (e.g.
# tree_13.2_33: c=4.48 from the MAH-like CMH vs ~6.9 from the full CMH).
#
# Here the CMH is built only from PRE-ACCRETION (distinct-halo) entries:
# with izroot[id] = mass[id].argmax() the accretion snapshot (TreeGen/
# SubEvo convention: the branch is still a distinct halo at izroot and is
# inside its parent from izroot-1 on), branch id contributes at snapshot
# j iff j >= izroot[id]. Distinct halos never contain one another, so no
# exclusion logic is needed, and the same rule works on evolved (_evo)
# trees, whose post-accretion entries are simply ignored.
#
# To evaluate c at an arbitrary snapshot iz_t (needed because the host
# potential in SubEvo is rebuilt every snapshot as NFW(M(z),c(z))), the
# tree is re-rooted at iz_t: only branches that are progenitors of the
# host AT iz_t count, i.e. every link in the branch's chain up to the
# main branch must have merged by iz_t (izroot > iz_t for the branch and
# for each non-host ancestor). L16 Sec.4.2 find <rho_-2> = C rhoc(z_-2)
# independent of identification redshift, which is what licenses using
# the same C, f at every z0 = z(iz_t).
#
# Constants (checked against L16, MNRAS 460, 1214): f = 0.02 (Sec.4.2-
# 4.3, App.B); C = 400 for Parkinson+08-style Monte Carlo trees (Sec.4.3
# footnote 7; also the value measured directly from COCO trees in
# Sec.4.2). C = 650 is ONLY for the analytic erfc CMH of eq.(7), not used
# here. L16 use M200c / c200 = r200/r_-2; we solve with the tree's own
# Bryan & Norman Delta_vir(z) instead, which is self-consistent for the
# <rho_-2> side (r_-2 and M_-2 are physical, definition-independent
# quantities) and returns c_vir = Rvir/r_-2 directly, as SubEvo's
# NFW(Mvir, c, Delta=Delta_BN) host needs. The residual inconsistency is
# that progenitor masses (and the f*M0 threshold) are BN virial masses,
# ~10-20% larger than M200c.
#
#########################################################################

def progenitor_mask(mass,ParentID,iz_t):
    """
    Branches of a SatGen tree that are progenitors of the host (id=0) as
    it exists at snapshot iz_t.

    Syntax:

        progenitor_mask(mass,ParentID,iz_t)

    where

        mass, ParentID: (Nbranch,Nz) tree arrays (TreeGen convention)
        iz_t: snapshot index at which the host is identified (int)

    A branch qualifies iff it merged into its parent before iz_t
    (izroot > iz_t) and its parent qualifies, recursively up to id=0.

    Return:

        boolean array of length Nbranch (True for the host itself)
    """
    Nbranch = mass.shape[0]
    izroot = mass.argmax(axis=1)
    parent = ParentID[np.arange(Nbranch),izroot]
    prog = np.zeros(Nbranch,dtype=bool)
    prog[0] = True
    merged = (izroot > iz_t) & (mass.max(axis=1) > 0.)
    merged[0] = True
    # propagate down the chain; converges in (max order) sweeps
    for _ in range(Nbranch):
        valid = parent >= 0
        new = np.zeros(Nbranch,dtype=bool)
        new[valid] = merged[valid] & prog[parent[valid]]
        new[0] = True
        if np.array_equal(new,prog):
            break
        prog = new
    return prog

def collapsed_mass_history_tree(mass,ParentID,iz_t=0,f=0.02):
    """
    Collapsed mass history of the host identified at snapshot iz_t,
    summed over ALL of its distinct (pre-accretion) progenitors above
    f*M(iz_t).

    Syntax:

        collapsed_mass_history_tree(mass,ParentID,iz_t=0,f=0.02)

    Return:

        CMH: [Msun], array of length Nz-iz_t, aligned with snapshots
            iz_t, iz_t+1, ... (CMH[0] = M0)
        M0: host mass at iz_t [Msun]
    """
    Nbranch,Nz = mass.shape
    M0 = mass[0,iz_t]
    izroot = mass.argmax(axis=1)
    prog = progenitor_mask(mass,ParentID,iz_t)
    j = np.arange(Nz)[None,:]
    distinct = j >= izroot[:,None]
    distinct[0,:] = True
    counted = prog[:,None] & distinct & (mass > 0.) & (mass >= f*M0)
    CMH = np.sum(np.where(counted,mass,0.),axis=0)[iz_t:]
    return CMH,M0

def _solve_c_Ludlow(CMH,M0,zs,Delta,A,c0,tol,maxiter):
    """
    Fixed-point solve of L16 eq.(6): <rho_-2>(c) = A rhoc(z_-2[c]),
    with zs[0] the identification redshift. Returns (c, z_-2, converged).
    """
    rhoc0 = co.rhoc(zs[0],cfg.h,cfg.Om,cfg.OL)
    c = c0
    for i in range(maxiter):
        z_2 = z_of_CMH(F_of_c(c),CMH,M0,zs)
        c_new = c_of_rho_m2(A*co.rhoc(z_2,cfg.h,cfg.Om,cfg.OL),Delta,rhoc0)
        if not np.isfinite(c_new):
            return c0,zs[0],False
        if abs(c_new-c) < tol:
            return c_new,z_2,True
        c = c_new
    return c0,zs[0],False

def concentration_history_Ludlow2016(mass,ParentID,redshift,Delta,c_guess,
    f=0.02,A=400.,tol=1e-3,maxiter=50):
    """
    Ludlow+2016 concentration of the host (id=0) at every snapshot,
    using the full collapsed mass history of the host re-rooted at each
    snapshot (see the block comment above).

    Syntax:

        concentration_history_Ludlow2016(mass,ParentID,redshift,Delta,
            c_guess,f=0.02,A=400.,tol=1e-3,maxiter=50)

    where

        mass, ParentID: (Nbranch,Nz) tree arrays
        redshift: snapshot redshifts, increasing (redshift[0]=z0)
        Delta: virial overdensity at each snapshot, e.g. cfg.Dvsample
            (array, length >= Nz)
        c_guess: starting guess per snapshot, e.g. the tree's Zhao host
            concentration[0,:]; snapshots with c_guess<=0 (SatGen -99
            dummy: host not resolved) are skipped and returned as -99
        f, A: L16 constants (0.02, 400 -- see block comment)

    Return:

        c: host concentration per snapshot (-99 where skipped; equal to
            c_guess where the iteration did not converge)
        z_2: formation redshift z_-2 per snapshot (NaN where skipped or
            not converged)
        converged: boolean per snapshot (False where skipped)
    """
    Nz = mass.shape[1]
    c = np.full(Nz,-99.)
    z_2 = np.full(Nz,np.nan)
    conv = np.zeros(Nz,dtype=bool)
    for iz in range(Nz):
        if not (c_guess[iz] > 0. and mass[0,iz] > 0.):
            continue
        CMH,M0 = collapsed_mass_history_tree(mass,ParentID,iz_t=iz,f=f)
        zs = redshift[iz:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ci,z2i,ok = _solve_c_Ludlow(CMH,M0,zs,Delta[iz],A,
                c_guess[iz],tol,maxiter)
        c[iz] = ci
        conv[iz] = ok
        if ok:
            z_2[iz] = z2i
    return c,z_2,conv
