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
# proportionality constant (A~400-900 depending on tree algorithm/
# cosmology; L16 quote A~900 for their own Parkinson+08 trees).
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

f_CMH = 0.02   # progenitor-mass threshold defining the CMH, in units of
               # M0 (the host's own z0 mass); fixed at 0.02 in L16
A_L16 = 900.   # <<< calibration constant in <rho_-2> = A*rhoc(z_-2).
               # L16 find A~900 for their own Parkinson+08-based trees;
               # subsequent papers applying the same algorithm to other
               # trees/cosmologies find A in the range ~650-900 (e.g.
               # Johnson+2019 quote A~400 for a somewhat different tree
               # setup). Since SatGen also uses Parkinson+08 trees, 900
               # is a reasonable starting point, but this should ideally
               # be recalibrated against a resolved N-body/SatGen
               # comparison sample if precision matters for your science
               # case.

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

def collapsed_mass_history(mass,f=f_CMH):
    """
    Collapsed mass history (CMH), computed directly from the mass array
    of a SatGen merger tree (i.e., the "mass" array as saved by
    TreeGen.py, or read back via np.load(...)['mass']).

    Syntax:

        collapsed_mass_history(mass,f=0.02)

    where

        mass: merger-tree mass array, shape (Nbranch,Nz), with
            mass[id,iz] = mass [Msun] of branch "id" at snapshot "iz",
            or -99. if branch "id" does not yet exist / no longer exists
            at snapshot "iz" (SatGen sentinel convention) (2d array)
        f: progenitor-mass threshold below which a branch does NOT
            contribute to the collapsed mass, in units of M0=mass[0,0]
            (float, default=0.02, following Ludlow+2016)

    Note: a SatGen tree is rooted at the z0 host (branch id=0), so every
    branch that exists (mass>0) at a given snapshot is, by construction,
    a progenitor of that host. The CMH at snapshot iz is therefore just
    the sum of mass[:,iz] over branches whose mass exceeds f*M0 --
    summed over ALL branch ids, not only the main branch. (Restricting
    the sum to id=0 gives the ordinary mass accretion history of the
    main branch, which is NOT the CMH, and will generally underestimate
    it before the last major merger.)

    Return:

        CMH: collapsed mass history [Msun], array of length Nz aligned
            with the tree's own snapshots (float array)
        M0: host mass at z0, i.e. mass[0,0] [Msun] (float)
    """
    M0 = mass[0,0]
    Mthresh = f * M0
    msk = mass >= Mthresh # the -99. sentinel automatically fails this
                           # test as long as Mthresh>0
    CMH = np.sum(np.where(msk,mass,0.),axis=0)
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
    spl = InterpolatedUnivariateSpline(zsample,CMHfrac-F,k=1)
    return brentq(spl,zsample[0],zsample[-1])

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
    return brentq(resid,c_lo,c_hi,xtol=1e-5)

def concentration_Ludlow2016(mass,z0=0.,Delta=200.,f=f_CMH,A=A_L16,
    h=None,Om=None,OL=None,c0=10.,tol=1e-3,maxiter=50):
    """
    Halo concentration from the Ludlow+2016 (Sec.4.3) physically-
    motivated model, using the collapsed mass history read directly off
    a SatGen merger tree.

    Syntax:

        concentration_Ludlow2016(mass,z0=0.,Delta=200.,f=0.02,A=900.,
            h=None,Om=None,OL=None,c0=10.,tol=1e-3,maxiter=50)

    where

        mass: merger-tree mass array, shape (Nbranch,Nz), as saved by
            TreeGen.py (mass[0,0]=M0, the z0 host mass) (2d array)
        z0: redshift at which the host is identified (float, default=0.;
            should match cfg.zsample[0]/cfg.z0)
        Delta: spherical overdensity defining M0 (float, default=200.,
            i.e. M0=M_200c; use cfg.Dvsample[0] instead if the tree's
            host mass is defined via Bryan & Norman virial overdensity)
        f: CMH progenitor-mass threshold, in units of M0 (float,
            default=0.02, per Ludlow+2016)
        A: calibration constant in <rho_-2>=A*rhoc(z_-2) (float,
            default=900.; see the A_L16 module-level comment)
        h,Om,OL: cosmological parameters (default=None, i.e. fall back
            to cfg.h,cfg.Om,cfg.OL)
        c0: initial guess for c, to seed the fixed-point iteration
            (float, default=10.)
        tol: convergence tolerance on c between iterations (float,
            default=1e-3)
        maxiter: maximum number of fixed-point iterations (int,
            default=50)

    Return:

        c: NFW concentration, c=Rvir/r_-2 (float; np.nan if the
            iteration fails to converge)
        z_2: the inferred formation redshift z_-2 (float)
        CMH: the collapsed mass history used [Msun] (array), in case you
            want to inspect/plot it as a diagnostic
    """
    if h is None: h = cfg.h
    if Om is None: Om = cfg.Om
    if OL is None: OL = cfg.OL

    CMH,M0 = collapsed_mass_history(mass,f=f)
    zsample = cfg.zsample[:len(CMH)]
    rhoc0 = co.rhoc(z0,h,Om,OL)

    c = c0
    z_2 = z0
    for i in range(maxiter):
        Mm2 = M0 * F_of_c(c)
        z_2 = z_of_CMH(Mm2/M0,CMH,M0,zsample)
        rho_target = A * co.rhoc(z_2,h,Om,OL)
        c_new = c_of_rho_m2(rho_target,Delta,rhoc0)
        if abs(c_new-c) < tol:
            c = c_new
            break
        c = c_new
    else:
        warnings.warn("concentration_Ludlow2016: fixed-point iteration "
            "did not converge after %d iterations (last |delta c|=%.3g)"
            " -- inspect the returned CMH for pathologies (e.g. a tree "
            "that is too shallow/under-resolved)."%
            (maxiter,abs(c_new-c)))

    return c,z_2,CMH