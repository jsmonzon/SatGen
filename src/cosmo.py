##################### cosmology-related functions #######################

# Arthur Fangzhou Jiang 2019 Hebrew University
# Arthur Fangzhou Jiang 2021 Caltech & Carnegie

# On 2021-05-04, added Benson+21 values of the PCH08 merger tree params

#########################################################################

import config as cfg

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import sys
# NOTE (jsm 2026-08-29): cosmolopy (formerly imported here as cdis/cden/cc/cper)
# has been removed as a dependency -- it requires a SWIG-compiled C extension
# that fails to build on modern toolchains (distutils/setuptools incompatibility
# on top of needing `swig` present at all), and only two of its functions were
# ever used in this file: perturbation.fgrowth() (in D(), below) and
# perturbation.transfer_function_EH() (in T(), below). Both are closed-form,
# non-integrating fitting functions, so they've been ported directly to numpy
# -- see _fgrowth_EH() and _TFmdm_set_cosm()/_TFmdm_onek_mpc() near the bottom
# of this section. _TFmdm_onek_mpc() is a line-for-line transcription of
# TFmdm_onek_mpc() in cosmolopy's EH/power.c (Eisenstein & Hu 1999, ApJ 511 5),
# restricted to the baryonic_effects=False, zero-massive-neutrino case that is
# the only one config.py ever uses. Validated against a real cosmolopy install
# (built in a separate env with setuptools<60) across k in [1e-4, 1e3] h/Mpc
# and z in [0, 50]: relative agreement ~1e-7, consistent with cosmolopy's C
# code using 32-bit float internally where this port uses float64.
#########################################################################

#---basics 

def rhoc(z,h=0.7,Om=0.3,OL=0.7):
    """
    Critical density [Msun kpc^-3] at redshift z.
    
    Syntax: 
    
        rhoc(z,h=0.7,Om=0.3,OL=0.7)
    
    where
        
        z: redshift (float or array)
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    return cfg.rhoc0 * h**2 * (Om*(1.+z)**3 + OL)
    
def rhom(z,h=0.7,Om=0.3,OL=0.7):
    """
    Mean density [Msun kpc^-3] at redshift z.
    
    Syntax: 
    
        rhom(z,h=0.7,Om=0.3,OL=0.7)
    
    where
        
        z: redshift (float or array)
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    return Omega(z,Om,OL) * rhoc(z,h,Om,OL)
    
def DeltaBN(z,Om=0.3,OL=0.7):
    """
    Virial overdensity of Bryan & Norman (1998).
    
    Syntax:
    
        DeltaBN(z, Om=0.3,OL=0.7)
        
    where
        
        z: redshift (float or array)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    x = Omega(z,Om,OL) - 1.
    return 18.*np.pi**2 + 82.*x - 39.*x**2

def Omega(z,Om=0.3,OL=0.7):
    """
    Matter density in units of the critical density, at redshift z.
    
    Syntax: 
    
        Omega(z, Om=0.3,OL=0.7)
        
    where
        
        z: redshift (float or array)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    fac = Om * (1.+z)**3
    return fac / (fac + OL)

def tdyn(z,h=0.7,Om=0.3,OL=0.7):
    """
    Halo dynamical time [Gyr] defined as
    
        R_vir/V_vir = sqrt(3 / 4 pi G Delta rho_crit) 
                    = sqrt(2 / Delta) * 1/H(z)
        
    Syntax: 
    
        tdyn(z,h=0.7,Om=0.3,OL=0.7)
    
    where 
        
        z: redshift (float or array)
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    
    Note that at high-z, this is rougly 0.1 times Hubble time 1/H(z).    
    """
    return np.sqrt(2./DeltaBN(z,Om,OL)) / H(z,h,Om,OL)

def Ndyn(z1,z2,h=0.7,Om=0.3,OL=0.7):
    """
    Number of halo dynamical times elapsed between redshift z1 and z2 
    (z1>z2).
    
    Syntax: 
    
        Ndyn(z1,z2,h=0.7,Om=0.3,OL=0.7)
        
    where
    
        z1: a higher redshift (float)
        z2: a lower redshift
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    return quad(dNdz, z1,z2, args=(h,Om,OL,),
        epsabs=1.e-7, epsrel=1.e-6,limit=10000)[0]
def dNdz(z,h,Om,OL):
    r"""
    Auxiliary function for the function Ndyn -- the integrand, dN/dz(z),
    for computing 
    
        N_dyn = int_z1^z2 dN/dz(z) dz
              = int_z1^z2 dt/dz(z) * 1/t_dyn(z) dz
    
    Syntax:
        
        dNdz(z,h,Om,OL)
    
    where
        
        z: redshift (float)
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    return dtdz(z,h,Om,OL) / tdyn(z,h,Om,OL)
def dtdz(z,h,Om,OL):
    """
    complementary function for computing N_dyn, it returns
        dt / dz
    i.e., cosmic time increment per redshift decrement 
    """
    z1 = z*(1.-cfg.eps)
    z2 = z*(1.+cfg.eps)
    t1 = t(z1,h,Om,OL) # t1>t2 because z1<z2
    t2 = t(z2,h,Om,OL)
    return (t1-t2) / (z1-z2)

def H(z,h=0.7,Om=0.3,OL=0.7):
    """
    Hubble constant [Gyr^-1] at redshift z.
    
    Syntax:
    
        H(z,h=0.7,Om=0.3,OL=0.7)
        
    where
        
        z: redshift (float or array)
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    return (h/9.778) * np.sqrt( Om*(1.+z)**3 + OL )

def E(z,Om=0.3,OL=0.7):
    """
    Hubble constant at redshift z in units of the Hubble constant at z=0.
        
        E(z):=H(z)/H0
    Syntax:
    
        E(z,Om=0.3,OL=0.7)
        
    where
    
        z: redshift (float or array)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    return np.sqrt( Om*(1.+z)**3 + OL )
    
def t(z,h=0.7,Om=0.3,OL=0.7):
    r"""
    Cosmic time [Gyr] (time since Big Bang).
    
    Syntax: 
    
        t(z,h=0.7,Om=0.3,OL=0.7)
    
    where
        
        z: redshift (float or array)
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    fac = OL / (1.+z)**3
    return (9.778/h) * 2./(3.*np.sqrt(OL)) * \
        np.log((np.sqrt(fac)+np.sqrt(fac+Om)) / np.sqrt(Om))

def tlkbk(z,h=0.7,Om=0.3,OL=0.7):
    """
    Lookback time [Gyr] at redshift z.
    
    Syntax: 
    
        tlkbk(z,h=0.7,Om=0.3,OL=0.7)
    
    where
        
        z: redshift (float or array)
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    return t(0.,h,Om,OL) - t(z,h,Om,OL) 

#------------------------- for EPS formalism ----------------------------
# - critical overdensity for collapse, 
# - transfer function, 
# - linear power spectrum, 
# - mass variance,
# - peak height,
# - Parkinson+08 algorithm
# - EPS conditional mass function & progenitor mass function
#   They all depend on the CosmoloPy library, and thus are grouped here. 

# critical overdensity for collapse
# ---------------------------------------------------------------------
# Pure-numpy replacements for cosmolopy.perturbation.fgrowth and
# .transfer_function_EH (see the NOTE at the top of this file). Not meant
# as general-purpose standalone utilities -- kept private (leading
# underscore) and used only by D() and T() below.
# ---------------------------------------------------------------------

def _fgrowth_EH(z, omega_M_0, unnormed=False):
    """Carroll, Press, & Turner (1992, ARA&A, 30, 499) growth factor,
    normalized to 1 at z=0. Line-for-line port of cosmolopy.perturbation
    .fgrowth, which always assumes a flat cosmology (omega_lambda_0 =
    1 - omega_M_0) internally regardless of the true omega_lambda_0 --
    reproduced here for exact numerical agreement."""
    omega = 1.0 / (1.0 + (1.0 - omega_M_0) / (omega_M_0 * (1.0 + z) ** 3.0))
    lamb = 1.0 - omega
    a = 1.0 / (1.0 + z)
    if unnormed:
        norm = 1.0
    else:
        norm = 1.0 / _fgrowth_EH(0.0, omega_M_0, unnormed=True)
    return (norm * (5.0 / 2.0) * a * omega /
            (omega ** (4.0 / 7.0) - lamb + (1.0 + omega / 2.0) * (1.0 + lamb / 70.0)))


def _TFmdm_set_cosm(omega_matter, omega_baryon, omega_hdm, degen_hdm,
                     omega_lambda, hubble, redshift):
    """Port of TFmdm_set_cosm() in cosmolopy's EH/power.c (Eisenstein & Hu
    1999, ApJ 511 5). Returns the scalar quantities TFmdm_onek_mpc needs,
    instead of setting C globals."""
    theta_cmb = 2.728 / 2.7

    if degen_hdm < 1:
        degen_hdm = 1
    num_degen_hdm = float(degen_hdm)

    if omega_baryon <= 0:
        omega_baryon = 1e-5
    if omega_hdm <= 0:
        omega_hdm = 1e-5

    omega_curv = 1.0 - omega_matter - omega_lambda
    omhh = omega_matter * hubble ** 2
    obhh = omega_baryon * hubble ** 2
    f_baryon = omega_baryon / omega_matter
    f_hdm = omega_hdm / omega_matter
    f_cdm = 1.0 - f_baryon - f_hdm
    f_cb = f_cdm + f_baryon
    f_bnu = f_baryon + f_hdm

    z_equality = 25000.0 * omhh / theta_cmb ** 4
    z_drag_b1 = 0.313 * omhh ** (-0.419) * (1 + 0.607 * omhh ** 0.674)
    z_drag_b2 = 0.238 * omhh ** 0.223
    z_drag = (1291 * omhh ** 0.251 / (1.0 + 0.659 * omhh ** 0.828) *
              (1.0 + z_drag_b1 * obhh ** z_drag_b2))
    y_drag = z_equality / (1.0 + z_drag)

    sound_horizon_fit = 44.5 * np.log(9.83 / omhh) / np.sqrt(1.0 + 10.0 * obhh ** 0.75)

    p_c = 0.25 * (5.0 - np.sqrt(1 + 24.0 * f_cdm))
    p_cb = 0.25 * (5.0 - np.sqrt(1 + 24.0 * f_cb))

    omega_denom = omega_lambda + (1.0 + redshift) ** 2 * (omega_curv + omega_matter * (1.0 + redshift))
    omega_lambda_z = omega_lambda / omega_denom
    omega_matter_z = omega_matter * (1.0 + redshift) ** 2 * (1.0 + redshift) / omega_denom

    growth_k0 = (z_equality / (1.0 + redshift) * 2.5 * omega_matter_z /
                 (omega_matter_z ** (4.0 / 7.0) - omega_lambda_z +
                  (1.0 + omega_matter_z / 2.0) * (1.0 + omega_lambda_z / 70.0)))

    alpha_nu = (f_cdm / f_cb * (5.0 - 2. * (p_c + p_cb)) / (5. - 4. * p_cb) *
                (1 + y_drag) ** (p_cb - p_c) *
                (1 + f_bnu * (-0.553 + 0.126 * f_bnu * f_bnu)) /
                (1 - 0.193 * np.sqrt(f_hdm * num_degen_hdm) + 0.169 * f_hdm * num_degen_hdm ** 0.2) *
                (1 + (p_c - p_cb) / 2 * (1 + 1 / (3. - 4. * p_c) / (7. - 4. * p_cb)) / (1 + y_drag)))
    alpha_gamma = np.sqrt(alpha_nu)
    beta_c = 1 / (1 - 0.949 * f_bnu)

    return dict(theta_cmb=theta_cmb, num_degen_hdm=num_degen_hdm, f_hdm=f_hdm,
                f_cb=f_cb, omhh=omhh, growth_k0=growth_k0, p_cb=p_cb,
                alpha_gamma=alpha_gamma, sound_horizon_fit=sound_horizon_fit,
                beta_c=beta_c)


def _TFmdm_onek_mpc(kk, p):
    """Port of TFmdm_onek_mpc() in cosmolopy's EH/power.c, given the params
    dict from _TFmdm_set_cosm(). Returns tf_cb (the CDM+baryon transfer
    function) -- matches index [0] of what cosmolopy's transfer_function_EH
    returns for baryonic_effects=False. kk is in Mpc^-1 and may be a scalar
    or numpy array."""
    kk = np.asarray(kk, dtype=float)
    theta_cmb = p['theta_cmb']; num_degen_hdm = p['num_degen_hdm']
    f_hdm = p['f_hdm']; omhh = p['omhh']; growth_k0 = p['growth_k0']
    p_cb = p['p_cb']; alpha_gamma = p['alpha_gamma']
    sound_horizon_fit = p['sound_horizon_fit']; beta_c = p['beta_c']

    qq = kk / omhh * theta_cmb ** 2

    y_freestream = (17.2 * f_hdm * (1 + 0.488 * f_hdm ** (-7.0 / 6.0)) *
                     (num_degen_hdm * qq / f_hdm) ** 2)
    temp1 = growth_k0 ** (1.0 - p_cb)
    temp2 = (growth_k0 / (1 + y_freestream)) ** 0.7
    growth_cb = (1.0 + temp2) ** (p_cb / 0.7) * temp1

    gamma_eff = omhh * (alpha_gamma + (1 - alpha_gamma) /
                         (1 + (kk * sound_horizon_fit * 0.43) ** 4))
    qq_eff = qq * omhh / gamma_eff

    tf_sup_L = np.log(2.71828 + 1.84 * beta_c * alpha_gamma * qq_eff)
    tf_sup_C = 14.4 + 325 / (1 + 60.5 * qq_eff ** 1.11)
    tf_sup = tf_sup_L / (tf_sup_L + tf_sup_C * qq_eff ** 2)

    qq_nu = 3.92 * qq * np.sqrt(num_degen_hdm / f_hdm)
    max_fs_correction = (1 + 1.2 * f_hdm ** 0.64 * num_degen_hdm ** (0.3 + 0.6 * f_hdm) /
                          (qq_nu ** (-1.6) + qq_nu ** 0.8))
    tf_master = tf_sup * max_fs_correction

    tf_cb = tf_master * growth_cb / growth_k0
    return tf_cb


def deltac(z,Om=0.3):
    """
    Critical linearized overdensity for spherical collapse.
    
    Syntax:
    
        delta_coll(z,Om=0.3)
        
    where
        
        z: redshift (float or array)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
    """
    return 1.686 / D(z,Om)
def D(z,Om=0.3):
    """
    Linear growth rate D(z).
    
    Syntax:
    
        D(z,Om=0.3)
    
    where
        
        z: redshift (float or array)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
    """
    return _fgrowth_EH(z,Om) 

# transfer function    
def T(k, **cosmo):
    """
    Transfer function of Eisenstein & Hu (1999 ApJ 511 5), with optional
    baryonic effects of Eisenstein & Hu (1997 ApJ 496 605), as 
    implemented in the CosmoloPy library.
    
    Syntax:
    
        T(k, **cosmo)
        
    where
    
        k: wave number [h Mpc^-1] (float or array)
        cosmo: cosmological parameters (dictionary defined in config.py)
    
    Note that if cosmo['m_WDM'] exists, multiply a correction factor, 
    following Bode+01 as cited by Lovell+14, to account for WDM effect.
    """ 
    h = cosmo['h']
    k = h*k # transfer_function_EH takes k in [Mpc^-1]
    if cosmo.get('baryonic_effects', False):
        raise NotImplementedError(
            "T(): baryonic_effects=True was never exercised after the "
            "cosmolopy removal -- only the tf_cb (no baryon-wiggle-fit) "
            "branch was ported. Port _TFmdm_onek_mpc's baryonic-effects "
            "companion (tf_fit.c) if you actually need this.")
    Ttmp = _TFmdm_onek_mpc(k, _TFmdm_set_cosm(
        cosmo['omega_M_0'], cosmo['omega_b_0'], cosmo['omega_n_0'],
        int(cosmo['N_nu']), cosmo['omega_lambda_0'], h, 0.0))
    if 'm_WDM' in cosmo:
        a = 0.05 * cosmo['m_WDM']**(-1.15) * \
            (cosmo['omega_M_0']/0.4)**0.15 * \
            (h/0.65)**1.3
        Ttmp = Ttmp * (1. + (a*k)**2 )**(-5)
    return Ttmp

# power spectrum
def P(k,z=0.,**cosmo):
    """
    Power spectrum. 
    
    Syntax:
    
        P(k,z=0.,**cosmo)
    
    where 
    
        k: wave number [h Mpc^-1] (float or array)
        z: redshift (default=0.)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    Om = cosmo['omega_M_0']
    ns = cosmo['n']
    if 'k0' not in cosmo: # i.e., not normalized yet
        return (T(k,**cosmo)*D(z,Om))**2. * (k/k0(**cosmo))**ns
    else:                 # i.e., already normalized to sigma_8
        return (T(k,**cosmo)*D(z,Om))**2. * (k/cosmo['k0'])**ns
def k0(**cosmo):
    """
    Normalization (k_0) of the primordial power spectrum 
    
        P_primordial(k) = (k/k_0)^n
    
    such that 
    
        sigma(R=8Mpc/h,z=0) = simga_8.
        
    Syntax:
    
        k0(**cosmo)
        
    where
    
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    cosmo['k0'] = 1./3000. # give a temporary, arbitrary normalization 
    k0tmp = cosmo['k0']
    s8 = cosmo['sigma_8']
    ns = cosmo['n']
    s8tmp = sigmaR(8.,**cosmo)
    return k0tmp * (s8tmp / s8 )**(2./ns)
def sigmaR(R,**cosmo):
    """
    Variance of density field smoothed over a spatial scale, linearly 
    extrapolated to z=0.
    
    That is, the integral of 
    
        k^3 / (2 pi^2) * P(k,z=0) * W(k,R)^2 dln(k)
        
    from ln(k) = -inf to +inf, where W(k,R) is the F.T of a window 
    function of size R.
    
    Syntax: 
    
        sigmaR(R,**cosmo)
    
    where
    
        R: the comoving spatial scale of interest [Mpc/h] (float)
        cosmo: cosmological parameters (dictionary defined in config.py)
        
    Return:
    
        the sqrt of the variance, i.e., sigma(R) (float) 
    """
    lnkc = np.log(1./R) # discontinuity of the integrand at ln(k_c)
    # divide the integral range at the discontinuity, ln(k_c), and 
    # integrate separately for the two parts
    S1, S1err = quad(dSdlnk, -50., lnkc, args=(R,cosmo),
                     epsabs=1e-7, epsrel=1e-6,limit=10000)
    S2, S2err = quad(dSdlnk, lnkc, 50., args=(R,cosmo),
                     epsabs=1e-7, epsrel=1e-6,limit=10000)
    S = S1+S2
    return np.sqrt(S)
def dSdlnk(lnk,R,cosmo):
    """
    Auxiliary function -- the integrand for "sigmaR".
    """
    k = np.exp(lnk)
    return DeltaSqr(k,z=0.,**cosmo) * W(k,R)**2
def DeltaSqr(k,z=0.,**cosmo):
    """
    Dimensionless power spectrum,
    
        Delta(k)^2 := k^3 P(k) / (2 pi^2), 
        
    which represents the contribution per log wavenumber of the power 
    spectrum to the variance.
    
    Syntax:
    
        DeltaSqr(k,z=0.,**cosmo)
    
    where 
    
        k: wave number [h Mpc^-1] (float or array)
        z: redshift (default=0.)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    return k**3 / cfg.TwoPisqr * P(k,z,**cosmo)
def W(k,R):
    """
    F.T. of a spherical tophat window function of a given spatial scale. 
    
    Syntax:
        
        W(k,R)
    
    where
        
        k: wave number [h Mpc^-1] (float or array)
        R: the comoving spatial scale of interest [Mpc/h] (float)
    """
    x = k*R
    j1 = (np.sin(x) - x*np.cos(x)) / x**2.
    return 3.*j1/x

# mass variance
def sigma(M,z=0.,**cosmo):
    """
    Variance of linearized density field, smoothed over a mass scale.
    
    Syntax:
    
        sigmaM(M,z=0.,**cosmo)
        
    where
    
        M: the mass scale of interest [M_sun] (float)
        z: redshift (default=0.)
        cosmo: cosmological parameters (dictionary defined in config.py)
    
    Note that this is a wrapper of the function "sigmaM". "sigmaM" only 
    takes a single mass as the input. This function can also take an 
    array of masses as the input, and return an array of sigma(M).
    
    Return:
    
        the sqrt of the variance, i.e., sigma(M,z) (float or array) 
    """
    Om = cosmo['omega_M_0']
    if np.isscalar(M):
        return sigmaM(M,**cosmo) * D(z,Om)
    else:
        return sigmaM_vec(M,**cosmo) * D(z,Om)
    
def sigmaM(M,**cosmo):
    """
    Variance of density field smoothed over a mass scale, linearly 
    extrapolated to z=0.
    
    Syntax:
    
        sigmaM(M,**cosmo)
        
    where
    
        M: the mass scale of interest [M_sun] (float)
        cosmo: cosmological parameters (dictionary defined in config.py)
        
    Return:
    
        the sqrt of the variance, i.e., sigma(M) (float) 
    """
    h = cosmo['h']
    Om = cosmo['omega_M_0']
    OL = cosmo['omega_lambda_0']
    if cosmo['MassVarianceChoice']==0:
        rho=rhom(0.,h,Om,OL) # [Msun kpc^-3]
        R = ( M / rho / cfg.FourPiOverThree )**(1./3.) # [kpc]
        R = R / 1000. * h # sigmaR takes R in [Mpc/h]
        return sigmaR(R,**cosmo)
    else:
        return cfg.sigmalgM_interp(np.log10(M))
sigmaM_vec = np.vectorize(sigmaM, doc="Vectorized 'sigmaM'")

# peak height
def nu(M,z=0,**cosmo):
    """
    Peak height, 
    
        delta_c / sigma(M,z)
    
    where delta_c = 1.686 is the critical overdensity for spherical 
    tophat collapse, and sigma(M,z) is the RMS density fluctuation in 
    spherical tophats of mass M at redshift z.
    
    Syntax:
    
        nu(M,z=0,**cosmo)
        
    where:
        
        M: the mass scale of interest [M_sun] (float or array)
        z: redshift (default=0.)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    return 1.686 / sigma(M,z,**cosmo)

# Parkinson+08 algorithm  

def dlnSdlnM(M,**cosmo):
    """
    The derivative of mass variance:
    
        dln(S)/dln(M) 
    
    where S=sigma(M,z=0)^2 is the mass variance linearly extrapolated to
    z=0.
    
    Syntax:
    
        dlnSdlnM(M,**cosmo)
    
    where
    
        M: the mass scale of interest [M_sun] (float or array)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    return 2.* dlnsigmadlnM(M,**cosmo)
def dlnsigmadlnM(M,**cosmo):
    """
    The derivative of the RMS density fluctuation in spherical tophats of 
    mass M:
    
        dln[sigma(M,z=0))]/dln(M) 
    
    Syntax:
    
        dlnSdlnM(M,**cosmo)
    
    where
    
        M: the mass scale of interest [M_sun] (float or array)
        cosmo: cosmological parameters (dictionary defined in config.py)
        
    Note that the Parkinson+08 alpha(M) factor is the absolute value,
    i.e., the negative, of this function.
    """
    M1 = (1.+cfg.eps)*M
    M2 = (1.-cfg.eps)*M
    sigma1 = sigma(M1,0.,**cosmo)
    sigma2 = sigma(M2,0.,**cosmo)
    return (np.log(sigma1) - np.log(sigma2))/(np.log(M1) - np.log(M2))

def UpdateGlobalVariables(**cosmo):
    """
    Update a few intermediate global variables that are repeatedly used 
    by the functions for the Parkinson+08 algorithm.
    
    Syntax:
    
        UpdateGlobalVariables(**cosmo)
        
    where 
    
        cosmo: cosmological parameters (dictionary defined in config.py)    
    """
    cfg.W0 = deltac(cfg.z0,cosmo['omega_M_0'])
    if cfg.M0>cfg.Mres:
        cfg.qres = min(cfg.Mres/cfg.M0,0.499) # 0.499 is a safety
    else:
        cfg.qres = min(cfg.Mmin/cfg.M0,0.499)
    cfg.sigmares = sigma(cfg.qres*cfg.M0,0.,**cosmo)
    cfg.sigma0 = sigma(cfg.M0,0.,**cosmo)
    cfg.sigmah = sigma(0.5*cfg.M0,0.,**cosmo)
    cfg.S0 = cfg.sigma0**2
    cfg.Sh = cfg.sigmah**2
    Sres = cfg.sigmares**2
    cfg.alphah = -dlnsigmadlnM(0.5*cfg.M0,**cosmo)
    cfg.ures = cfg.sigma0/np.sqrt(Sres-cfg.S0)
    Vres = Sres / (Sres - cfg.S0)**1.5
    Vh = cfg.Sh / (cfg.Sh - cfg.S0)**1.5
    cfg.beta = np.log(Vres/Vh) / np.log(2.*cfg.qres)
    cfg.B = 2.0**cfg.beta * Vh
    cfg.mu = cfg.alphah if cfg.gamma1>=0. else \
        - np.log(cfg.sigmares/cfg.sigmah) / np.log(2.*cfg.qres)
    cfg.eta = cfg.beta - 1. - cfg.gamma1*cfg.mu
    cfg.NupperOverdW = NupperOverdW()
    cfg.dW = dW()
   
def R(q,**cosmo): 
    """
    The factor 
    
        R(q),
        
    as in Parkinson+08 Eq.(A3).
    
    Syntax:
        
        R(q,**cosmo)
    
    where
        
        q: M_1 / M_0, where M_1 is the mass of a progenitor of M_0
            (float or array)
        cosmo: cosmological parameters (dictionary defined in config.py)
        
    Note that this function uses global variables.
    On 2021-05-04, we add the Benson+21 modification to the "V" factor.
    """
    M1 = q*cfg.M0
    S1 = sigma(M1,0.,**cosmo)**2
    V = S1 / (S1 - cfg.S0)**1.5
    V = V * (1.- cfg.S0/S1)**cfg.gamma3 # <<< Benson+21 modification
    fac1 = -dlnsigmadlnM(M1,**cosmo) / cfg.alphah
    fac2 = V / (cfg.B * q**cfg.beta)
    fac3=((2.*q)**cfg.mu *sigma(M1,0.,**cosmo)/cfg.sigmah)**cfg.gamma1
    Rtmp = fac1 * fac2 * fac3
    #if Rtmp>1.0: # <<< a safety check, may remove if turned out useless
    #    print("Warning: R(q=%g)=%g>1, fac1=%g,fac2=%g,fac3=%g"%\
    #        (q,Rtmp,fac1,fac2,fac3))
    return Rtmp
    
def dW():
    """
    Timestep for the Parkinson+08 method.
    
    Syntax:
        
        dW()
        
    Note that this function uses global variables.
    """
    dW1 = 0.1 * cfg.Root2 * np.sqrt(cfg.Sh-cfg.S0)
    dW2 = 0.1 / cfg.NupperOverdW
    return min(dW1,dW2)
def NupperOverdW():
    """
    The integral used to determine the timestep, dW, i.e.,
    
        N_upper/dW, 
        
    where N_upper is given by Parkinson+08 Eqs.(A3),(A5).
    
    Syntax:
        
        NupperOverdW()
        
    Note that this function uses global variables.
    """
    A = cfg.Root2OverPi * cfg.B * cfg.alphah * cfg.G0 \
        / 2.**(cfg.mu*cfg.gamma1) * (cfg.W0/cfg.sigma0)**cfg.gamma2 \
        * (cfg.sigmah/cfg.sigma0)**cfg.gamma1 # this is the Parkinson+08
        # S(q) in Eq.(A2) apart from the factor q^(eta-1).  
    if cfg.qres>=(0.5-cfg.eps):
        I = cfg.eps
    else:
        if np.abs(cfg.eta)>cfg.eps:
            I = (0.5**cfg.eta - cfg.qres**cfg.eta)/cfg.eta
        else:
            I = - np.log(2.*cfg.qres)
    return A * I

def J(ures):
    r"""
    J(u_res), as given by Eq.(A7) of Parkinson+08.
    
    Syntax:
    
        J(ures)
        
    where:
    
        ures: sigma(M_0) / [S(M1)-S(M0)]^1/2 (float)
    """
    return quad(dJdu,0.,ures,epsabs=1e-7,epsrel=1e-6,limit=50)[0]
J_vec = np.vectorize(J, doc="Vectorized 'J(u_res)' function")
def dJdu(u):
    """
    Integrand of J.
    
    Note that this function uses the global variable, cfg.gamma1.
    """
    return (1.+1./u**2)**(cfg.gamma1/2.)
    
def F():
    """
    The smooth accretion fraction, M_smooth / M_0, during a timestep dW,
    as in Parkinson+08 eq.(A6).
    
    Syntax:
    
        F()
    
    Note that this function uses global variables.
    """
    return min(0.5, cfg.Root2OverPi * cfg.Jures_interp(cfg.ures) * \
           cfg.G0/cfg.sigma0 * (cfg.W0/cfg.sigma0)**cfg.gamma2 * cfg.dW)
           
def DrawProgenitors(**cosmo):
    """
    Draw progenitor masses using the Parkinson+08 method.
    
    Syntax:
         
         DrawProgenitors(**cosmo)
         
    where
    
        cosmo: cosmological parameters (dictionary defined in config.py)
         
    Return 
        
        mass of main progenitor (float),
        mass of secondary progenitor (float, =0. if only one progenitor),
        number of progenitors (int, either 1 or 2)
    """
    r1 = np.random.random()
    Nupper = cfg.NupperOverdW * cfg.dW
    Np = 0 # initialize
    if r1 > Nupper:
        M1 = cfg.M0 * (1.-F())
        M2 = 0.
    else:
        r2 = np.random.random()
        q = (cfg.qres**cfg.eta + \
            r2*(2.**(-cfg.eta) - cfg.qres**cfg.eta))**(1./cfg.eta)
        r3 = np.random.random()
        if (r3<R(q,**cosmo)):
            Mtmp1 = cfg.M0 * (1.-F()-q)
            Mtmp2 = cfg.M0 * q
            M1 = max(Mtmp1,Mtmp2)
            M2 = min(Mtmp1,Mtmp2)
        else:
            M1 = cfg.M0 * (1.-F()) 
            M2 = 0.
    if M1>cfg.Mres: Np += 1
    if M2>cfg.Mres: Np += 1
    return M1,M2,Np

# EPS conditional mass function & progenitor mass function

def Masterisk(z=0.,height=1.,**cosmo):
    """
    The Press-Schechter mass, or, more generally, the mass corresponding 
    to a density peak of a given height at a given redshift [M_sun].
    
    Syntax:
    
        Masterisk(z=0.,nu=1.,**cosmo)
    
    where
        
        z: redshift (default=0.)
        height: peak height (default=1.)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    return brentq(FindMasterisk, 1e1, 1e17, args=(z,height,cosmo), 
        xtol=1e-5, rtol=1e-3, maxiter=100)

def FindMasterisk(M,z,height,cosmo):
    """
    Auxiliary function for the function "Masterisk".
    """
    return nu(M,z,**cosmo) - height
    
def dNdlnM1(M1,z1,M0,z0,**cosmo):
    """
    The EPS progenitor mass function (PMF), 
    
        dN/dln(M_1) = M_0/M_1 dP/dln(M_1),
        
    i.e., the mean number of progenitors with mass in the logarithmic 
    mass bin [ln(M_1), ln(M_1)+dln(M_1)].
    
    Strictly speaking, the PMF is dN/dM_1, related to dN/dln(M_1) by 
    
        dN/dM_1 = 1/M_1 * dN/dln(M_1) := n_EPS(M_1,z_1|M_0,z_0)
    
    where n_EPS(M_1,z_1|M_0,z_0)dM_1 is the mean number of progenitors of 
    mass [M_1,M_1+dM_1] at redshift z_1, which belong to descendent 
    halos of mass M_0 at redshift z_0, as defined in, e.g., 
    Jiang & van den Bosch (2014) Eq.(3).
    
    Syntax:
    
        dNdlnM1(M1,z1,M0,z0,**cosmo)
        
    where
    
        M1: progenitor mass [M_sun] (float or array)
        z1: progenitor redshift (float)
        M0: descendent mass [M_sun] (float)
        z0: descendent redshift (float)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    return M0/M1*dPdlnM1(M1,z1,M0,z0,**cosmo)

def dPdlnM1(M1,z1,M0,z0,**cosmo):
    """
    The EPS conditional mass function (CMF), 
    
        dP/dln(M_1), 
        
    or equivalently,
    
        dP/dln(M_1/M_0).
    
    Strictly speaking, the CMF is dP/dM_1, related to dP/dln(M_1) by
    
        dP/dM_1 = 1/M_1 * dP/dln(M_1) := P(M_1,z_1|M_0,z_0),
    
    where P(M_1,z_1|M_0,z_0)dM_1 is the mass fraction of halos of mass 
    M_0 at redshift z_0 that was contained in progenitors of mass 
    [M_1,M_1+dM_1] at z_1>z_0, as defined in e.g., 
    Jiang & van den Bosch (2014) Eq.(1).
    
    Syntax:
    
        dNdlnM1(M1,z1,M0,z0,**cosmo)
        
    where
    
        M1: progenitor mass [M_sun] (float or array)
        z1: progenitor redshift (float)
        M0: descendent mass [M_sun] (float)
        z0: descendent redshift (float)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    Om = cosmo['omega_M_0']
    S1 = sigma(M1,0.,**cosmo)**2.
    S0 = sigma(M0,0.,**cosmo)**2.
    W1 = deltac(z1,Om)
    W0 = deltac(z0,Om)
    return fEPS(S1,W1,S0,W0) * S1 * (-dlnSdlnM(M1,**cosmo))
def fEPS(S1,W1,S0,W0):
    """
    The conditional probability density 
    
         f_EPS(S_1,W_1|S_0,W_0)
         
    as in f_EPS(S_1,W_1|S_0,W_0)dS_1, which is the probability for a 
    random walk passing through (S_0,W_0) to excute a first-upcrossing of 
    a higher overdensity barrier, W=W_1, at [S1,S1+dS1].
    
    Syntax:
    
        fEPS(S1,W1,S0,W0)
        
    where
    
        S1: variance of density fluctuations on the mass scale of the 
            progenitor mass, M_1, linearly extrapolated to z=0 
            (float or array)
        W1: critical overdensity for collapse at redshift z_1 (float)
        S0: variance of density fluctuations on the mass scale of the 
            descendent mass, M_0, linearly extrapolated to z=0 (float)
        W0: critical overdensity for collapse at redshift z_0 (float)
        
    Note that what we implement here is NOT the original 
    spherical-collapse conditional probability density function, but an 
    empirical fit to the Millenium simulation by Cole+08 Eq.(7). The 
    reason is that this function and the functions calling it are used as 
    benchmarks for testing if Monte-Carlo merger trees are accurate 
    compared to merger trees from simulations. While we don't think the 
    Millenium result is the ultimate representation of trees in 
    simulations (as the halo finding and linking procedure of the 
    Millenium simulations are not necessarily optimal), it is a commonly 
    used benchmark (e.g., by Parkinson+08 and Jiang & van den Bosch 14). 
    If a newer fit based on better simulations becomes available, we 
    shall replace the Cole+08 fit with the newer result. 
    """
    DeltaS = S1-S0
    v10 = (W1-W0)/np.sqrt(DeltaS)
    return 0.2 *v10**0.75 /DeltaS *np.exp(-0.1 *v10**3) # empirical 
    #return cfg.Root1Over2Pi*v10/DeltaS*np.exp(-0.5*v10**2) # SC EPS
    
def NGTM1(M1,z1,M0,z0,**cosmo):
    """
    The cumulative EPS progenitor mass function, 
    
        N(>M_1,z_1|M_0,z_0).
    
    Syntax:
    
        NGTM1(M1,z1,M0,z0,**cosmo)
    
    where:
    
        M1: progenitor mass [M_sun] (float or array)
        z1: progenitor redshift (float)
        M0: target halo mass [M_sun] (float)
        z0: target halo redshift (float)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    a = np.log(M1)
    b = np.log(M0)
    return quad(dNGTM1dlnM1, a, b, args=(z1,M0,z0,cosmo),
        epsabs=1e-4, epsrel=1e-3,limit=100)[0]
def dNGTM1dlnM1(lnM1,z1,M0,z0,cosmo):
    """
    The integrand for "NGTM1".
    """
    M1 = np.exp(lnM1)
    return dNdlnM1(M1,z1,M0,z0,**cosmo)
    
def MGTM1(M1,z1,M0,z0,**cosmo):
    """
    The cumulative mass-weighted EPS progenitor mass function, 
    
        M(>M_1,z_1|M_0,z_0).
    
    Syntax:
    
        MGTM1(M1,z1,M0,z0,**cosmo)
    
    where:
    
        M1: progenitor mass [M_sun] (float or array)
        z1: progenitor redshift (float)
        M0: target halo mass [M_sun] (float)
        z0: target halo redshift (float)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    a = np.log(M1)
    b = np.log(M0)
    return quad(dMGTM1dlnM1, a, b, args=(z1,M0,z0,cosmo),
        epsabs=1e-4, epsrel=1e-3,limit=100)[0]
def dMGTM1dlnM1(lnM1,z1,M0,z0,cosmo):
    """
    The integrand for "MGTM1".
    """
    M1 = np.exp(lnM1)
    return M1*dNdlnM1(M1,z1,M0,z0,**cosmo)
 
# unevolved subhalo mass functions
   
def dNdlnmaM0_all(x,gamma,alpha,beta,zeta):
    """
    Jiang & van den Bosch (2014) fitting function for the total
    unevolved subhalo mass function, 
    
        dN/dln(x) = gamma x^alpha exp(-beta x^zeta)
    
    with x: = m_acc/M_0 the mass at infall divided by the host mass.
    
    Syntax:
    
        dNdlnmaM0_all(x,gamma,alpha,beta,zeta)
        
    where
    
        x: m_acc / M_0
        gamma: normalization
        alpha: slope
        beta: parameter for the location of decay
        zeta: parameter for the steepness of decay
    """
    return gamma* x**alpha * np.exp(-beta*x**zeta)

def dNdlnmaM0_1st(x,gamma1,gamma2,alpha1,alpha2,beta,zeta):
    """
    Jiang & van den Bosch (2014) fitting function for the 1st-order 
    unevolved subhalo mass function, 
    
        dN/dln(x) = [gamma1 x^alpha1 + gamma2 x^alpha2] exp(-beta x^zeta)
    
    with x: = m_acc/M_0 the mass at infall divided by the host mass.
    
    Syntax:
    
        dNdlnmaM0_1st(x,gamma1,gamma2,alpha1,alpha2,beta,zeta)
        
    where
    
        x: m_acc / M_0
        gamma1: normalization for the first power-law component
        alpha1: slope of the first power law
        gamma2: normalization for the second power-law component
        alpha2: slope of the second power law
        beta: parameter for the location of decay
        zeta: parameter for the steepness of decay
    """
    return (gamma1*x**alpha1+gamma2*x**alpha2)*np.exp(-beta*x**zeta)

def fsub_pred(M0, z0, level=1, **cosmo):
    """
    Predicted value for f_{sub} based on N_dyn, the dynamical age of
    a halo. Uses the method described in Section 4.2 of Jiang &
    van den Bosch (2016), "Paper 1".
        
    Syntax:
    
        fsub_pred(M0, z0,level,**cosmo)
        
    where
    
        M0: halo mass at redshift of observation (float)
        z0: halo redshift of observation (float)
        level: 1 or 2, i.e., the fraction of mass bound into
               level 1+ subhaloes or level 2+ subhaloes
        cosmo: dictionary of cosmological parameters
    Note:
        The Ndyn function called computes the number of dynamical
        times between two redshifts, but the dynamical time differs
        from that defined in Jiang & van den Bosch (2016) by a factor
        of pi/2.
    """
    Om = cosmo['omega_M_0']
    OL = cosmo['omega_lambda_0']
    h = cosmo['h']
    f = 0.5
    alpha_f = 0.815 * np.exp(-2. * f**3.) / f**0.707
    omegat_f = np.sqrt(2.*np.log(alpha_f + 1.))
    rhs = deltac(z0,Om) + omegat_f * np.sqrt(
          sigma(f*M0,**cosmo)**2. - sigma(M0,**cosmo)**2.)
    eqn = lambda zf: deltac(zf,Om) - rhs
    zform = brentq(eqn, 0., 1000., 
            xtol=1e-5, rtol=1e-3, maxiter=100)
    Nt = Ndyn(zform,z0,h,Om,OL) / (np.pi / 2.)

    if(level == 1):
        return 0.325/Nt**0.6 - 0.075
    elif(level == 2):
        return 0.0461/Nt**1.3 - 0.0035
    else:
        sys.exit("Invalid subhalo level chosen for fsub!")
