################## Functions for galaxy-halo connection ####################

# Arthur Fangzhou Jiang 2019, HUJI
# Arthur Fangzhou Jiang 2020, Caltech
# Sheridan Beckwith Green 2020, Yale

#########################################################################

import numpy as np

import config as cfg
import aux
import profiles as pr
import cosmo as co
import math
import astropy.constants as c
import astropy.units as u
G_const = c.G.to(u.kpc**3 / (u.solMass * u.s**2))

from lmfit import minimize, Parameters

#########################################################################


def dynamical_time(radius, mass):
    return 1/np.sqrt((G_const*mass)/(radius**3))

def integrate_SFH(SFR, thalo):
    """
    Integrate the star formation history (SFH) accounting for stellar mass loss.
    
    Parameters:
    SFR : array
        Star formation rate at each time step (Msun/Gyr).
    thalo : array
        Array of time values corresponding to SFR (in Gyr).
    
    Returns:
    SFH : array
        Cumulative stellar mass formed over time, accounting for mass loss.
    """
    f_lost = 0.05 * np.log(1 + thalo / 0.0014)
    delta_t = np.diff(thalo)
    mstar_insitu = delta_t * SFR[:-1]
    SFH = np.cumsum(mstar_insitu * (1 - f_lost[:-1]))
    
    return SFH, f_lost

#---SFR-Vmax relation   

def SFR_B19(v_Mpeak, z, V_0=2.151, V_a=-1.658, V_la=1.680, V_z=-0.233, 
           eps_0=0.109, eps_a=-3.441, eps_la=5.079, eps_z=-0.781, 
           alpha_0=-5.598, alpha_a=-20.731, alpha_la=13.455, alpha_z=-1.321, 
           beta_0=-1.911, beta_a=0.395, beta_z=0.747, 
           gamma_0=-1.699, gamma_a=4.206, gamma_z=-0.809, 
           delta_0=0.055): 
    """
    Compute the star formation rate (SFR) for star-forming galaxies.
    
    Parameters:
        v_Mpeak : float
            Peak circular velocity of the halo (km/s)
        z : float
            Redshift

    Returns:
        SFR_SF : float
            Star formation rate for star-forming galaxies.
    """

    a = 1 / (1 + z)
    # Compute log10(V) and V
    log_V = V_0 + V_a * (1 - a) + V_la * np.log(1 + z) + V_z * z
    V = 10 ** log_V

    # Compute log10(epsilon) and epsilon
    log_eps = eps_0 + eps_a * (1 - a) + eps_la * np.log(1 + z) + eps_z * z
    eps = 10 ** log_eps

    # Compute alpha and beta
    alpha = alpha_0 + alpha_a * (1 - a) + alpha_la * np.log(1 + z) + alpha_z * z
    beta = beta_0 + beta_a * (1 - a) + beta_z * z

    # Compute log10(gamma) and gamma
    log_gamma = gamma_0 + gamma_a * (1 - a) + gamma_z * z
    gamma = 10 ** log_gamma

    # Compute v
    v = v_Mpeak / V

    # Compute SFR_SF
    SFR_SF = eps * ((v**alpha + v**beta)**-1 + gamma * np.exp(- (np.log10(v)**2) / (2 * delta_0**2)))*(u.solMass/u.yr)
    SFR = SFR_SF.to(u.solMass/u.Gyr).value

    return SFR

#---galaxy-size-halo-structure relation   

def Reff(Rv,c2):
    """
    Effective radius (3D half-stellar-mass radius) of a galaxy, given
    the halo virial radius and concentration, using the empirical formula
    of Jiang+19 (MN, 488, 4801) eq.6
    
        R_eff = 0.02 (c/10)^-0.7 R_vir
    
    Syntax:
    
        Reff(Rv,c2)
        
    where
    
        Rv: virial radius [kpc] (float or array)
        c2: halo concentration defined as R_vir / r_-2, where r_-2 is the
            radius at which dln(rho)/dln(r) = -2 (float or array)
    """
    return 0.02 * (c2/10.)**(-0.7) * Rv

def Reff_A24(lgMs, saga_type="All"):

    """_summary_

    Returns: log effective size (R50 in kpc)
        _type_: _description_
    """

    SAGA_BF = {
    "SAGA All": {"slope": 0.26458366571273345, "intercept": -2.100690006286266, "scatter": 0.1906663381273509},
    "SAGA Q": {"slope": 0.1994499157457564, "intercept": -1.6041844860688481, "scatter": 0.14507035619243538},
    "SAGA SF": {"slope": 0.3012187938834186, "intercept": -2.4024451095768806, "scatter": 0.19285279643803752},
    "Iso": {"slope": 0.3013391193852031, "intercept": -2.4832747367616874, "scatter": 0.2292941864255589},
    "SAGAbg": {"slope": 0.2672172918734746, "intercept": -2.071862467897147, "scatter": 0.22267752553295456},
    "All": {"slope": 0.2791014019742783, "intercept": -2.2146932772595345, "scatter": 0.22718136165752814},}

    slope = SAGA_BF[saga_type]["slope"]
    intercept = SAGA_BF[saga_type]["intercept"]

    return slope * lgMs + intercept

    
#---stellar-halo-mass relation

def lgMs_B18(lgMv, z=0, return_epsilon=False):
    """
    Calculate the stellar mass for a given peak halo mass and redshift based on UniverseMachine2018.
    
    Parameters:
        Mpeak (float): Log10 of peak historical halo mass (in solar masses).
        z (float): Redshift.
        params (dict): Dictionary of parameters for the SMHM relation.

    Returns:
        float: Log10 of the stellar mass (in solar masses).
        
    """

    params = {'EFF_0': -1.434595, # taken from the "smhm_med_params.txt" file
            'EFF_0_A': 1.831346,
            'EFF_0_A2': 1.368294,
            'EFF_0_Z': -0.2169429,
            'M_1': 12.03538,
            'M_1_A': 4.556205,
            'M_1_A2': 4.417054,
            'M_1_Z': -0.7313717,
            'ALPHA': 1.963342,
            'ALPHA_A': -2.315609,
            'ALPHA_A2': -1.732084,
            'ALPHA_Z': 0.177598,
            'BETA': 0.4817875,
            'BETA_A': -0.8405796,
            'BETA_Z': -0.4706532,
            'DELTA': 0.4108514,
            'GAMMA': -1.034197,
            'GAMMA_A': -3.100399,
            'GAMMA_Z': -1.054511,
            'CHI2': 157.1985}
    
    a = 1.0 / (1.0 + z)
    a1 = a - 1.0
    lna = np.log(a)
    
    # Calculate z-dependent parameters
    zparams = {}
    zparams['m_1'] = params['M_1'] + a1 * params['M_1_A'] - lna * params['M_1_A2'] + z * params['M_1_Z']
    zparams['sm_0'] = (zparams['m_1'] + params['EFF_0'] + a1 * params['EFF_0_A'] - lna * params['EFF_0_A2'] + z * params['EFF_0_Z'])
    zparams['alpha'] = params['ALPHA'] + a1 * params['ALPHA_A'] - lna * params['ALPHA_A2'] + z * params['ALPHA_Z']
    zparams['beta'] = params['BETA'] + a1 * params['BETA_A'] + z * params['BETA_Z']
    zparams['delta'] = params['DELTA']
    zparams['gamma'] = 10 ** (params['GAMMA'] + a1 * params['GAMMA_A'] + z * params['GAMMA_Z'])
    
    # Compute the stellar mass
    dm = lgMv - zparams['m_1']
    dm2 = dm / zparams['delta']
    lgMs = (zparams['sm_0'] - np.log10(10 ** (-zparams['alpha'] * dm) + 10 ** (-zparams['beta'] * dm)) + zparams['gamma'] * np.exp(-0.5 * (dm2 ** 2)))

    if return_epsilon:
        return 10**(lgMs - lgMv)
    else:
        return lgMs

def lgMs_B13(lgMv,z=0, return_epsilon=False):
    r"""
    Log stellar mass [M_sun] given log halo mass and redshift, using the 
    fitting function by Behroozi+13.
    
    Syntax:
    
        lgMs_B13(lgMv,z)
    
    where 
        lgMv: log virial mass [Msun] (float or array)
        z: redshift (float) (default=0.)
    """
    a = 1./(1.+z)
    v = v_B13(a)
    e0 = -1.777
    ea = -0.006
    ez = 0.000
    ea2 = -0.119
    M0 = 11.514
    Ma = -1.793
    Mz = -0.251
    lge = e0 + (ea*(a-1.)+ez*z)*v + ea2*(a-1.)
    lgM = M0 + (Ma*(a-1.)+Mz*z)*v

    lgMs =  lge+lgM + f_B13(lgMv-lgM,a) - f_B13(0.,a)

    if return_epsilon:
        return 10**(lgMs - lgMv)
    else:
        return lgMs

def v_B13(a):
    r"""
    Auxiliary function for lgMs_B13.
    """
    return np.exp(-4.*a**2)

def f_B13(x,a):
    r"""
    Auxiliary function for lgMs_B13.
    """
    a0 = -1.412
    aa = 0.731
    az = 0.0
    d0 = 3.508
    da = 2.608
    dz = -0.043
    g0 = 0.316
    ga = 1.319
    gz = 0.279
    v = v_B13(a)
    z = 1./a-1.
    alpha = a0 + (aa*(a-1.)+az*z)*v
    delta = d0 + (da*(a-1.)+dz*z)*v
    gamma = g0 + (ga*(a-1.)+gz*z)*v
    return delta*(np.log10(1.+np.exp(x)))**gamma/(1.+np.exp(10**(-x)))-\
        np.log10(1.+10**(alpha*x))

def lgMs_RP17(lgMv,z=0, return_epsilon=False):
    """
    Log stellar mass [M_sun] given log halo mass and redshift, using the 
    fitting function by Rodriguez-Puebla+17.
    
    Syntax:
    
        lgMs_RP17(lgMv,z)
    
    where 
    
        lgMv: log virial mass [M_sun] (float or array)
        z: redshift (float) (default=0.)
    """
    a = 1./(1.+z)
    v = v_RP17(a)
    e0 = -1.758
    ea = 0.110
    ez = -0.061
    ea2 = -0.023
    M0 = 11.548
    Ma = -1.297
    Mz = -0.026
    lge = e0 + (ea*(a-1.)+ez*z)*v + ea2*(a-1.)
    lgM = M0 + (Ma*(a-1.)+Mz*z)*v

    lgMs = lge+lgM + f_RP17(lgMv-lgM,a) - f_RP17(0.,a)

    if return_epsilon:
        return 10**(lgMs - lgMv)
    else:
        return lgMs

def v_RP17(a):
    """
    Auxiliary function for lgMs_RP17.
    """
    return np.exp(-4.*a**2)

def f_RP17(x,a):
    r"""
    Auxiliary function for lgMs_RP17.
    
    Note that RP+17 use 10**( - alpha*x) while B+13 used 10**( +alpha*x).
    """
    a0 = 1.975
    aa = 0.714
    az = 0.042
    d0 = 3.390
    da = -0.472
    dz = -0.931
    g0 = 0.498
    ga = -0.157
    gz = 0.0
    v = v_RP17(a)
    z = 1./a-1.
    alpha = a0 + (aa*(a-1.)+az*z)*v
    delta = d0 + (da*(a-1.)+dz*z)*v
    gamma = g0 + (ga*(a-1.)+gz*z)*v
    return delta*(np.log10(1.+np.exp(x)))**gamma/(1.+np.exp(10**(-x)))-\
        np.log10(1.+10**( - alpha*x))

#---halo-response patterns

def slope(X,choice='NIHAO'):
    """
    Logarithmic halo density slope at 0.01 R_vir, as a function of the 
    stellar-to-halo-mass ratio X, based on simulation results.
    
    Syntax:
    
        slope(X,choice='NIHAO')
        
    where
    
        X: M_star / M_vir (float or array)
        choice: choice of halo response -- 
            'NIHAO' (default, Tollet+16, mimicking strong core formation)
            'APOSTLE' (Bose+19, mimicking no core formation)
    """
    if choice=='NIHAO':
        s0 = X / 8.77e-3
        s1 = X / 9.44e-5
        return np.log10(26.49*(1.+s1)**(-0.85) + s0**1.66) + 0.158
    elif choice=='APOSTLE':
        s0 = X / 8.77e-3
        return np.log10( 20. + s0**1.66 ) + 0.158 

def c2c2DMO(X,choice='NIHAO'):
    """
    The ratio between the baryon-influenced concentration c_-2 and the 
    dark-matter-only c_-2, as a function of the stellar-to-halo-mass
    ratio, based on simulation results. 
    
    Syntax:
    
        c2c2DMO(X,choice='NIHAO')
        
    where
    
        X: M_star / M_vir (float or array)
        choice: choice of halo response -- 
            'NIHAO' (default, Tollet+16, mimicking strong core formation)
            'APOSTLE' (Bose+19, mimicking no core formation)
    """
    if choice=='NIHAO':
        #return 1. + 227.*X**1.45 - 0.567*X**0.131 # <<< Freundlich+20
        return 1.2 + 227.*X**1.45 - X**0.131 # <<< test
    elif choice=='APOSTLE':
        return 1. + 227.*X**1.45
        
#---concentration-mass-redshift relations

def c2_Zhao09(Mv,t,version='zhao'):
    """
    Halo concentration from the mass assembly history, using the Zhao+09
    relation.
    
    Syntax:
    
        c2_Zhao09(Mv,t,version)
        
    where
    
        Mv: main-branch virial mass history [M_sun] (array)
        t: the time series of the main-branch mass history (array of the
            same size as Mv)
        version: 'zhao' or 'vdb' for the different versions of the
                 fitting function parameters (string)
    
    Note that we need Mv and t in reverse chronological order, i.e., in 
    decreasing order, such that Mv[0] and t[0] is the instantaneous halo
    mass and time.
    
    Note that Mv is the Bryan and Norman 98 M_vir.
    
    Return:
        
        halo concentration c R_vir / r_-2 (float)
    """
    if(version == 'vdb'):
        coeff1 = 3.40
        coeff2 = 6.5
    elif(version == 'zhao'):
        coeff1 = 3.75
        coeff2 = 8.4
    idx = aux.FindNearestIndex(Mv,0.04*Mv[0])
    return 4.*(1.+(t[0]/(coeff1*t[idx]))**coeff2)**0.125
    
def lgc2_DM14(Mv,z=0.):
    r"""
    Halo concentration given virial mass and redshift, using the 
    fitting formula from Dutton & Maccio 14 (eqs.10-11)
    
    Syntax:
    
        lgc2_DM14(Mv,z=0.)
    
    where 
    
        Mv: virial mass, M_200c [M_sun] (float or array)
        z: redshift (float or array of the same size as Mv,default=0.)
        
    Note that this is for M_200c, for the BN98 M_vir, use DM14 eqs.12-13
    instead. 
    
    Note that the parameters are for the Planck(2013) cosmology.
    
    Return:
    
        log of halo concentration c_-2 = R_200c / r_-2 (float or array)
    """
    # <<< concentration from NFW fit
    #a = 0.026*z - 0.101 # 
    #b = 0.520 + (0.905-0.520) * np.exp(-0.617* z**1.21)
    # <<< concentration from Einasto fit
    a = 0.029*z - 0.130
    b = 0.459 + (0.977-0.459) * np.exp(-0.490* z**1.303) 
    return a*np.log10(Mv*cfg.h/10**12.)+b

def c2_DK15(Mv,z=0.,n=-2):
    """
    Halo concentration from Diemer & Kravtsov 15 (eq.9).
    
    Syntax:
    
        c2_DK15(Mv,z)
        
    where
    
        Mv: virial mass, M_200c [M_sun] (float or array)
        z: redshift (float or array of the same size as Mv,default=0.)
        n: logarithmic slope of power spectrum (default=-2 or -2.5 for
            typical values of LCDM, but more accurately, use the power
            spectrum to calculate n)
    
    Note that this is for M_200c.
    Note that the parameters are for the median relation
    
    Return:
        
        halo concentration R_200c / r_-2 (float)
    """
    cmin = 6.58 + 1.37*n
    vmin = 6.82 + 1.42*n
    v = co.nu(Mv,z,**cfg.cosmo)
    fac = v / vmin
    return 0.5*cmin*(fac**(-1.12)+fac**1.69)

#---halo contraction model

def contra_Hernquist(r,h,d,A=0.85,w=0.8):
    """
    Returns contracted halo profile given baryon profile and initial halo 
    profile, following the model of Gnedin+04.
    
    Syntax:
    
        contra(r,h,d)
        
    where
    
        r: initial radii at which we evaluate the mass profile [kpc]
            (array)
        h: initial NFW halo profile (object of the NFW class as defined
            in profiles.py)
        d: baryon profile (object of the Hernquist class as defined in
            profiles.py)
        A: coefficient in the relation between the orbit-averaged radius 
            of a particle that is currently in a shell and the instant
            radius of the shell: <r>/r_vir = A (r/r_vir)^w 
            (default=0.85)
        w: power-law index in the relation between the orbit-averaged
            radius and instant radius (default=0.8)
            
    Note that there is halo-to-halo variation in the values of A and w,
    which is discussed in Gnedin+11. Here we ignore the halo-to-halo 
    variation and adopt the fiducial values A=0.85 and w=0.8 as in 
    Gnedin+04.
    
    Note that the input halo object "h" is for the total mass profile,
    which includes an initial baryon mass distribution that is assumed
    to be -similar to the initial DM profile, i.e.,
    
        M_dm,i = (1-f_b) M_i(r)
        M_b,i = f_b M_i(r)
            
    Return:
    
        contracted radii, r_f [kpc] (array of the same length as r)
        enclosed DM mass at r_f [M_sun] (array of the same length as r) 
    """
    # prepare variables
    Mv = h.Mh
    c = h.ch
    rv = h.rh
    fc = h.f(c)
    Mb = d.Mb
    rb = d.r0
    fb = Mb/Mv
    xb = rb/rv
    x = r/rv
    xave = A * x**w
    rave = xave * rv # orbit-averaged radii
    # compute y_0
    a = 2.*fb*(1.+xb)**2 * fc / (xb*c)**2
    fdm = 1.-fb
    s = 0.5/a 
    p = 1.+2.*w
    sqrtQ1 = np.sqrt( (fdm/(3.*a))**3 + s**2 )
    sqrtQw = np.sqrt( (fdm/p)**p / a**3 + s**2 )
    y1 = (sqrtQ1 + s)**(1./3.) - (sqrtQ1 - s)**(1./3.)
    yw = (sqrtQw + s)**(1./p) - (sqrtQw - s)**(1./p)
    em2a = np.exp(-2.*a) 
    y0 = y1*em2a + yw*(1.-em2a)
    # compute exponent b
    b = 2.*y0/(1.-y0)*(2./xb-4.*c/3.)/(2.6+fdm/(a*y0**(2.*w)))
    # compute the contraction ratio y(x)=r_f / r
    Mi = h.M(rave)
    t0 = 1./(fdm + d.M(y0**w *rave)/Mi)
    t1 = 1./(fdm + d.M(rave)/Mi)
    embx = np.exp(-b*x)
    y = t0*embx + t1*(1.-embx)
    rf = y*r
    return rf, fdm*h.M(r)
    
def contra_exp(r,h,d,A=0.85,w=0.8):
    """
    Returns contracted halo profile given baryon profile and initial halo 
    profile, following the model of Gnedin+04.
    
    Similar to "contra_Hernquist", but here we assume the final baryon
    distribution to be an exponential disk, instead of a spherical 
    Hernquist profile
    
    Syntax:
    
        contra(r,h,d)
        
    where
    
        r: initial radii at which we evaluate the mass profile [kpc]
            (array)
        h: initial NFW halo profile (object of the NFW class as defined
            in profiles.py)
        d: baryon profile (object of the exponential class as defined in
            profiles.py)
        A: coefficient in the relation between the orbit-averaged radius 
            of a particle that is currently in a shell and the instant
            radius of the shell: <r>/r_vir = A (r/r_vir)^w 
            (default=0.85)
        w: power-law index in the relation between the orbit-averaged
            radius and instant radius (default=0.8)
            
    Note that there is halo-to-halo variation in the values of A and w,
    which is discussed in Gnedin+11. Here we ignore the halo-to-halo 
    variation and adopt the fiducial values A=0.85 and w=0.8 as in 
    Gnedin+04.
    
    Note that the input halo object "h" is for the total mass profile,
    which includes an initial baryon mass distribution that is assumed
    to be -similar to the initial DM profile, i.e.,
    
        M_dm,i = (1-f_b) M_i(r)
        M_b,i = f_b M_i(r)
            
    Return:
    
        contracted radii, r_f [kpc] (array of the same length as r)
        enclosed DM mass at r_f [M_sun] (array of the same length as r) 
    """
    # prepare variables
    Mv = h.Mh
    c = h.ch
    rv = h.rh
    fc = h.f(c)
    Mb = d.Mb
    rb = d.r0
    fb = Mb/Mv
    xb = rb/rv
    x = r/rv
    xave = A * x**w
    rave = xave * rv # orbit-averaged radii
    # compute y_0
    a = fb * fc / (xb*c)**2
    fdm = 1.-fb
    s = 0.5/a 
    p = 1.+2.*w
    sqrtQ1 = np.sqrt( (fdm/(3.*a))**3 + s**2 )
    sqrtQw = np.sqrt( (fdm/p)**p / a**3 + s**2 )
    y1 = (sqrtQ1 + s)**(1./3.) - (sqrtQ1 - s)**(1./3.)
    yw = (sqrtQw + s)**(1./p) - (sqrtQw - s)**(1./p)
    em2a = np.exp(-2.*a) 
    y0 = y1*em2a + yw*(1.-em2a)
    # compute exponent b
    b = 2.*y0/(1.-y0)*(2./(3.*xb)-4.*c/3.)/(2.6+fdm/(a*y0**(2.*w)))
    # compute the contraction ratio y(x)=r_f / r
    Mi = h.M(rave)
    t0 = 1./(fdm + d.M(y0**w *rave)/Mi)
    t1 = 1./(fdm + d.M(rave)/Mi)
    embx = np.exp(-b*x)
    y = t0*embx + t1*(1.-embx)
    rf = y*r
    return rf, fdm*h.M(r)
    
def contra(r,h,d,A=0.85,w=0.8):
    """
    Returns contracted halo profile given baryon profile and initial halo 
    profile, following the model of Gnedin+04.
    
    Syntax:
    
        contra(r,h,d)
        
    where
    
        r: initial radii at which we evaluate the mass profile [kpc]
            (array)
        h: initial NFW halo profile (object of the NFW class as defined
            in profiles.py)
        d: baryon profile (object of the Hernquist class as defined in
            profiles.py)
        A: coefficient in the relation between the orbit-averaged radius 
            of a particle that is currently in a shell and the instant
            radius of the shell: <r>/r_vir = A (r/r_vir)^w 
            (default=0.85)
        w: power-law index in the relation between the orbit-averaged
            radius and instant radius (default=0.8)
            
    Note that there is halo-to-halo variation in the values of A and w,
    which is discussed in Gnedin+11. Here we ignore the halo-to-halo 
    variation and adopt the fiducial values A=0.85 and w=0.8 as in 
    Gnedin+04.
    
    Note that the input halo object "h" is for the total mass profile,
    which includes an initial baryon mass distribution that is assumed
    to be -similar to the initial DM profile, i.e.,
    
        M_dm,i = (1-f_b) M_i(r)
        M_b,i = f_b M_i(r)
            
    Return:
    
        the contracted DM profile (object of the Dekel class as defined
            in profiles.py) 
        contracted radii, r_f [kpc] (array of the same length as r)
        enclosed DM mass at r_f [M_sun] (array of the same length as r) 
    """
    # contract
    if isinstance(d,pr.Hernquist):
        rf,Mdmf = contra_Hernquist(r,h,d,A,w)
    elif isinstance(d,pr.exp):
        rf,Mdmf = contra_exp(r,h,d,A,w)
    # fit contracted profile
    params = Parameters()
    params.add('Mv', value=(1.-d.Mb/h.Mh)*h.Mh, vary=False)
    params.add('c', value=h.ch,min=1.,max=100.)
    params.add('a', value=1.,min=-2.,max=2.)
    out = minimize(fobj_Dekel, params, args=(rf,Mdmf,h.Deltah,h.z)) 
    MvD = out.params['Mv'].value
    cD = out.params['c'].value
    aD = out.params['a'].value
    return pr.Dekel(MvD,cD,aD),rf,Mdmf 
def fobj_Dekel(p, xdata, ydata, Delta, z):
    """
    Auxiliary function for "contra" -- objective function for fitting
    a Dekel+ profile to the contracted halo
    """
    h = pr.Dekel(p['Mv'].value,p['c'].value,p['a'].value,Delta=Delta,z=z)
    ymodel = h.M(xdata)
    return (ydata - ymodel) / ydata



def Moster_epsilon(M):

    M1 = 11.78
    epsilon_N = 0.15
    beta = 1.78
    gamma = 0.57

    # Compute epsilon(M, z)
    epsilon = 2 * epsilon_N / ((M / M1)**(-beta) + (M / M1)**gamma)
    return epsilon

def epsilon_M_z(M, z, param_type):
    """
    Compute epsilon(M, z)
    
    Parameters:
    - M: Mass (float or array)
    - z: Redshift (float or array)
    - M0: Base mass constant
    - Mz: Mass redshift factor
    - epsilon_0: Base epsilon constant
    - epsilon_z: Epsilon redshift factor
    - beta_0: Base beta constant
    - beta_z: Beta redshift factor
    - gamma: Gamma constant
    
    Returns:
    - epsilon: The computed epsilon(M, z)
    """

    if param_type == "Moster":

        M0=11.339
        Mz=0.692
        epsilon_0=0.005
        epsilon_z=0.689
        beta_0=3.344
        beta_z=-2.079
        gamma=0.966
        
    elif param_type == "Oleary":

        M0=11.34829
        Mz=0.654238
        epsilon_0=0.009010
        epsilon_z=0.596666
        beta_0=3.094621
        beta_z=-2.019841
        gamma=1.107304
        
    elif param_type == "Nadler":

        M0=11.34829
        Mz=0.654238
        epsilon_0=0.009010
        epsilon_z=0.596666
        beta_0= 2.64
        beta_z=-2.019841
        gamma=1.107304

    elif param_type == "Read":

        M0=11.34829
        Mz=0.654238
        epsilon_0=0.009010
        epsilon_z=0.596666
        beta_0= 2.34
        beta_z=-2.019841
        gamma=1.107304
    
    # Scale factor a
    a = 1 / (z + 1)
    
    # Log10(M1(z))
    M1 = 10**(M0 + (Mz*(1-a)))
    
    # epsilon_N(z)
    epsilon_N = epsilon_0 + epsilon_z*(1-a)
    
    # beta(z)
    beta = beta_0 + beta_z*(1-a)
    
    # Compute epsilon(M, z)
    epsilon = 2 * epsilon_N / ((M / M1)**(-beta) + (M / M1)**gamma)
    
    return epsilon


def dex_sampler(log_arr, dex=0.2, N_samples=1):

    if N_samples==1:
        scatter = np.random.normal(loc=0, scale=dex, size=(log_arr.shape[0])) # the standard normal PDF

    elif N_samples>1:
        scatter = np.random.normal(loc=0, scale=dex, size=(N_samples, log_arr.shape[0])) # the standard normal PDF
    return log_arr + scatter
