### A series of SHMRs that increase in complexity ###
### Main purpose of having so many is to explore degeneracies in posteriors ###
import numpy as np

def simple(theta, lgMh_2D):

    """_summary_
    Convert from halo mass to stellar mass in the simplest way possible

    Args:
        lgMh_2D (np.ndarray): 2D halo mass array
        theta_0: power law slope 
        theta_1: log normal scatter

    Returns:
        np.ndarray: 2D stellar mass array
    """

    M_star_a = 10.5
    M_halo_a = 12

    lgMs_2D = theta[0]*(lgMh_2D-M_halo_a) + M_star_a
    scatter_2D = np.random.normal(loc=0, scale=theta[1], size=(lgMs_2D.shape))
    return lgMs_2D + scatter_2D

def anchor(theta, lgMh_2D):

    """_summary_
    Convert by accounting for uncertainty in the anchor point

    Args:
        lgMh_2D (np.ndarray): 2D halo mass array
        theta_0: power law slope 
        theta_1: log normal scatter
        theta_2: the stellar mass anchor point

    Returns:
        np.ndarray: 2D stellar mass array
    """

    M_star_a = theta[2]
    M_halo_a = 12

    lgMs_2D = theta[0]*(lgMh_2D-M_halo_a) + M_star_a
    scatter_2D = np.random.normal(loc=0, scale=theta[1], size=(lgMs_2D.shape))
    return lgMs_2D + scatter_2D

def curve(theta, lgMh_2D):

    """_summary_
    Convert by accounting for a non-powerpoint relation

    Args:
        lgMh_2D (np.ndarray): 2D halo mass array
        theta_0: power law slope 
        theta_1: log normal scatter
        theta_2: the stellar mass anchor point
        theta_3: quadratic term to curve the relation

    Returns:
        np.ndarray: 2D stellar mass array
    """

    M_star_a = theta[2]
    M_halo_a = 12

    lgMs_2D = theta[0]*(lgMh_2D-M_halo_a) + theta[3]*(lgMh_2D-M_halo_a)**2 + M_star_a
    scatter_2D = np.random.normal(loc=0, scale=theta[1], size=(lgMs_2D.shape))
    return lgMs_2D + scatter_2D

def sigma(theta, lgMh_2D):

    """_summary_
    Convert from halo mass to stellar mass with a growing scatter

    Args:
        lgMh_2D (np.ndarray): 2D halo mass array
        theta_0: power law slope 
        theta_1: log normal scatter
        theta_2: the stellar mass anchor point
        theta_3: quadratic term to curve the relation
        theta_4: slope of scatter as function of log halo mass

    Returns:
        np.ndarray: 2D stellar mass array
    """

    M_star_a = theta[2]
    M_halo_a = 12

    sigma = theta[1] + theta[4]*(lgMh_2D - M_halo_a)
    sigma[sigma < 0] = theta[1]

    lgMs_2D = theta[0]*(lgMh_2D-M_halo_a) + theta[3]*(lgMh_2D-M_halo_a)**2 + M_star_a
    scatter_2D = np.random.normal(loc=0, scale=sigma, size=(lgMs_2D.shape))
    return lgMs_2D + scatter_2D

def redshift(theta, lgMh_2D, z_2D):

    """_summary_
    Convert from halo mass to stellar mass with scatter in Ms
    Now Ms* is based on z_acc

    Args:
        lgMh_2D (np.ndarray): 2D halo mass array
        theta_0: power law slope 
        theta_1: log normal scatter
        theta_2: the stellar mass anchor point
        theta_3: quadratic term to curve the relation
        theta_4: slope of scatter as function of log halo mass
        theta_5: the strength of the redshift dependance on the stellar mass anchor point

    Returns:
        np.ndarray: 2D stellar mass array
    """

    M_star_a = theta[2] * (1+z_2D)**theta[5]
    M_halo_a = 12

    sigma = theta[1] + theta[4]*(lgMh_2D - M_halo_a)
    sigma[sigma < 0] = theta[1]

    lgMs_2D = theta[0]*(lgMh_2D-M_halo_a) + theta[3]*(lgMh_2D-M_halo_a)**2 + M_star_a
    scatter_2D = np.random.normal(loc=0, scale=sigma, size=(lgMs_2D.shape))
    return lgMs_2D + scatter_2D

def deterministic(theta, lgMh_2D, z_2D):

    """_summary_
    Convert from halo mass to stellar mass with no scatter

    Args:
        lgMh_2D (np.ndarray): 2D halo mass array
        theta_0: power law slope 
        theta_1: the stellar mass anchor point
        theta_2: quadratic term to curve the relation
        theta_3: the strength of the redshift dependance on the stellar mass anchor point

    Returns:
        np.ndarray: 2D stellar mass array
    """

    M_star_a = theta[1] * (1+z_2D)**theta[3]
    M_halo_a = 12

    lgMs_2D = theta[0]*(lgMh_2D-M_halo_a) + theta[2]*(lgMh_2D-M_halo_a)**2 + M_star_a
    return lgMs_2D 


def lgMs_D22(lgMv, a=1.82, log_e=-1.5):

    """
    returns the determinisitic stellar mass [M_sun]
    """
    lgMs = log_e + 12.5 + a*lgMv - a*12.5
    return lgMs


def lgMs_B13(lgMv,z=0.):
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
    return lge+lgM + f_B13(lgMv-lgM,a) - f_B13(0.,a)
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

def lgMs_RP17(lgMv,z=0.):
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
    return lge+lgM + f_RP17(lgMv-lgM,a) - f_RP17(0.,a)
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


# def Nsigma(theta, lgMh_2D, Nsamples=3):

#     """_summary_
#     Convert from halo mass to stellar mass sampling sigma more than once!
#     Nsamples is not a free parameter! In principle it could be...?

#     Args:
#         lgMh_2D (np.ndarray): 2D halo mass array
#         theta_0: power law slope
#         theta_1: quadratic term to curve the function
#         theta_2: log normal scatter

#     Returns:
#         np.ndarray: 2D stellar mass array
#     """

#     M_star_a = 10 
#     M_halo_a = 12

#     lgMs_2D = theta[0]*(lgMh_2D-M_halo_a) + theta[1]*(lgMh_2D-M_halo_a)**2 + M_star_a
#     scatter_3D = np.random.normal(loc=0, scale=theta[2], size=(Nsamples, lgMs_2D.shape[0], lgMs_2D.shape[1]))
#     stack = scatter_3D + lgMs_2D[np.newaxis, :, :]
#     return np.vstack(stack)