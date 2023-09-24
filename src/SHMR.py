### A series of SHMRs that increase in complexity ###
### Main purpose of having so many is to explore degeneracies in posteriors ###
import numpy as np

def deterministic(theta, lgMh_2D):

    """_summary_
    Convert from halo mass to stellar mass with no scatter

    Args:
        lgMh_2D (np.ndarray): 2D halo mass array
        a0: power law slope
        a1: quadratic term to curve the function

    Returns:
        np.ndarray: 2D stellar mass array
    """

    M_star_a = 10 # these are the anchor points
    M_halo_a = 11.67

    lgMs_2D = theta[0]*(lgMh_2D-M_halo_a) + theta[1]*(lgMh_2D-M_halo_a)**2 + M_star_a
    return lgMs_2D

def sigma(theta, lgMh_2D):

    """_summary_
    Convert from halo mass to stellar mass with scatter

    Args:
        lgMh_2D (np.ndarray): 2D halo mass array
        a0: power law slope
        a1: quadratic term to curve the function
        a2: log normal scatter

    Returns:
        np.ndarray: 2D stellar mass array
    """

    M_star_a = 10 
    M_halo_a = 11.67

    lgMs_2D = theta[0]*(lgMh_2D-M_halo_a) + theta[1]*(lgMh_2D-M_halo_a)**2 + M_star_a
    scatter_2D = np.random.normal(loc=0, scale=theta[2], size=(lgMs_2D.shape))
    return lgMs_2D + scatter_2D

def Nsigma(theta, lgMh_2D, Nsamples=3):

    """_summary_
    Convert from halo mass to stellar mass sampling sigma more than once!
    Nsamples is not a free parameter! In principle it could be...?

    Args:
        lgMh_2D (np.ndarray): 2D halo mass array
        a0: power law slope
        a1: quadratic term to curve the function
        a2: log normal scatter

    Returns:
        np.ndarray: 2D stellar mass array
    """

    M_star_a = 10 
    M_halo_a = 11.67

    lgMs_2D = theta[0]*(lgMh_2D-M_halo_a) + theta[1]*(lgMh_2D-M_halo_a)**2 + M_star_a
    scatter_3D = np.random.normal(loc=0, scale=theta[2], size=(Nsamples, lgMs_2D.shape[0], lgMs_2D.shape[1]))
    stack = scatter_3D + lgMs_2D[np.newaxis, :, :]
    return np.vstack(stack)

def sigmaGrow(theta, lgMh_2D):

    """_summary_
    Convert from halo mass to stellar mass with a growing scatter

    Args:
        lgMh_2D (np.ndarray): 2D halo mass array
        a0: power law slope
        a1: quadratic term to curve the function
        a2: initial value of sigma
        a3: the power law slope of sigma (needs to be negative)

    Returns:
        np.ndarray: 2D stellar mass array
    """

    M_star_a = 10
    M_halo_a = 11.67

    sigma = theta[2] + theta[3]*(lgMh_2D - M_halo_a)

    lgMs_2D = theta[0]*(lgMh_2D-M_halo_a) + theta[1]*(lgMh_2D-M_halo_a)**2 + M_star_a
    scatter_2D = np.random.normal(loc=0, scale=sigma, size=(lgMs_2D.shape))
    return lgMs_2D + scatter_2D

def anchor(theta, lgMh_2D):

    """_summary_
    Convert from halo mass to stellar mass with scatter
    Ms* is a free parameter

    Args:
        lgMh_2D (np.ndarray): 2D halo mass array
        a0: power law slope
        a1: quadratic term to curve the function
        a2: log normal scatter
        a3: the stellar mass anchor point

    Returns:
        np.ndarray: 2D stellar mass array
    """

    M_star_a = theta[3]
    M_halo_a = 11.67

    lgMs_2D = theta[0]*(lgMh_2D-M_halo_a) + theta[1]*(lgMh_2D-M_halo_a)**2 + M_star_a
    scatter_2D = np.random.normal(loc=0, scale=theta[2], size=(lgMs_2D.shape))
    return lgMs_2D + scatter_2D

def anchorRedshift(theta, lgMh_2D, z_2D):

    """_summary_
    Convert from halo mass to stellar mass with scatter in Ms
    Now Ms* is based on z_acc

    Args:
        lgMh_2D (np.ndarray): 2D halo mass array
        a0: power law slope
        a1: quadratic term to curve the function
        a2: log normal scatter
        a3: intial Ms*
        a4: power on z_acc

    Returns:
        np.ndarray: 2D stellar mass array
    """

    M_star_a = theta[3] * (1+z_2D)**theta[4]
    M_halo_a = 11.67

    lgMs_2D = theta[0]*(lgMh_2D-M_halo_a) + theta[1]*(lgMh_2D-M_halo_a)**2 + M_star_a
    scatter_2D = np.random.normal(loc=0, scale=theta[2], size=(lgMs_2D.shape))
    return lgMs_2D + scatter_2D