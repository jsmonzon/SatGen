############################# orbit class ###############################

# Arthur Fangzhou Jiang 2016, HUJI --- original version

# Arthur Fangzhou Jiang 2019, HUJI, UCSC --- revisions:
# - no longer convert speed unit from kpc/Gyr to km/s 
# - improved dynamical-friction (DF) (see profiles.py for more details)

# Arthur Fangzhou Jiang 2020, Caltech --- revision:
# - added ram-pressure (RP) drag due to dark-matter self-interaction 

#########################################################################

from profiles import ftot

import numpy as np
from scipy.integrate import ode
from scipy.optimize import root_scalar
import profiles
import config as cfg
import init
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#########################################################################

#---
class orbit(object):
    """
    Class for orbit and orbit integration in axisymetric potential
    
    Syntax:
    
        o = orbit(xv,potential=None)
        
    which initializes an orbit object "o", where
    
        xv: the phase-space coordinate in a cylindrical frame 
            [R,phi,z,VR,Vphi,Vz] [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr]
            (numpy array)
        potential: host potential (a density profile object, as defined 
            in profiles.py, or a list such objects that constitute a 
            composite potential)
        
    Attributes:
    
        o.t: integration time [Gyr] (float, list, or array)  
        o.xv: phase-space coordinates in a cylindrical frame
            [R,phi,z,VR,Vphi,Vz] [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] 
            which are initialized to be the value put in by hand when 
            the orbit object is created, and are updated once the method 
            o.integrate is called
            (numpy array)
        o.tList: time sequence
        o.xvList: coordinates along the time sequence
    
    and conditionally also: (only available if potential is not None and 
    spherically symmetric) 
    
        o.rperi: peri-center distance [kpc] (float)
        o.rapo: apo-center distance [kpc] (float)
        
    Methods:
    
        o.integrate(t,potential,m=None,CoulombLogChoice=None):
            updates the coordinates o.xv by integrates over time "t" in 
            the "potential", using scipy.integrate.ode, and considering 
            dynamical friction if "m" and "CoulombLogChoice" are provided 
        
    Arthur Fangzhou Jiang, 2016-10-27, HUJI
    Arthur Fangzhou Jiang, 2019-08-21, UCSC
    """
    def __init__(self,xv,potential=None):
        r"""
        Initialize an orbit by specifying the phase-space coordinates 
        in a cylindrical frame.
        
        Syntax:
            
            o = orbit(xv, potential=None)
        
        where 
        
            xv: phase-space coordinates in a cylindrical frame
                [R,phi,z,VR,Vphi,Vz] 
                [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] 
                (numpy array)
            potential: host potential (a density profile object, as 
                defined in profiles.py, or a list such objects that 
                constitute a composite potential)
                (dafault=None, i.e., when initializing an orbit, do not
                specify the potential, but if provided, a few more 
                attributes attributes are triggered, including o.rperi
                and o.rapo, and maybe more, to be determined)
        """
        self.xv = xv # instantaneous coordinates, initialized by input
        self.t = 0. # instantaneous time, initialized here to be 0. 
        self.tList = [] # initialize time sequencies
        self.xvList = []
        if potential is not None: 
            pass # <<< to be added: self.rperi and self.rapo etc 
    def integrate(self,t,potential,m=None,sigmamx=None,Xd=None):
        r"""
        Integrate orbit over time "t" [Gyr], using methods that comes 
        with scipy.integrate.ode; and update coorinates to be the values 
        at the end of t.
        
        Syntax:
        
            o.integrate(t,potential,m=None,sigmamx=None)
        
        where
        
            t: time [Gyr] (float, list, or numpy array)
            potential: host potential (a profile object, as defined 
                in profile.py, or a list of such objects which 
                altogether constitute the host potential)
            m: satellite mass [M_sun]  (float or None)
                (default is None; if provided, dynamical friction is 
                triggered)
            sigmamx: self-interaction cross section over particle mass 
                [cm^2/g] or [2.08889e-10 kpc^2/M_sun] (float or None)
                (default is None; if provided, ram-pressure drag is 
                triggered)    
            Xd: coefficient for ram-pressure drag as in Kummer+18
                (default is None; if sigmamx provided, ram-pressure drag
                is triggered, then Xd must be provided)
                
        Note that in case when t is list or array, attributes such as 
        
            .tList
            .xvList
        
        which are lists, will start from empty and get appended new 
        value for each timestep; while attributes
        
            .t
            .xv
            
        store the instantaneous time and coordinates atthe  end of t.     
        """
        solver = ode(f,jac=None).set_integrator(
            #'vode', 
            'dopri5',
            nsteps=500, # default=500
            max_step = 0.1, # default=0.0 
            rtol=1e-5, # default = 1e-6
            atol=1e-5, # default = 1e-12
            )
        solver.set_initial_value(self.xv, self.t)
        solver.set_f_params(potential,m,sigmamx,Xd)            
        if isinstance(t, list) or isinstance(t,np.ndarray): 
            for tt in t:               
                solver.integrate(tt)
                self.t = tt
                self.xv = solver.y
                self.tList.append(self.t) 
                self.xvList.append(self.xv)
        else: # i.e., if t is a scalar
            solver.integrate(t)
            self.xv = solver.y
            self.t = solver.t
            self.tList.append(self.t) 
            self.xvList.append(self.xv)
        self.tArray = np.array(self.tList)
        self.xvArray = np.array(self.xvList)

def f(t,y,p,m,sigmamx,Xd):
    r"""
    Returns right-hand-side functions of the EOMs for orbit integration:
    
        d R / d t = VR    
        d phi / d t = Vphi / R
        d z / d t = Vz
        d VR / dt = Vphi^2 / R + fR 
        d Vphi / dt = - VR * Vphi / R + fphi
        d Vz / d t = fz
        
    for use in the method ".integrate" of the "orbit" class.
        
    Syntax:
            
        f(t,y,p,m)
            
    where 
        
        t: integration time [Gyr] (float)
        y: phase-space coordinates in a cylindrical frame
            [R,phi,z,VR,Vphi,Vz] [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] 
            (numpy array)
        p: host potential (a density profile object, as defined 
            in profiles.py, or a list of such objects that constitute a 
            composite potential)
        m: satellite mass [M_sun] (float or None) 
            (If m is None, dynamical friction is ignored)
        sigmamx: self-interaction cross section over particle mass 
            [cm^2/g] or [2.08889e-10 kpc^2/M_sun] (float or None)
            (If sigmamx is None, ram-pressure is ignored)

    Note that fR, fphi, fz are the R-, phi- and z-components of the 
    acceleration at phase-space location y, computed by the function 
    "ftot" defined in profiles.py. 

    Return: the list of
    
        [VR,    
         Vphi / R
         Vz,
         Vphi^2 / R + fR ,
         - VR * Vphi / R + fphi,
         fz]

    i.e., the right-hand side of the EOMs describing the evolution of the
    phase-space coordinates in a cylindrical frame
    """
    R, phi, z, VR, Vphi, Vz = y
    fR, fphi, fz = ftot(p,y,m,sigmamx,Xd)
    R = max(R,1e-6) # safety
    return [VR, 
        Vphi/R, 
        Vz,
        Vphi**2./R + fR,
        - VR*Vphi/R + fphi,
        fz]

def E_orbital(xv, potential):
    """
    Measure the orbital energy from SatGen Orbits at a specific time (t).

    Parameters:
    -----------
    xv : numpy.ndarray
        Phase-space coordinates in Cartesian reference frame:
        [x, y, z, vx, vy, vz] in units of [kpc, kpc, kpc, kpc/Gyr, kpc/Gyr, kpc/Gyr].
    
    potential : object or list of objects
        Host potential (a density profile object, as defined in profiles.py) 
        or a list of such objects that constitute a composite potential.

    Returns:
    --------
    float
        The total orbital energy (kinetic + potential).
    """

    if not isinstance(xv, np.ndarray) or xv.shape != (6,):
        raise ValueError("xv must be a NumPy array of shape (6,)")

    # Compute kinetic energy
    KE = 0.5 * (xv[3]**2 + xv[4]**2 + xv[5]**2)

    # Compute radius
    radius = np.sqrt(xv[0]**2 + xv[1]**2 + xv[2]**2)
    PE = potential.Phi(radius) #this is negative by convention!

    return KE + PE #should be negative!

def L_orbital(xv):
    """
    Compute the magnitude of the angular momentum vector L = r x v.

    Parameters:
    -----------
    xv : numpy.ndarray
        Phase-space coordinates in Cartesian reference frame:
        [x, y, z, vx, vy, vz] in units of [kpc, kpc, kpc, kpc/Gyr, kpc/Gyr, kpc/Gyr].

    Returns:
    --------
    float
        The magnitude of the angular momentum vector.
    """

    if not isinstance(xv, np.ndarray) or xv.shape != (6,):
        raise ValueError("xv must be a NumPy array of shape (6,)")

    # Position vector r = (x, y, z)
    r = xv[:3]  # [x, y, z]

    # Velocity vector v = (vx, vy, vz)
    v = xv[3:]  # [vx, vy, vz]

    # Compute angular momentum vector L = r x v
    L_vector = np.cross(r, v)

    # Return the magnitude of L
    return np.linalg.norm(L_vector)


def find_pericenter_apocenter(xv, potential):
    """
    Compute the pericenter and apocenter of an orbit by solving for the 
    roots of the orbital equation.

    Parameters:
    -----------
    xv : numpy.ndarray
        Phase-space coordinates in Cartesian reference frame:
        [x, y, z, vx, vy, vz] in units of [kpc, kpc, kpc, kpc/Gyr, kpc/Gyr, kpc/Gyr].
    
    potential : object or list of objects
        Host potential (a density profile object, as defined in profiles.py) 
        or a list of such objects that constitute a composite potential.

    Returns:
    --------
    tuple (r_peri, r_apo)
        The pericenter and apocenter distances in kpc.
    """

    def orbital_equation(r, E_orbit, L, potential):
        """Equation whose roots correspond to pericenter and apocenter."""
        
        Phi_r = potential.Phi(r)
        return (1 / r**2) + (2*(Phi_r - E_orbit)) / L**2

    # Compute total orbital energy
    E_orbit = E_orbital(xv, potential)

    # Compute angular momentum magnitude
    L = L_orbital(xv)

    # Set an initial bracket for root-finding
    r_min, r_max = 0.01, 2000  # Reasonable astrophysical limits (in kpc)
    r_prime = np.linalg.norm(xv[:3])

    # Find pericenter (smallest r)
    peri_result = root_scalar(orbital_equation, args=(E_orbit, L, potential),
                              bracket=[r_min, r_prime], method='brentq')

    # Find apocenter (largest r)
    apo_result = root_scalar(orbital_equation, args=(E_orbit, L, potential),
                             bracket=[r_prime, r_max], method='brentq')

    if not (peri_result.converged and apo_result.converged):
        raise RuntimeError("Root-finding for pericenter or apocenter failed.")

    return np.array([peri_result.root, apo_result.root])


def resample_orbit(tree_file, save_file, select_average=False, seed_index=None, plot=False):
    """
    Resamples the orbital coordinates of subhalos based on the method from Li et al. (2020),
    using an NFW profile for the parent halo. The updated coordinates are saved in a .npz file.
    
    Parameters:
    -----------
    tree_file : str
        Path to the input file containing halo tree data.
    save_file : str
        Path to save the resampled data.
    seed_index : int, optional
        Seed for random number generation (default: None).
    plot : bool, optional
        If True, plots histograms comparing original and resampled coordinates (default: False).
    
    Returns:
    --------
    None
    """
    np.random.seed(seed_index)
    data = np.load(tree_file)
    xvs_og = np.copy(data["coordinates"])  # Original coordinates
    xvs_copy = np.copy(xvs_og)  # Modified coordinates
    
    Nhalo = data["mass"].shape[0]
    
    for sub_ii in range(1, Nhalo):  # Skip the host (sub_ii = 0)
        try:
            acc_index_ii = np.nonzero(xvs_og[sub_ii])[0][0]
            parent_ii = data["ParentID"][sub_ii, acc_index_ii]

            hp_ii = profiles.NFW(
                data["mass"][parent_ii, acc_index_ii], 
                data["concentration"][parent_ii, acc_index_ii], 
                Delta=cfg.Dvsample[acc_index_ii], 
                z=cfg.zsample[acc_index_ii]
            )

            if select_average:
                vel_ratio, gamma = 1.15, 2.6
                #vel_ratio, gamma = init.ZZLi2020_fixed(hp_ii, data["mass"][sub_ii, acc_index_ii], cfg.zsample[acc_index_ii])
                xvs_copy[sub_ii, acc_index_ii] = init.orbit_from_Li2020(hp_ii, vel_ratio, gamma)  
        
            else:
                vel_ratio, gamma = init.ZZLi2020(hp_ii, data["mass"][sub_ii, acc_index_ii], cfg.zsample[acc_index_ii])
                xvs_copy[sub_ii, acc_index_ii] = init.orbit_from_Li2020(hp_ii, vel_ratio, gamma)  
        
        except IndexError:
            pass
            #print(f"Skipping subhalo {sub_ii} due to missing data.")

    np.savez(save_file, 
        redshift=cfg.zsample,
        CosmicTime=cfg.tsample,
        mass=data["mass"],
        order=data["order"],
        ParentID=data["ParentID"],
        VirialRadius=data["VirialRadius"],
        concentration=data["concentration"],
        coordinates=xvs_copy,
    )
    
    if plot:
        # Extract nonzero values for both original and resampled coordinates
        mask = xvs_og[:,:,0] != 0
        labels = ["r (kpc)", "$\\theta$ (rad)", "z (kpc)", "vr (kpc/Gyr)", "v$\\theta$ (rad/Gyr)", "vz (kpc/Gyr)"]
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
        
        for i, ax in enumerate(axes.flat):
            ax.hist(xvs_og[:,:,i][mask], bins=15, edgecolor="white", alpha=0.6, label="Original")
            ax.hist(xvs_copy[:,:,i][mask], bins=15, edgecolor="white", alpha=0.3, label="Resampled")
            ax.set_xlabel(labels[i])
            ax.set_yscale("log")
        
        axes[0,0].legend()
        plt.tight_layout()
        plt.show()


