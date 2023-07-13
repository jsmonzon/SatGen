import numpy as np
import matplotlib.pyplot as plt
import galhalo
import jax
import jax.numpy as jnp

@jax.jit
def cumulative(Ms, mass_bins):
    N = jnp.histogram(Ms, bins=mass_bins)[0] #just need the counts
    Nsub = np.sum(N)
    stat = Nsub-jnp.cumsum(N) 
    return jnp.insert(stat, 0, Nsub) #to add the missing index

@jax.jit
def CSMF(Ms, mass_bins, full=False):

    counts = jnp.apply_along_axis(cumulative, 1, Ms, mass_bins=mass_bins) 
    quant = jnp.percentile(counts, jnp.array([5, 50, 95]), axis=0, method="nearest")

    if full == True:
        return counts
    else:
        return quant

def plot_CSMF(mass_bins, quant):

    plt.figure(figsize=(8, 8))
    plt.plot(mass_bins[1:], quant[1], label="median", color="black")
    plt.fill_between(mass_bins[1:], y1=quant[0], y2=quant[2], alpha=0.2, color="grey")
    plt.grid(alpha=0.4)
    plt.yscale("log")
    plt.xlabel("log m$_{stellar}$ (M$_\odot$)", fontsize=15)
    plt.ylabel("log N (> m$_{stellar}$)", fontsize=15)
    plt.legend()
    plt.show()

def H2H_CSMF(scatter_quant, norm=False):

    if norm==True:
        stat = (scatter_quant[:,2]-scatter_quant[:,0])/scatter_quant[:,1]
    else:
        stat = (scatter_quant[:,2]-scatter_quant[:,0])
    return stat

@jax.jit
def differential(Ms, mass_bins, binsize): # nested function
    N = jnp.histogram(Ms, bins=mass_bins)[0]
    return N/binsize

@jax.jit
def SHMF(Ms, mass_bins, plot=False, full=False):
    binsize = (mass_bins[1] - mass_bins[0])
    counts = jnp.apply_along_axis(differential, 1, Ms, mass_bins=mass_bins, binsize=binsize) 

    SHMF_ave = jnp.average(counts/binsize, axis=0)
    SHMF_std = jnp.std(counts/binsize, axis=0)

    if plot==True:
        plt.figure(figsize=(8, 8))

        plt.plot(mass_bins, SHMF_ave, label="average", marker="o", color="black")
        plt.plot(mass_bins, SHMF_ave+SHMF_std, label="1 $\sigma$", marker=".", ls=":", color="grey")
        plt.plot(mass_bins, SHMF_ave-SHMF_std, marker=".", ls=":", color="grey")
        plt.yscale("log")
        plt.xlabel("log (m/M)", fontsize=20)
        plt.ylabel("log[ dN / dlog(m/M) ]", fontsize=20)
        plt.legend()
        plt.show()
    
    if full == True:
        return counts
    else:
        return SHMF_ave, SHMF_std

def mass_rank(mass):

    rank = np.flip(np.argsort(mass,axis=1), axis=1) # rank the subhalos from largest to smallest
    ranked_mass = np.take_along_axis(mass, rank, axis=1) # this is it!!!

    return rank, ranked_mass
    

def SHMF_old(mass, mass_min=-4, Nbins=50, plot=True):
 
    mass_frac = mass/np.nanmax(mass) #normalizing by host mass
    mass_frac[:, 0] = np.nan # removing the host mass from the matrix
    ana_mass = mass_frac

    # mass_frac = mass/np.max(mass) #normalizing by host mass
    # mass_frac[:, 0] = 0.0  # removing the host mass from the matrix
    # zero_mask = mass_frac != 0.0 
    # ana_mass = np.log10(np.where(zero_mask, mass_frac, np.nan))  # up until here the stats are good
    
    def histofunc(mass, bins=False): # nested function
        if bins==True:
            return np.histogram(mass, range=(mass_min, 0), bins=Nbins)
        else:
            return np.histogram(mass, range=(mass_min, 0), bins=Nbins)[0]

    # now to start counting!
    m_counts, bins = histofunc(ana_mass[0], bins=True)  # to be keep in memory, only needs to be measured once
    binsize = (bins[1] - bins[0])
    bincenters = 0.5 * (bins[1:] + bins[:-1])

    I = np.apply_along_axis(histofunc, 1, ana_mass)  # this applies the histogram to the whole matrix

    SHMF_ave = np.average(I, axis=0)

    SHMF_std = np.std(I, axis=0)
        
    if plot == True:
        plt.figure(figsize=(8, 8))

        plt.plot(bincenters, np.log10(SHMF_ave/binsize), label="average", marker="o", color="black")
        plt.plot(bincenters, np.log10((SHMF_ave+SHMF_std)/binsize), label="1 $\sigma$", marker=".", ls=":", color="grey")
        plt.plot(bincenters, np.log10((SHMF_ave-SHMF_std)/binsize), marker=".", ls=":", color="grey")

        plt.xlabel("log (m/M)", fontsize=20)
        plt.ylabel("log[ dN / dlog(m/M) ]", fontsize=20)
        plt.legend()
        plt.show()

    return bincenters, np.array([SHMF_ave, SHMF_std])


def CSMF_old(Ms, Npix=50, down=5, up=95, plot=True, full=False):

    """
    calculates the cumulative satellite mass function given a number of mass ind
    input mass should not be in log space
    """
    # the same x-array for all the CSMFs
    mass_range = np.logspace(3,10,Npix)
    #cleaning up the excessive padding
    #max_real = Ms.shape[1] - np.sum(np.isnan(Ms),axis=1)

    #now to start counting!
    I = np.zeros((Npix, Ms.shape[0]))

    for i,val in enumerate(mass_range):
        I[i] = np.sum(Ms > np.log10(val),axis=1)
            
    CSMF_quant = np.percentile(I.transpose(), [down, 50, up], axis=0) # the percentiles
        
    if plot==True:
        
        plt.figure(figsize=(8, 8))

        plt.plot(mass_range, CSMF_quant[1, :], label="median", color="black")
        plt.fill_between(mass_range, y1=CSMF_quant[0, :], y2=CSMF_quant[2, :], alpha=0.2, color="grey")
        plt.grid(alpha=0.4)
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("log m$_{stellar}$ (M$_\odot$)", fontsize=15)
        plt.ylabel("log N (> m$_{stellar}$)", fontsize=15)
        plt.legend()
        plt.show()

    if full == True:
        return I.transpose()
    else:
        return CSMF_quant

def CSMF_1D(Ms, Npix=50, plot=True):

    """
    calculates the cumulative satellite mass function given a number of mass ind
    input mass should not be in log space
    """
    # the same x-array for all the CSMFs
    mass_range = np.logspace(3,10,Npix)
    I = [np.sum(Ms > np.log10(i)) for i in mass_range]
                    
    if plot==True:
        
        plt.figure(figsize=(8, 8))

        plt.plot(mass_range, I, color="black")
        plt.grid(alpha=0.4)
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("log m$_{stellar}$ (M$_\odot$)", fontsize=15)
        plt.ylabel("log N (> m$_{stellar}$)", fontsize=15)
        plt.show()

    return I

def scatter_stat(CSMF, det_CSMF):

    if len(CSMF.shape) == 4:
        print("averaging over the samples!")
        CSMF = np.average(CSMF,axis=0)
    
    Nreal = CSMF.shape[0]
    Mpix = CSMF.shape[2]

    stat = (CSMF[:,2]-CSMF[:,0])

    print("returning a", Nreal, "by", Mpix, "matrix")
    return stat
           

def plot_single_real(file):

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm  

    tree = np.load(file)
    mass = tree["mass"]
    time = tree["CosmicTime"]

    nhalo = 10
    select = [12,13,14,15,30,45,67,99,80,23]
    
    colors = cm.viridis(np.linspace(0, 1, nhalo))

    plt.figure(figsize=(6,6))

    for i in range(nhalo):
        plt.plot(time, mass[select[i]], color=colors[i])
    plt.xlabel("Gyr", fontsize=15)
    plt.ylabel("halo mass (M$_{\odot}$)", fontsize=15)
    plt.yscale("log")
    plt.ylim(1e5,1e12)
    plt.show()

# def wow():
#     halo = np.load("../etc/halo_mass_PDF_full.npy")
#     plt.plot(halo[:,0], halo[:,1], lw=4, label="0.15 dex")
#     plt.xlim(11,15)
#     plt.axvline(12, label="MW", color="black", ls="--", lw=4)
#     plt.xlabel("Host Halo Mass", fontsize=15)
#     plt.legend(fontsize=15)
#     plt.show()