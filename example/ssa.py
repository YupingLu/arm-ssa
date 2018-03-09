import numpy as np

def SSA(Y,L,period_groups):
    T = Y.size
    assert L <= T/2
    K = T - L + 1

    # Form the trajectory matrix and find the eigen decomp
    X = np.zeros((L,K))
    for i in range(K): X[:,i] = Y[i:(i+L)]
    lamda,P = np.linalg.eig(np.dot(X,X.T))

    # Find the dominant frequency of each eigenvector
    f   = np.zeros(lamda.size)
    fs  = np.fft.fftfreq(f.size,1.)
    ix  = np.argsort(fs)
    fs  = fs[ix]
    eps = 0.99*(fs[1]-fs[0])
    for i in range(f.size):
        ps = np.abs(np.fft.fft(P[:,i]))**2
        ps = ps[ix]
        f[i] = fs[ps.argmax()]
    f = np.abs(f)

    # convert periodicity into frequency
    fgroups = 1/np.asarray(period_groups,dtype=float)
    fgroups = np.hstack([0,fgroups])
    
    # Build an approximation of X by taking a subset of the
    # decomposition. This approximation is formed by taking
    # eigenvectors whose dominant frequency is close to the targetted
    # values.
    Xt = np.zeros((fgroups.size,)+X.shape)
    for i in range(f.size):
        g = np.where(np.abs(fgroups-f[i]) < eps)[0]
        if g.size == 0: continue
        Xt[g[0]] += np.dot(np.outer(P[:,i],P[:,i]),X)

    # Now we reconstruct the signal by taking a mean of all the
    # approximations.
    Yt = np.zeros((fgroups.size,Y.size))
    c  = np.zeros((fgroups.size,Y.size))
    for g in range(fgroups.size):
        for i in range(K): 
            Yt[g,i:(i+L)] += Xt[g,:,i]
            c [g,i:(i+L)] += 1
    Yt /= c
    
    return Yt

def GetResidual(Y,window=120,groups=[12,6,4,3,2]):
    """Extracts anomalies from a signal.

    Parameters
    ----------
    Y : numpy.ndarray
        the one-dimensional, non-masked array of data
    window : int
        the window size, the maximum periodicity you wish to search
        for in the input signal
    groups: array-like of ints
        a list of data periodicities to extract from the input signal

    Returns
    -------
    R : numpy.ndarray
        the signal minus the trend and the group periodicities
        specified

    """
    Yt = SSA(Y,window,groups)
    R  = Y-Yt.sum(axis=0)
    return R
    
if __name__ == "__main__":
    from netCDF4 import Dataset
    import pylab as plt

    dset = Dataset("/Users/yupinglu/OneDrive/project/ARM/code/gpp.nc")
    i    = np.abs(dset.variables["lat"][...]+  5.).argmin()
    j    = np.abs(dset.variables["lon"][...]-300.).argmin()
    t    = dset.variables["time"][...  ]/365.+1850.
    gpp  = dset.variables["gpp" ][:,i,j]*86400000.0
    groups = [60,12]
    decomp = SSA(gpp,120,groups)
    
    fig,axs = plt.subplots(nrows=len(groups)+2,tight_layout=True)
    axs[0].plot(t,gpp,'-')
    axs[0].set_ylim(gpp.min(),gpp.max())    
    for g in range(len(groups)+1):
        axs[g].plot(t,decomp[g],'-')
    axs[-1].plot(t,gpp-decomp.sum(axis=0),'-')
    for i in range(len(groups)+2):
        axs[i].set_ylabel("g m-2 d-1")
    plt.show()
