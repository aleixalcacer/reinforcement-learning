import torch
from scipy.optimize import LinearConstraint
import numpy as np
from numpy.matlib import repmat
from scipy.stats import sem
import statsmodels.api as sm


def acf_gen(s, W, T, deltaT, rho):
    """ FUNCTION: computes generated autocorrelation
    INPUTS: s       = spectrum
            W       = spectral components (n_k,n_s,n_s)
            T       = max time to integrate over
            deltaT  = lags to compute
            rho     = state probability vector
    """
    deltaT = np.asarray(deltaT)
    n_s = len(rho)
    n_t = len(deltaT)
    Wd = np.array([W[:, i, i] for i in range(n_s)]).T  # (n_s,n_k)
    ACgen = np.zeros((T+1, n_t))
    for t in range(T+1):
        rho_t = np.einsum('i, kij, k', rho, W, s**t)
        rho_t = rho_t/rho_t.sum()
        ACgen_t = repmat(s.reshape(-1, 1), 1, n_t) # (n_k,n_t) convention
        ACgen_t = ACgen_t**deltaT
        ACgen_t = Wd@ACgen_t # (n_s,n_k) x (n_k,n_t) -> (n_s,n_t)
        ACgen_t = rho_t@ACgen_t # (,n_s) x (n_s,n_t) -> (n_t,)
        ACgen[t,:] = ACgen_t
    return ACgen

def acf_sum(s, W, T, deltaT, rho, sumT=True, sum_deltaT=True):
    """ sums over acf components """
    ACgen = acf_gen(s, W, T, deltaT, rho)
    if sumT:
        if sum_deltaT:
            return ACgen.sum()
        else:
            return ACgen.sum(0)
    else:
        if sum_deltaT:
            return ACgen.sum(1)
        else:
            return ACgen



def constraints_stochmat(W):
    """
    FUNCTION: linear constraints which ensure resulting evolution matrix is a stochastic matrix
    INPUTS: W = (n_k, n_s, n_s) spectral weights
    """

    tol = 1e-6
    n_k = W.shape[0]
    # sum_{j,k} W[k,i,j]*s_k = 1
    lc1_stochmat = LinearConstraint(A=W.T.sum(0), lb=1-tol, ub=1+tol)
    # sum_{k} W[k,i,j]*s_k >= 0
    lc2_stochmat = LinearConstraint(A=W.T.reshape((-1, n_k)), lb=-tol, ub=np.inf)
    return lc1_stochmat, lc2_stochmat

def estimate_acf(env, data, d=0):
    """
    FUNCTION: estimate episodic (temporal, semantic, spatial) autocorrelation
    INPUTS: data = (n_t, n_samp, n_vars) matrix of samples
    """

    """
        FUNCTION: estimate episodic (time space, semantic space) autocorrelation
        INPUTS: data = (n_t, n_samp, n_vars) matrix of samples
        """

    # data = env.states_transformed.values[data]
    print(data.shape)

    n_samp = data.shape[0]
    n_t = data.shape[1]
    n_vars = data.shape[2]

    AC_samp = np.zeros((n_t, n_samp, n_vars))
    axis = [0]

    for var in axis:
        for samp in range(n_samp):
            acor = sm.tsa.acf(data[samp, :, var], nlags=n_t - 1, fft=False)
            AC_samp[:, samp, var] = np.abs(acor)

    # AC_samp = AC_samp[:, :, axis].reshape(n_t, n_samp, -1).mean(axis=2)
    AC_samp = AC_samp[:, :, axis].reshape(n_t, n_samp, -1)
    AC = AC_samp.mean(axis=1)

    if n_samp > 1:
        AC_sem = sem(AC_samp, axis=1)
    else:
        AC_sem = np.zeros(AC.shape)

    return AC, AC_sem