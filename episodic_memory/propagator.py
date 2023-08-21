import torch
import numpy as np

from scipy.optimize import minimize
from functools import partial

from autocorrelation import acf_sum, constraints_stochmat


class Propagator(object):
    def __init__(self, generator, sigma=1.0, tau=1.0):
        self.generator = generator
        self.n_states = self.generator.n_states
        # Check sigma is grater than 0 and convert to float
        if sigma <= 0:
            raise ValueError("Sigma must be greater than 0")
        self.sigma = float(sigma)

        # Check sigma is grater than 0 and convert to float
        if tau <= 0:
            raise ValueError("Tau must be greater than 0")
        self.tau = float(tau)

        self.compute_kernels()

    def compute_kernels(self, power_spec=None):
        """
        Computes the propagator kernels.
        """
        if power_spec is None:
            self.S = torch.diag(torch.exp(self.generator.V / self.tau))  # power spectrum matrix
        else:
            self.S = torch.diag(power_spec)

        self.P = self.generator.G @ self.S @ self.generator.W

        # Suppress imaginary components using absolute value
        self.P = torch.real(self.P)

    def min_auto_cf(self, T=2, lags=(1, 2), rho_init=None, maxiter=1000):
        """
        FUNCTION: sets spectrum to minimize autocorrelation at lags summed over times
        INPUTS: T           = maximum sampled timesteps to consider ACF
                lags        = list of time offsets, ACF sum over lags is minimized
                rho_init    = ACF initialized from this distribution, 'start', 'stationary', int (state), array
                maxiter     = maximum # iterations during optimization
        """
        # FIXME inequality constraints violated using scipy.minimize
        # https://scikit-optimize.github.io - forest_minimize broken
        # TODO try hyperopt, pyomo, gurobi?

        if type(rho_init) is int:
            rho0 = torch.zeros(self.n_states)
            rho0[rho_init] = 1.0
        elif rho_init.size == self.n_states:
            rho0 = rho_init
        else:
            raise ValueError(
                "unknown setting for initial distribution in acf calculation"
            )

        x0 = np.diag(self.S)  # initialize at currently set spectrum
        W = np.array(self.generator.spectral_matrix())
        fun = partial(acf_sum, W=W, T=T, deltaT=lags, rho=rho0)
        options = {"maxiter": maxiter}

        lc1_stochmat, lc2_stochmat = constraints_stochmat(W)

        opt = minimize(
            fun,
            x0,
            method=None,
            constraints=[lc1_stochmat, lc2_stochmat],
            tol=None,
            callback=None,
            options=options,
        )

        s_opt = torch.Tensor(opt.x)

        self.compute_kernels(power_spec=s_opt)



