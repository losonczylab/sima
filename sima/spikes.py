from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range

import time

import numpy as np
from scipy.stats import uniform
import sys
from warnings import warn


def get_poisson_spikes(seed=11111, rate=5, steps=1000, deltat=1 / 30):
    """
    Generate a poisson spike train

    Parameters
    ----------
    seed : int, optional
        Random number generator seed.
    rate : int
        Mean firing rate across the spike train (in Hz).
    steps : int
        Number of time steps in spike train.
    deltat : int
        Width of each time bin (in seconds).

    Returns
    -------
    spikes : array
        Array of length equal to steps containing binary values.

    """
    np.random.seed(seed)
    spikes = np.zeros(steps)
    spikes[[
        step for step in range(steps) if uniform.rvs() <= rate * deltat]] = 1.0
    return spikes


def nextpow2(value):
    """
    Find exponent such that 2^exponent is equal to or greater than abs(value).

    Parameters
    ----------
    value : int

    Returns
    -------
    exponent : int

    """
    exponent = 0
    avalue = np.abs(value)
    while avalue > np.power(2, exponent):
        exponent += 1
    return exponent


def axcov(data, maxlag=1):
    """
    Compute the autocovariance of data at lag = -maxlag:0:maxlag

    Parameters
    ----------
    data : array
        Array containing fluorescence data
    maxlag : int
        Number of lags to use in autocovariance calculation

    Returns
    -------
    axcov : array
        Autocovariances computed from -maxlag:0:maxlag

    """
    data = data - np.mean(data)
    bins = np.size(data)
    xcov = np.fft.fft(data, np.power(2, nextpow2(2 * bins - 1)))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag + 1)]])
    return np.real(xcov)


def spike_inference(fluor, sigma=None, gamma=None, mode="correct",
                    verbose=False):
    """
    Infer the most likely discretized spike train underlying a fluorescence
    trace.

    Parameters
    ----------
    fluor : ndarray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    sigma : float, optional
        Standard deviation of the noise distribution.  If no value is given,
        then sigma is estimated from the data.
    gamma : float, optional
        Gamma is 1 - timestep/tau, where tau is the time constant of the AR(1)
        process.  If no value is given, then gamma is estimated from the data.
    mode : {'correct', 'robust'}, optional
        The method for estimating sigma. The 'robust' method overestimates the
        noise by assuming that gamma = 1. Default: 'correct'.
    verbose : bool, optional
        Whether to print status updates. Default: False.

    Returns
    -------
    inference : ndarray of float
        The inferred normalized spike count at each time-bin.  Values are
        normalized to the maximium value over all time-bins.
    fit : ndarray of float
        The inferred denoised fluorescence signal at each time-bin.
    parameters : dict
        Dictionary with values for 'sigma', 'gamma', and 'baseline'.

    References
    ----------
    * Pnevmatikakis et al. 2015. Submitted (arXiv:1409.2903).
    * Machado et al. 2015. Submitted.
    * Vogelstein et al. 2010. Journal of Neurophysiology. 104(6): 3691-3704.

    """
    try:
        import cvxopt.umfpack as umfpack
        from cvxopt import matrix, spdiag
        import picos
    except ImportError:
        raise ImportError('Spike inference requires picos package.')

    if verbose:
        sys.stdout.write('Spike inference...')

    if sigma is None or gamma is None:
        gamma, sigma = estimate_parameters([fluor], gamma, sigma, mode)

    # Make spike generating matrix (eye, but with -g on diag below main diag)
    gen = spdiag([1 for step in range(fluor.size)])
    for step in range(fluor.size):
        if step > 0:
            gen[step, step - 1] = -gamma

    # Use spike generating matrix to initialize other constraint variables
    gen_vec = gen * matrix(np.ones(fluor.size))
    gen_ones = matrix(np.ones(fluor.size))
    umfpack.linsolve(gen, gen_ones)

    # Initialize variables in our problem
    prob = picos.Problem()
    calcium_fit = prob.add_variable('calcium_fit', fluor.size)
    init_calcium = prob.add_variable('init_calcium', 1)
    baseline = prob.add_variable('baseline', 1)

    # Define constraints and objective
    prob.add_constraint(init_calcium > 0)
    prob.add_constraint(gen * calcium_fit > 0)
    res = abs(matrix(fluor.astype(float)) - calcium_fit - baseline - gen_ones *
              init_calcium)
    prob.add_constraint(res < sigma * np.sqrt(fluor.size))
    prob.set_objective('min', calcium_fit.T * gen_vec)

    # Run solver
    start_time = time.time()
    try:
        prob.solve(solver='mosek', verbose=False)
    except ImportError:
        prob.solver_selection()
        prob.solve(verbose=False)

    if verbose:
        sys.stdout.write("done!\n" +
                         "Status: " + prob.status +
                         "; Value: " + str(prob.obj_value()) +
                         "; Time: " + str(time.time() - start_time) +
                         "; Baseline = " + str(baseline.value) + "\n")

    # Return calcium model fit and spike inference
    fit = np.squeeze(np.asarray(calcium_fit.value)[np.arange(0, fluor.size)] +
                     baseline.value)
    inference = np.squeeze(np.asarray(gen * matrix(fit)))
    parameters = {'gamma': gamma, 'sigma': sigma,
                  'baseline': baseline.value[0]}
    return inference, fit, parameters


def estimate_parameters(fluor, gamma=None, sigma=None, mode="correct"):
    """
    Use the autocovariance to estimate the scale of noise and indicator tau

    Parameters
    ----------
    fluor : list of ndarray
        One dimensional arrays containing the fluorescence intensities with
        one array entry per time-bin, and one list entry per fluorescence
        time-series for which the same parameters are to be fit.
    gamma : float, optional
        Gamma is 1 - timestep/tau, where tau is the time constant of the AR(1)
        process.  If no value is given, then gamma is estimated from the data.
    sigma : float, optional
        Standard deviation of the noise distribution.  If no value is given,
        then sigma is estimated from the data.
    mode : {'correct', 'robust'}, optional
        The method for estimating sigma. The 'robust' method overestimates the
        noise by assuming that gamma = 1. Default: 'correct'.

    Returns
    -------
    gamma : float
        Gamma is 1 - dt/tau, where tau is the time constant of the AR(1)
        process.
    sigma : float, optional
        Standard deviation of the noise distribution.

    """
    # Use autocovariance (cv) to estimate gamma:
    #     cv(t)/cv(t-1) = gamma, if t > 1
    #
    # Gamma estimates will be unreliable if the mean firing rate over the
    # provided data is low such that autocovariance does not depend on gamma!
    #
    # Note that this equation is only strictly true if spiking is Poisson
    if gamma is None:
        covars = []
        for trace in fluor:
            lags = min(50, len(trace) // 2 - 1)
            covars.append(axcov(trace, lags)[(lags+2):(lags+4)])
        covar = np.sum(covars, axis=0)
        gamma = covar[1] / covar[0]
        if gamma >= 1.0:
            warn('Warning: gamma parameter is estimated to be one!')
            gamma = 0.98

    # Use autocovariance (cv) to estimate sigma:
    #     sqrt((gamma*cv(t-1)-cv(t))/gamma) = gamma, if t == 1
    #
    # If gamma was estimated poorly, approximating the above by assuming
    # that gammma equals 1 will overestimate the noise and generate
    # more "robust" spike inference output
    if sigma is None:
        lags = 1
        covar = np.sum(
            [axcov(trace, lags) for trace in fluor],
            axis=0
        ) / sum(len(trace) for trace in fluor)
        if np.logical_not(set([mode]).issubset(["correct", "robust"])):
            mode = "correct"

        # Correct method; assumes gamma estimate is accurate
        if mode == "correct":
            sigma = np.sqrt((gamma * covar[1] - covar[0]) / gamma)

        # Robust method; assumes gamma is approximately 1
        if mode == "robust":
            sigma = np.sqrt(covar[1] - covar[0])

        # Ensure we aren't returning garbage
        # We should hit this case in the true noiseless data case
        if np.isnan(sigma) or sigma < 0:
            warn('Warning: sigma parameter is estimated to be zero!')
            sigma = 0
    return gamma, sigma
