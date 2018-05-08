"""
epnev's method is based on code by Eftychios Pnevmatikakis:
https://github.com/epnev/constrained_foopsi_python

which was published under the GPL v2:

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see http://www.gnu.org/licenses/.
"""

from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range

import time

import numpy as np
from scipy.stats import uniform
from scipy.signal import welch
from scipy.linalg import toeplitz
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


def default_psd_opts():
    """
    Return default options for psd method


    Returns
    -------
    dict : dictionary
        Default options for psd method

    """
    return {  # Default option values
        'method': 'cvx',  # solution method (no other currently supported)
        'bas_nonneg': True,  # bseline strictly non-negative
        'noise_range': (.25, .5),  # frequency range for averaging noise PSD
        'noise_method': 'logmexp',  # method of averaging noise PSD
        'lags': 5,  # number of lags for estimating time constants
        'resparse': 0,  # times to resparse original solution (not supported)
        'fudge_factor': 1,  # fudge factor for reducing time constant bias
        'verbosity': False,  # display optimization details
    }


def spike_inference(fluor, sigma=None, gamma=None, mode="correct",
                    ar_order=2, psd_opts=None, verbose=False):
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
    mode : {'correct', 'robust', 'psd'}, optional
        The method for estimating sigma. The 'robust' method overestimates
        the noise by assuming that gamma = 1. The 'psd' method estimates
        sigma from the PSD of the fluorescence data. Default: 'correct'.
    ar_order : int, optional
        Autoregressive model order. Only implemented for 'psd' method.
        Default: 2
    psd_opts : dictionary
        Dictionary of options for the psd method; if None, default options
        will be used. Default: None
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
        from cvxopt import matrix, spdiag, spmatrix, solvers
        import picos
    except ImportError:
        raise ImportError('Spike inference requires picos package.')

    if verbose:
        sys.stdout.write('Spike inference...')

    if psd_opts is None:
        opts = default_psd_opts()
    else:
        opts = psd_opts

    if sigma is None or gamma is None:
        gamma, sigma = estimate_parameters(
            [fluor], gamma, sigma, mode, ar_order, psd_opts)

    # Initialize variables in our problem
    prob = picos.Problem()

    if mode == "psd":
        T = len(fluor)
        # construct deconvolution matrix  (sp = gen*c)
        gen = spmatrix(1., range(T), range(T), (T, T))

        for i in range(ar_order):
            gen = gen + spmatrix(
                float(-gamma[i]), range(i + 1, T), range(T - i - 1), (T, T))

        gr = np.roots(np.concatenate([np.array([1]), -gamma.flatten()]))
        # decay vector for initial fluorescence
        gd_vec = np.max(gr)**np.arange(T)
        gen_vec = gen * matrix(np.ones(fluor.size))

        # Define variables
        calcium_fit = prob.add_variable('calcium_fit', fluor.size)
        baseline = prob.add_variable('baseline', 1)
        if opts['bas_nonneg']:
            b_lb = 0
        else:
            b_lb = np.min(fluor)

        prob.add_constraint(baseline >= b_lb)

        init_calcium = prob.add_variable('init_calcium', 1)
        prob.add_constraint(init_calcium >= 0)

        # Add constraints
        prob.add_constraint(gen * calcium_fit >= 0)
        res = abs(matrix(fluor.astype(float)) - calcium_fit -
                  baseline * matrix(np.ones(fluor.size)) -
                  matrix(gd_vec) * init_calcium)

    else:
        # Make spike generating matrix
        # (eye, but with -g on diag below main diag)
        gen = spdiag([1 for step in range(fluor.size)])
        for step in range(fluor.size):
            if step > 0:
                gen[step, step - 1] = -gamma

        # Use spike generating matrix to initialize other constraint variables
        gen_vec = gen * matrix(np.ones(fluor.size))
        gen_ones = matrix(np.ones(fluor.size))
        umfpack.linsolve(gen, gen_ones)

        # Initialize variables in our problem
        calcium_fit = prob.add_variable('calcium_fit', fluor.size)
        init_calcium = prob.add_variable('init_calcium', 1)
        baseline = prob.add_variable('baseline', 1)

        # Define constraints and objective
        prob.add_constraint(init_calcium > 0)
        prob.add_constraint(gen * calcium_fit > 0)
        res = abs(matrix(fluor.astype(float)) - calcium_fit -
                  baseline -
                  gen_ones * init_calcium)

    prob.add_constraint(res < sigma * np.sqrt(fluor.size))
    prob.set_objective('min', calcium_fit.T * gen_vec)

    # Run solver
    start_time = time.time()
    try:
        prob.solve(solver='mosek', verbose=False)
    except ImportError:
        warn('MOSEK is not installed. Spike inference may be VERY slow!')
        prob.solver_selection()
        prob.solve(verbose=False)

    if verbose:
        sys.stdout.write("done!\n" +
                         "Status: " + prob.status +
                         "; Value: " + str(prob.obj_value()) +
                         "; Time: " + str(time.time() - start_time) +
                         "; Baseline = " + str(baseline.value) + "\n")

    # if problem in infeasible due to low noise value then project onto the
    # cone of linear constraints with cvxopt
    if mode == "psd" and prob.status in [
            'prim_infeas_cer', 'dual_infeas_cer', 'primal infeasible']:
        warn("Original problem infeasible. "
             "Adjusting noise level and re-solving")
        # setup quadratic problem with cvxopt
        solvers.options['show_progress'] = verbose
        ind_rows = range(T)
        ind_cols = range(T)
        vals = np.ones(T)

        cnt = 2  # no of constraints (init_calcium and baseline)
        ind_rows += range(T)
        ind_cols += [T] * T
        vals = np.concatenate((vals, np.ones(T)))

        ind_rows += range(T)
        ind_cols += [T + cnt - 1] * T
        vals = np.concatenate((vals, np.squeeze(gd_vec)))

        P = spmatrix(vals, ind_rows, ind_cols, (T, T + cnt))
        H = P.T * P
        Py = P.T * matrix(fluor.astype(float))
        sol = solvers.qp(
            H, -Py, spdiag([-gen, -spmatrix(1., range(cnt), range(cnt))]),
            matrix(0., (T + cnt, 1)))
        xx = sol['x']
        fit = np.array(xx[:T])
        inference = np.array(gen * matrix(fit))
        fit = np.squeeze(fit)

        baseline = np.array(xx[T + 1]) + b_lb
        init_calcium = np.array(xx[-1])
        sigma = np.linalg.norm(
            fluor - fit - init_calcium * gd_vec - baseline) / np.sqrt(T)
        parameters = {'gamma': gamma, 'sigma': sigma,
                      'baseline': baseline}
    else:
        # Return calcium model fit and spike inference
        fit = np.squeeze(
            np.asarray(calcium_fit.value)[np.arange(0, fluor.size)] +
            baseline.value)
        inference = np.squeeze(np.asarray(gen * matrix(fit)))
        parameters = {'gamma': gamma, 'sigma': sigma,
                      'baseline': baseline.value[0]}

    return inference, fit, parameters


def estimate_sigma(fluor, range_ff=(0.25, 0.5), method='logmexp'):
    """
    Estimate noise power through the power spectral density over the range of
    large frequencies

    Parameters
    ----------
    fluor : nparray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    range_ff : 2-tuple, optional, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is
        averaged. Default: (0.25, 0.5)
    method : {'mean', 'median', 'logmexp'}, optional
        method of averaging: Mean, median, exponentiated mean of logvalues.
        Default: 'logmexp'

    Returns
    -------
    sigma       : noise standard deviation
    """

    if len(fluor) < 256:
        nperseg = len(fluor)
    else:
        nperseg = 256

    ff, Pxx = welch(fluor, nperseg=nperseg)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1, ind2)
    Pxx_ind = Pxx[ind]
    sigma = {
        'mean': lambda Pxx_ind: np.sqrt(np.mean(Pxx_ind / 2)),
        'median': lambda Pxx_ind: np.sqrt(np.median(Pxx_ind / 2)),
        'logmexp': lambda Pxx_ind: np.sqrt(
            np.exp(np.mean(np.log(Pxx_ind / 2))))
    }[method](Pxx_ind)

    return sigma


def estimate_gamma(fluor, sigma, p=2, lags=5, fudge_factor=1):
    """
    Estimate AR model parameters through the autocovariance function

    Parameters
    ----------
    fluor : nparray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    sigma : float
        noise standard deviation, estimated if None.
    p : positive integer, optional
        order of AR system. Default: 2
    lags : positive integer, optional
        number of additional lags where he autocovariance is computed.
        Default: 5
    fudge_factor : float (0< fudge_factor <= 1), optional
        shrinkage factor to reduce bias. Default: 1

    Returns
    -------
    gamma : estimated coefficients of the AR process
    """

    if sigma is None:
        sigma = estimate_sigma(fluor)

    lags += p
    xc = axcov(fluor, lags)
    xc = xc[:, np.newaxis]

    A = toeplitz(xc[lags + np.arange(lags)],
                 xc[lags + np.arange(p)]) - sigma**2 * np.eye(lags, p)
    gamma = np.linalg.lstsq(A, xc[lags + 1:])[0]
    if fudge_factor < 1:
        gr = fudge_factor * np.roots(
            np.concatenate([np.array([1]), -gamma.flatten()]))
        gr = (gr + gr.conjugate()) / 2
        gr[gr > 1] = 0.95
        gr[gr < 0] = 0.15
        gamma = np.poly(gr)
        gamma = -gamma[1:]

    return gamma.flatten()


def estimate_parameters(fluor, gamma=None, sigma=None, mode="correct",
                        ar_order=2, psd_opts=None):
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
    mode : {'correct', 'robust', 'psd'}, optional
        The method for estimating sigma. The 'robust' method overestimates
        the noise by assuming that gamma = 1. The 'psd' method estimates
        sigma from the PSD of the fluorescence data. Default: 'correct'.
    ar_order : int, optional
        Autoregressive model order. Only implemented for 'psd' method.
        Default: 2
    psd_opts : dictionary
        Dictionary of options for the psd method; if None, default options
        will be used. Default: None

    Returns
    -------
    gamma : float
        Gamma is 1 - dt/tau, where tau is the time constant of the AR(1)
        process.
    sigma : float, optional
        Standard deviation of the noise distribution.

    """

    if mode == "psd":
        if psd_opts is None:
            opts = default_psd_opts()
        else:
            opts = psd_opts

        mega_trace = np.concatenate([trace for trace in fluor])

        sigma = estimate_sigma(
            mega_trace, opts['noise_range'], opts['noise_method'])
        gamma = estimate_gamma(
            mega_trace, sigma, ar_order, opts['lags'], opts['fudge_factor'])

    else:

        # Use autocovariance (cv) to estimate gamma:
        #     cv(t)/cv(t-1) = gamma, if t > 1
        #
        # Gamma estimates will be unreliable if the mean firing rate over the
        # provided data is low such that autocovariance does not depend on
        # gamma!
        #
        # Note that this equation is only strictly true if spiking is Poisson

        if gamma is None:
            covars = []
            for trace in fluor:
                lags = min(50, len(trace) // 2 - 1)
                covars.append(axcov(trace, lags)[(lags + 2):(lags + 4)])
            covar = np.sum(covars, axis=0)
            gamma = covar[1] / covar[0]
            if gamma >= 1.0:
                warn('Warning: gamma parameter is estimated to be one!')
                gamma = 0.98

        if sigma is None:
            # Use autocovariance (cv) to estimate sigma:
            #     sqrt((gamma*cv(t-1)-cv(t))/gamma) = gamma, if t == 1
            #
            # If gamma was estimated poorly, approximating the above by
            # assuming that gammma equals 1 will overestimate the noise and
            # generate more "robust" spike inference output
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
