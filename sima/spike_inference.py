# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 10:44:55 2014

@author: tamachado
"""
import numpy as np
from scipy import signal
from scipy.stats import uniform, norm


def get_poisson_spikes(seed=11111, rate=5, steps=1000, deltat=1./30.):
    """
    Generate a poisson spike train
    """
    np.random.seed(seed)
    spikes = np.zeros(steps)
    spikes[[step for step in xrange(steps) if uniform.rvs()
            <= rate*deltat]] = 1.0
    return spikes


def nextpow2(value):
    """
    Find exponent such that 2^exponent is equal to or greater than abs(value).
    """
    exponent = 0
    avalue = np.abs(value)
    while avalue > np.power(2, exponent):
        exponent += 1
    return exponent


def axcov(data, maxlag=1):
    """
    Compute the autocovariance of data at lag = -maxlag:0:maxlag
    """
    data = data - np.mean(data)
    bins = np.size(data)
    xcov = np.fft.fft(data, np.power(2, nextpow2(2*bins-1)))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size-maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag+1)]])
    return np.real(xcov)


def spike_inference(fluor, noise=None, gamma=None, verbose=False,
                    mode="correct"):
    """
    Infer most likely discretized spike train underlying a fluorescence trace
    """
    try:
        import cvxopt.umfpack as umfpack
        from cvxopt import matrix, spdiag
        import picos
    except ImportError:
        raise ImportError('Spike inference requires picos package.')

    if verbose:
        sys.stdout.write('Spike inference...')

    if noise is None or gamma is None:
        gamma, noise = estimate_parameters(fluor, gamma, noise, mode)

    # Make spike generating matrix (eye, but with -g on diag below main diag)
    gen = spdiag([1 for step in xrange(fluor.size)])
    for step in xrange(fluor.size):
        if step > 0:
            gen[step, step-1] = -gamma

    # Use spike generating matrix to initialize other constraint variables
    gen_vec = gen*matrix(np.ones(fluor.size))
    gen_ones = matrix(np.ones(fluor.size))
    umfpack.linsolve(gen, gen_ones)

    # Initialize variables in our problem
    prob = picos.Problem()
    calcium_fit = prob.add_variable('calcium_fit', fluor.size)
    init_calcium = prob.add_variable('init_calcium', 1)
    baseline = prob.add_variable('baseline', 1)

    # Define constraints and objective
    prob.add_constraint(init_calcium > 0)
    prob.add_constraint(gen*calcium_fit > 0)
    res = abs(matrix(fluor)-calcium_fit-baseline-gen_ones*init_calcium)
    prob.add_constraint(res < noise * np.sqrt(fluor.size))
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
                         "; Value: " + str(prob.obj_value())
                         + "; Time: " + str(time.time()-start_time) +
                         "; Baseline = " + str(baseline.value) + "\n")

    # Return calcium model fit and spike inference
    fit = np.squeeze(np.asarray(calcium_fit.value)[np.arange(0, fluor.size)]
                     + baseline.value)
    inference = np.squeeze(np.asarray(gen*matrix(fit)))
    return (inference, fit)


def estimate_parameters(fluor, gamma=None, sigma=None, mode="correct"):
    """
    Use the autocovariance to estimate the scale of noise and indicator tau
    """
    # Use autocovariance (cv) to estimate gamma:
    #     cv(t)/cv(t-1) = gamma, if t > 1
    #
    # Gamma estimates will be unreliable if the mean firing rate over the
    # provided data is low such that autocovariance does not depend on gamma!
    #
    # Note that this equation is only strictly true if spiking is Poisson
    if gamma is None:
        lags = 50
        if fluor.size > lags*2+1:
            covar = axcov(fluor, lags)/fluor.size
            gamma = covar[lags+3]/covar[lags+2]
        else:
            raise Exception  # TODO: what if this condition fails?

    # Use autocovariance (cv) to estimate sigma:
    #     sqrt((gamma*cv(t-1)-cv(t))/gamma) = gamma, if t == 1
    #
    # If gamma was estimated poorly, approximating the above by assuming
    # that gammma equals 1 will overestimate the noise and generate
    # more "robust" spike inference output
    if sigma is None:
        lags = 1
        covar = axcov(fluor, lags)/fluor.size
        if np.logical_not(set([mode]).issubset(["correct", "robust"])):
            mode = "correct"

        # Correct method; assumes gamma estimate is accurate
        if mode == "correct":
            sigma = np.sqrt((gamma*covar[1]-covar[0])/gamma)

        # Robust method; assumes gamma is approximately 1
        if mode == "robust":
            sigma = np.sqrt(covar[1]-covar[0])

        # Ensure we aren't returning garbage
        # We should hit this case in the true noiseless data case
        if np.isnan(sigma):
            sys.stderr.write(
                "Warning: sigma parameter is estimated to be zero!\n")
            sigma = 0
    return (gamma, sigma)


if __name__ == '__main__':

    import time as time
    import sys as sys
    import seaborn as sns
    import matplotlib.mlab as ml
    import matplotlib.pyplot as plt

    #########
    # PART 1: Make model calcium data
    #########

    # Data parameters
    RATE = 1         # mean firing rate of poisson spike train (Hz)
    STEPS = 5000     # number of time steps in data
    TAU = 0.6        # time constant of calcium indicator (seconds)
    DELTAT = 1./30.  # time step duration (seconds)
    SIGMA = 0.1      # standard deviation of gaussian noise
    SEED = 2222      # random number generator seed
    NTRACE = 5       # number of data traces to generate

    # Make a poisson spike trains
    SPIKES = [get_poisson_spikes(deltat=DELTAT, rate=RATE,
                                 steps=STEPS, seed=SEED+i)
              for i in xrange(NTRACE)]
    SPIKES = np.asarray(SPIKES)

    # Convolve with kernel to make calcium signal
    np.random.seed(SEED)
    GAMMA = 1-(DELTAT/TAU)
    CALCIUM = signal.lfilter([1], [1, -GAMMA], SPIKES)
    TIME = np.linspace(0, STEPS*DELTAT, STEPS)

    # Make fluorescence traces with random gaussian noise and baseline
    FLUORS = [CALCIUM[i, ] + norm.rvs(scale=SIGMA, size=STEPS) + uniform.rvs()
              for i in xrange(NTRACE)]
    FLUORS = np.asarray(FLUORS)

    #########
    # PART 2:  Estimate model parameters and perform spike inference
    #########

    # Perform spike inference on all simulated fluorescence traces
    INFERENCE = np.zeros([STEPS, NTRACE])
    FITS = np.zeros([STEPS, NTRACE])

    # Jointly estimate gamma on traces concatenated together
    [joint_gamma_est, joint_sigma_est] = estimate_parameters(
        FLUORS.reshape(FLUORS.size), mode="correct")

    for x in xrange(NTRACE):

        # Estimate noise and decay parameters
        [gamma_est, sigma_est] = estimate_parameters(
            FLUORS[x, ], mode="correct", gamma=joint_gamma_est)
        print "tau = {tau},  sigma = {sigma}".format(
            tau=DELTAT/(1-gamma_est), sigma=sigma_est)

        # Run spike inference
        INFERENCE[:, x], FITS[:, x] = spike_inference(
            FLUORS[x, ], noise=sigma_est, gamma=joint_gamma_est, verbose=True)

    #########
    # PART 3: Plot results
    #########

    # Close all open figures
    plt.close("all")

    # Set up plotting style
    sns.set(context="talk", rc={"figure.figsize": [20, 6]}, style="white")
    sns.set_palette("muted", desat=.6)
    tck = [0, .5, 1]
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, facecolor='w')

    # Which cell to plot in first figure
    cp = 0

    # Plot the simulated data and model fit from the first result
    plt.axes(ax1)
    sns.tsplot(FLUORS[cp, ], ax=ax1, time=TIME)
    sns.tsplot(FITS[:, cp], ax=ax1, time=TIME, color="red")
    ax1.set_ylabel("Data and Fit")
    plt.yticks(
        np.round([FLUORS[cp].min(), FLUORS[cp].mean(), FLUORS[cp].max()], 1))

    # Plot the true spike train
    plt.axes(ax2)
    plt.bar(TIME, SPIKES[cp, ], color="DimGray", width=DELTAT)
    ax2.set_ylabel("True Spikes")
    plt.yticks(tck)
    plt.ylim(-.1, 1.1)

    # Get true positives and false positives
    spike_cutoff = 0.1
    i_times = ml.find(INFERENCE[:, cp] > spike_cutoff)  # inferred spikes
    t_times = ml.find(SPIKES[cp, :])  # true spikes
    sInds = np.intersect1d(i_times, t_times)  # indices of true positives
    wInds = np.setdiff1d(i_times, t_times)   # indices of false positives
    tp = float(sInds.size)/float(i_times.size)  # true positive rate
    fp = float(wInds.size)/(STEPS-float(t_times.size))  # false positive rate

    # Plot the spike inference
    plt.axes(ax3)
    plt.bar(
        TIME[sInds], np.ones(sInds.size),
        color="LightGrey", edgecolor="LightGrey", width=DELTAT)
    plt.bar(
        TIME[wInds], np.ones(wInds.size),
        color="Red", edgecolor="Red", width=DELTAT)
    plt.bar(
        TIME, INFERENCE[:, 0]/INFERENCE[:, 0].max(),
        color="DimGray", edgecolor="DimGray", width=DELTAT)
    ax3.set_xlabel("Time (Seconds)")
    ax3.set_ylabel("Spike Inference")
    sns.despine(bottom=True, left=True)
    plt.yticks(tck)
    plt.ylim(-.1, 1.1)
    plt.title(
        "TP rate = " + str(round(tp, 2)) + "; FP rate = " + str(round(fp, 2)))

    # Plot all traces and inference
    plt.figure(5, facecolor='w')
    plt.subplot(211)
    plt.imshow(FLUORS, aspect="auto", interpolation="none")
    plt.colorbar()
    plt.subplot(212)
    plt.imshow(INFERENCE.transpose(), aspect="auto", interpolation="none")
    plt.colorbar()

    plt.show()
