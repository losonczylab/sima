from __future__ import division
from builtins import object
# Unit tests for sima/spikes.py
# Tests follow conventions for NumPy/SciPy available at
# https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt

# use assert_() and related functions over the built in assert to ensure tests
# run properly, beardless of how python is started.
from scipy import signal
from scipy.stats import uniform, norm

from numpy.testing import (
    assert_,
    assert_equal,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_raises,
    assert_array_equal,
    dec,
    TestCase,
    run_module_suite,
    assert_allclose)

import sima
import sima.spikes
import numpy as np

_has_picos = True
try:
    import picos
except ImportError:
    _has_picos = False

def setup():
    pass


def teardown():
    pass


class Test_Spike_Inference(object):

    def setup(self):

        #########
        # PART 1: Make model calcium data
        #########

        # Data parameters
        RATE = 1       # mean firing rate of poisson spike train (Hz)
        STEPS = 100   # number of time steps in data
        STEPS_LONG = 5000   # number of time steps in data
        TAU = 0.6      # time constant of calcium indicator (seconds)
        DELTAT = 1/30  # time step duration (seconds)
        self.sigma = 0.1    # standard deviation of gaussian noise
        SEED = 2222    # random number generator seed

        # Make a poisson spike trains
        self.spikes = sima.spikes.get_poisson_spikes(
            deltat=DELTAT, rate=RATE, steps=STEPS, seed=SEED)

        # longer time-series for parameter estimation
        self.spikes_long = sima.spikes.get_poisson_spikes(
            deltat=DELTAT, rate=RATE, steps=STEPS_LONG, seed=SEED)

        # Convolve with kernel to make calcium signal
        np.random.seed(SEED)
        self.gamma = 1 - (DELTAT / TAU)
        CALCIUM = signal.lfilter([1], [1, -self.gamma], self.spikes)
        CALCIUM_LONG = signal.lfilter([1], [1, -self.gamma], self.spikes_long)

        # Make fluorescence traces with random gaussian noise and baseline
        self.fluors = CALCIUM + norm.rvs(
            scale=self.sigma, size=STEPS) + uniform.rvs()
        self.fluors_long = CALCIUM_LONG + norm.rvs(
            scale=self.sigma, size=STEPS_LONG) + uniform.rvs()

    def teardown(self):
        pass

    @dec.skipif(not _has_picos)
    def test_estimate_parameters(self):
        gamma_est, sigma_est = sima.spikes.estimate_parameters(
            [self.fluors_long], mode="correct")
        assert_(abs(gamma_est - self.gamma) < 0.01)
        assert_(abs(sigma_est - self.sigma) < 0.01)

    # @dec.skipif(not _has_picos)
    @dec.knownfailureif(True)  # fails w/o mosek
    def test_spike_inference(self):
        inference, fits, params = sima.spikes.spike_inference(
            self.fluors, mode='correct')
        assert_(np.linalg.norm(inference - self.spikes) < 1.0)

    # @dec.skipif(not _has_picos)
    @dec.knownfailureif(True)  # fails w/o mosek
    def test_spike_inference_psd(self):
        inference, fits, params = sima.spikes.spike_inference(
            self.fluors, mode='psd')
        assert_(np.linalg.norm(inference - self.spikes) < 1.0)

    # @dec.skipif(not _has_picos)
    @dec.knownfailureif(True)  # fails w/o mosek
    def test_spike_inference_correct_parameters(self):

        # Run spike inference
        inference, fits, params = sima.spikes.spike_inference(
            self.fluors, sigma=self.sigma, gamma=self.gamma)
        assert_(np.linalg.norm(inference - self.spikes) < 0.51)
        assert_equal(params['gamma'], self.gamma)
        assert_equal(params['sigma'], self.sigma)


if __name__ == "__main__":
    run_module_suite()
