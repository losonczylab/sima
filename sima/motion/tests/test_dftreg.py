from future import standard_library
standard_library.install_aliases()
# Unit tests for sima/motion/dftreg.py
# Tests follow conventions for NumPy/SciPy avialble at
# https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt

# use assert_() and related functions over the built in assert to ensure tests
# run properly, regardless of how python is started.
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

import sima.motion.dftreg as dftreg

import numpy as np

def test_shifted_corr():
    reference = np.random.RandomState(seed=0).normal(
        size=(128, 128)) * np.ones((10, 128, 128))
    shifts = np.random.RandomState(seed=0).randint(
        -32, 33, size=(reference.shape[0]-1, 2))
    shifted = reference
    for i, s in enumerate(shifts):
        for axis in [0,1]:
            shifted[i+1] = np.roll(shifted[i+1], -s[axis], axis)

    reg = dftreg.register(shifted)
    reg_shift = np.array(zip(reg[1], reg[0]))

    assert_equal(reg_shift[1:]-reg_shift[0]-shifts, 0)

if __name__ == "__main__":
    run_module_suite()
