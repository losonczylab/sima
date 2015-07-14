# Unit tests for sima/motion/dftreg.py
# Tests follow conventions for NumPy/SciPy avialble at
# https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt
# use assert_() and related functions over the built in assert to ensure tests
# run properly, regardless of how python is started.

from future import standard_library
from numpy.testing import assert_array_equal, run_module_suite
import sima.motion.dftreg as dftreg
import numpy as np
standard_library.install_aliases()


def test_dftreg_register():
    # make test image
    reference = np.random.RandomState(seed=0).normal(
        size=(128, 128)) * np.ones((10, 128, 128))

    # generate random shifts
    shifts = np.random.RandomState(seed=0).randint(
        -32, 33, size=(reference.shape[0]-1, 2))

    # apply shifts to frames (but not first frame)
    shifted = reference
    for i, s in enumerate(shifts):
        for axis in [0, 1]:
            shifted[i+1] = np.roll(shifted[i+1], -s[axis], axis)

    # estimate shifts using the first frame as reference
    estimated_shifts = dftreg._register(shifted, num_images_for_mean=1,
                                        randomise_frames=False)

    # reshape estimated shifts, removing first frame results (will be 0,0)
    # so that size test does not fail
    estimated_shifts = np.array(
        list(zip(
            estimated_shifts[0][1:],
            estimated_shifts[1][1:])))

    # test
    assert_array_equal(shifts, estimated_shifts)

if __name__ == "__main__":
    run_module_suite()
