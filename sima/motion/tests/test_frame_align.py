# Unit tests for sima/motion/_hmm.py
# Tests follow conventions for NumPy/SciPy avialble at
# https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt

# use assert_() and related functions over the built in assert to ensure tests
# run properly, regaurdless of how python is started.
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

import sima.motion.frame_align
from sima import misc
from sima import Sequence
from sima.misc import example_hdf5, example_tiff

import warnings
import cPickle as pickle

from scipy.weave import build_tools
import os
import tempfile
import numpy as np
from numpy.linalg import det


def setup():
    # setup is run prior to executing any of the included tests

    # setup ensures that a temporary directory is established either locally
    # or in the most secure location possible.

    global tmp_dir
    tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')

    try:
        os.mkdir(tmp_dir)
    except:
        pass

    tmp_dir = build_tools.configure_temp_dir(tmp_dir)


def test_shifted_corr():
    reference = np.random.RandomState(seed=0).normal(size=(10, 20, 30, 3))
    shifts = np.array([2, -4, 7])
    shifted = reference
    for i, s in enumerate(shifts):
        shifted = np.roll(shifted, -s, i)
    assert_almost_equal(
        sima.motion.frame_align.shifted_corr(reference, shifted, shifts), 1.)

def teardown():
    # teardown is executed after all of the tests in this file have comlpeted

    # remove any files remaining in the temporyary directory and delete the
    # folder

    global tmp_dir

    if os.path.exists(tmp_dir):
        for root, dirs, files in os.walk(tmp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    os.rmdir(tmp_dir)


def test_base_alignment():
    reference = np.random.RandomState(seed=0).normal(size=(10, 20, 30, 3))
    shifts = np.array([2, -4, 7])
    shifted = reference
    for i, s in enumerate(shifts):
        shifted = np.roll(shifted, -s, i)
    estimated_shifts = sima.motion.frame_align.base_alignment(
        reference, shifted)
    assert_array_equal(shifts, estimated_shifts)


def test_pyramid_align():
    # test volume
    reference = np.random.RandomState(seed=0).normal(size=(30, 128, 256, 3))
    shifts = np.array([3, 15, -20])
    shifted = reference
    for i, s in enumerate(shifts):
        shifted = np.roll(shifted, -s, i)
    estimated_shifts = sima.motion.frame_align.pyramid_align(
        reference, shifted)
    assert_array_equal(shifts, estimated_shifts)

    # test plane
    reference = np.random.RandomState(seed=0).normal(size=(1, 128, 256, 3))
    shifts = np.array([0, -10, 23])
    shifted = reference
    for i, s in enumerate(shifts):
        shifted = np.roll(shifted, -s, i)
    estimated_shifts = sima.motion.frame_align.pyramid_align(
        reference, shifted)
    assert_array_equal(shifts, estimated_shifts)


if __name__ == "__main__":
    run_module_suite()
