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

import sima
from sima.motion import hmm
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


# def test_estimate_movement_model():
#     shifts = [np.array([[[0, 0]],
#                         [[0, 7]],
#                         [[-1, 5]],
#                         [[-1, 6]],
#                         [[-1, 6]],
#                         [[-1, 5]]])]
#     expected = (np.array([[0.02, 0.075], [0.075, 1.25]]),
#                 np.array([[0.83364924, 0.00700411],
#                           [0.00700411, 0.87939634]]),
#                 np.array([[-0.1555321, -0.52939923],
#                           [-4.44306493, -4.02117885]]),
#                 np.array([-0.66666667,  4.83333333]))
#
#     for x, y in zip(
#             hmm._estimate_movement_model(shifts, 10), expected):
#         assert_array_almost_equal(x, y)
#
#     expected = (np.diag([0.01, 0.01]),
#                 np.zeros((2, 2)),
#                 np.array([[0.9880746, -4.20179324],
#                           [-4.20179324, -9.39166108]]),
#                 np.array([0., 0.]))
#     for x, y in zip(
#             hmm._estimate_movement_model([np.zeros((10, 1, 2))], 10),
#             expected):
#         assert_array_almost_equal(x, y)


def test_threshold_gradient():
    test = [np.arange(4) + 4 * i for i in range(4)]
    res = np.zeros((4, 4), dtype=bool)
    res[3, 3] = True
    assert_equal(hmm._threshold_gradient(np.array([test]))[0], res)


def test_initial_distribution():
    initial_dist = hmm._initial_distribution(
        np.diag([0.9, 0.9]),
        10 * np.ones((2, 2)), np.array([-1, 1]))
    assert_almost_equal(initial_dist(0), 0.00754154839)


def test_lookup_tables():
    min_displacements = np.array([0, -1, -1])
    max_displacements = np.array([0, 1, 1])
    log_markov_matrix = np.ones((1, 2, 2))

    position_tbl, transition_tbl, log_markov_tbl = hmm._lookup_tables(
        [min_displacements, max_displacements + 1], log_markov_matrix)

    pos_tbl = [[0, int(i / 3) - 1, i % 3 - 1] for i in range(9)]
    assert_array_equal(position_tbl, pos_tbl)
    assert_(all(transition_tbl[range(9), range(8, -1, -1)] == 4))
    assert_(all(log_markov_tbl == 1))


def test_backtrace():
    states = [i * 10 + np.arange(5) for i in range(3)]
    position_tbl = np.array([[i % 5 - 2, int(i / 5) - 2] for i in range(25)])
    backpointer = [np.arange(5) for i in range(2)]

    traj = hmm._backtrace(2, backpointer, states, position_tbl)
    assert_array_equal(traj, [[0, -2], [0, 0], [0, 2]])


class Test_HiddenMarkov2D(object):
    # Tests related to the MCImagingDataset class are grouped together in a
    # class. Test classes can have their own setup/teardown methods

    def setup(self):
        for frame in Sequence.create('HDF5', example_hdf5(), 'yxt'):
            break
        frame_shifts = [np.array([[[0, 0]], [[-5, -10]]])]
        self.frame_shifts = [np.array([[[5, 10]], [[0, 0]]])]
        self.correlations = [np.array([[1], [0.9301478]])]

        shifted = frame.copy()
        shifted = np.roll(shifted, -frame_shifts[0][1, 0, 1], axis=2)
        shifted = np.roll(shifted, -frame_shifts[0][1, 0, 0], axis=1)
        frames = np.array([frame, shifted])

        self.hm2d = hmm.HiddenMarkov2D(n_processes=1, verbose=False)
        self.dataset = sima.ImagingDataset(
            [Sequence.create('ndarray', frames)], None)

    def test_pixel_distribution(self):
        assert_almost_equal(
            hmm._pixel_distribution(self.dataset),
            ([1110.20196533], [946000.05906352]))

    def test_correlation_based_correction(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            shifts = sima.motion.frame_align.PlaneTranslation2D(
                n_processes=1).estimate(self.dataset)

        for shift, shift_ in zip(shifts, self.frame_shifts):
            assert_array_equal(shift, shift_)

    def test_whole_frame_shifting(self):
        reference, variances = hmm._whole_frame_shifting(
            self.dataset, self.frame_shifts)
        ref_shape = np.array(self.dataset.frame_shape)
        ref_shape[1:3] += self.frame_shifts[0][0, 0]
        assert_array_equal(reference.shape, ref_shape)
        assert_equal(len(np.where(variances > 0)[0]), 0)

    @dec.knownfailureif(True)  # TODO: fix displacements.pkl so this passes
    def test_hmm(self):
        global tmp_dir

        frames = Sequence.create('TIFF', example_tiff())
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            corrected = self.hm2d.correct(
                [frames], os.path.join(tmp_dir, 'test_hmm.sima'))

        with open(misc.example_data() + '/displacements.pkl', 'rb') as fh:
            displacements = [d.reshape((20, 1, 128, 2))
                             for d in pickle.load(fh)]

        displacements_ = [seq.displacements for seq in corrected]
        assert_almost_equal(displacements_, displacements)

    def test_hmm_tmp(self):  # TODO: remove when displacements.pkl is updated
        global tmp_dir
        frames = Sequence.create('TIFF', example_tiff())
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            corrected = self.hm2d.correct(
                [frames], os.path.join(tmp_dir, 'test_hmm_2.sima'))
        with open(misc.example_data() + '/displacements.pkl', 'rb') as fh:
            displacements = [d.reshape((20, 1, 128, 2))
                             for d in pickle.load(fh)]
        displacements_ = [seq.displacements for seq in corrected]
        diffs = displacements_[0] - displacements[0]
        assert_(
            ((diffs - diffs.mean(axis=2).mean(axis=1).mean(axis=0)) > 1).mean()
            <= 0.001)

    def test_hmm_missing_frame(self):
        global tmp_dir
        frames = Sequence.create('TIFF', example_tiff())
        masked_seq = frames.mask([(5, None, None)])
        corrected = self.hm2d.correct(
            [masked_seq], os.path.join(tmp_dir, 'test_hmm_3.sima'))
        assert_(all(np.all(np.isfinite(seq.displacements))
                    for seq in corrected))
        assert_(np.prod(corrected.frame_shape) > 0)

    def test_hmm_missing_row(self):
        global tmp_dir
        frames = Sequence.create('TIFF', example_tiff())
        mask = np.zeros(frames.shape[1:-1], dtype=bool)
        mask[:, 20, :] = True
        masked_seq = frames.mask([(None, mask, None)])
        corrected = self.hm2d.correct(
            [masked_seq],
            os.path.join(tmp_dir, 'test_hmm_missing_row.sima'))
        assert_(
            all(np.all(np.isfinite(seq.displacements)) for seq in corrected))

    def test_hmm_missing_column(self):
        global tmp_dir
        frames = Sequence.create('TIFF', example_tiff())
        mask = np.zeros(frames.shape[1:-1], dtype=bool)
        mask[:, :, 30] = True
        masked_seq = frames.mask([(None, mask, None)])
        corrected = self.hm2d.correct(
            [masked_seq],
            os.path.join(tmp_dir, 'test_hmm_missing_column.sima'))
        assert_(all(np.all(np.isfinite(seq.displacements))
                    for seq in corrected))


if __name__ == "__main__":
    run_module_suite()
