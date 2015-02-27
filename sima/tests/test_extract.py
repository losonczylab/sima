from __future__ import division
from builtins import range
from past.utils import old_div
from builtins import object
# Unit tests for sima/extract.py
# Tests follow conventions for NumPy/SciPy available at
# https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt

# use assert_() and related functions over the built in assert to ensure tests
# run properly, beardless of how python is started.
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

import os
import shutil
import sima
import numpy as np


def setup():
    pass


def teardown():
    pass


class Test_ConstantData(object):

    def setup(self):
        data = np.ones((10, 2, 6, 8, 2))
        data[:, 0, 1:4, 2:6, 1] = 1000

        seq = sima.Sequence.create('ndarray', data)
        self.dataset = sima.ImagingDataset([seq], savedir=None)

    def test_polygon_roi(self):
        roi = sima.ROI.ROI(
            polygons=[[2, 1, 0], [6, 1, 0], [6, 4, 0], [2, 4, 0]],
            im_shape=(2, 6, 8))
        rois = sima.ROI.ROIList([roi])
        signals = self.dataset.extract(
            rois=rois, signal_channel=1, remove_overlap=True, n_processes=2,
            demix_channel=None)

        assert_equal(signals['raw'][0], 1.)

    def test_boolean_mask_roi(self):
        mask = np.array([[[False, False, False, False, False, False],
                          [False, False, True, True, True, True],
                          [False, False, True, True, True, True],
                          [False, False, True, True, True, True]]])
        roi = sima.ROI.ROI(mask=mask)
        rois = sima.ROI.ROIList([roi])
        signals = self.dataset.extract(
            rois=rois, signal_channel=1, remove_overlap=True, n_processes=2,
            demix_channel=None)

        assert_array_equal(signals['raw'], 1.)

    def test_non_boolean_mask(self):
        mask = np.array([[[0., 0., 0., 0., 0., 0.],
                          [0., 0., 0.4, 0.3, 0.2, 0.1],
                          [0., 0., 0.4, 0.3, 0.2, 0.1],
                          [0., 0., 0.4, 0.3, 0.2, 0.1]]])
        roi = sima.ROI.ROI(mask=mask)
        rois = sima.ROI.ROIList([roi])
        signals = self.dataset.extract(
            rois=rois, signal_channel=1, remove_overlap=True, n_processes=2,
            demix_channel=None)

        assert_array_equal(signals['raw'][0], 3.)

    def test_overlapping_rois(self):
        polygon = [[2, 1, 0], [5, 1, 0], [5, 3, 0], [2, 3, 0]]
        mask = np.array([[[False, False, False, False, False, False],
                          [False, False, False, False, False, False],
                          [False, False, False, True, True, True],
                          [False, False, False, True, True, True]]])
        rois = sima.ROI.ROIList(
            [sima.ROI.ROI(polygons=polygon, im_shape=(2, 6, 8)),
             sima.ROI.ROI(mask=mask)])
        no_overlap_signals = self.dataset.extract(
            rois=rois, signal_channel=1, remove_overlap=True, n_processes=1,
            demix_channel=None)
        overlap_signals = self.dataset.extract(
            rois=rois, signal_channel=1, remove_overlap=False, n_processes=2,
            demix_channel=None)

        assert_array_equal(no_overlap_signals['raw'], 1.)
        assert_array_equal(overlap_signals['raw'], 1.)


class Test_VaryingData(object):

    def setup(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
        try:
            os.mkdir(self.tmp_dir)
        except:
            pass
        pattern = np.tile([0.4, 0.8, 1.2, 1.6], (3, 1))
        data = np.ones((20, 2, 6, 8, 2))
        for t in range(20):
            data[t, 1, 1:4, 2:6, 0] = np.roll(pattern, t, 1) * 1000

        path = os.path.join(self.tmp_dir, "test_extract.sima")
        seq = sima.Sequence.create('ndarray', data)
        self.dataset = sima.ImagingDataset([seq], savedir=path)

    def teardown(self):
        shutil.rmtree(self.tmp_dir)

    def test_polygon_roi(self):
        roi = sima.ROI.ROI(
            polygons=[[2, 1, 1], [6, 1, 1], [6, 4, 1], [2, 4, 1]],
            im_shape=(2, 6, 8))
        rois = sima.ROI.ROIList([roi])
        signals = self.dataset.extract(
            rois=rois, signal_channel=0, remove_overlap=True, n_processes=2,
            demix_channel=None)

        assert_array_equal(signals['raw'][0], 1.)
        self.dataset.export_signals(
            os.path.join(self.tmp_dir, "export_signals_test.csv"))

    def test_boolean_mask_roi(self):
        mask = np.array([[[False, False, False, False, False, False],
                          [False, False, False, False, False, False],
                          [False, False, False, False, False, False],
                          [False, False, False, False, False, False]],
                         [[False, False, False, False, False, False],
                          [False, False, True, True, True, True],
                          [False, False, True, True, True, True],
                          [False, False, True, True, True, True]]])
        roi = sima.ROI.ROI(mask=mask)
        rois = sima.ROI.ROIList([roi])
        signals = self.dataset.extract(
            rois=rois, signal_channel=0, remove_overlap=True, n_processes=1,
            demix_channel=None)

        assert_array_equal(signals['raw'], 1.)

    def test_non_boolean_mask_roi(self):
        mask = np.array([[[0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.]],
                         [[0., 0., 0., 0., 0., 0.],
                          [0., 0., 0.4, 0.3, 0.2, 0.1],
                          [0., 0., 0.4, 0.3, 0.2, 0.1],
                          [0., 0., 0.4, 0.3, 0.2, 0.1]]])
        roi = sima.ROI.ROI(mask=mask)
        rois = sima.ROI.ROIList([roi])
        signals = self.dataset.extract(
            rois=rois, signal_channel=0, remove_overlap=True, n_processes=2,
            demix_channel=None)

        expected_output = np.array(
            [sum(np.roll(np.array([-0.6, -0.2, 0.2, 0.6]), t, 0)
             * np.array([0.4, 0.3, 0.2, 0.1])) * 3. + 3. for t in range(20)])

        assert_array_almost_equal(signals['raw'][0][0], expected_output)

    def test_overlapping_rois(self):
        polygon = [[2, 1, 1], [5, 1, 1], [5, 3, 1], [2, 3, 1]]
        mask = np.array([[[False, False, False, False, False, False],
                          [False, False, False, False, False, False],
                          [False, False, False, False, False, False],
                          [False, False, False, False, False, False]],
                         [[False, False, False, False, False, False],
                          [False, False, False, False, False, False],
                          [False, False, False, True, True, True],
                          [False, False, False, True, True, True]]])
        rois = sima.ROI.ROIList(
            [sima.ROI.ROI(polygons=polygon, im_shape=(2, 6, 8)),
             sima.ROI.ROI(mask=mask)])
        no_overlap_signals = self.dataset.extract(
            rois=rois, signal_channel=0, remove_overlap=True, n_processes=1,
            demix_channel=None)
        overlap_signals = self.dataset.extract(
            rois=rois, signal_channel=0, remove_overlap=False, n_processes=2,
            demix_channel=None)

        roi1_expected_no_overlap = np.array(
            [sum(np.array([-0.6, -0.2, 0.2, 0.6])
             * np.roll(np.array([0.5, 0.25, 0.25, 0.0]), -t, 0)) + 1.
             for t in range(20)])
        roi2_expected_no_overlap = np.array(
            [sum(np.array([-0.6, -0.2, 0.2, 0.6])
             * np.roll(np.array([0.0, 0.25, 0.25, 0.5]), -t, 0)) + 1.
             for t in range(20)])
        roi1_expected_overlap = np.array(
            [sum(np.array([-0.6, -0.2, 0.2, 0.6]) *
             np.roll(np.array([old_div(1, 3.), old_div(1, 3.), old_div(1, 3.), 0.0]), -t, 0)) + 1.
             for t in range(20)])
        roi2_expected_overlap = np.array(
            [sum(np.array([-0.6, -0.2, 0.2, 0.6]) *
             np.roll(np.array([0.0, old_div(1, 3.), old_div(1, 3.), old_div(1, 3.)]), -t, 0)) + 1.
             for t in range(20)])

        assert_array_almost_equal(
            no_overlap_signals['raw'][0][0], roi1_expected_no_overlap)
        assert_array_almost_equal(
            no_overlap_signals['raw'][0][1], roi2_expected_no_overlap)
        assert_array_almost_equal(
            overlap_signals['raw'][0][0], roi1_expected_overlap)
        assert_array_almost_equal(
            overlap_signals['raw'][0][1], roi2_expected_overlap)


class Test_MissingData(object):
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_empty_roi(self):
        data = np.ones((10, 2, 6, 8, 2))
        data[:, 0, 1:4, 2:6, 1] = 1000

        seq = sima.Sequence.create('ndarray', data)
        dataset = sima.ImagingDataset([seq], savedir=None)

        mask1 = np.array([[[False, False, False, False, False, False],
                          [False, False, False, False, False, False],
                          [False, False, False, False, False, False],
                          [False, False, False, False, False, False]]])
        roi1 = sima.ROI.ROI(mask=mask1)
        mask2 = np.array([[[False, False, False, False, False, False],
                          [False, False, True, True, True, True],
                          [False, False, True, True, True, True],
                          [False, False, True, True, True, True]]])
        roi2 = sima.ROI.ROI(mask=mask2)
        rois = sima.ROI.ROIList([roi1, roi2])
        signals = dataset.extract(
            rois=rois, signal_channel=1, remove_overlap=True, n_processes=1,
            demix_channel=None)

        assert_array_equal(signals['raw'][0][0, :], np.nan)
        assert_(np.all(np.isfinite(signals['raw'][0][1, :])))

    def test_missing_frame(self):
        data = np.ones((10, 2, 6, 8, 2))
        data[:, 0, 1:4, 2:6, 1] = 1000
        data[3:5, ...] = np.nan

        seq = sima.Sequence.create('ndarray', data)
        dataset = sima.ImagingDataset([seq], savedir=None)

        mask1 = np.array([[[False, False, False, False, False, False],
                          [False, True, True, False, False, False],
                          [False, True, True, False, False, False],
                          [False, False, False, False, False, False]]])
        roi1 = sima.ROI.ROI(mask=mask1)
        mask2 = np.array([[[False, False, False, False, False, False],
                          [False, False, True, True, True, True],
                          [False, False, True, True, True, True],
                          [False, False, True, True, True, True]]])
        roi2 = sima.ROI.ROI(mask=mask2)
        rois = sima.ROI.ROIList([roi1, roi2])
        signals = dataset.extract(
            rois=rois, signal_channel=1, remove_overlap=True, n_processes=1,
            demix_channel=None)

        assert_array_equal(signals['raw'][0][:, 3:5], np.nan)
        assert_(np.all(np.isfinite(signals['raw'][0][:, :3])))
        assert_(np.all(np.isfinite(signals['raw'][0][:, 5:])))

    def test_missing_frame_overlapping_rois(self):
        data = np.ones((10, 2, 6, 8, 2))
        data[:, 0, 1:4, 2:6, 1] = 1000
        data[3:5, ...] = np.nan

        seq = sima.Sequence.create('ndarray', data)
        dataset = sima.ImagingDataset([seq], savedir=None)

        mask1 = np.array([[[False, False, False, False, False, False],
                          [False, True, True, False, False, False],
                          [False, True, True, False, False, False],
                          [False, False, False, False, False, False]]])
        roi1 = sima.ROI.ROI(mask=mask1)
        mask2 = np.array([[[False, False, False, False, False, False],
                          [False, False, True, True, True, True],
                          [False, False, True, True, True, True],
                          [False, False, True, True, True, True]]])
        roi2 = sima.ROI.ROI(mask=mask2)
        rois = sima.ROI.ROIList([roi1, roi2])
        signals = dataset.extract(
            rois=rois, signal_channel=1, remove_overlap=False, n_processes=1,
            demix_channel=None)

        assert_array_equal(signals['raw'][0][:, 3:5], np.nan)
        assert_(np.all(np.isfinite(signals['raw'][0][:, :3])))
        assert_(np.all(np.isfinite(signals['raw'][0][:, 5:])))

#     @dec.knownfailureif(True)
#     def test_partial_missing_data(self):
#         raise NotImplemented

#     @dec.knownfailureif(True)
#     def test_partial_missing_data_overlapping_rois(self):
#         raise NotImplemented

if __name__ == "__main__":
    run_module_suite()
