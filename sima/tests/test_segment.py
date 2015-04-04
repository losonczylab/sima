from builtins import next
from builtins import object
# Unit tests for sima/sequence.py
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

import numpy as np

import os
import shutil

import sima
import sima.segment
import sima.ROI
from sima.misc import example_tiffs, example_tiff


tmp_dir = None


def setup():
    global tmp_dir

    tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')

    try:
        os.mkdir(tmp_dir)
    except:
        pass


def teardown():
    global tmp_dir

    shutil.rmtree(tmp_dir)

def _gaussian_2d(xy, xy0, xysig):
    X, Y = np.meshgrid(xy[0], xy[1])
    return np.exp(-0.5*(((X-xy0[0]) / xysig[0])**2 + ((Y-xy0[1]) / xysig[1])**2))


class TestSegment(object):

    def setup(self):
        global tmp_dir

        self.filepath = os.path.join(tmp_dir, "test_imaging_dataset.sima")
        self.tiff_ds = sima.ImagingDataset(
            [sima.Sequence.create('TIFF', example_tiff(), 1, 1)],
            self.filepath)

    def teardown(self):
        shutil.rmtree(self.filepath)

    def test_postprocess(self):
        roi_xy = [np.arange(self.tiff_ds.sequences[0].shape[2]),
                  np.arange(self.tiff_ds.sequences[0].shape[3])]

        rois = sima.ROI.ROIList([
            sima.ROI.ROI(_gaussian_2d(roi_xy, (10,10), (10,10))),
            sima.ROI.ROI(_gaussian_2d(roi_xy, (40,40), (10,10))),
            sima.ROI.ROI(_gaussian_2d(roi_xy, (70,70), (10,10))),
            sima.ROI.ROI(_gaussian_2d(roi_xy, (100,100), (10,10)))])

        tobool = sima.segment.SparseROIsFromMasks(n_processes=2)
        smooth = sima.segment.SmoothROIBoundaries(n_processes=2)
        rois = smooth.apply(tobool.apply(rois))

        refrois = [
            [np.array([[  1.,   1.,   0.],
                       [ 28.,   1.,   0.],
                       [ 30.,   4.,   0.],
                       [ 30.,  28.,   0.],
                       [ 26.,  30.,   0.],
                       [  2.,  30.,   0.],
                       [  1.,  27.,   0.],
                       [  1.,   1.,   0.]])],
            [np.array([[ 30.,  29.,   0.],
                       [ 57.,  29.,   0.],
                       [ 60.,  30.,   0.],
                       [ 60.,  57.,   0.],
                       [ 59.,  60.,   0.],
                       [ 31.,  60.,   0.],
                       [ 29.,  57.,   0.],
                       [ 29.,  36.,   0.],
                       [ 30.,  29.,   0.]])],
            [np.array([[ 60.,  59.,   0.],
                       [ 87.,  59.,   0.],
                       [ 90.,  60.,   0.],
                       [ 90.,  87.,   0.],
                       [ 89.,  90.,   0.],
                       [ 61.,  90.,   0.],
                       [ 59.,  87.,   0.],
                       [ 59.,  66.,   0.],
                       [ 60.,  59.,   0.]])],
            [np.array([[  90.,   89.,    0.],
                       [ 117.,   89.,    0.],
                       [ 120.,   90.,    0.],
                       [ 120.,  117.,    0.],
                       [ 119.,  120.,    0.],
                       [  91.,  120.,    0.],
                       [  89.,  117.,    0.],
                       [  89.,   96.,    0.],
                       [  90.,   89.,    0.]])]]

        for refroi, roi in zip(refrois, rois):
            assert_array_equal(refroi[0], roi.coords[0])


if __name__ == "__main__":
    run_module_suite()
