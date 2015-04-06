import sys
from distutils.version import LooseVersion

import numpy as np
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

from sima import ImagingDataset, Sequence, ROI
from sima.misc import example_data, example_tiff
from sima import segment
from sima.segment.ca1pc import cv2_available


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


def test_extract_rois():
    return


@dec.knownfailureif(
    sys.version_info > (3, 0) and
    LooseVersion(np.__version__) < LooseVersion('1.9.0'))
def test_STICA():
    ds = ImagingDataset.load(example_data())
    method = segment.STICA(components=5)
    method.append(segment.SparseROIsFromMasks(min_size=50))
    method.append(segment.SmoothROIBoundaries(radius=3))
    method.append(segment.MergeOverlapping(0.5))
    ds.segment(method)


@dec.skipif(not cv2_available)
def test_PlaneNormalizedCuts():
    ds = ImagingDataset.load(example_data())[:, :, :, :50, :50]
    affinty_method = segment.BasicAffinityMatrix(num_pcs=5)
    method = segment.PlaneWiseSegmentation(
        segment.PlaneNormalizedCuts(affinty_method))
    ds.segment(method)


@dec.skipif(not cv2_available)
def test_PlaneCA1PC():
    ds = ImagingDataset.load(example_data())[:, :, :, :50, :50]
    method = segment.PlaneCA1PC(num_pcs=5)
    ds.segment(method)


class TestPostprocess(object):

    def setup(self):
        global tmp_dir

        self.filepath = os.path.join(tmp_dir, "test_imaging_dataset.sima")
        self.tiff_ds = ImagingDataset(
            [Sequence.create('TIFF', example_tiff(), 1, 1)],
            self.filepath)

    def teardown(self):
        shutil.rmtree(self.filepath)

    def test_postprocess(self):
        roi_xy = [np.arange(self.tiff_ds.sequences[0].shape[2]),
                  np.arange(self.tiff_ds.sequences[0].shape[3])]

        rois = ROI.ROIList([
            ROI.ROI(_gaussian_2d(roi_xy, (10,10), (10,10))),
            ROI.ROI(_gaussian_2d(roi_xy, (40,40), (10,10))),
            ROI.ROI(_gaussian_2d(roi_xy, (70,70), (10,10))),
            ROI.ROI(_gaussian_2d(roi_xy, (100,100), (10,10)))])

        tobool = segment.SparseROIsFromMasks(n_processes=2)
        smooth = segment.SmoothROIBoundaries(n_processes=2)
        rois = smooth.apply(tobool.apply(rois))

        if sys.version_info > (3, 0):
            refrois = [
                [np.array([[  1.,   1.,   0.],
                           [ 25.,   1.,   0.],
                           [ 27.,   4.,   0.],
                           [ 28.,   7.,   0.],
                           [ 28.,  13.,   0.],
                           [ 26.,  19.,   0.],
                           [ 24.,  22.,   0.],
                           [ 12.,  28.,   0.],
                           [  8.,  28.,   0.],
                           [  4.,  27.,   0.],
                           [  1.,  24.,   0.],
                           [  1.,   1.,   0.]])],
                [np.array([[ 35.,  23.,   0.],
                           [ 44.,  23.,   0.],
                           [ 47.,  24.,   0.],
                           [ 53.,  28.,   0.],
                           [ 55.,  31.,   0.],
                           [ 57.,  37.,   0.],
                           [ 57.,  43.,   0.],
                           [ 55.,  49.,   0.],
                           [ 53.,  52.,   0.],
                           [ 45.,  56.,   0.],
                           [ 41.,  57.,   0.],
                           [ 37.,  57.,   0.],
                           [ 33.,  56.,   0.],
                           [ 29.,  54.,   0.],
                           [ 26.,  51.,   0.],
                           [ 24.,  48.,   0.],
                           [ 23.,  45.,   0.],
                           [ 23.,  36.,   0.],
                           [ 24.,  33.,   0.],
                           [ 28.,  27.,   0.],
                           [ 35.,  23.,   0.]])],
                [np.array([[ 65.,  53.,   0.],
                           [ 74.,  53.,   0.],
                           [ 77.,  54.,   0.],
                           [ 83.,  58.,   0.],
                           [ 85.,  61.,   0.],
                           [ 87.,  67.,   0.],
                           [ 87.,  73.,   0.],
                           [ 85.,  79.,   0.],
                           [ 83.,  82.,   0.],
                           [ 75.,  86.,   0.],
                           [ 71.,  87.,   0.],
                           [ 67.,  87.,   0.],
                           [ 63.,  86.,   0.],
                           [ 59.,  84.,   0.],
                           [ 56.,  81.,   0.],
                           [ 54.,  78.,   0.],
                           [ 53.,  75.,   0.],
                           [ 53.,  66.,   0.],
                           [ 54.,  63.,   0.],
                           [ 58.,  57.,   0.],
                           [ 65.,  53.,   0.]])],
                [np.array([[  95.,   83.,    0.],
                           [ 104.,   83.,    0.],
                           [ 107.,   84.,    0.],
                           [ 113.,   88.,    0.],
                           [ 115.,   91.,    0.],
                           [ 117.,   97.,    0.],
                           [ 117.,  103.,    0.],
                           [ 115.,  109.,    0.],
                           [ 113.,  112.,    0.],
                           [ 105.,  116.,    0.],
                           [ 101.,  117.,    0.],
                           [  97.,  117.,    0.],
                           [  93.,  116.,    0.],
                           [  89.,  114.,    0.],
                           [  86.,  111.,    0.],
                           [  84.,  108.,    0.],
                           [  83.,  105.,    0.],
                           [  83.,   96.,    0.],
                           [  84.,   93.,    0.],
                           [  88.,   87.,    0.],
                           [  95.,   83.,    0.]])]]
        else:
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
