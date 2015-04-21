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

import matplotlib.path as mplPath

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
    return np.exp(-0.5*(((X-xy0[0]) / xysig[0])**2 +
                        ((Y-xy0[1]) / xysig[1])**2))


def test_extract_rois():
    return


@dec.knownfailureif(
    sys.version_info > (3, 0) and
    LooseVersion(np.__version__) < LooseVersion('1.9.0'))
def test_STICA():
    ds = ImagingDataset.load(example_data())
    method = segment.STICA(components=5)
    method.append(segment.SparseROIsFromMasks(min_size=50))
    method.append(segment.SmoothROIBoundaries(tolerance=1,min_verts=8))
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

        centers = [(10, 10), (40, 40), (70, 70), (100, 100)]

        roi_xy = [np.arange(self.tiff_ds.sequences[0].shape[2]),
                  np.arange(self.tiff_ds.sequences[0].shape[3])]

        rois = ROI.ROIList([
            ROI.ROI(_gaussian_2d(roi_xy, center, (10, 10)))
            for center in centers])

        tobool = segment.SparseROIsFromMasks(n_processes=2)
        smooth = segment.SmoothROIBoundaries(n_processes=2)
        rois = smooth.apply(tobool.apply(rois))

        assert_(len(rois) == len(centers))
        for roi in rois:
            polygon = mplPath.Path(roi.coords[0][:, :2])
            for nc, center in enumerate(centers):
                if polygon.contains_point(center):
                    centers.pop(nc)
                    break
        assert_(len(centers) == 0)

if __name__ == "__main__":
    run_module_suite()
