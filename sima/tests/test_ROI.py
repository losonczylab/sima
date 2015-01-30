# Unit tests for sima/ROI.py
# Tests follow conventions for NumPy/SciPy available at
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

from sima import ROI
import os
import tempfile
import numpy as np


def setup():
    return


def teardown():
    return


def test_poly2mask():
    poly1 = [[0, 0], [0, 1], [1, 1], [1, 0]]
    poly2 = [[0, 1], [0, 2], [2, 2], [2, 1]]
    masks = ROI.poly2mask([poly1, poly2], (3, 3))
    assert_equal(
        masks[0].todense(),
        np.matrix([[True, False, False],
                   [True, True, False],
                   [False, False, False]], dtype=bool))


def test_mask2poly():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 3:7] = True
    multi_poly = ROI.mask2poly(mask)
    assert_equal(
        np.array(multi_poly[0].exterior.coords),
        np.array([[7., 3.5, 0.],
                  [7., 2.5, 0.],
                  [6.5, 2., 0.],
                  [3.5, 2., 0.],
                  [3., 2.5, 0.],
                  [3., 3.5, 0.],
                  [3.5, 4., 0.],
                  [6.5, 4., 0.],
                  [7., 3.5, 0.]]))


class TestROI(object):

    def test_ROI(self):
        roi = ROI.ROI(
            polygons=[[0, 0], [0, 1], [1, 1], [1, 0]],
            im_shape=(2, 2))

        assert_equal(
            roi.coords,
            [np.array([[0., 0., 0.], [0., 1., 0.], [1., 1., 0.], [1., 0., 0.],
                       [0., 0., 0.]])])

        assert_equal(
            roi.mask[0].todense(),
            np.matrix([[True, False], [False, False]], dtype=bool))


class TestROIList(object):

    def test_transform(self):
        roi = ROI.ROI(
            polygons=[[0, 0], [0, 2], [2, 2], [2, 0]],
            im_shape=(3, 3))
        transforms = [np.array([[1, 0, 0], [0, 1, 0]])]  # one per plane

        roi_list = ROI.ROIList([roi])
        assert_equal(
            roi_list.transform(transforms)[0].coords,
            roi.coords)


if __name__ == "__main__":
    run_module_suite()
