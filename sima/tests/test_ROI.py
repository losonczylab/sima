# Unit tests for sima/ROI.py
# Tests follow conventions for NumPy/SciPy avialble at 
# https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt

# use assert_() and related functions over the built in assert to ensure tests
# run properly, regaurdless of how python is started.
from numpy.testing import (assert_, assert_equal, assert_almost_equal,
        assert_array_almost_equal, assert_raises, assert_array_equal,
        dec, TestCase, run_module_suite, assert_allclose)

from sima import ROI
import os
import tempfile
import numpy as np


def setup():
    return

def teardown():
    return

def test_poly2mask():
    poly1 = [[0,0], [0,1], [1,1], [1,0]]
    poly2 = [[0,1], [0,2], [2,2], [2,1]]
    mask = ROI.poly2mask([poly1, poly2], (3, 3))
    assert_equal(
        mask.todense(),
        np.matrix([[ True, False, False],
                   [ True,  True, False],
                   [False, False, False]], dtype=bool))


class TestROI(object):
    def test_ROI(self):
        roi = ROI.ROI(
            polygons=[[0, 0], [0, 1], [1, 1], [1, 0]], 
            im_shape=(2, 2))

        assert_equal(
            roi.coords,
            [np.array([[ 0.,  0.],[ 0.,  1.],[ 1.,  1.],[ 1.,  0.],[ 0.,  0.]])])

        assert_equal(
            roi.mask.todense(), 
            np.matrix([[ True, False],[False, False]], dtype=bool))


if __name__ == "__main__":
    run_module_suite()
