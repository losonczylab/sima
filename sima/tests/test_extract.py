# Unit tests for sima/extract.py
# Tests follow conventions for NumPy/SciPy avialble at 
# https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt

# use assert_() and related functions over the built in assert to ensure tests
# run properly, regaurdless of how python is started.
from numpy.testing import (assert_, assert_equal, assert_almost_equal,
        assert_array_almost_equal, assert_raises, assert_array_equal,
        dec, TestCase, run_module_suite, assert_allclose)

from sima import extract
import os
import tempfile
import numpy as np


def setup():
    return

def teardown():
    return

def test_extract_rois():
    return

if __name__ == "__main__":
    run_module_suite()
