# Unit tests for sima/imaging.py
# Tests follow conventions for NumPy/SciPy avialble at
# https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt

# use assert_() and related functions over the built in assert to ensure tests
# run properly, regaurdless of how python is started.
from numpy.testing import run_module_suite
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


from sima import ImagingDataset
from scipy.weave import build_tools
import os
import tempfile
import numpy as np


def setup():
    global tmp_dir

    try:
        tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
        os.mkdir(tmp_dir)
    except:
        pass

    tmp_dir = build_tools.configure_temp_dir(tmp_dir)


def teardown():
    global tmp_dir

    if os.path.exists(tmp_dir):
        for root, dirs, files in os.walk(tmp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    os.rmdir(tmp_dir)


class TestImagingDataset(object):

    def test_ImagingDataset_3d(self):
        global tmp_dir

        from sima import Sequence
        from sima.misc import example_hdf5
        path = example_hdf5()
        seq = Sequence.create('HDF5', path, 'tzyxc')
        filepath = os.path.join(tmp_dir, "test_ImagingDataset_3d.sima")
        ds = ImagingDataset([seq, seq], filepath)
        assert_equal((ds.num_sequences,) + (ds.num_frames,) + ds.frame_shape,
                     (2, 20, 3, 50, 100, 1))


if __name__ == "__main__":
    run_module_suite()
