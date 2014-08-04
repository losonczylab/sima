from numpy.testing import (assert_, assert_equal, assert_almost_equal,
        assert_array_almost_equal, assert_raises, assert_array_equal,
        dec, TestCase, run_module_suite, assert_allclose)

from sima import motion
from scipy.weave import build_tools
import os
import tempfile
import numpy as np


def setup():
    global tmp_dir

    try:
        tmp_dir = os.path.join(os.path.dirname(__file__),'tmp')
        os.mkdir(tmp_dir)
    except:
        pass

    tmp_dir = build_tools.configure_temp_dir(tmp_dir)


def teardown():
    global tmp_dir
    """
    if os.path.exists(tmp_dir):
        for root, dirs, files in os.walk(tmp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    os.rmdir(tmp_dir)
    """

def test_hmm():
    global tmp_dir

    #frames = np.ones((100, 128, 128))
    #motion.hmm([[frames]],tmp_dir)

    assert_equal(1,2)


class Test_MCImagingDataset(object):
    def test_estimate_displacements(self):
        assert_almost_equal(1,2)

if __name__ == "__main__":
    run_module_suite()
