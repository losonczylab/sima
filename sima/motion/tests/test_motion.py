from numpy.testing import (
    assert_,
    assert_equal,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_raises, assert_array_equal,
    dec,
    TestCase,
    run_module_suite,
    assert_allclose)
import sima
import sima.motion
import sima.motion.frame_align
from sima import Sequence
from sima.misc import example_volume
import os
import numpy as np


def setup():
    global tmp_dir
    tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
    try:
        os.mkdir(tmp_dir)
    except:
        pass


def teardown():
    if os.path.exists(tmp_dir):
        for root, dirs, files in os.walk(tmp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))

            for name in dirs:
                os.rmdir(os.path.join(root, name))

    os.rmdir(tmp_dir)


class Test_ResonantCorrection(object):

    def setup(self):
        self.strategy = sima.motion.ResonantCorrection(
            sima.motion.HiddenMarkov2D(n_processes=1, verbose=False), 5)
        self.dataset = sima.ImagingDataset(
            [Sequence.create('HDF5', example_volume(), 'tzyxc')], None)

    def test_estimate(self):
        displacements = self.strategy.estimate(self.dataset)
        assert_(all((np.all(d >= 0) for d in displacements)))
        assert_(np.diff(displacements[0][0, 0, :, -1])[0] == 5)


if __name__ == '__main__':
    run_module_suite()
