from numpy.testing import (assert_, assert_equal, assert_almost_equal,
        assert_array_almost_equal, assert_raises, assert_array_equal,
        dec, TestCase, run_module_suite, assert_allclose)

from sima import motion
from sima import misc
from sima.iterables import MultiPageTIFF

import warnings
import cPickle as pickle

from scipy.weave import build_tools
import os
import tempfile
import numpy as np
from numpy.linalg import det

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

    if os.path.exists(tmp_dir):
        for root, dirs, files in os.walk(tmp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    os.rmdir(tmp_dir)


def test_descrete_transition_prob():
    cov_matrix = np.array([[  1.64473684e-03,  -6.85307018e-05],
                           [ -6.85307018e-05,   2.71838450e-02]])
    transition_probs = lambda x, x0: 1. / np.sqrt(
        2 * np.pi * det(cov_matrix)) * np.exp(
        -np.dot(x - x0, np.linalg.solve(cov_matrix, x - x0)) / 2.)
    
    assert_almost_equal(motion._discrete_transition_prob(np.array([1., 1.]), 
        np.array([0., 0.]),transition_probs, 8), 0)
    assert_almost_equal(motion._discrete_transition_prob(np.array([0., 0.]), 
        np.array([0., 0.]),transition_probs, 8), 3.09625122)


def test_estimate_movement_model():
    shifts = np.array([[ 0.,  0., -1., -1., -1., -1.],
                       [ 0.,  7.,  5.,  6.,  6.,  5.]])
    expected = (np.array([[ 0.02 ,  0.075],[ 0.075,  1.25 ]]), 
                np.array([[ 0.83364924,  0.00700411],[ 0.00700411,  0.87939634]]),
                np.array([[-0.1555321 , -0.52939923],[-4.44306493, -4.02117885]]))

    assert_array_almost_equal(motion._estimate_movement_model(shifts,10),expected)

    assert_array_almost_equal(motion._estimate_movement_model(np.zeros((2,10)),10),
                        (np.diag([0.01,0.01]),np.zeros((2,2)),
                         np.array([[ 0.9880746 , -4.20179324],
                                   [-4.20179324, -9.39166108]])))


def test_threshold_gradient():
    test = [np.arange(4)+4*i for i in range(4)]
    res = np.zeros((4,4),dtype=bool)
    res[3,3] = True
    assert_equal(motion._threshold_gradient(np.array([test]))[0],res)


def test_hmm():
    global tmp_dir

    frames = MultiPageTIFF(misc.example_images())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        corrected = motion.hmm([[frames]],os.path.join(tmp_dir,'test_hmm.sima'),
                               artifact_channels=[0],verbose=False)

    displacements=np.array([])
    with open(misc.example_data()+'/displacements.pkl','rb') as fh:
        displacements = pickle.load(fh)

    assert_almost_equal(corrected._displacements,displacements)

class Test_MCImagingDataset(object):

    def test_estimate_displacements(self):
        assert_almost_equal(1,1)


if __name__ == "__main__":
    run_module_suite()
