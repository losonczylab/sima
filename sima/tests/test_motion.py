# Unit tests for sima/motion.py
# Tests follow conventions for NumPy/SciPy avialble at 
# https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt

# use assert_() and related functions over the built in assert to ensure tests
# run properly, regaurdless of how python is started.
from numpy.testing import (assert_, assert_equal, assert_almost_equal,
        assert_array_almost_equal, assert_raises, assert_array_equal,
        dec, TestCase, run_module_suite, assert_allclose)

from sima import motion
from sima import misc
from sima.iterables import MultiPageTIFF
from sima.misc import example_tiff

import warnings
import cPickle as pickle

from scipy.weave import build_tools
import os
import tempfile
import numpy as np
from numpy.linalg import det


def setup():
    # setup is run prior to executing any of the included tests

    # setup ensures that a temporary directory is established either locally
    # or in the most secure location possible.

    global tmp_dir   
    tmp_dir = os.path.join(os.path.dirname(__file__),'tmp')
    
    try:
        os.mkdir(tmp_dir)
    except:
        pass

    tmp_dir = build_tools.configure_temp_dir(tmp_dir)


def teardown():
    # teardown is executed after all of the tests in this file have comlpeted

    # remove any files remaining in the temporyary directory and delete the 
    # folder

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
                np.array([[ 0.83364924,  0.00700411],
                          [ 0.00700411,  0.87939634]]),
                np.array([[-0.1555321 , -0.52939923],
                          [-4.44306493, -4.02117885]]))

    assert_array_almost_equal(
        motion._estimate_movement_model(shifts,10),expected)

    assert_array_almost_equal(
        motion._estimate_movement_model(np.zeros((2,10)),10),
                                        (np.diag([0.01,0.01]),np.zeros((2,2)),
                                         np.array([[ 0.9880746 , -4.20179324],
                                         [-4.20179324, -9.39166108]])))


def test_threshold_gradient():
    test = [np.arange(4)+4*i for i in range(4)]
    res = np.zeros((4,4),dtype=bool)
    res[3,3] = True
    assert_equal(motion._threshold_gradient(np.array([test]))[0],res)


def test_initial_distribution():
    initial_dist = motion._initial_distribution(
        np.diag([0.9,0.9]), 
        10*np.ones((2,2)), np.array([-1,1]))
    assert_almost_equal(initial_dist(0),0.00754154839)


def test_lookup_tables():
    min_displacements = np.array([-8,-5])
    max_displacements = np.array([5,14])
    min_displacements = np.array([-1,-1])
    max_displacements = np.array([1,1])

    log_markov_matrix = np.ones((2,2))

    num_columns = 2
    references = np.ones((1,2,2))

    offset = np.array([0,0])

    position_tbl, transition_tbl, log_markov_matrix_tbl, slice_tbl = \
            motion._lookup_tables(min_displacements, max_displacements,
                           log_markov_matrix, num_columns, references,
                           offset)

    pos_tbl = [[i%3-1,int(i/3)-1] for i in range(9)]
    assert_array_equal(position_tbl,pos_tbl)
    assert_(all(transition_tbl[range(9),range(8,-1,-1)]==4))
    assert_(all(log_markov_matrix_tbl==1))


def test_backtrace():
    states = [i*10+np.arange(5) for i in range(3)]
    position_tbl = np.array([[i%5-2,int(i/5)-2] for i in range(25)])
    backpointer = [np.arange(5) for i in range(2)]

    traj = motion._backtrace(2,backpointer,states,position_tbl)
    assert_array_equal(traj,[[0,-2],[0,0],[0,2]])

def test_hmm():
    global tmp_dir

    frames = MultiPageTIFF(misc.example_tiff())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        corrected = motion.hmm([[frames]],os.path.join(tmp_dir,'test_hmm.sima'),
                               artifact_channels=[0],verbose=False)

    displacements=np.array([])
    with open(misc.example_data()+'/displacements.pkl','rb') as fh:
        displacements = pickle.load(fh)

    assert_almost_equal(corrected._displacements,displacements)


class Test_MCImagingDataset(object):
    # Tests related to the MCImagingDataset class are grouped together in a 
    # class. Test classes can have their own setup/teardown methods

    def setup(self):
        for frame in MultiPageTIFF(example_tiff()):
            break
        self.frame_shifts = np.array([[0,-5],[0,-10]])
        self.correlations = np.array([1,0.5])

        shifted = frame.copy()
        shifted = np.roll(shifted,-self.frame_shifts[1,1],axis=1)
        shifted = np.roll(shifted,-self.frame_shifts[0,1],axis=0)
        frames = [frame,shifted]

        self.mc_ds = motion._MCImagingDataset([[frames]])

        self.valid_rows = {}
        self.valid_rows[0] = \
            [np.ones((self.mc_ds.num_rows*self.mc_ds.num_frames),dtype=bool)]

    def test_createcycles(self):
        cycles = self.mc_ds._create_cycles([np.ones((2,5,5)),np.ones((5,10,10))])
        assert_equal(type(cycles[0]),motion._MCCycle)
        assert_equal(type(cycles[1]),motion._MCCycle)

    def test_pixel_distribution(self):
        assert_almost_equal(
            self.mc_ds._pixel_distribution(),([ 1110.20196533],[ 946000.05906352]))

    def test_correlation_based_correction(self):        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            shifts,corrections = \
                self.mc_ds._correlation_based_correction(self.valid_rows)
        
        assert_array_equal(shifts,self.frame_shifts)
        assert_array_equal(corrections,self.correlations)

    def test_whole_frame_shifting(self):
        reference,variances,offset = \
            self.mc_ds._whole_frame_shifting(self.frame_shifts,
                                             self.correlations)

        ref_shape = np.concatenate(
            ([1], -self.frame_shifts[:,1] +[self.mc_ds.num_rows,
                self.mc_ds.num_columns]))
        assert_array_equal(reference.shape,ref_shape)
        assert_equal(len(np.where(variances>0)[0]),0)
        assert_array_equal(offset,-self.frame_shifts[:,1])

    def test_detect_artifact(self):
        valid_rows = self.mc_ds._detect_artifact([0])
        assert_equal(valid_rows,self.valid_rows)

    def test_estimate_gains(self):
        # This test needs to be finished
        for frame in MultiPageTIFF(example_tiff()):
            break


if __name__ == "__main__":
    run_module_suite()
