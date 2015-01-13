# Unit tests for sima/sequence.py
# Tests follow conventions for NumPy/SciPy avialble at
# https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt

# use assert_() and related functions over the built in assert to ensure tests
# run properly, regaurdless of how python is started.
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

import sima
from sima.misc import example_tiffs


def setup():
    return


def teardown():
    return


class TestSequence(object):

    def test_create_TIFFs(self):
        seq = sima.Sequence.create(
            'TIFFs', [[example_tiffs(), example_tiffs()],
                      [example_tiffs(), example_tiffs()],
                      [example_tiffs(), example_tiffs()],
                      [example_tiffs(), example_tiffs()]])
        assert_equal(seq.shape, (3, 4, 173, 173, 2))


if __name__ == "__main__":
    run_module_suite()
