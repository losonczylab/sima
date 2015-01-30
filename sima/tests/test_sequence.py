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
from sima.misc import example_tiffs, example_tiff


def setup():
    return


def teardown():
    return


class TestSequence(object):

    def setup(self):
        self.tiff_seq = sima.Sequence.create('TIFF', example_tiff(), 2, 2)

    def test_create_tiffs(self):
        seq = sima.Sequence.create(
            'TIFFs', [[example_tiffs(), example_tiffs()],
                      [example_tiffs(), example_tiffs()],
                      [example_tiffs(), example_tiffs()],
                      [example_tiffs(), example_tiffs()]])
        assert_equal(seq.shape, (3, 4, 173, 173, 2))

    def test__get_frame_tiff(self):
        it = iter(self.tiff_seq)
        assert_array_equal(next(it), self.tiff_seq._get_frame(0))
        assert_array_equal(next(it), self.tiff_seq._get_frame(1))

    # @dec.knownfailureif(True)
    # def test_export_hdf5(self):
    #     raise NotImplemented

    # @dec.knownfailureif(True)
    # def test_export_tiff8(self):
    #     raise NotImplemented

    # @dec.knownfailureif(True)
    # def test_export_tiff16(self):
    #     raise NotImplemented


if __name__ == "__main__":
    run_module_suite()
