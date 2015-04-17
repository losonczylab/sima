from builtins import next
from builtins import object
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

import numpy as np

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

    def test_get_frame_tiff(self):
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

class TestMaskedSequence(object):

    def setup(self):
        self.tiff_seq = sima.Sequence.create('TIFF', example_tiff(), 2, 2)
        self.masked_mask = np.zeros((5, 2, 128, 256, 2), dtype=bool)

    def test_no_masks(self):
        masked = self.tiff_seq.mask([])

        assert_equal(masked.shape, (5, 2, 128, 256, 2))
        assert_(np.all(np.isfinite(masked)))

    def test_mask_all(self):
        masked = self.tiff_seq.mask([(None, None, None, None)])

        assert_equal(masked.shape, (5, 2, 128, 256, 2))
        assert_(np.all(np.isnan(masked)))

    def test_explicit_mask_all(self):
        mask_all = np.ones((128, 256), dtype=bool)
        masked = self.tiff_seq.mask([(range(5), range(2), mask_all, range(2))])

        assert_equal(masked.shape, (5, 2, 128, 256, 2))
        assert_(np.all(np.isnan(masked)))

    def test_explicit_3d_mask_all(self):
        mask_all = np.ones((2, 128, 256), dtype=bool)
        masked = self.tiff_seq.mask([(range(5), mask_all, range(2))])

        assert_equal(masked.shape, (5, 2, 128, 256, 2))
        assert_(np.all(np.isnan(masked)))

    def test_all_1_plane(self):
        masked = self.tiff_seq.mask([(None, 0, None, None)])

        self.masked_mask[:, 0, :, :, :] = True

        assert_equal(masked.shape, (5, 2, 128, 256, 2))
        assert_(np.all(np.isnan(masked)[self.masked_mask]))
        assert_(np.all(np.isfinite(masked)[~self.masked_mask]))

    def test_all_1_plane_list(self):
        masked = self.tiff_seq.mask([(None, [0], None, None)])

        self.masked_mask[:, 0, :, :, :] = True

        assert_equal(masked.shape, (5, 2, 128, 256, 2))
        assert_(np.all(np.isnan(masked)[self.masked_mask]))
        assert_(np.all(np.isfinite(masked)[~self.masked_mask]))

    def test_all_1_frame(self):
        masked = self.tiff_seq.mask([(3, None, None, None)])

        self.masked_mask[3, :, :, :, :] = True

        assert_equal(masked.shape, (5, 2, 128, 256, 2))
        assert_(np.all(np.isnan(masked)[self.masked_mask]))
        assert_(np.all(np.isfinite(masked)[~self.masked_mask]))

    def test_all_1_channel(self):
        masked = self.tiff_seq.mask([(None, None, None, 1)])

        self.masked_mask[:, :, :, :, 1] = True

        assert_equal(masked.shape, (5, 2, 128, 256, 2))
        assert_(np.all(np.isnan(masked)[self.masked_mask]))
        assert_(np.all(np.isfinite(masked)[~self.masked_mask]))

    def test_all_1_mask(self):
        frame_mask = np.random.random((128, 256))
        frame_mask = frame_mask > 0.5
        masked = self.tiff_seq.mask([(None, None, frame_mask, None)])

        self.masked_mask[:, :, frame_mask, :] = True

        assert_equal(masked.shape, (5, 2, 128, 256, 2))
        assert_(np.all(np.isnan(masked)[self.masked_mask]))
        assert_(np.all(np.isfinite(masked)[~self.masked_mask]))

    def test_3d_mask(self):
        frame_mask = np.random.random((2, 128, 256))
        frame_mask = frame_mask > 0.5
        masked = self.tiff_seq.mask([(None, frame_mask, None)])

        self.masked_mask[:, frame_mask, :] = True

        assert_equal(masked.shape, (5, 2, 128, 256, 2))
        assert_(np.all(np.isnan(masked)[self.masked_mask]))
        assert_(np.all(np.isfinite(masked)[~self.masked_mask]))


    def test_static_and_single_frame_mask(self):
        frame_mask = np.random.random((128, 256))
        frame_mask = frame_mask > 0.5
        static_mask = np.random.random((128, 256))
        static_mask = static_mask > 0.3
        masked = self.tiff_seq.mask([(None, None, static_mask, None),
                                     (2, 1, frame_mask, 0)])

        self.masked_mask[:, :, static_mask, :] = True
        self.masked_mask[2, 1, frame_mask, 0] = True

        assert_equal(masked.shape, (5, 2, 128, 256, 2))
        assert_(np.all(np.isnan(masked)[self.masked_mask]))
        assert_(np.all(np.isfinite(masked)[~self.masked_mask]))

    def test_all_mask_types(self):
        frame_mask_2d = np.random.random((128, 256))
        frame_mask_2d = frame_mask_2d > 0.1
        static_mask_2d = np.random.random((128, 256))
        static_mask_2d = static_mask_2d > 0.1
        frame_mask_3d = np.random.random((2, 128, 256))
        frame_mask_3d = frame_mask_3d > 0.1
        static_mask_3d = np.random.random((2, 128, 256))
        static_mask_3d = static_mask_3d > 0.1
        masked = self.tiff_seq.mask([([1, 3], 0, frame_mask_2d, 1),
                                     (None, 1, static_mask_2d, 0),
                                     ([4], frame_mask_3d, [0]),
                                     (None, static_mask_3d, [1])])

        self.masked_mask[1, 0, frame_mask_2d, 1] = True
        self.masked_mask[3, 0, frame_mask_2d, 1] = True
        self.masked_mask[:, 1, static_mask_2d, 0] = True
        self.masked_mask[4, frame_mask_3d, 0] = True
        self.masked_mask[:, static_mask_3d, 1] = True

        assert_equal(masked.shape, (5, 2, 128, 256, 2))
        assert_(np.all(np.isnan(masked)[self.masked_mask]))
        assert_(np.all(np.isfinite(masked)[~self.masked_mask]))


if __name__ == "__main__":
    run_module_suite()
