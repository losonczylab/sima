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


from sima import ImagingDataset, Sequence, ROI
from sima.misc import example_hdf5, example_imagej_rois
from scipy.weave import build_tools
import os
import shutil
# import tempfile
import numpy as np
from PIL import Image
import h5py

tmp_dir = None


def setup():
    global tmp_dir

    tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')

    try:
        os.mkdir(tmp_dir)
    except:
        pass

    tmp_dir = build_tools.configure_temp_dir(tmp_dir)


def teardown():
    global tmp_dir

    shutil.rmtree(tmp_dir)


def test_imaging_dataset_3d():
    global tmp_dir

    path = example_hdf5()
    seq = Sequence.create('HDF5', path, 'yxt')

    filepath = os.path.join(tmp_dir, "test_imaging_dataset_3d.sima")
    ds = ImagingDataset([seq, seq], filepath)
    assert_equal((ds.num_sequences,) + (ds.num_frames,) + ds.frame_shape,
                 (2, 40, 1, 128, 256, 1))


class TestImagingDataset(object):

    def setup(self):
        global tmp_dir

        path = example_hdf5()
        seq = Sequence.create('HDF5', path, 'yxt')
        self.filepath = os.path.join(tmp_dir, "test_imaging_dataset.sima")
        self.ds = ImagingDataset([seq, seq], self.filepath)

    def teardown(self):
        shutil.rmtree(self.filepath)

    def test_time_averages(self):
        averages = self.ds.time_averages
        assert_equal(self.ds.frame_shape, averages.shape)
        # Check it twice, since second time should load from a saved pkl
        averages2 = self.ds.time_averages
        assert_equal(self.ds.frame_shape, averages2.shape)

    def test_export_averages_tiff16(self):
        time_avg_path = os.path.join(self.filepath, 'time_avg_Ch2.tif')
        self.ds.export_averages(
            [time_avg_path], fmt='TIFF16', scale_values=False)
        assert_equal(self.ds.time_averages[0, ..., 0].astype('uint16'),
                     np.array(Image.open(time_avg_path)))

    def test_export_averages_tiff8(self):
        time_avg_path = os.path.join(self.filepath, 'time_avg_Ch2.tif')
        self.ds.export_averages(
            [time_avg_path], fmt='TIFF8', scale_values=False)
        assert_equal(self.ds.time_averages[0, ..., 0].astype('uint8'),
                     np.array(Image.open(time_avg_path)))

    def test_export_averages_hdf5(self):
        time_avg_path = os.path.join(self.filepath, 'time_avg.h5')
        self.ds.export_averages(time_avg_path, fmt='HDF5', scale_values=False)

        h5_time_avg = h5py.File(time_avg_path, 'r')['time_average']
        assert_equal(self.ds.time_averages.astype('uint16'), h5_time_avg)
        assert_equal(self.ds.channel_names, h5_time_avg.attrs['channel_names'])
        dim_labels = [dim.label for dim in h5_time_avg.dims]
        assert_equal(['z', 'y', 'x', 'c'], dim_labels)

    def test_add_and_delete_rois(self):
        rois = ROI.ROIList.load(example_imagej_rois(), fmt='ImageJ')
        self.ds.add_ROIs(rois, 'rois')
        assert_equal(len(self.ds.ROIs), 1)

        self.ds.add_ROIs(rois, 'rois2')
        assert_equal(len(self.ds.ROIs), 2)

        assert_equal(sorted(self.ds.ROIs.keys()), ['rois', 'rois2'])
        assert_equal(len(self.ds.ROIs['rois']), 2)

        # This should quietly do nothing
        self.ds.delete_ROIs('foo')

        self.ds.delete_ROIs('rois')
        assert_equal(len(self.ds.ROIs), 1)
        self.ds.delete_ROIs('rois2')
        assert_equal(len(self.ds.ROIs), 0)

        # This should quietly do nothing
        self.ds.delete_ROIs('foo')

    def test_rois(self):
        assert_equal(len(self.ds.ROIs), 0)

    # @dec.knownfailureif(True)
    # def test_import_transformed_rois(self):
    #     raise NotImplemented

    # @dec.knownfailureif(True)
    # def test_export_signals(self):
    #     raise NotImplemented


if __name__ == "__main__":
    run_module_suite()
