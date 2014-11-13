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
from sima.misc import example_data
from sima import segment


def setup():
    return


def teardown():
    return


def test_extract_rois():
    return


def test_PlaneSTICA():
    ds = ImagingDataset.load(example_data())
    method = segment.PlaneSTICA(components=5)
    ds.segment(method)


def test_PlaneNormalizedCuts():
    ds = ImagingDataset.load(example_data())[:, :, :, :50, :50]
    affinty_method = segment.BasicAffinityMatrix(num_pcs=5)
    method = segment.PlaneWiseSegmentationStrategy(
        segment.PlaneNormalizedCuts(affinty_method))
    ds.segment(method)


def test_PlaneCA1PC():
    ds = ImagingDataset.load(example_data())[:, :, :, :50, :50]
    method = segment.PlaneCA1PC(num_pcs=5)
    ds.segment(method)


if __name__ == "__main__":
    run_module_suite()
