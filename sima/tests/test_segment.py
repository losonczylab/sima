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

from sima import extract
from sima import ImagingDataset
from sima.misc import example_data
from sima import segment
import os
import tempfile
import numpy as np


def setup():
    return


def teardown():
    return


def test_extract_rois():
    return


def test_stica():
    ds = ImagingDataset.load(example_data())
    rois = segment.stica(ds, channel=0, components=5)
    assert_equal(len(rois), 3)
    assert_(rois[0].mask.toarray()[22, 39])
    assert_(not rois[0].mask.toarray()[50, 50])

if __name__ == "__main__":
    run_module_suite()
