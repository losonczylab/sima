"""SIMA: Python package for sequential image analysis.
Licensed under the GNU GPL version 2 or later.
Documentation: http://www.losonczylab.org/sima
Version 1.1.0"""

from sima.imaging import ImagingDataset
from sima.sequence import Sequence

from numpy.testing import Tester
test = Tester().test

__version__ = '1.1.1'
