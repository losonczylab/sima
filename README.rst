Overview
--------
The SIMA (Sequential IMage Analysis) package facilitates
analysis of time-series imaging data arising from fluorescence
microscopy.
The functionality of this package includes:

- correction of motion artifacts
- segmentation of imaging fields into regions of interest (ROIs)
- extraction of dynamic signals from ROIs

The included ROI Buddy software provides a graphical user interface
(GUI) supporting the following functionality:

- editing of ROIs resulting from automated segmentation
- registration of ROIs across separate imaging sessions


Installation and Use
--------------------
For complete documentation go to <http://www.losonczylab.org/sima>


Dependencies
-------------

- Python 2.7 (http://python.org)
- numpy >= 1.6.2 (http://www.scipy.org)
- scipy >= 0.13.0 (http://www.scipy.org)
- matplotlib >= 1.2.1 (http://matplotlib.org)
- scikit-image >= 0.9.3 (http://scikit-image.org)
- shapely >= 1.2.14 (https://pypi.python.org/pypi/Shapely)

Optional dependencies
---------------------

- OpenCV 2.4 (http://opencv.org), required for segmentation, registration of
  ROIs across multiple datasets, and the ROI Buddy GUI
- h5py >= 2.3.1 (http://http://www.h5py.org), required for HDF5 file format
- pylibtiff (https://code.google.com/p/pylibtiff/), required for more efficient
  handling of large TIFF files
- mdp (http://mdp-toolkit.sourceforge.net), required for ICA demixing of channels


License
-------
Unless otherwise specified in individual files, all code is

Copyright (C) 2014  The Trustees of Columbia University in the City of New York.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
