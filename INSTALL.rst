************
Installation
************

.. Contents::

Introduction
============

We recommended that you perform a binary installation. You should not attempt
to build SIMA if you are not familiar with compiling software from sources.

On Windows, we recommend using a scientific python distribution for installing
the various prerequisites. Recommended distributions are:

* Python(x,y) (http://code.google.com/p/pythonxy/)
* WinPython (http://code.google.com/p/winpython/)
* Anaconda (https://store.continuum.io/cshop/anaconda)
* EPD (http://www.enthought.com/products/epd.php)

For Mac OS X, we recommend installing the prerequisites, especially OpenCV,
using a package manager, such as MacPorts (http://www.macports.org).

Prerequisites
=============

SIMA depends on various freely available, open source software that must be
installed prior to using SIMA:

* Python 2.7 (http://python.org)
* numpy >= 1.6.2 (http://www.scipy.org)
* scipy >= 0.13.0 (http://www.scipy.org)
* matplotlib >= 1.2.1 (http://matplotlib.org)
* scikit-image >= 0.9.3 (http://scikit-image.org)
* shapely >= 1.2.14 (https://pypi.python.org/pypi/Shapely)

Depending on the features and data formats you wish to use, you may also need
to install the following packages:

* OpenCV 2.4 (http://opencv.org), required for segmentation, registration of
  ROIs across multiple datasets, and the ROI Buddy GUI
* scikit-learn >= 0.11 (http://scikit-learn.org), required for stICA segmentation
* h5py >= 2.3.1 (http://http://www.h5py.org), required for HDF5 file format
* pylibtiff (https://code.google.com/p/pylibtiff/), required for more efficient
  handling of large TIFF files
* mdp (http://mdp-toolkit.sourceforge.net), required for ICA demixing of channels

If you build the package from source, you may also need:

* Cython (http://cython.org)


SIMA installation
=================

Linux / Mac OS X
----------------

The SIMA package can be installed from the python package index::

    $ pip install sima

The easy_install tool can also be used::

    $ easy_install sima

Source code can be downloaded from https://pypi.python.org/pypi/sima.  If you
download the source, you can install the package with setuptools::

    $ cython sima/*.pyx
    $ python setup.py build
    $ sudo python setup.py install

Windows
-------

On Windows, you can simply execute the Windows installer that can be downloaded
from https://pypi.python.org/pypi/sima.  If you run Windows Vista or Windows 7,
you may need to right-click on the installer and select "Run as Administrator".

If building SIMA from source or using pip or easy_install on Windows, you may
also need to follow these `instructions for compiling the Cython extensions
<https://github.com/cython/cython/wiki/64BitCythonExtensionsOnWindows>`_.

