************
Installation
************

.. Contents::

Introduction
============

We recommended that you perform a binary installation. You should not 
attempt to build SIMA if you are not 
familiar with compiling software from sources.

On Windows, we recommend using a scientific python distribution for
installing the various prerequisites. Recommended distributions are:

* Python(x,y) (http://code.google.com/p/pythonxy/)
* WinPython (http://code.google.com/p/winpython/)
* Anaconda (https://store.continuum.io/cshop/anaconda)
* EPD (http://www.enthought.com/products/epd.php)

For Mac OS X, we recommend installing the prerequisites, especially OpenCV,
using a package manager, such as MacPorts (http://www.macports.org).

Prerequisites
=============

SIMA depends on various freely available, open source software
that must be installed prior to using SIMA:

* Python 2.7 (http://python.org)
* numpy & scipy (http://www.scipy.org)
* scikit-image (http://scikit-image.org)
* matplotlib (http://matplotlib.org)
* shapely (https://pypi.python.org/pypi/Shapely)
* mdp (http://mdp-toolkit.sourceforge.net)
* OpenCV 2.4 (http://opencv.org)

Depending on the data formats you wish to use, you may also
need to install the following packages:

* h5py >= 2.3.1 (http://http://www.h5py.org), required for HDF5 file format
* pylibtiff (https://code.google.com/p/pylibtiff/), required for more 
  efficient handling of large TIFF files

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

Source code can be downloaded from https://pypi.python.org/pypi/sima.
If you download the source, you can install the package with setuptools::

    $ python setup.py build
    $ sudo python setup.py install

Windows
-------

On Windows, you can simply execute the Windows installer that can 
be downloaded from https://pypi.python.org/pypi/sima.
If you run Windows Vista or Windows 7, you may need to right-click 
on the installer and select "Run as Administrator".

If building from source or using pip or easy_install on Windows,
you may also need compilers and other Windows (7) stuff:

* http://go.microsoft.com/?linkid=7729279
* http://www.microsoft.com/en-us/download/details.aspx?id=29
* http://www.microsoft.com/en-us/download/details.aspx?id=3138

