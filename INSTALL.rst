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

If you build the package from source, you may also need:

* Cython (http://cython.org)


Binary installation
===================


Linux / Mac OS X
----------------

Pip can be installed from the python package index::

    $ pip install sima

Windows
-------

On Windows, you can simply execute this `installer<>_`. 
If you run Windows Vista or Windows 7, you may need to right-click 
on the installer and select "Run as Administrator".


Building from source
====================

Source code can be downloaded from https://pypi.python.org/pypi/sima.

Linux
-----

If you download the source, you can use setuptools::

    $ python setup.py build
    $ sudo python setup.py install

Windows
-------
We recommend using 32bit `WinPython <http://winpython.sourceforge.net/>`_.

You may also need compilers and other Windows (7) stuff:

* http://www.microsoft.com/en-us/download/details.aspx?id=29
* http://www.microsoft.com/en-us/download/details.aspx?id=3138


Mac OS X
--------
