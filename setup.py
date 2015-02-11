#!/usr/bin/env python
import sys

if 'setuptools' in sys.modules or any(
        s.startswith('bdist') for s in sys.argv) or any(
        s.startswith('develop') for s in sys.argv):
    from setuptools import setup as setup
    from setuptools import Extension
else:  # special case for runtests.py
    from distutils.core import setup as setup
    from distutils.extension import Extension

import numpy
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False


extensions = [
    Extension(
        'sima.motion._motion',
        sources=['sima/motion/_motion.%s' % ('pyx' if USE_CYTHON else 'c')],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        'sima.segment._opca',
        sources=['sima/segment/_opca.%s' % ('pyx' if USE_CYTHON else 'c')],
        include_dirs=[numpy.get_include()],
    )
]

if USE_CYTHON:
    extensions = cythonize(extensions)

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)
Operating System :: MacOS
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Topic :: Scientific/Engineering

"""
setup(
    name="sima",
    version="1.0.3",
    packages=['sima',
              'sima.misc',
              'sima.motion',
              'sima.motion.tests',
              'sima.segment',
              'sima.segment.tests',
              ],
    #   scripts = [''],
    #
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        'numpy>=1.8',
        'scipy>=0.13.0',
        'scikit-image>=0.9.3',
        'shapely>=1.2.14',
        'scikit-learn>=0.11',
        'pillow>=2.6.1',
    ],
    package_data={
        'sima': [
            'tests/*.py',
            'tests/data/example.sima/*',
            'tests/data/example.tif',
            'tests/data/example.h5',
            'tests/data/example-volume.h5',
            'tests/data/imageJ_ROIs.zip',
            'tests/data/example-tiffs/*.tif',
        ]
    },
    #
    # metadata for upload to PyPI
    author="The SIMA Development Team",
    author_email="software@losonczylab.org",
    description="Software for analysis of sequential imaging data",
    license="GNU GPLv2",
    keywords="imaging microscopy neuroscience segmentation",
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    ext_modules=extensions,
    # setup_requires=['setuptools_cython'],
    url="http://www.losonczylab.org/sima/",
    platforms=["Linux", "Mac OS-X", "Windows"],
    #
    # could also include long_description, download_url, classifiers, etc.
)
