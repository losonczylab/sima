#!/usr/bin/env python
import sys

if 'setuptools' in sys.modules or any(
        s.startswith('bdist') for s in sys.argv) or any(
        s.startswith('develop') for s in sys.argv):
    from setuptools import setup as setup
else:
    from distutils.core import setup as setup

from distutils.extension import Extension
import numpy

extensions = [
    Extension(
        'sima._motion',
        include_dirs=[numpy.get_include()],
        sources=['sima/_motion.c']
    ),
    Extension(
        'sima._opca',
        include_dirs=[numpy.get_include()],
        sources=['sima/_opca.c']
    )
]

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved
Programming Language :: Python
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: MacOS

"""
setup(
    name="sima",
    version="0.3.1",
    packages=['sima', 'sima.misc'],
    #   scripts = [''],
    #
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        'numpy>=1.6.2',
        'scipy>=0.13.0',
        'matplotlib>=1.2.1',
        'scikit-image>=0.9.3',
        'shapely>=1.2.14',
        # 'h5py>=2.3.1',
        # 'cv2>=2.4.8',
    ],
    package_data={
        'sima': ['tests/*.py',
                 'tests/data/example.sima/*',
                 'tests/data/example.tif',
                 'tests/data/example.h5',
                 'tests/data/imageJ_ROIs.zip',
                 ]
    },
    #
    # metadata for upload to PyPI
    author="Patrick Kaifosh, Jeffrey Zaremba, Nathan Danielson",
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
