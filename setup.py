#!/usr/bin/env python
import sys
import os

if 'setuptools' in sys.modules or any(
        s.startswith('bdist') for s in sys.argv) or any(
        s.startswith('develop') for s in sys.argv):
    from setuptools import setup as setup
else:
    from distutils.core import setup as setup

from distutils.extension import Extension
from distutils.ccompiler import new_compiler
import numpy
import shutil
import tempfile

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False


def hasfunction(cc, funcname, include=None, extra_postargs=None):
    # From http://stackoverflow.com/questions/
    #            7018879/disabling-output-when-compiling-with-distutils
    tmpdir = tempfile.mkdtemp(prefix='hasfunction-')
    devnull = oldstderr = None
    try:
        try:
            fname = os.path.join(tmpdir, 'funcname.c')
            f = open(fname, 'w')
            if include is not None:
                f.write('#include %s\n' % include)
            f.write('int main(void) {\n')
            f.write('    %s;\n' % funcname)
            f.write('}\n')
            f.close()
            # Redirect stderr to /dev/null to hide any error messages
            # from the compiler.
            # This will have to be changed if we ever have to check
            # for a function on Windows.
            devnull = open('/dev/null', 'w')
            oldstderr = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            objects = cc.compile([fname], output_dir=tmpdir,
                                 extra_postargs=extra_postargs)
            cc.link_executable(objects, os.path.join(tmpdir, "a.out"))
        except Exception:
            return False
        return True
    finally:
        if oldstderr is not None:
            os.dup2(oldstderr, sys.stderr.fileno())
        if devnull is not None:
            devnull.close()
        shutil.rmtree(tmpdir)


def detect_openmp():
    # From http://mdanalysis.googlecode.com/git/package/setup.py
    "Does this compiler support OpenMP parallelization?"
    compiler = new_compiler()
    print "Attempting to autodetect OpenMP support... ",
    hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
    needs_gomp = hasopenmp
    if not hasopenmp:
        compiler.add_library('gomp')
        hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
        needs_gomp = hasopenmp
    if hasopenmp:
        print "Compiler supports OpenMP"
    else:
        print "Did not detect OpenMP support."
    return hasopenmp, needs_gomp

has_openmp, needs_gomp = detect_openmp()
parallel_args = ['-fopenmp'] if has_openmp else []
parallel_libraries = ['gomp'] if needs_gomp else []

extensions = [
    Extension(
        'sima.motion._motion',
        sources=['sima/motion/_motion.%s' % ('pyx' if USE_CYTHON else 'c')],
        include_dirs=[numpy.get_include()],
        extra_compile_args=parallel_args,
        extra_link_args=parallel_args,
    ),
    Extension(
        'sima._opca',
        sources=['sima/_opca.%s' % ('pyx' if USE_CYTHON else 'c')],
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
    version="1.0.0-dev",
    packages=['sima', 'sima.misc', 'sima.motion', 'sima.motion.tests'],
    #   scripts = [''],
    #
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        'numpy>=1.8',
        'scipy>=0.13.0',
        'matplotlib>=1.2.1',
        'scikit-image>=0.9.3',
        'shapely>=1.2.14',
    ],
    package_data={
        'sima': ['tests/*.py',
                 'tests/data/example.sima/*',
                 'tests/data/example.tif',
                 'tests/data/example.h5',
                 'tests/data/imageJ_ROIs.zip',
                 'tests/data/example-tiffs/*.tif',
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
