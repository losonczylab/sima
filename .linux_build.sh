#!/bin/bash

TRAVIS_PYTHON_VERSION=$1
TESTMODE=$2

uname -a
free -m
df -h
ulimit -a
sudo apt-get update -qq
sudo apt-get install -qq libatlas-dev libatlas-base-dev liblapack-dev gfortran
sudo cp /usr/lib/x86_64-linux-gnu/libm.so /usr/lib/x86_64-linux-gnu/libm.so.6
if [[ "$TRAVIS_PYTHON_VERSION" == "py27" ]]; then
  wget http://repo.continuum.io/miniconda/Miniconda-3.7.0-Linux-x86_64.sh -O miniconda.sh;
else
  wget http://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O miniconda.sh;
fi
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
deps='pip numpy>=1.8 scipy>=0.13 nose shapely h5py cython scikit-image pep8 flake8 scikit-learn sphinx>=1.3.1 numpydoc pillow future'
if [[ "$TRAVIS_PYTHON_VERSION" == "py27" ]]; then export PY_VERSION="2.7"; fi
if [[ "$TRAVIS_PYTHON_VERSION" == "py34" ]]; then export PY_VERSION="3.4"; fi
echo "NBD_PY_VERSION:"
echo $PY_VERSION
echo "NBD_TRAVIS_PYTHON_VERSION:"
echo $TRAVIS_PYTHON_VERSION
conda create -q -n test-environment python=$PY_VERSION $deps
source activate test-environment
if [[ "$TRAVIS_PYTHON_VERSION" == "py27" ]]; then conda install -c https://conda.binstar.org/anaconda opencv; fi
# As of Sphinx 1.3.1, napoleon is now in the main Sphinx package
# - pip install sphinxcontrib-napoleon
# Not really sure how the old version of Sphinx is getting installed...
pip install -U Sphinx>=1.3.1
pip install picos
if [ "${TESTMODE}" == "full" ]; then pip install coverage coveralls; fi
python -V
echo $PATH
cp /usr/lib/x86_64-linux-gnu/libm.so /home/travis/miniconda/envs/test-environment/bin/../lib/libm.so.6
ldconfig -p
conda list -e
