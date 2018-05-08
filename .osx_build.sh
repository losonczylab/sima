#!/bin/bash

TRAVIS_PYTHON_VERSION=$1

if [[ "$TRAVIS_PYTHON_VERSION" == "py27" ]]; then export PY_VERSION="python2"; fi
if [[ "$TRAVIS_PYTHON_VERSION" == "py35" ]]; then export PY_VERSION="python3"; fi

brew update
brew outdated cmake || brew upgrade cmake
brew outdated $PY_VERSION || brew upgrade $PY_VERSION

if [[ "$TRAVIS_PYTHON_VERSION" == "py35" ]]; then brew prune; fi
if [[ "$TRAVIS_PYTHON_VERSION" == "py35" ]]; then brew linkapps python3; fi

brew install hdf5

if [[ "$TRAVIS_PYTHON_VERSION" == "py27" ]]; then
    sudo pip2 install numpy scipy matplotlib scikit-image shapely scikit-learn Pillow future toolz nose h5py Cython flake8 sphinx numpydoc picos
else
    sudo pip3 install numpy scipy matplotlib scikit-image shapely scikit-learn Pillow future toolz nose h5py Cython flake8 sphinx numpydoc picos
fi
