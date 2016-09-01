#!/bin/bash

TRAVIS_PYTHON_VERSION=$1

if [[ "$TRAVIS_PYTHON_VERSION" == "py27" ]]; then export PY_VERSION="python"; fi
if [[ "$TRAVIS_PYTHON_VERSION" == "py35" ]]; then export PY_VERSION="python3"; fi

brew update
# brew install cmake || brew outdated cmake || brew upgrade cmake
brew outdated cmake || brew upgrade cmake
brew install $PY_VERSION

if [[ "$TRAVIS_PYTHON_VERSION" == "py35" ]]; then brew prune; fi
if [[ "$TRAVIS_PYTHON_VERSION" == "py35" ]]; then brew linkapps python3; fi

brew install homebrew/science/hdf5

if [[ "$TRAVIS_PYTHON_VERSION" == "py27" ]]; then
  sudo pip install numpy scipy matplotlib scikit-image shapely scikit-learn Pillow future toolz nose h5py cython pep8 flake8 sphinx numpydoc picos
else
  sudo pip3 install numpy scipy matplotlib scikit-image shapely scikit-learn Pillow future toolz nose h5py cython pep8 flake8 sphinx numpydoc picos
fi
