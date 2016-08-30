# Dockerfile for core SIMA functionality.
#
# https://github.com/losonczylab/sima
#
# To build:
#   docker build -t sima_core .
# To run a python session:
#  docker run -it --rm sima_core
#

FROM debian:jessie

MAINTAINER Jeff Zaremba <jzaremba@gmail.com>

ENV DEBIAN_FRONTEND "noninteractive"

RUN apt-get update -qq && apt-get install -qq \
    curl \
    gcc \
    libatlas-base-dev \
    libatlas-dev \
    libfreetype6-dev \
    libgeos-dev \
    liblapack-dev \
    libpng-dev \
    python \
    python2.7-dev \
    && apt-get clean

# Install pip
RUN curl --silent --retry 5 https://bootstrap.pypa.io/get-pip.py | python2.7

# Required for building C libraries, must be installed first
RUN pip install Cython
RUN pip install numpy

RUN pip install \
    future \
    pillow \
    scikit-image \
    scikit-learn \
    scipy \
    shapely

RUN mkdir /sima
COPY . /sima
WORKDIR /sima

RUN python setup.py install

CMD python
