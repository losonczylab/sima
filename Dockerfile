# Container running SIMA in Python 2.7 with additional optional
# dependencies.
#
# https://github.com/losonczylab/sima
#
# To build:
#   docker build -t losonczylab/sima -f .
# 
# To run the image with X forwarding enabled:
#   docker run -it --rm --net=host --env="DISPLAY" -v $HOME/.Xauthority:/root/.Xauthority:rw
#       -v /PATH/TO/DATA:/data --name sima losonczylab/sima bash
#

FROM debian:jessie

MAINTAINER Jeff Zaremba <jzaremba@gmail.com>

ENV DEBIAN_FRONTEND "noninteractive"

RUN apt-get update -qq && apt-get install -qq \
    build-essential \
    cmake \
    curl \
    libatlas-base-dev \
    libatlas-dev \
    libfreetype6-dev \
    libgeos-dev \
    libhdf5-dev \
    liblapack-dev \
    libpng-dev \
    python \
    python-tk \
    python2.7-dev \
    unzip \
    && apt-get clean

# Install pip
RUN curl --silent --retry 5 https://bootstrap.pypa.io/get-pip.py | python2.7

# Required for building C libraries, must be installed first
RUN pip install Cython
RUN pip install numpy

# Install required SIMA dependencies
RUN pip install \
    future \
    pillow \
    scikit-image \
    scikit-learn \
    scipy \
    shapely

# Install optional SIMA packages
RUN pip install \
    bottleneck \
    h5py \
    matplotlib \
    MDP \
    picos

# On first-run matplotlib needs to build a font list
RUN python -c "import matplotlib.pyplot"

# Build and install OpenCV
RUN mkdir /opencv && \
    cd /opencv && \
    curl -s http://kent.dl.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.13/opencv-2.4.13.zip -o opencv-2.4.13.zip && \
    unzip -qq opencv-2.4.13.zip && \
    rm opencv-2.4.13.zip && \
    mkdir opencv-2.4.13/build && \
    cd opencv-2.4.13 && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_BUILD_DIR=/opencv/opencv-2.4.13/build . && \
    make && \
    make install && \
    make clean

# Copy in and install sima
RUN mkdir /sima
COPY . /sima
WORKDIR /sima
RUN python setup.py install

# Download example data for workflow.py
RUN curl -s http://www.losonczylab.org/workflow_data.zip -o /sima/examples/workflow_data.zip && \
    cd /sima/examples && \
    unzip -qq /sima/examples/workflow_data.zip && \
    rm /sima/examples/workflow_data.zip

WORKDIR /sima

CMD python
