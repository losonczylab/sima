# cython: profile=True

STUFF = 'HI' # Fixes cython issue. See link below:
# http://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu

import itertools as it

import cython
import numpy as np
cimport numpy as np

# from sima.misc import pairwise

INT_TYPE = np.int
FLOAT_TYPE = np.float
ctypedef np.int_t INT_TYPE_t
ctypedef np.float_t FLOAT_TYPE_t

def _fast_ocorr(dataset, np.ndarray[INT_TYPE_t, ndim=2] pixel_pairs, channel=0):
    cdef Py_ssize_t num_pairs = pixel_pairs.shape[0]
    cdef np.ndarray[FLOAT_TYPE_t] correlations = np.zeros(num_pairs)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] time_avg = dataset.time_averages[channel]
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] X
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] Y
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] ostdevs = np.zeros(
        (dataset.num_rows, dataset.num_columns))
    cdef int pair_idx, p
    cdef Py_ssize_t a0, a1, b0, b1
    pixels_ = set()
    for a0, a1, b0, b1 in pixel_pairs:
        pixels_.add((a0, a1))
        pixels_.add((b0, b1))
    cdef np.ndarray[INT_TYPE_t, ndim=2] pixels = np.concatenate(
        [np.reshape(pix, (1,2)) for pix in pixels_])
    cdef int num_pixels = pixels.shape[0]
    for cycle_idx, cycle in enumerate(dataset):
        for frame_idx, frames in enumerate(pairwise(cycle)):
            X = np.nan_to_num(frames[0][channel] - time_avg)
            Y = np.nan_to_num(frames[1][channel] - time_avg)
            if not frame_idx % 100:
                print frame_idx
            ostdevs += X * Y
            for pair_idx in range(num_pairs):
                a0 = pixel_pairs[pair_idx, 0]
                a1 = pixel_pairs[pair_idx, 1]
                b0 = pixel_pairs[pair_idx, 2]
                b1 = pixel_pairs[pair_idx, 3]
                correlations[pair_idx] += (
                    X[a0, a1] * Y[b0, b1] + Y[a0, a1] * X[b0, b1])
    return (ostdevs, correlations, pixels)


def _Z_update(np.ndarray[FLOAT_TYPE_t, ndim=2] Z,
              np.ndarray[FLOAT_TYPE_t, ndim=2] U, data):
    # Calculate: Z = np.dot(np.dot(U, data.T), np.dot(O, data))
    cdef np.ndarray[FLOAT_TYPE_t] X
    cdef np.ndarray[FLOAT_TYPE_t] UX
    cdef np.ndarray[FLOAT_TYPE_t] X_
    cdef np.ndarray[FLOAT_TYPE_t] UX_
    cdef np.ndarray[FLOAT_TYPE_t] X__
    cdef np.ndarray[FLOAT_TYPE_t] UX__
    Z.fill(0.)

    i = iter(data)
    X__ = next(i)
    UX__ = np.dot(U, X__)
    X_ = next(i)
    UX_ = np.dot(U, X_)
    X_ = X_

    Z += np.outer(UX_, X__)

    for X in i:
        UX = np.dot(U, X)
        Z += np.outer(UX__ + UX, X_)

        X__ = X_
        X_ = X
        UX__ = UX_
        UX_ = UX

    Z += np.outer(UX__, X_)
    # for X, Y in pairwise(data):
    #     if t == 0:
    #         UX = np.dot(U, X)
    #     else:
    #         UX = UY
    #     UY = np.dot(U, Y)
    #     Z += np.outer(UX, Y)
    #     Z += np.outer(UY, X)
    #     t += 1
    Z *= 0.5


def pairwise(iterable):
    a, b = it.tee(iterable)
    next(b, None)
    return it.izip(a, b)
