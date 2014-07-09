STUFF = 'HI' # Fixes cython issue. See link below:
# http://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu

import itertools as it

import cython
import numpy as np
cimport numpy as np

INT_TYPE = np.int
FLOAT_TYPE = np.float
ctypedef np.int_t INT_TYPE_t
ctypedef np.float_t FLOAT_TYPE_t

def pairwise(iterable):
    a, b = it.tee(iterable)
    next(b, None)
    return it.izip(a, b)

def _fast_ocorr(dataset, np.ndarray[INT_TYPE_t, ndim=2] pixel_pairs, channel=0):
    cdef Py_ssize_t num_pairs = pixel_pairs.shape[0]
    cdef np.ndarray[FLOAT_TYPE_t] correlations = np.zeros(num_pairs)
    #cdef np.ndarray[INT_TYPE_t] pair
    cdef np.ndarray[float, ndim=2] X
    cdef np.ndarray[float, ndim=2] Y
    cdef int pair_idx, p
    cdef Py_ssize_t a0, a1, b0, b1
    cdef float i0, i1
    pixels_ = set()
    for a0, a1, b0, b1 in pixel_pairs:
        pixels_.add((a0, a1))
        pixels_.add((b0, b1))
    cdef np.ndarray[INT_TYPE_t, ndim=2] pixels = np.concatenate(
        [np.reshape(pix, (1,2)) for pix in pixels_])
    cdef int num_pixels = pixels.shape[0]
    cdef np.ndarray[FLOAT_TYPE_t] means = np.zeros(num_pixels)
    cdef np.ndarray[FLOAT_TYPE_t] offset_stdevs = np.zeros(num_pixels)
    for cycle in dataset:
        for frame_idx, frames in enumerate(pairwise(cycle)):
            X = np.nan_to_num(frames[0][channel])
            Y = np.nan_to_num(frames[1][channel])
            print frame_idx
            for p in range(num_pixels):
                i0 = X[pixels[p, 0], pixels[p, 1]]
                i1 = Y[pixels[p, 0], pixels[p, 1]]
                means[p] += i0
                offset_stdevs[p] += i0 * i1
            for pair_idx in range(num_pairs):
                a0 = pixel_pairs[pair_idx, 0]
                a1 = pixel_pairs[pair_idx, 1]
                b0 = pixel_pairs[pair_idx, 2]
                b1 = pixel_pairs[pair_idx, 3]
                correlations[pair_idx] += (
                    X[a0, a1] * Y[b0, b1] + Y[a0, a1] * X[b0, b1])
    return (means, offset_stdevs, correlations, pixels)
