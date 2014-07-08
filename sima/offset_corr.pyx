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

def _offset_corrs(dataset, np.ndarray[INT_TYPE_t, ndim=2] pixel_pairs, channel=0):
    """
    Calculate the offset correlation for specified pixel pairs.

    Parameters
    -----------
    dataset : sima.ImagingDataset
        The dataset to be used.
    pixel_pairs : list of tuple of tuple of int
        The pairs of pixels, indexed ((y0, x0), (y1, x1)) for
        which the correlation is to be calculated.
    channel : int, optional
        The channel to be used for estimating the pixel correlations.
        Defaults to 0.

    Returns
    -------
    correlations: dict
        A dictionary whose keys are the elements of the pixel_pairs
        input list, and whose values are the calculated offset
        correlations.
    """
    cdef Py_ssize_t num_pairs = pixel_pairs.shape[0]
    cdef np.ndarray[FLOAT_TYPE_t] correlations = np.zeros(num_pairs)
    cdef np.ndarray[INT_TYPE_t] pair
    cdef np.ndarray[float, ndim=2] X
    cdef np.ndarray[float, ndim=2] Y
    cdef int pair_idx
    cdef int a0, a1, b0, b1
    pixels = set()
    for a0, a1, b0, b1 in pixel_pairs:
        pixels.add((a0, a1))
        pixels.add((b0, b1))
    means = {x: 0. for x in pixels}
    offset_stdevs = {x: 0. for x in pixels}
    for cycle in dataset:
        for frame_idx, frames in enumerate(pairwise(cycle)):
            X = np.nan_to_num(frames[0][channel])
            Y = np.nan_to_num(frames[1][channel])
            print frame_idx
            for a in pixels:
                a0 = X[a[0], a[1]]
                a1 = Y[a[0], a[1]]
                means[a] += a0
                offset_stdevs[a] += a0 * a1
            print 'pairs'
            for pair_idx in range(num_pairs):
                a0 = pixel_pairs[pair_idx, 0]
                a1 = pixel_pairs[pair_idx, 1]
                b0 = pixel_pairs[pair_idx, 2]
                b1 = pixel_pairs[pair_idx, 3]
                correlations[pair_idx] += (
                    X[a0, a1] * Y[b0, b1] +
                    Y[a0, a1] * X[b0, b1])
    for pixel in pixels:
        means[pixel] /= dataset.num_frames - 1.
        offset_stdevs /= dataset.num_frames - 1.
        offset_stdevs[pixel] = np.sqrt(
            max(0., offset_stdevs[pixel] - means[pixel] ** 2))
    for pair_idx, pair in enumerate(pixel_pairs):
        correlations[pair_idx] /= 2. * (dataset.num_frames - 1)
        correlations[pair_idx] = np.nan_to_num(
            (correlations[pair_idx] - means[pair[0]] * means[pair[1]]) /
            (offset_stdevs[(pair[0], pair[1])] * offset_stdevs[(pair[2], pair[3])])
        )
    return {((PAIR[0], PAIR[1]), (PAIR[2], PAIR[3])): correlations[pair_idx]
            for pair_idx, PAIR in enumerate(pixel_pairs)}


