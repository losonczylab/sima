STUFF = 'HI' # Fixes cython issue. See link below:
# http://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu

import warnings
import itertools as it

import cython
# import cython.parallel
import numpy as np
cimport numpy as np

INT_TYPE = np.int
FLOAT_TYPE = np.float
ctypedef np.int_t INT_TYPE_t
ctypedef np.float_t FLOAT_TYPE_t


def transitions(
        np.ndarray[INT_TYPE_t] previousStateIDs,
        np.ndarray[FLOAT_TYPE_t] log_markov_matrix_lookup,
        np.ndarray[FLOAT_TYPE_t] logPold,
        positionLookup,
        np.ndarray[INT_TYPE_t, ndim=2] transitionLookup):
    cdef Py_ssize_t maxLen = len(positionLookup)
    cdef np.ndarray[INT_TYPE_t] tmpMap = - np.ones(maxLen, dtype='int')
    cdef np.ndarray[INT_TYPE_t] tmpStateIds = np.empty(maxLen, dtype='int')
    cdef np.ndarray[INT_TYPE_t] tmpBackpointer = np.empty(maxLen, dtype='int')
    cdef np.ndarray[FLOAT_TYPE_t] tmpLogP = np.empty(maxLen, dtype='float')
    cdef Py_ssize_t old_index, mapped_index, tmpIndex, k, count, num_transitions
    cdef FLOAT_TYPE_t lp
    count = 0
    num_transitions = len(transitionLookup)
    for old_index in xrange(len(previousStateIDs)):
        for k in xrange(num_transitions):  # TODO: parallelize
            tmpIndex = transitionLookup[k, previousStateIDs[old_index]]
            if tmpIndex != -1:
                #identify temporary location of tmpIndex
                mapped_index = tmpMap[tmpIndex]
                lp = log_markov_matrix_lookup[k] + logPold[old_index]
                if mapped_index == -1:
                    mapped_index = count
                    count += 1
                    tmpStateIds[mapped_index] = tmpIndex
                    tmpMap[tmpIndex] = mapped_index
                    tmpLogP[mapped_index] = lp
                    tmpBackpointer[mapped_index] = old_index
                # Compute probability of starting here and getting there.
                # If most likely way of getting there, then update backpointer
                elif lp > tmpLogP[mapped_index]:
                    tmpLogP[mapped_index] = lp
                    tmpBackpointer[mapped_index] = old_index
    return tmpStateIds[0:count], tmpLogP[0:count], tmpBackpointer[0:count]


def slice_lookup(np.ndarray[FLOAT_TYPE_t, ndim=3] references,
                 np.ndarray[INT_TYPE_t, ndim=2] positionLookup,
                 Py_ssize_t num_columns, np.ndarray[INT_TYPE_t] offset):
    # create lookup tables for min and max indices
    # 0. start index for reference row
    # 1. stop index for reference row
    # 2. start index for row
    # 3. stop index for row
    cdef Py_ssize_t i, rowIdx, posIdx, x_shift
    cdef Py_ssize_t low_ref_idx, high_ref_idx, low_frame_idx, high_frame_idx
    cdef np.ndarray[Py_ssize_t] nonNanIndices
    cdef np.ndarray[INT_TYPE_t] minNonNanIndices, maxNonNanIndices
    cdef np.ndarray[INT_TYPE_t, ndim=3] sliceLookup

    minNonNanIndices = np.empty(references.shape[0], dtype='int')
    maxNonNanIndices = np.empty(references.shape[0], dtype='int')
    for i, row in enumerate(references[:, :, 0]):
        nonNanIndices = np.nonzero(np.logical_not(np.isnan(row)))[0]
        try:
            minNonNanIndices[i] = nonNanIndices[0]
            maxNonNanIndices[i] = nonNanIndices[-1] + 1
        except IndexError:
            minNonNanIndices[i] = 0
            maxNonNanIndices[i] = 0
    assert np.all(minNonNanIndices <= maxNonNanIndices)
    sliceLookup = np.empty([references.shape[0], positionLookup.shape[0], 4],
                           dtype='int')
    for posIdx, position in enumerate(positionLookup):
        x_shift = position[1]
        low_ref_idx = max(min(offset[1] + x_shift, references.shape[1]), 0)
        high_ref_idx = max(
            min(offset[1] + x_shift + num_columns, references.shape[1]), 0)
        assert low_ref_idx <= high_ref_idx
        if x_shift >= 0:
            low_frame_idx = 0
            high_frame_idx = low_frame_idx + high_ref_idx - low_ref_idx
        else:
            high_frame_idx = num_columns
            low_frame_idx = high_frame_idx - high_ref_idx + low_ref_idx
        assert low_frame_idx <= high_frame_idx
        for rowIdx in range(references.shape[0]):
            sliceLookup[rowIdx, posIdx, 0] = min(
                maxNonNanIndices[rowIdx],
                max(minNonNanIndices[rowIdx], low_ref_idx))
            sliceLookup[rowIdx, posIdx, 1] = min(
                maxNonNanIndices[rowIdx],
                max(minNonNanIndices[rowIdx], high_ref_idx))
            sliceLookup[rowIdx, posIdx, 2] = max(
                0, low_frame_idx + sliceLookup[rowIdx, posIdx, 0] - low_ref_idx
            )
            sliceLookup[rowIdx, posIdx, 3] = max(
                0,
                high_frame_idx + sliceLookup[rowIdx, posIdx, 1] - high_ref_idx
            )
    assert np.all(sliceLookup[:, :, 0] <= sliceLookup[:, :, 1])
    assert np.all(sliceLookup[:, :, 2] <= sliceLookup[:, :, 3])
    assert np.all(sliceLookup[:, :, :2] <= references.shape[1])
    assert np.all(sliceLookup[:, :, :2] >= 0)
    assert np.all(sliceLookup[:, :, 2:] <= num_columns)
    assert np.all(sliceLookup[:, :, 2:] >= 0)
    return sliceLookup


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)
def log_observation_probabilities_generalized(
        np.ndarray[FLOAT_TYPE_t, ndim=1] tmpLogP,
        np.ndarray[INT_TYPE_t, ndim=1] tmpStateIds,
        np.ndarray[FLOAT_TYPE_t, ndim=2] im,
        np.ndarray[FLOAT_TYPE_t, ndim=2] logImP,
        np.ndarray[FLOAT_TYPE_t, ndim=2] logImFac,
        np.ndarray[FLOAT_TYPE_t, ndim=4] scaled_references,
        np.ndarray[FLOAT_TYPE_t, ndim=4] logScaledRefs,
        np.ndarray[INT_TYPE_t, ndim=2] positions,
        np.ndarray[INT_TYPE_t, ndim=2] positionLookup):

    cdef Py_ssize_t i, j, index, chan, Z, Y, X
    cdef INT_TYPE_t z, y, x
    cdef FLOAT_TYPE_t logp, ninf, nan
    ninf = -float('inf')
    nan = float(np.nan)
    Z = scaled_references.shape[0]
    Y = scaled_references.shape[1]
    X = scaled_references.shape[2]

    for i in range(tmpLogP.shape[0]):  # cython.parallel.prange(tmpLogP.shape[0], nogil=True):
        index = tmpStateIds[i]
        logp = 0.0
        for j in range(im.shape[0]):
            z = positions[j, 0] + positionLookup[index, 0]
            y = positions[j, 1] + positionLookup[index, 1]
            x = positions[j, 2] + positionLookup[index, 2]
            if 0 <= x and 0 <= y and 0 <= z and z < Z and y < Y and x < X:
                for chan in range(im.shape[1]):
                    if scaled_references[z, y, x, chan] == nan:
                        logp += logImP[j, chan]
                    else:
                        logp += im[j, chan] * logScaledRefs[z, y, x, chan] - \
                           scaled_references[z, y, x, chan] - logImFac[j, chan]
            else:
                for chan in range(im.shape[1]):
                    logp += logImP[j, chan]
        tmpLogP[i] += logp


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)
def log_observation_probabilities(
        np.ndarray[FLOAT_TYPE_t, ndim=1] tmpLogP,
        np.ndarray[INT_TYPE_t, ndim=1] tmpStateIds,
        np.ndarray[FLOAT_TYPE_t, ndim=3] im,
        np.ndarray[FLOAT_TYPE_t, ndim=3] logImP,
        np.ndarray[FLOAT_TYPE_t, ndim=3] logImFac,
        np.ndarray[FLOAT_TYPE_t, ndim=3] scaled_references,
        np.ndarray[FLOAT_TYPE_t, ndim=3] logScaledRefs,
        int frame_row,
        np.ndarray[INT_TYPE_t, ndim=3] sliceLookup,
        np.ndarray[INT_TYPE_t, ndim=2] positionLookup,
        np.ndarray[INT_TYPE_t] offset,
        int num_reference_rows):

    cdef Py_ssize_t reference_row, minFrame, maxFrame, i, j, jj, k, index, chan
    cdef double logp, ninf
    ninf = -float('inf')

    for i in range(tmpLogP.shape[0]):  # cython.parallel.prange(tmpLogP.shape[0], nogil=True):
        index = tmpStateIds[i]
        reference_row = frame_row + positionLookup[index, 0] + offset[0]
        if reference_row < 0 or reference_row >= num_reference_rows:
            # return -inf if row is outside the bounds of the reference image
            tmpLogP[i] = ninf
        else:
            minFrame = sliceLookup[reference_row, index, 2]
            maxFrame = sliceLookup[reference_row, index, 3]
            logp = 0.0
            for chan in range(im.shape[2]):
                for j in range(0, minFrame):
                    logp += logImP[frame_row, j, chan]
                jj = sliceLookup[reference_row, index, 0]
                for j in range(minFrame, maxFrame):
                    logp += im[frame_row, j, chan] * \
                        logScaledRefs[reference_row, jj, chan] - \
                        scaled_references[reference_row, jj, chan] - \
                        logImFac[frame_row, j, chan]
                    jj += 1
                for j in range(maxFrame, logImP.shape[1]):
                    logp += logImP[frame_row, j, chan]
            tmpLogP[i] += logp

@cython.boundscheck(False)  # turn off bounds-checking for entire function
def _align_frame(
        np.ndarray[FLOAT_TYPE_t, ndim=4] frame,
        np.ndarray[INT_TYPE_t, ndim=3] displacements,
        corrected_frame_size):
    """Correct a frame based on previously estimated displacements.

    Parameters
    ----------
    frame : array
        Uncorrected imaging frame from each each channel.
    displacements : array
        The displacements, adjusted so that (0,0) corresponds to the corner.
        Shape: (num_planes, num_rows, 2).

    Returns
    -------
    array : float32
        The corrected frame, with unobserved locations indicated as NaN.
    """
    cdef np.ndarray[FLOAT_TYPE_t, ndim=4] corrected_frame = np.zeros(
        corrected_frame_size)
    cdef np.ndarray[INT_TYPE_t, ndim=3] count = np.zeros(
        corrected_frame_size[:-1], dtype=int)
    cdef int num_cols, p, i, j, x, y, z, c, y_idx, x_idx
    num_cols = frame.shape[2]
    if displacements.shape[2] == 2:
        y_idx = 0
        x_idx = 1
    else:
        y_idx = 1
        x_idx = 2
    for p in range(frame.shape[0]):
        for i in range(frame.shape[1]):
            if y_idx == 1:
                z = p + displacements[p, i, 0]
            else:
                z = p
            y = i + displacements[p, i, y_idx]
            for j in range(num_cols):
                x = displacements[p, i, x_idx] + j
                count[z, y, x] += 1
                for c in range(frame.shape[3]):
                    corrected_frame[z, y, x, c] += frame[p, i, j, c]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (corrected_frame.T / count.T).T

def observation_counts(
        frame_shape,
        np.ndarray[INT_TYPE_t, ndim=3] displacements,
        corrected_frame_size):
    """Correct a frame based on previously estimated displacements.

    Parameters
    ----------
    frame_shape : tuple of int
        (num_planes, num_rows, num_columns)
    displacements : array
        The displacements, adjusted so that (0,0) corresponds to the corner.
        Shape: (num_rows, 2).

    Returns
    -------
    counts : ndarray
        The number of times each pixel has been observed
    """
    cdef np.ndarray[INT_TYPE_t, ndim=3] count = np.zeros(corrected_frame_size, dtype=int)
    cdef int num_cols, plane_idx, i, j, x, y
    num_cols = frame_shape[2]
    for plane_idx in range(frame_shape[0]):
        for i in range(frame_shape[1]):
            y = i + displacements[plane_idx, i, 0]
            for j in range(num_cols):
                x = displacements[plane_idx, i, 1] + j
                count[plane_idx, y, x] += 1
    return count
