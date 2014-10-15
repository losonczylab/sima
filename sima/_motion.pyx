# Written by Patrick Kaifosh, pwk2108@columbia.edu
#
# If you use this software, please cite the following paper:
# Kaifosh et al. 2013. Nature Neuroscience. 16(9): 1182-4.
#
# The HMM approach to motion correction of laser scanning microscopy
# data was first used in the following paper:
# Dombeck et al. 2007. Neuron. 56(1): 43-57.

STUFF = 'HI' # Fixes cython issue. See link below:
# http://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu

import warnings
import itertools as it

import cython
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
    cdef Py_ssize_t old_index, mapped_index, tmpIndex, k, count
    cdef FLOAT_TYPE_t lp
    count = 0
    for old_index in xrange(len(previousStateIDs)):
        for k in xrange(9):  # TODO: Parallelize
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

    minNonNanIndices = np.empty(references.shape[1], dtype='int')
    maxNonNanIndices = np.empty(references.shape[1], dtype='int')
    for i, row in enumerate(references[0]):
        nonNanIndices = np.nonzero(np.logical_not(np.isnan(row)))[0]
        minNonNanIndices[i] = nonNanIndices[0]
        maxNonNanIndices[i] = nonNanIndices[-1]
    sliceLookup = np.empty([references.shape[1], positionLookup.shape[0], 4],
                           dtype='int')
    for posIdx, position in enumerate(positionLookup):
        x_shift = position[1]
        if x_shift >= 0:
            low_ref_idx = min(offset[1] + x_shift, references.shape[2] - 1)
            high_ref_idx = min(references.shape[2], low_ref_idx + num_columns)
            low_frame_idx = 0
            high_frame_idx = low_frame_idx + high_ref_idx - low_ref_idx
        else:
            low_ref_idx = max(offset[1] + x_shift, 0)
            high_ref_idx = max(offset[1] + x_shift + num_columns, 1)
            high_frame_idx = num_columns
            low_frame_idx = high_frame_idx - high_ref_idx + low_ref_idx
        for rowIdx in range(references.shape[1]):
            sliceLookup[rowIdx, posIdx, 0] = max(minNonNanIndices[rowIdx],
                                                 low_ref_idx)
            sliceLookup[rowIdx, posIdx, 1] = min(maxNonNanIndices[rowIdx],
                                                 high_ref_idx)
            sliceLookup[rowIdx, posIdx, 2] = low_frame_idx + \
                sliceLookup[rowIdx, posIdx, 0] - low_ref_idx
            sliceLookup[rowIdx, posIdx, 3] = high_frame_idx + \
                sliceLookup[rowIdx, posIdx, 1] - high_ref_idx
    return sliceLookup


@cython.boundscheck(False)  # turn of bounds-checking for entire function
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

    cdef Py_ssize_t reference_row, minFrame, maxFrame, i, j, jj, k, index
    cdef double logp

    for i in range(tmpLogP.shape[0]):  # TODO: parallelize
        index = tmpStateIds[i]
        reference_row = frame_row + positionLookup[index, 0] + offset[0]
        if reference_row < 0 or reference_row >= num_reference_rows:
            # return -inf if row is outside the bounds of the reference image
            tmpLogP[i] = -float('inf')
        else:
            minFrame = sliceLookup[reference_row, index, 2]
            maxFrame = sliceLookup[reference_row, index, 3]
            logp = 0.0
            for chan in range(im.shape[0]):
                for j in range(0, minFrame):
                    logp += logImP[chan, frame_row, j]
                jj = sliceLookup[reference_row, index, 0]
                for j in range(minFrame, maxFrame):
                    logp += im[chan, frame_row, j] * \
                        logScaledRefs[chan, reference_row, jj] - \
                        scaled_references[chan, reference_row, jj] - \
                        logImFac[chan, frame_row, j]
                    jj += 1
                for j in range(maxFrame, logImP.shape[2]):
                    logp += logImP[chan, frame_row, j]
            tmpLogP[i] += logp

def _align_frame(
        np.ndarray[FLOAT_TYPE_t, ndim=2] frame,
        np.ndarray[INT_TYPE_t, ndim=2] displacements,
        corrected_frame_size):
    """Correct a frame based on previously estimated displacements.

    Parameters
    ----------
    frame : list of array
        Uncorrected imaging frame from each each channel.
    displacements : array
        The displacements, adjusted so that (0,0) corresponds to the corner.
        Shape: (num_rows, 2).

    Returns
    -------
    array : float32
        The corrected frame, with unobserved locations indicated as NaN.
    """
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] corrected_frame = np.zeros(
        corrected_frame_size)
    cdef np.ndarray[INT_TYPE_t, ndim=2] count = np.zeros(corrected_frame_size, dtype=int)
    cdef int num_cols, i, j, x, y
    num_cols = frame.shape[1]
    for i in range(frame.shape[0]):
        y = i + displacements[i, 0]
        for j in range(num_cols):
            x = displacements[i, 1] + j
            count[y, x] += 1
            corrected_frame[y, x] += frame[i, j]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return corrected_frame / count
