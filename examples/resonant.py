"""
When acquiring data imaging data with a resonant scanner, the data
acquired when imaging the same positions can be substantially different
depending no whether the resonant scanner is moving in one direction
or the other when passing over that row. This can cause problems when
trying to motion correct the data, since even rows are collected while
scanning in one direction and odd rows are colleced by scanning
in the other direction.

The function defined here addresses this issue by using only the even
rows to estimate the displacments, and then uses those displacements
to motion-correct the entire dataset.
"""

import shutil

import numpy as np
import itertools as it

from sima import ImagingDataset
from sima.motion import HiddenMarkov2D, HiddenMarkov3D
from sima.motion.motion import _trim_coords, MotionEstimationStrategy


def resonant_motion_correction(
        sequences, save_dir, channel_names, correction_channels,
        num_states_retained, max_displacement, trim_criterion, dims=2):
    """HMM motion correction of data acquired with resonant scanning.

    Parameters
    ----------
    sequences, save_dir, channel_names, correction_channels,
    num_states_retained, max_displacement, trim_criterion
        See __init__() and correct() methods of HiddenMarkov2D.
    dims : (2, 3), optional
        Whether to correct for 2- or 3-dimensional displacements.
        Default: 2.

    Returns
    dataset : sima.ImagingDataset
        The motion corrected dataset.
    """

    tmp_savedir = save_dir + '.tmp.sima'

    if dims is 2:
        hmm = HiddenMarkov2D(
            n_processes=4, verbose=False,
            num_states_retained=num_states_retained,
            max_displacement=(max_displacement[0] / 2, max_displacement[1]))
    elif dims is 3:
        hmm = HiddenMarkov3D(
            granularity=(3, 8), n_processes=4, verbose=False,
            num_states_retained=num_states_retained,
            max_displacement=((max_displacement[0]/2,) + max_displacement[1:]),
        )

    # Motion correction of the even rows
    sliced_set = hmm.correct(
        [seq[:, :, ::2] for seq in sequences], tmp_savedir, channel_names,
        correction_channels)

    # corrected_sequences = []
    displacements = []
    for seq_idx, sequence in enumerate(sequences):
        # Repeat the displacements for all rows and multiple y-shifts by 2
        disps = sliced_set.sequences[seq_idx]._base.displacements
        disps = np.repeat(disps, 2, axis=2)
        disps[:, :, :, 0] *= 2

        # Subtract off the phase offset from every other line
        displacements.append(disps)

    displacements = MotionEstimationStrategy._make_nonnegative(displacements)

    disp_dim = displacements[0].shape[-1]
    max_disp = np.max(list(it.chain.from_iterable(d.reshape(-1, disp_dim)
                           for d in displacements)),
                      axis=0)
    raw_shape = np.array(sequences[0].shape)[1: -1]  # (z, y, x)

    if len(max_disp) == 2:  # if 2D displacements
        max_disp = np.array([0, max_disp[0], max_disp[1]])

    corrected_shape = raw_shape + max_disp

    corrected_sequences = [s.apply_displacements(d, corrected_shape)
                           for s, d in zip(sequences, displacements)]

    planes, rows, columns = _trim_coords(
        trim_criterion, displacements, raw_shape, corrected_shape)

    corrected_sequences = [sequence[:, planes, rows, columns]
                           for sequence in corrected_sequences]

    # Save full corrected dataset and remove tempdir
    imSet = ImagingDataset(
        corrected_sequences, save_dir, channel_names=channel_names)
    shutil.rmtree(tmp_savedir)

    return imSet
