import itertools as it

import numpy as np

import _motion as mc


def trim_coords(trim_criterion, displacements, raw_shape, untrimmed_shape):
    """The coordinates used to trim the corrected imaging data."""
    assert len(raw_shape) == 3
    assert len(untrimmed_shape) == 3
    if trim_criterion is None:
        trim_criterion = 1.
    if isinstance(trim_criterion, (float, int)):
        obs_counts = sum(_observation_counts(raw_shape, d, untrimmed_shape)
                         for d in it.chain(*displacements))
        num_frames = sum(len(x) for x in displacements)
        occupancy = obs_counts.astype(float) / num_frames
        row_occupancy = occupancy.sum(axis=2).sum(axis=0) / (
            raw_shape[0] * raw_shape[2])
        good_rows = row_occupancy + 1e-8 >= trim_criterion
        row_min = np.nonzero(good_rows)[0].min()
        row_max = np.nonzero(good_rows)[0].max() + 1
        col_occupancy = occupancy.sum(axis=1).sum(axis=0) / np.prod(
            raw_shape[:2])
        good_cols = col_occupancy + 1e-8 >= trim_criterion
        col_min = np.nonzero(good_cols)[0].min()
        col_max = np.nonzero(good_cols)[0].max() + 1
        rows = slice(row_min, row_max)
        columns = slice(col_min, col_max)
    else:
        raise TypeError('Invalid type for trim_criterion')
    return rows, columns


def _observation_counts(raw_shape, displacements, untrimmed_shape):
    cnt = np.zeros(untrimmed_shape, dtype=int)
    if displacements.ndim == 2:
        for plane in range(raw_shape[0]):
            y, x = displacements[plane]
            cnt[plane, y:(y + raw_shape[1]), x:(x + raw_shape[2])] += 1
        return cnt
    elif displacements.ndim == 3:
        return mc.observation_counts(raw_shape, displacements, untrimmed_shape)
