import itertools as it
import abc

import numpy as np

import sima
import _motion as mc


def add_with_offset(array1, array2, offset):
    """

    >>> from sima.motion.motion import add_with_offset
    >>> import numpy as np
    >>> a1 = np.zeros((4, 4))
    >>> a2 = np.ones((1, 2))
    >>> add_with_offset(a1, a2, (1, 2))
    >>> np.array_equal(a1[1:2, 2:4], a2)
    True

    """
    slices = tuple(slice(o, o + e) for o, e in zip(offset, array2.shape))
    array1[slices] += array2


class MotionEstimationStrategy(object):
    __metaclass__ = abc.ABCMeta

    @classmethod
    def _make_nonnegative(cls, displacements):
        min_displacement = np.min(
            list(it.chain.from_iterable(d.reshape(-1, d.shape[-1])
                                        for d in displacements)),
            axis=0)
        return [d - min_displacement for d in displacements]

    @abc.abstractmethod
    def _estimate(self, dataset):
        return

    def estimate(self, dataset):
        """Estimate the displacements for a dataset.

        Parameters
        ----------
        dataset : sima.ImagingDataset

        Returns
        -------
        displacements : list of ndarray of int
        """
        shifts = self._estimate(dataset)
        assert np.any(np.all(x is not np.ma.masked for x in shift)
                      for shift in it.chain.from_iterable(shifts))
        assert np.all(
            np.all(x is np.ma.masked for x in shift) or
            not np.any(x is np.ma.masked for x in shift)
            for shift in it.chain.from_iterable(shifts))
        shifts = self._make_nonnegative(shifts)
        assert np.any(np.all(x is not np.ma.masked for x in shift)
                      for shift in it.chain.from_iterable(shifts))
        assert np.all(
            np.all(x is np.ma.masked for x in shift) or
            not np.any(x is np.ma.masked for x in shift)
            for shift in it.chain.from_iterable(shifts))
        return shifts

    def correct(self, sequences, savedir, channel_names=None, info=None,
                correction_channels=None, trim_criterion=None):
        """Create a motion-corrected dataset.

        Parameters
        ----------
        sequences : list of list of iterable
            Iterables yielding frames from imaging cycles and channels.
        savedir : str
            The directory used to store the dataset. If the directory
            name does not end with .sima, then this extension will
            be appended.
        channel_names : list of str, optional
            Names for the channels. Defaults to ['0', '1', '2', ...].
        info : dict
            Data for the order and timing of the data acquisition.
            See sima.ImagingDataset for details.
        correction_channels : list of int, optional
            Information from the channels corresponding to these indices
            will be used for motion correction. By default, all channels
            will be used.
        trim_criterion : float, optional
            The required fraction of frames during which a location must
            be within the field of view for it to be included in the
            motion-corrected imaging frames. By default, only locations
            that are always within the field of view are retained.

        Returns
        -------
        dataset : sima.ImagingDataset
            The motion-corrected dataset.
        """
        if correction_channels:
            correction_channels = [
                sima.misc.resolve_channels(c, channel_names, len(sequences[0]))
                for c in correction_channels]
            mc_sequences = [s[:, :, :, :, correction_channels]
                            for s in sequences]
        else:
            mc_sequences = sequences
        displacements = self.estimate(sima.ImagingDataset(mc_sequences, None))
        disp_dim = displacements[0].shape[-1]
        max_disp = np.max(list(it.chain.from_iterable(d.reshape(-1, disp_dim)
                               for d in displacements)),
                          axis=0)
        frame_shape = np.array(sequences[0].shape)[1: -1]  # (z, y, x)
        if len(max_disp) == 2:  # if 2D displacements
            frame_shape[1:3] += max_disp
        else:  # if 3D displacements
            frame_shape += max_disp
        corrected_sequences = [s.apply_displacements(d, frame_shape)
                               for s, d in zip(sequences, displacements)]
        planes, rows, columns = _trim_coords(
            trim_criterion, displacements, sequences[0].shape[1:4],
            frame_shape)
        corrected_sequences = [
            s[:, planes, rows, columns] for s in corrected_sequences]
        return sima.ImagingDataset(
            corrected_sequences, savedir, channel_names=channel_names)


def _trim_coords(trim_criterion, displacements, raw_shape, untrimmed_shape):
    """The coordinates used to trim the corrected imaging data."""
    epsilon = 1e-8
    assert len(raw_shape) == 3
    assert len(untrimmed_shape) == 3
    if trim_criterion is None:
        trim_criterion = 1.
    if trim_criterion == 0.:
        trim_criterion = epsilon
    if not isinstance(trim_criterion, (float, int)):
        raise TypeError('Invalid type for trim_criterion')
    obs_counts = sum(_observation_counts(raw_shape, d, untrimmed_shape)
                     for d in it.chain.from_iterable(displacements))
    num_frames = sum(len(x) for x in displacements)
    occupancy = obs_counts.astype(float) / num_frames

    plane_occupancy = occupancy.sum(axis=2).sum(axis=1) / (
        raw_shape[1] * raw_shape[2])
    good_planes = plane_occupancy + epsilon > trim_criterion
    plane_min = np.nonzero(good_planes)[0].min()
    plane_max = np.nonzero(good_planes)[0].max() + 1

    row_occupancy = occupancy.sum(axis=2).sum(axis=0) / (
        raw_shape[0] * raw_shape[2])
    good_rows = row_occupancy + epsilon > trim_criterion
    row_min = np.nonzero(good_rows)[0].min()
    row_max = np.nonzero(good_rows)[0].max() + 1

    col_occupancy = occupancy.sum(axis=1).sum(axis=0) / np.prod(
        raw_shape[:2])
    good_cols = col_occupancy + epsilon > trim_criterion
    col_min = np.nonzero(good_cols)[0].min()
    col_max = np.nonzero(good_cols)[0].max() + 1

    rows = slice(row_min, row_max)
    columns = slice(col_min, col_max)
    planes = slice(plane_min, plane_max)
    return planes, rows, columns


def _observation_counts(raw_shape, displacements, untrimmed_shape):
    cnt = np.zeros(untrimmed_shape, dtype=int)
    if displacements.ndim == 1:
        z, y, x = displacements
        cnt[z:(z + raw_shape[0]),
            y:(y + raw_shape[1]),
            x:(x + raw_shape[2])] = 1
    elif displacements.ndim == 2:
        for plane in range(raw_shape[0]):
            d = list(displacements[plane])
            if len(d) == 2:
                d = [0] + d
            cnt[plane + d[0],
                d[1]:(d[1] + raw_shape[1]),
                d[2]:(d[2] + raw_shape[2])] += 1
    elif displacements.ndim == 3:
        if displacements.shape[-1] == 2:
            return mc.observation_counts(raw_shape, displacements,
                                         untrimmed_shape)
        else:
            for plane, p_disp in enumerate(displacements):
                for row, r_disp in enumerate(p_disp):
                    add_with_offset(cnt, np.ones((1, 1, raw_shape[2])),
                                    r_disp + np.array([plane, row, 0]))
    else:
        raise ValueError
    return cnt
