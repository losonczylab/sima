import itertools as it
import abc

import numpy as np

import sima
import _motion as mc


class MotionEstimationStrategy(object):
    __metaclass__ = abc.ABCMeta

    @classmethod
    def _make_nonnegative(cls, displacements):
        min_displacement = np.min(
            list(it.chain(*[d.reshape(-1, d.shape[-1])
                            for d in displacements])),
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
        return self._make_nonnegative(self._estimate(dataset))

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
        max_disp = np.max(
            list(it.chain(*[d.reshape(-1, disp_dim) for d in displacements])),
            axis=0)
        frame_shape = np.array(sequences[0].shape)[1:]
        if len(max_disp) == 2:
            frame_shape[1:3] += max_disp
        else:
            frame_shape[:3] += max_disp
        corrected_sequences = [s.apply_displacements(d, frame_shape)
                               for s, d in zip(sequences, displacements)]
        rows, columns = _trim_coords(trim_criterion, displacements,
                                     sequences[0].shape[1:4], frame_shape[:3])
        corrected_sequences = [
            s[:, :, rows, columns] for s in corrected_sequences]
        return sima.ImagingDataset(
            corrected_sequences, savedir, channel_names=channel_names)


def _trim_coords(trim_criterion, displacements, raw_shape, untrimmed_shape):
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
    if displacements.ndim == 1:
        z, y, x = displacements
        cnt[z:(z + raw_shape[0]),
            y:(y + raw_shape[1]),
            x:(x + raw_shape[2])] = 1
        return cnt
    elif displacements.ndim == 2:
        for plane in range(raw_shape[0]):
            y, x = displacements[plane]
            cnt[plane, y:(y + raw_shape[1]), x:(x + raw_shape[2])] += 1
        return cnt
    elif displacements.ndim == 3:
        return mc.observation_counts(raw_shape, displacements, untrimmed_shape)
    else:
        raise ValueError
