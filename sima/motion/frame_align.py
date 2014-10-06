import itertools as it
import multiprocessing
import warnings

import numpy as np
try:
    from bottleneck import nanmean
except ImportError:
    from scipy import nanmean

import sima.misc
from sima.misc.align import align_cross_correlation
from misc import trim_coords as _trim_coords

# Setup global variables used during parallelized whole frame shifting
lock = 0
namespace = 0


def estimate(
        sequences, max_displacement=None, method='correlation',
        n_processes=None, partitions=None):
    """Estimate whole-frame displacements based on pixel correlations.

    Parameters
    ----------
    max_displacement : array
        see estimate_displacements

    Returns
    -------
    shifts : array
        (2, num_frames*num_cycles)-array of integers giving the
        estimated displacement of each frame
    correlations : array
        (num_frames*num_cycles)-array giving the correlation of
        each shifted frame with the reference
    n_processes : (None, int)
        Number of pool processes to spawn to parallelize frame alignment
    partitions : tuple of int, optional
        The number of partitions in y and x respectively. The alignement
        will be calculated separately on each partition and then the
        results compared. Default: calculates an appropriate value based
        on max_displacement and the frame shape.
    """
    shape = sequences[0].shape
    DIST_CRITERION = 2.
    if partitions is None:
        if max_displacement is None:
            partitions = (2, 2)
        else:
            partitions = np.maximum(
                np.array(shape[2:4]) / (3 * np.array(max_displacement)),
                [1, 1])
    dy = shape[2] / partitions[0]
    dx = shape[3] / partitions[1]

    shifts_record = []
    corr_record = []
    for ny, nx in it.product(range(partitions[0]), range(partitions[1])):
        partioned_sequences = [s[:, :, ny*dy:(ny+1)*dy, nx*dx:(nx+1)*dx]
                               for s in sequences]
        shifts, correlations = _frame_alignment_base(
            partioned_sequences, max_displacement, method, n_processes)
        shifts_record.append(shifts)
        corr_record.append(correlations)
        if len(shifts_record) > 1:
            first_shifts = []  # shifts to be return
            second_shifts = []
            out_corrs = []  # correlations to be returned
            for corrs, shifts in zip(zip(*corr_record), zip(*shifts_record)):
                corr_array = np.array(corrs)  # (partions, frames, planes)
                shift_array = np.array(shifts)
                assert corr_array.ndim is 3 and shift_array.ndim is 4
                second, first = np.argpartition(corr_array, -2, axis=0)[-2:]
                first_shifts.append(np.concatenate(
                    [np.expand_dims(first.choose(s), -1)
                     for s in np.rollaxis(shift_array, -1)],
                    axis=-1))
                second_shifts.append(np.concatenate(
                    [np.expand_dims(second.choose(s), -1)
                     for s in np.rollaxis(shift_array, -1)],
                    axis=-1))
                out_corrs.append(first.choose(corr_array))
            if np.mean([np.sum((f - s)**2, axis=-1)
                        for f, s in zip(first_shifts, second_shifts)]
                       ) < (DIST_CRITERION ** 2):
                break
    try:
        return first_shifts, out_corrs
    except NameError:  # single parition case
        return shifts, correlations


def _frame_alignment_base(
        sequences, max_displacement=None, method='correlation',
        n_processes=None):
    """Estimate whole-frame displacements based on pixel correlations.

    Parameters
    ----------
    max_displacement : array
        see estimate_displacements

    Returns
    -------
    shifts : array
        (2, num_frames*num_cycles)-array of integers giving the
        estimated displacement of each frame
    correlations : array
        (num_frames*num_cycles)-array giving the correlation of
        each shifted frame with the reference
    n_processes : (None, int)
        Number of pool processes to spawn to parallelize frame alignment
    """

    if n_processes is None:
        n_pools = multiprocessing.cpu_count() / 2
    else:
        n_pools = n_processes
    if n_pools == 0:
        n_pools = 1

    global namespace
    global lock
    namespace = multiprocessing.Manager().Namespace()
    namespace.offset = np.zeros(2, dtype=int)
    namespace.pixel_counts = np.zeros(sequences[0].shape[1:])  # TODO: int?
    namespace.pixel_sums = np.zeros(sequences[0].shape[1:]).astype('float64')
    # NOTE: float64 gives nan when divided by 0
    namespace.shifts = [
        np.zeros(seq.shape[:2] + (2,), dtype=int) for seq in sequences]
    namespace.correlations = [np.empty(seq.shape[:2]) for seq in sequences]

    lock = multiprocessing.Lock()
    pool = multiprocessing.Pool(processes=n_pools, maxtasksperchild=1)

    for cycle_idx, cycle in zip(it.count(), sequences):
        if n_processes > 1:
            map_generator = pool.imap_unordered(
                _align_frame,
                zip(it.count(), cycle, it.repeat(cycle_idx),
                    it.repeat(method), it.repeat(max_displacement)),
                chunksize=1 + len(cycle) / n_pools)
        else:
            map_generator = it.imap(
                _align_frame,
                zip(it.count(), cycle, it.repeat(cycle_idx),
                    it.repeat(method), it.repeat(max_displacement)))

        # Loop over generator and calculate frame alignments
        while True:
            try:
                next(map_generator)
            except StopIteration:
                break

    # TODO: align planes to minimize shifts between them
    pool.close()
    pool.join()

    def _align_planes(shifts):
        """Align planes to minimize shifts between them."""
        mean_shift = nanmean(list(it.chain(*it.chain(*shifts))), axis=0)
        # calculate alteration of shape (num_planes, dim)
        alteration = (mean_shift - mean_shift[0]).astype(int)
        for seq in shifts:
            seq -= alteration

    shifts = namespace.shifts
    _align_planes(shifts)

    # make all the shifts non-negative
    min_shift = np.min(list(it.chain(*it.chain(*shifts))), axis=0)
    shifts = [s - min_shift for s in shifts]

    return shifts, namespace.correlations


def frame_alignment(
        sequences, savedir, channel_names=None, info=None,
        method='correlation', max_displacement=None,
        correction_channels=None, trim_criterion=None, verbose=True):
    """Align whole frames.

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
    method : {'correlation', 'ECC'}
        Alignment method to be used.
    max_displacement : array of int, optional
        The maximum allowed displacement magnitudes in [y,x]. By
        default, arbitrarily large displacements are allowed.
    correction_channels : list of int, optional
        Information from the channels corresponding to these indices
        will be used for motion correction. By default, all channels
        will be used.
    trim_criterion : float, optional
        The required fraction of frames during which a location must
        be within the field of view for it to be included in the
        motion-corrected imaging frames. By default, only locations
        that are always within the field of view are retained.
    verbose : boolean, optional
        Whether to print the progress status. Defaults to True.

    Returns
    -------
    dataset : sima.ImagingDataset
        The motion-corrected dataset.

    """
    if correction_channels:
        correction_channels = [
            sima.misc.resolve_channels(c, channel_names, len(sequences[0]))
            for c in correction_channels]
        mc_sequences = [s[:, :, :, :, correction_channels] for s in sequences]
    else:
        mc_sequences = sequences
    if method == 'correlation':
        displacements, correlations = estimate(mc_sequences, max_displacement)
    elif method == 'ECC':
        # http://docs.opencv.org/trunk/modules/video/doc
        # /motion_analysis_and_object_tracking.html#findtransformecc
        #
        # http://xanthippi.ceid.upatras.gr/people/evangelidis/ecc/
        raise NotImplementedError
    else:
        raise ValueError("Unrecognized option for 'method'")
    max_disp = np.max(list(it.chain(*it.chain(*displacements))), axis=0)
    frame_shape = np.array(sequences[0].shape)[1:]
    frame_shape[1:3] += max_disp
    corrected_sequences = [
        s.apply_displacements(d, frame_shape)
        for s, d in zip(sequences, displacements)]
    rows, columns = _trim_coords(
        trim_criterion, displacements, sequences[0].shape[1:4], frame_shape[:3]
    )
    corrected_sequences = [
        s[:, :, rows, columns] for s in corrected_sequences]
    return sima.ImagingDataset(
        corrected_sequences, savedir, channel_names=channel_names)


def _align_frame(inputs):
    """Aligns single frames and updates reference image.
    Called by _frame_alignment_correlation to parallelize the alignment

    Parameters
    ----------
    frame_idx : int
        The index of the current frame
    frame : array
        (num_planes, num_rows, num_columns, num_chanels) array of raw data
    cycle_idx : int
        The index of the current cycle
    method : string
        Method to use for correlation calculation
    max_displacement : list of int
        See motion.hmm

    There is no return, but shifts and correlations in the shared namespace
    are updated.

    """

    frame_idx, frame, cycle_idx, method, max_displacement = inputs

    # Pulls in the shared namespace and lock across all processes
    global namespace
    global lock

    def _update_sums_and_counts(
            pixel_sums, pixel_counts, offset, shift, plane, plane_idx):
        """Updates pixel sums and counts of the reference image each frame"""
        ref_indices = [offset + shift[plane_idx],
                       offset + shift[plane_idx] + plane.shape[:-1]]
        assert pixel_sums.ndim == 4
        pixel_counts[plane_idx][ref_indices[0][0]:ref_indices[1][0],
                                ref_indices[0][1]:ref_indices[1][1]
                                ] += np.isfinite(plane)
        pixel_sums[plane_idx][ref_indices[0][0]:ref_indices[1][0],
                              ref_indices[0][1]:ref_indices[1][1]
                              ] += np.nan_to_num(plane)
        assert pixel_sums.ndim == 4
        return pixel_sums, pixel_counts

    def _resize_arrays(shift, pixel_sums, pixel_counts, offset, frame_shape):
        """Enlarge storage arrays if necessary."""
        l = - np.minimum(0, shift + offset)
        r = np.maximum(
            # 0, shift + offset + np.array(sequences[0].shape[2:-1]) -
            0, shift + offset + np.array(frame_shape[1:-1]) -
            np.array(pixel_sums.shape[1:-1])
        )
        assert pixel_sums.ndim == 4
        if np.any(l > 0) or np.any(r > 0):
            # adjust Y
            pre_shape = (pixel_sums.shape[0], l[0]) + pixel_sums.shape[2:]
            post_shape = (pixel_sums.shape[0], r[0]) + pixel_sums.shape[2:]
            pixel_sums = np.concatenate(
                [np.zeros(pre_shape), pixel_sums, np.zeros(post_shape)],
                axis=1)
            pixel_counts = np.concatenate(
                [np.zeros(pre_shape), pixel_counts, np.zeros(post_shape)],
                axis=1)
            # adjust X
            pre_shape = pixel_sums.shape[:2] + (l[1], pixel_sums.shape[3])
            post_shape = pixel_sums.shape[:2] + (r[1], pixel_sums.shape[3])
            pixel_sums = np.concatenate(
                [np.zeros(pre_shape), pixel_sums, np.zeros(post_shape)],
                axis=2)
            pixel_counts = np.concatenate(
                [np.zeros(pre_shape), pixel_counts, np.zeros(post_shape)],
                axis=2)
            offset += l
        assert pixel_sums.ndim == 4
        assert np.prod(pixel_sums.shape) < 4 * np.prod(frame_shape)
        return pixel_sums, pixel_counts, offset

    for p, plane in zip(it.count(), frame):
        # if frame_idx in invalid_frames:
        #     correlations[i] = np.nan
        #     shifts[:, i] = np.nan
        with lock:
            any_check = np.any(namespace.pixel_counts[p])
            if not any_check:
                corrs = namespace.correlations
                corrs[cycle_idx][frame_idx][p] = 1
                namespace.correlations = corrs
                s = namespace.shifts
                s[cycle_idx][frame_idx][p][:] = 0
                namespace.shifts = s
                namespace.pixel_sums, namespace.pixel_counts = \
                    _update_sums_and_counts(
                        namespace.pixel_sums, namespace.pixel_counts,
                        namespace.offset,
                        namespace.shifts[cycle_idx][frame_idx], plane, p)
        if any_check:
            # recompute reference using all aligned images
            with lock:
                p_sums = namespace.pixel_sums[p]
                p_counts = namespace.pixel_counts[p]
                p_offset = namespace.offset
                shifts = namespace.shifts
            with warnings.catch_warnings():  # ignore divide by 0
                warnings.simplefilter("ignore")
                reference = p_sums / p_counts
            if method == 'correlation':
                if max_displacement is not None and np.all(
                        max_displacement > 0):
                    min_shift = np.min(list(it.chain(*it.chain(*shifts))),
                                       axis=0)
                    max_shift = np.max(list(it.chain(*it.chain(*shifts))),
                                       axis=0)
                    displacement_bounds = np.array(
                        [np.minimum(max_shift - max_displacement, min_shift),
                         np.maximum(min_shift + max_displacement, max_shift)])
                else:
                    displacement_bounds = None
                shift, p_corr = align_cross_correlation(
                    reference, plane, displacement_bounds)
            elif method == 'ECC':
                raise NotImplementedError
                # cv2.findTransformECC(reference, plane)
            else:
                raise ValueError('Unrecognized alignment method')
            with lock:
                s = namespace.shifts
                s[cycle_idx][frame_idx][p][:] = shift - p_offset
                namespace.shifts = s
                corrs = namespace.correlations
                corrs[cycle_idx][frame_idx][p] = p_corr
                namespace.correlations = corrs

            with lock:
                namespace.pixel_sums, namespace.pixel_counts, namespace.offset\
                    = _resize_arrays(
                        namespace.shifts[cycle_idx][frame_idx][p],
                        namespace.pixel_sums, namespace.pixel_counts,
                        namespace.offset, frame.shape)
                namespace.pixel_sums, namespace.pixel_counts \
                    = _update_sums_and_counts(
                        namespace.pixel_sums, namespace.pixel_counts,
                        namespace.offset,
                        namespace.shifts[cycle_idx][frame_idx], plane, p)
