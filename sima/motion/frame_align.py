from __future__ import absolute_import
from __future__ import division
from builtins import next
from builtins import zip
from builtins import map
from builtins import range
from past.utils import old_div
from builtins import object
import itertools as it
import multiprocessing
import warnings

import numpy as np
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean
import scipy.ndimage.filters

from . import motion
from sima.misc.align import align_cross_correlation

# Setup global variables used during parallelized whole frame shifting
lock = 0
namespace = 0


class Struct(object):

    def __init__(self, **entries):
        self.__dict__.update(entries)


class PlaneTranslation2D(motion.MotionEstimationStrategy):

    """Estimate 2D translations for each plane.

    Parameters
    ----------
    max_displacement : array of int, optional
        The maximum allowed displacement magnitudes in [y,x]. By
        default, arbitrarily large displacements are allowed.
    method : {'correlation', 'ECC'}
        Alignment method to be used.
    n_processes : int, optional
        Number of pool processes to spawn to parallelize frame alignment.
        Defaults to 1.

    """

    def __init__(self, max_displacement=None, method='correlation',
                 n_processes=1, **method_kwargs):
        self._params = dict(locals())
        del self._params['self']

    def _estimate(self, dataset):
        """Estimate whole-frame displacements based on pixel correlations.

        Parameters
        ----------

        Returns
        -------
        shifts : array
            (2, num_frames*num_cycles)-array of integers giving the
            estimated displacement of each frame

        """
        params = self._params
        return _frame_alignment_base(
            dataset, params['max_displacement'], params['method'],
            params['n_processes'], **params['method_kwargs'])[0]


def _frame_alignment_base(
        dataset, max_displacement=None, method='correlation', n_processes=1,
        **method_kwargs):
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
    n_processes : int, optional
        Number of pool processes to spawn to parallelize frame alignment.
        Defaults to 1.

    """

    if n_processes < 1:
        raise ValueError('n_processes must be at least 1')

    global namespace
    global lock
    if n_processes > 1:
        namespace = multiprocessing.Manager().Namespace()
    else:
        namespace = Struct()
    namespace.offset = np.zeros(3, dtype=int)
    namespace.pixel_counts = np.zeros(dataset.frame_shape)  # TODO: int?
    namespace.pixel_sums = np.zeros(dataset.frame_shape).astype('float64')
    # NOTE: float64 gives nan when divided by 0
    namespace.shifts = [
        np.zeros(seq.shape[:2] + (3,), dtype=int) for seq in dataset]
    namespace.correlations = [np.empty(seq.shape[:2]) for seq in dataset]
    namespace.min_shift = np.zeros(3)
    namespace.max_shift = np.zeros(3)

    lock = multiprocessing.Lock()
    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes, maxtasksperchild=1)

    for cycle_idx, cycle in zip(it.count(), dataset):
        chunksize = min(1 + old_div(len(cycle), n_processes), 200)
        if n_processes > 1:
            map_generator = pool.imap_unordered(
                _align_frame,
                zip(it.count(), cycle, it.repeat(cycle_idx),
                    it.repeat(method), it.repeat(max_displacement)),
                chunksize=chunksize)
        else:
            map_generator = map(
                _align_frame,
                zip(it.count(), cycle, it.repeat(cycle_idx),
                    it.repeat(method), it.repeat(max_displacement),
                    it.repeat(method_kwargs)))

        # Loop over generator and calculate frame alignments
        while True:
            try:
                next(map_generator)
            except StopIteration:
                break

    if n_processes > 1:
        pool.close()
        pool.join()

    def _align_planes(shifts):
        """Align planes to minimize shifts between them."""
        mean_shift = nanmean(np.concatenate(shifts), axis=0)
        # calculate alteration of shape (num_planes, dim)
        alteration = (mean_shift - mean_shift[0]).astype(int)
        for seq in shifts:
            seq -= alteration

    shifts = [s[..., 1:] for s in namespace.shifts]
    correlations = namespace.correlations

    del namespace.pixel_counts
    del namespace.pixel_sums
    del namespace.shifts
    del namespace.correlations

    _align_planes(shifts)
    return shifts, correlations


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

    (frame_idx, frame, cycle_idx, method,
        max_displacement, method_kwargs) = inputs
    if max_displacement is not None:
        max_displacement = [0] + list(max_displacement)

    # Pulls in the shared namespace and lock across all processes
    global namespace
    global lock

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
                namespace.pixel_sums, namespace.pixel_counts, \
                    namespace.offset = _update_reference(
                        namespace.pixel_sums, namespace.pixel_counts,
                        namespace.offset, [p, 0, 0], np.expand_dims(plane, 0))
        if any_check:
            # recompute reference using all aligned images
            with lock:
                p_sums = namespace.pixel_sums[p]
                p_counts = namespace.pixel_counts[p]
                offset = namespace.offset
                min_shift = namespace.min_shift
                max_shift = namespace.max_shift
            with warnings.catch_warnings():  # ignore divide by 0
                warnings.simplefilter("ignore")
                reference = old_div(p_sums, p_counts)
            if method == 'correlation':
                if max_displacement is not None and np.all(
                        np.array(max_displacement) >= 0):
                    displacement_bounds = offset + np.array(
                        [np.minimum(max_shift - max_displacement, min_shift),
                         np.maximum(min_shift + max_displacement, max_shift) +
                         1], dtype=int)
                else:
                    displacement_bounds = None
                shift = pyramid_align(np.expand_dims(reference, 0),
                                      np.expand_dims(plane, 0),
                                      bounds=displacement_bounds,
                                      **method_kwargs)
                if displacement_bounds is not None and shift is not None:
                    assert np.all(shift >= displacement_bounds[0])
                    assert np.all(shift <= displacement_bounds[1])
                    assert np.all(abs(shift - offset) <= max_displacement)
            elif method == 'ECC':
                raise NotImplementedError
                # cv2.findTransformECC(reference, plane)
            else:
                raise ValueError('Unrecognized alignment method')
            with lock:
                s = namespace.shifts
                if shift is None:  # if no shift could be calculated
                    try:
                        shift = s[cycle_idx][frame_idx - 1][p] + offset
                    except IndexError:
                        shift = s[cycle_idx][frame_idx][p] + offset
                s[cycle_idx][frame_idx][p][:] = shift - offset
                namespace.shifts = s

            with lock:
                shift = namespace.shifts[cycle_idx][frame_idx][p]
                namespace.pixel_sums, namespace.pixel_counts, \
                    namespace.offset = _update_reference(
                        namespace.pixel_sums, namespace.pixel_counts,
                        namespace.offset, [p] + list(shift)[1:],
                        np.expand_dims(plane, 0))
                namespace.min_shift = np.minimum(shift, min_shift)
                namespace.max_shift = np.maximum(shift, max_shift)


def _update_reference(sums, counts, offset, displacement, image):
    displacement = np.array(displacement)
    sums = _resize_array(sums, displacement + offset, image.shape)
    counts = _resize_array(counts, displacement + offset, image.shape)
    offset = np.maximum(offset, -displacement)
    sums, counts = _update_sums_and_counts(
        sums, counts, offset, displacement, image)
    return sums, counts, offset


def _update_sums_and_counts(
        pixel_sums, pixel_counts, offset, shift, image):
    """Updates pixel sums and counts of the reference image each frame

    >>> from sima.motion.frame_align import _update_sums_and_counts
    >>> import numpy as np
    >>> pixel_counts = np.zeros((4, 5, 5, 2))
    >>> pixel_sums = np.zeros((4, 5, 5, 2))
    >>> plane = 2 * np.ones((1, 2, 3, 2))
    >>> pixel_sums, pixel_counts = _update_sums_and_counts(
    ...     pixel_sums, pixel_counts, [0, 0, 0], [3, 1, 2], plane)
    >>> np.all(pixel_sums[3, 1:3, 2:5] == 2)
    True
    >>> np.all(pixel_counts[3, 1:3, 2:5] == 1)
    True

    """
    assert pixel_sums.ndim == 4
    assert pixel_sums.ndim == image.ndim
    offset = np.array(offset)
    shift = np.array(shift)
    disp = offset + shift
    motion.add_with_offset(pixel_sums, np.nan_to_num(image), disp)
    motion.add_with_offset(pixel_counts, np.isfinite(image), disp)
    assert pixel_sums.ndim == 4
    return pixel_sums, pixel_counts


def _resize_array(array, displacement, frame_shape):
    """Enlarge storage arrays if necessary.

    >>> from sima.motion.frame_align import _resize_array
    >>> import numpy as np
    >>> a = np.ones((2, 128, 128, 5))
    >>> a = _resize_array(a, (0, -1, 2), (2, 128, 128, 5))
    >>> a.shape == (2, 129, 130, 5)
    True

    """
    pad_width = np.zeros((len(array.shape), 2))
    pad_width[:3, 0] = - np.minimum(0, displacement)
    pad_width[:3, 1] = np.maximum(
        0, np.array(displacement) + np.array(frame_shape[:-1]) -
        np.array(array.shape[:-1]))
    if np.any(pad_width):
        array = np.pad(array, tuple(tuple(x) for x in pad_width.astype(int)),
                       'constant')
    return array


class VolumeTranslation(motion.MotionEstimationStrategy):

    """Translate 3D volumes to maximize the correlation.

    Parameters
    ----------
    max_displacement : array of int, optional
        The maximum allowed displacement magnitudes in [z,y,x]. By
        default, arbitrarily large displacements are allowed.
    criterion : float, optional
        The number of standard deviations below the mean correlation that
        a frame's correlation can have following displacement for the
        displacement to be considered valid. Invalid displacements will be
        masked.

    """

    def __init__(self, max_displacement=None, criterion=None):
        if not (criterion is None or
                isinstance(criterion, (int, int, float))):
            raise ValueError('Criterion must be a number')
        self._params = dict(locals())
        del self._params['self']

    def _estimate(self, dataset):
        reference = next(iter(next(iter(dataset))))
        sums = np.zeros_like(reference)
        counts = np.zeros_like(reference)
        offset = np.zeros(3, dtype=int)
        displacements = []
        correlations = []
        disp_range = np.array([[0, 0, 0], [0, 0, 0]])
        for sequence in dataset:
            seq_displacements = []
            seq_correlations = []
            for frame in sequence:
                if self._params['max_displacement'] is not None:
                    bounds = np.array([
                        np.minimum(
                            disp_range[1] - self._params['max_displacement'],
                            disp_range[0]),
                        np.maximum(
                            disp_range[0] + self._params['max_displacement'],
                            disp_range[1])]) + offset
                else:
                    bounds = None
                displacement = pyramid_align(
                    reference, frame, bounds=bounds) - offset
                seq_displacements.append(displacement)
                disp_range[0] = np.minimum(disp_range[0], displacement)
                disp_range[1] = np.maximum(disp_range[1], displacement)
                sums, counts, offset = _update_reference(
                    sums, counts, offset, displacement, frame)
                if self._params['criterion'] is not None:
                    seq_correlations.append(
                        shifted_corr(reference, frame, offset + displacement))
                reference = old_div(sums, counts)
            displacements.append(np.array(seq_displacements))
            correlations.append(np.array(seq_correlations))
        if self._params['criterion'] is not None:
            threshold = np.concatenate(correlations).mean() - \
                self._params['criterion'] * np.std(
                    np.concatenate(correlations))
            for seq_idx, seq_correlations in enumerate(correlations):
                if np.any(seq_correlations < threshold):
                    displacements[seq_idx] = np.ma.array(
                        displacements[seq_idx],
                        mask=np.outer(seq_correlations < threshold,
                                      np.ones(3)))
        assert np.all(
            np.all(x is np.ma.masked for x in shift) or
            not np.any(x is np.ma.masked for x in shift)
            for shift in it.chain.from_iterable(displacements))
        return displacements


def shifted_corr(reference, image, displacement):
    """Calculate the correlation between the reference and the image shifted
    by the given displacement.

    Parameters
    ----------
    reference : np.ndarray
    image : np.ndarray
    displacement : np.ndarray

    Returns
    -------
    correlation : float

    """

    ref_cuts = np.maximum(0, displacement)
    ref = reference[ref_cuts[0]:, ref_cuts[1]:, ref_cuts[2]:]
    im_cuts = np.maximum(0, -displacement)
    im = image[im_cuts[0]:, im_cuts[1]:, im_cuts[2]:]
    s = np.minimum(im.shape, ref.shape)
    ref = ref[:s[0], :s[1], :s[2]]
    im = im[:s[0], :s[1], :s[2]]
    ref -= nanmean(ref.reshape(-1, ref.shape[-1]), axis=0)
    ref = np.nan_to_num(ref)
    im -= nanmean(im.reshape(-1, im.shape[-1]), axis=0)
    im = np.nan_to_num(im)
    assert np.all(np.isfinite(ref)) and np.all(np.isfinite(im))
    corr = nanmean(
        [old_div(np.sum(i * r), np.sqrt(np.sum(i * i) * np.sum(r * r))) for
         i, r in zip(np.rollaxis(im, -1), np.rollaxis(ref, -1))])
    return corr


def pyr_down_3d(image, axes=None):
    """Downsample an image along the specified axes.

    Parameters
    ----------
    image : ndarray
        The image to be downsampled.
    axes : tuple of int
        The axes along which the downsampling is to occur.  Defaults to
        downsampling on all axes.

    """

    stdevs = [1.05 if i in axes else 0 for i in range(image.ndim)]
    filtered_image = scipy.ndimage.filters.gaussian_filter(image, stdevs)
    slices = tuple(slice(None, None, 2) if i in axes else slice(None)
                   for i in range(filtered_image.ndim))
    return filtered_image[slices]


def base_alignment(reference, target, bounds=None):
    return align_cross_correlation(reference, target, bounds)[0]


def within_bounds(displacement, bounds):
    if bounds is None:
        return True
    assert len(displacement) == bounds.shape[1]
    return np.all(bounds[0] <= displacement) and \
        np.all(bounds[1] > displacement)


def pyramid_align(reference, target, min_shape=32, max_levels=None,
                  bounds=None):
    """
    Parameters
    ----------
    min_shape : int or tuple of int
    bounds : ndarray of int
        Shape: (2, D).

    """

    if max_levels is None:
        max_levels = np.inf
    assert bounds is None or np.all(bounds[0] < bounds[1])
    smallest_shape = np.minimum(reference.shape[:-1], target.shape[:-1])
    axes_bool = smallest_shape >= 2 * np.array(min_shape)
    if max_levels > 0 and np.any(axes_bool):
        axes = np.nonzero(axes_bool)[0]
        new_bounds = None if bounds is None else old_div(
            bounds, (1 + axes_bool))

        if bounds is None:
            new_bounds = None
        else:
            new_bounds = np.empty(bounds.shape, dtype=int)
            new_bounds[0] = np.floor(
                old_div(bounds[0].astype(float), (1 + axes_bool)))
            new_bounds[1] = np.ceil(
                old_div(bounds[1].astype(float), (1 + axes_bool)))

        disp = pyramid_align(pyr_down_3d(reference, axes),
                             pyr_down_3d(target, axes),
                             min_shape, max_levels - 1, new_bounds)
        if disp is None:
            return disp
        best_corr = -np.inf
        best_displacement = None
        for adjustment in it.product(
                *[list(range(-1, 2)) if a else list(range(1))
                  for a in axes_bool]):
            displacement = (1 + axes_bool) * disp + np.array(adjustment)
            if within_bounds(displacement, bounds):
                corr = shifted_corr(reference, target, displacement)
                if corr > best_corr:
                    best_corr = corr
                    best_displacement = displacement
        if best_displacement is None:
            warnings.warn('Could not align all frames.')
        return best_displacement
    else:
        return base_alignment(reference, target, bounds)
