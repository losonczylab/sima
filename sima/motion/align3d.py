try:
    from future_builtins import zip
except ImportError:  # Python 3.x
    pass
import itertools as it

import numpy as np
import scipy.ndimage.filters
try:
    from bottleneck import nanmean
except ImportError:
    from scipy.stats import nanmean

import motion
from sima.misc.align import align_cross_correlation


class VolumeTranslation(motion.MotionEstimationStrategy):
    """Translate 3D volumes to maximize the correlation.

    Parameters
    ----------
    max_displacement : array of int, optional
        The maximum allowed displacement magnitudes in [z,y,x]. By
        default, arbitrarily large displacements are allowed.
    """

    def __init__(self, max_displacement=None):
        self._max_displacement = max_displacement

    def _estimate(self, dataset):
        reference = next(iter(next(iter(dataset))))
        displacements = []
        disp_range = np.array([[0, 0, 0], [0, 0, 0]])
        for sequence in dataset:
            seq_displacements = []
            for frame in sequence:
                if self._max_displacement is not None:
                    bounds = np.array([
                        np.minimum(disp_range[1] - self._max_displacement,
                                   disp_range[0]),
                        np.maximum(disp_range[0] + self._max_displacement,
                                   disp_range[1])])
                else:
                    bounds = None
                seq_displacements.append(
                    pyramid_align(reference, frame, bounds=bounds))
                bounds[0] = np.minimum(bounds[0], seq_displacements[-1])
                bounds[1] = np.maximum(bounds[1], seq_displacements[-1])
            displacements.append(np.array(seq_displacements))
        return displacements


def shifted_corr(reference, image, displacement):
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
    return np.mean([np.sum(i * r) / np.sqrt(np.sum(i * i) * np.sum(r * r))
                    for i, r in zip(np.rollaxis(im, 1), np.rollaxis(ref, 1))])


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
        np.all(bounds[1] >= displacement)


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
    smallest_shape = np.minimum(reference.shape[:-1], target.shape[:-1])
    axes_bool = smallest_shape >= 2 * np.array(min_shape)
    if max_levels > 0 and np.any(axes_bool):
        axes = np.nonzero(axes_bool)[0]
        new_bounds = None if bounds is None else bounds / (1 + axes_bool)
        disp = pyramid_align(pyr_down_3d(reference, axes),
                             pyr_down_3d(target, axes),
                             min_shape, max_levels - 1, new_bounds)
        best_corr = -np.inf
        best_displacement = None
        for adjustment in it.product(range(-1, 2), range(-1, 2), range(-1, 2)):
            displacement = (1 + axes_bool) * disp + np.array(adjustment)
            if within_bounds(displacement, bounds):
                corr = shifted_corr(reference, target, displacement)
                if corr > best_corr:
                    best_corr = corr
                    best_displacement = displacement
        return best_displacement
    else:
        return base_alignment(reference, target, bounds)
