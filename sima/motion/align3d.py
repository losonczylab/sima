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


from sima.misc.align import align_cross_correlation

# for frame in seq:
#     pyramid_align(reference, frame)


class WholeVolumeShifting(object):

    def estimate(self, dataset):
        reference = next(next(iter(dataset)))
        displacements = []
        for sequence in dataset:
            seq_displacements = []
            for frame in sequence:
                seq_displacements.append(pyramid_align(reference, frame))
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
    """
    Parameters
    ----------
    image : ndarray
    axes : tuple of int
        The axes along which the downsampling is to occur
    """
    stdevs = [1.05 if i in axes else 0 for i in range(image.ndim)]
    filtered_image = scipy.ndimage.filters.gaussian_filter(image, stdevs)
    slices = tuple(slice(None, None, 2) if i in axes else slice(None)
                   for i in range(filtered_image.ndim))
    return filtered_image[slices]


def base_alignment(reference, target):
    return align_cross_correlation(reference, target)[0]


def pyramid_align(reference, target, min_shape=32, max_levels=None):
    """
    Parameters
    ----------
    min_shape : int or tuple of int
    """
    if max_levels is None:
        max_levels = np.inf
    smallest_shape = np.minimum(reference.shape[:-1], target.shape[:-1])
    axes_bool = smallest_shape >= 2 * np.array(min_shape)
    if max_levels > 0 and np.any(axes_bool):
        axes = np.nonzero(axes_bool)[0]
        disp = pyramid_align(pyr_down_3d(reference, axes),
                             pyr_down_3d(target, axes),
                             min_shape,
                             max_levels-1)
        best_corr = -np.inf
        best_displacement = None
        for adjustment in it.product(range(-1, 2), range(-1, 2), range(-1, 2)):
            displacement = (1 + axes_bool) * disp + np.array(adjustment)
            corr = shifted_corr(reference, target, displacement)
            if corr > best_corr:
                best_corr = corr
                best_displacement = displacement
        return best_displacement
    else:
        return base_alignment(reference, target)
