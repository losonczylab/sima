from distutils.version import LooseVersion

import numpy as np
from scipy import ndimage
from scipy.ndimage import measurements
from skimage.filter import threshold_otsu
try:
    import cv2
except ImportError:
    cv2_available = False
else:
    cv2_available = LooseVersion(cv2.__version__) >= LooseVersion('2.4.8')

import sima.misc
from .segment import (
    SegmentationStrategy, ROIFilter, CircularityFilter, PostProcessingStep)
from .normcut import BasicAffinityMatrix, PlaneNormalizedCuts
from .segment import _check_single_plane
from sima.ROI import ROI, ROIList


class CA1PCNucleus(PostProcessingStep):
    """Return ROI structures containing CA1 pyramidal cell somata.

    Parameters
    ----------
    cuts : list of sima.normcut.CutRegion
        The segmented regions identified by normalized cuts.
    circularity_threhold : float
        ROIs with circularity below threshold are discarded. Default: 0.5.
    min_roi_size : int, optional
        ROIs with fewer than min_roi_size pixels are discarded. Default: 20.
    min_cut_size : int, optional
        No ROIs are made from cuts with fewer than min_cut_size pixels.
        Default: 30.
    channel : int, optional
        The index of the channel to be used.
    x_diameter : int, optional
        The estimated x-diameter of the nuclei in pixels
    y_diameter : int, optional
        The estimated y_diameter of the nuclei in pixels

    Returns
    -------
    sima.ROI.ROIList
        ROI structures each corresponding to a CA1 pyramidal cell soma.
    """

    def __init__(self, channel=0, x_diameter=8, y_diameter=None):
        if y_diameter is None:
            y_diameter = x_diameter
        self._channel = channel
        self._x_diameter = x_diameter
        self._y_diameter = y_diameter

    def apply(self, rois, dataset):
        channel = sima.misc.resolve_channels(self._channel,
                                             dataset.channel_names)
        processed_im = _processed_image_ca1pc(
            dataset, channel, self._x_diameter, self._y_diameter)[0]
        shape = processed_im.shape[:2]
        ROIs = ROIList([])
        for roi in rois:
            roi_indices = np.nonzero(roi.mask[0])
            roi_indices = np.ravel_multi_index(roi_indices, shape)

            # pixel values in the cut
            vals = processed_im.flat[roi_indices]

            # indices of those values below the otsu threshold
            # if all values are identical, continue without adding an ROI
            try:
                roi_indices = roi_indices[vals < threshold_otsu(vals)]
            except ValueError:
                continue

            # apply binary opening and closing to the surviving pixels
            # expand the shape by 1 in all directions to correct for edge
            # effects of binary opening/closing
            twoD_indices = [np.unravel_index(x, shape) for x in roi_indices]
            mask = np.zeros([x + 2 for x in shape])
            for indices in twoD_indices:
                mask[indices[0] + 1, indices[1] + 1] = 1
            mask = ndimage.binary_closing(ndimage.binary_opening(mask))
            mask = mask[1:-1, 1:-1]
            roi_indices = np.where(mask.flat)[0]

            # label blobs in each cut
            labeled_array, num_features = measurements.label(mask)
            for feat in range(num_features):
                blob_inds = np.where(labeled_array.flat == feat + 1)[0]

                twoD_indices = [np.unravel_index(x, shape) for x in blob_inds]
                mask = np.zeros(shape)
                for x in twoD_indices:
                    mask[x] = 1

                ROIs.append(ROI(mask=mask))

        return ROIs


def _clahe(image, x_tile_size=10, y_tile_size=10, clip_limit=20):
    """Perform contrast limited adaptive histogram equalization (CLAHE)."""
    if not cv2_available:
        raise ImportError('OpenCV >= 2.4.8 required')
    transform = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=(
                                    int(image.shape[1] / float(x_tile_size)),
                                    int(image.shape[0] / float(y_tile_size))))
    return transform.apply(sima.misc.to8bit(image))


def _unsharp_mask(image, mask_weight, image_weight=1.35, sigma_x=10,
                  sigma_y=10):
    """Perform unsharp masking on an image.."""
    if not cv2_available:
        raise ImportError('OpenCV >= 2.4.8 required')
    return cv2.addWeighted(
        sima.misc.to8bit(image), image_weight,
        cv2.GaussianBlur(sima.misc.to8bit(image), (0, 0), sigma_x, sigma_y),
        -mask_weight, 0)


def _processed_image_ca1pc(dataset, channel_idx=-1, x_diameter=10,
                           y_diameter=10):
    """Create a processed image for identifying CA1 pyramidal cell ROIs."""
    unsharp_mask_mask_weight = 0.5
    im = dataset.time_averages[..., channel_idx]
    result = []
    for plane_idx in np.arange(dataset.frame_shape[0]):
        result.append(_unsharp_mask(
            _clahe(im[plane_idx], x_diameter, y_diameter),
            unsharp_mask_mask_weight, 1 + unsharp_mask_mask_weight,
            x_diameter, y_diameter))
    return np.array(result)


class AffinityMatrixCA1PC(BasicAffinityMatrix):

    def __init__(self, channel=0, max_dist=None, spatial_decay=None,
                 num_pcs=75, x_diameter=10, y_diameter=10, verbose=False):
        super(AffinityMatrixCA1PC, self).__init__(
            channel, max_dist, spatial_decay, num_pcs, verbose)
        self._params.x_diameter = x_diameter
        self._params.y_diameter = y_diameter

    def _setup(self, dataset):
        super(AffinityMatrixCA1PC, self)._setup(dataset)
        channel = sima.misc.resolve_channels(self._params.channel,
                                             dataset.channel_names)
        processed_image = _processed_image_ca1pc(
            dataset, channel, self._params.x_diameter,
            self._params.y_diameter)[0]
        time_avg = processed_image
        std = np.std(time_avg)
        time_avg = np.minimum(time_avg, 2 * std)
        self._time_avg = np.maximum(time_avg, -2 * std)
        self._dm = time_avg.max() - time_avg.min()

    def _weight(self, r0, r1):
        w = super(AffinityMatrixCA1PC, self)._weight(r0, r1)
        dy = r1[0] - r0[0]
        dx = r1[1] - r0[1]
        m = -np.Inf
        for xt in range(dx + 1):
            if dx != 0:
                ya = r0[0] + 0.5 + max(0., xt - 0.5) * float(dy) / float(dx)
                yb = r0[0] + 0.5 + min(xt + 0.5, dx) * float(dy) / float(dx)
            else:
                ya = r0[0]
                yb = r1[0]
            ym = int(min(ya, yb))
            yM = int(max(ya, yb))
            for yt in range(ym, yM + 1):
                m = max(m, self._time_avg[yt, r0[1] + xt])
        return w * np.exp(-3. * m / self._dm)


class PlaneCA1PC(SegmentationStrategy):
    """Segmentation method designed for finding CA1 pyramidal cell somata.

    Parameters
    ----------
    channel : int, optional
        The channel whose signals will be used in the calculations.
    num_pcs : int, optional
        The number of principle components to be used in the calculations.
        Default: 75.
    max_dist : tuple of int, optional
        Defaults to (2, 2).
    spatial_decay : tuple of int, optional
        Defaults to (2, 2).
    max_pen : float
        Iterative cutting will continue as long as the cut cost is less
        than max_pen.
    cut_min_size, cut_max_size : int
        Regardless of the cut cost, iterative cutting will not be
        performed on regions with fewer pixels than min_size and will
        always be performed on regions larger than max_size.
    circularity_threhold : float
        ROIs with circularity below threshold are discarded. Default: 0.5.
    min_roi_size : int, optional
        ROIs with fewer than min_roi_size pixels are discarded. Default: 20.
    min_cut_size : int, optional
        No ROIs are made from cuts with fewer than min_cut_size pixels.
        Default: 30.
    x_diameter : int, optional
        The estimated x-diameter of the nuclei in pixels. Default: 8
    y_diameter : int, optional
        The estimated x-diameter of the nuclei in pixels. Default: 8

    Notes
    -----
    This method begins with the normalized cuts procedure. The affinities
    used for normalized cuts also depend on the time-averaged image,
    which is used to increase the affinity between pixels within the
    same dark patch (putative nucleus).

    Following the normalized cut procedure, the algorithm attempts to
    identify a nucleus within each cut.

    * Otsu thresholding of the time-averaged image after processing with
      CLAHE and unsharp mask procedures.
    * Binary opening and closing of the resulting ROIs.
    * Rejection of ROIs not passing the circularity and size requirements.

    See also
    --------
    sima.segment.normcut

    Warning
    -------
    In version 1.0, this method currently only works on datasets with a
    single plane, or in conjunction with
    :class:`sima.segment.PlaneWiseSegmentation`.

    """
    def __init__(
            self, channel=0, num_pcs=75, max_dist=None, spatial_decay=None,
            cut_max_pen=0.01, cut_min_size=40, cut_max_size=200, x_diameter=10,
            y_diameter=10, circularity_threhold=.5, min_roi_size=20,
            min_cut_size=30, verbose=False):
        super(PlaneCA1PC, self).__init__()
        affinity_method = AffinityMatrixCA1PC(
            channel, max_dist, spatial_decay, num_pcs, x_diameter, y_diameter,
            verbose)
        self._normcut_method = PlaneNormalizedCuts(
            affinity_method, cut_max_pen, cut_min_size, cut_max_size)
        self._normcut_method.append(
            ROIFilter(lambda r: r.size >= min_cut_size))
        self._normcut_method.append(
            CA1PCNucleus(channel, x_diameter, y_diameter))
        self._normcut_method.append(
            ROIFilter(lambda r: r.size >= min_roi_size))
        self._normcut_method.append(CircularityFilter(circularity_threhold))

    @_check_single_plane
    def _segment(self, dataset):
        return self._normcut_method.segment(dataset)
