from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
from builtins import object
import abc
import itertools as it
from multiprocessing import Pool

import numpy as np
from scipy import sparse, ndimage
from skimage.measure import approximate_polygon

from sima.ROI import ROI, ROIList, mask2poly
from future.utils import with_metaclass


class SegmentationStrategy(with_metaclass(abc.ABCMeta, object)):

    """Abstract class implementing the interface for segmentation strategies.

    This class can be subclassed to create a concrete segmentation
    strategy by implementing a :func:`_segment()` method. As well,
    any existing segmentation strategy can be extended by adding a
    :class:`PostProcessingStep` with the :func:`append` method.

    """

    def __init__(self):
        self._post_processing_steps = []

    def segment(self, dataset):
        """Apply the segmentation method to a dataset.

        Parameters
        ----------
        dataset : ImagingDataset
            The dataset whose affinity matrix is being calculated.
            return self._segment(dataset)

        Returns
        -------
        rois : sima.ROI.ROIList
            A list of sima.ROI ROI objects.
        """
        rois = self._segment(dataset)
        for step in self._post_processing_steps:
            rois = step.apply(rois, dataset)
        return rois

    def append(self, post_processing_step):
        """Add a post processing step.

        Parameters
        ----------
        post_processing_step : PostProcessingStep

        """
        self._post_processing_steps.append(post_processing_step)

    @abc.abstractmethod
    def _segment(self, dataset):
        """Implementation of the segmentation strategy.

        This abstract method must be implemented by any subclass of
        SegmentationStrategy.

        Parameters
        ----------
        dataset : sima.ImagingDataset
            The dataset to be segmented.

        Returns
        -------
            ROIs : sima.ROI.ROIList
                The segmented regions of interest.
        """
        raise NotImplementedError


def _check_single_plane(func):
    """Decorator to check that dataset has a single plane"""

    def checked_func(self, dataset):
        if dataset.frame_shape[0] != 1:
            raise ValueError('This segmentation strategy requires a '
                             'dataset with exactly one plane.')
        return func(self, dataset)
    return checked_func


class PlaneWiseSegmentation(SegmentationStrategy):

    """Segmentation approach with each plane segmented separately.

    Parameters
    ----------
    plane_strategy : SegmentationStrategy or list of SegmentationStrategy
        The strategies to be applied to each plane.

    Examples
    --------
    One use is to apply the same segmentation method to multiple layers.
    For example, the user may wish to apply the PlaneCA1PC strategy separately
    to multiple well-separated planes in which distinct cell bodies are imaged:

    >>> from sima.segment import PlaneWiseSegmentation, PlaneCA1PC
    >>> layer_strategy = PlaneCA1PC()
    >>> strategy = PlaneWiseSegmentation(layer_strategy)

    Alternatively, one may wish to use different segmentation strategies on
    each plane. For example, to segment one plane of dendrites with stICA and
    one plane of cell bodies with the CA1PC strategy, a PlaneWiseSegmentation
    strategy can be created as follows:

    >>> from sima.segment import PlaneWiseSegmentation, PlaneCA1PC, STICA
    >>> strategy = PlaneWiseSegmentation([STICA(), PlaneCA1PC()])

    """

    def __init__(self, plane_strategy):
        super(PlaneWiseSegmentation, self).__init__()
        self.strategy = plane_strategy

    def _segment(self, dataset):
        def set_z(roi, z):
            old_mask = roi.mask
            roi.mask = [
                sparse.lil_matrix(old_mask[0].shape, dtype=old_mask[0].dtype)
                for _ in range(z)] + [old_mask[0]]

        rois = ROIList([])
        if isinstance(self.strategy, list):
            if len(self.strategy) != dataset.frame_shape[0]:
                raise Exception('There is not exactly one strategy per plane.')
            iterator = list(
                zip(self.strategy, list(range(dataset.frame_shape[0]))))
        elif isinstance(self.strategy, SegmentationStrategy):
            iterator = list(zip(it.repeat(self.strategy),
                                list(range(dataset.frame_shape[0]))))

        for strategy, plane_idx in iterator:
            plane_rois = strategy.segment(dataset[:, :, plane_idx])
            for roi in plane_rois:
                set_z(roi, plane_idx)
            rois.extend(plane_rois)
        return rois


class PostProcessingStep(with_metaclass(abc.ABCMeta, object)):

    """Abstract class representing the interface for post processing
    steps that can be appended to a segmentation method to modify the
    the segmented ROIs.

    Examples
    --------
    To apply a binary opening to all ROIs, a subclass of PostProcessingStep can
    be created as follows:

    >>> from scipy import ndimage
    >>> import sima.segment
    >>> class BinaryOpening(sima.segment.PostProcessingStep):
    ...     def apply(self, rois, dataset=None):
    ...         for r in rois:
    ...             r.mask = ndimage.binary_opening(r.mask)
    ...         return rois

    We can then append an instance of this class to any segmentation strategy.

    >>> strategy = sima.segment.STICA()
    >>> strategy.append(BinaryOpening())

    """
    # TODO: method for clearing memory after the step is applied

    @abc.abstractmethod
    def apply(self, rois, dataset=None):
        """Apply the post-processing step to rois from a dataset.

        Parameters
        ----------
        rois : sima.ROI.ROIList
            The ROIs to be post-processed.
        dataset : sima.ImagingDataset
            The dataset from which the ROIs were segmented.

        Returns
        -------
        ROIs : sima.ROI.ROIList
            The post-processed ROIs.
        """
        return


class ROIFilter(PostProcessingStep):

    """Post-processing step for generic filtering of ROIs.

    ROIs produced by the segmentation are filtered to retain
    only the ROIs that cause the specified function to evaluate
    to be True.

    Parameters
    ----------
    func : function
        A boolean-valued function taking arguments (rois, dataset)
        that is used to filter the ROIs.

    Example
    -------

    To select ROIs with a size (i.e. number of non-zero pixels) of at least 20
    pixels and no more than 50 pixels, the following ROIFilter can be created:

    >>> from sima.segment import ROIFilter
    >>> size_filter = ROIFilter(lambda roi: roi.size >= 20 and roi.size <= 50)

    """

    def __init__(self, func):
        self._valid = func

    def apply(self, rois, dataset=None):
        return ROIList([r for r in rois if self._valid(r)])


class CircularityFilter(ROIFilter):

    """Post-processing step to filter ROIs based on circularity.

    Parameters
    ----------
    circularity_threhold : float, optional
        ROIs with circularity below threshold are discarded. Default: 0.5.
        Range: 0 to 1.
    """

    def __init__(self, circularity_threhold=0.5):
        def f(roi, dataset=None):
            mask = roi.mask[0].todense()
            poly_pts = np.array(mask2poly(mask)[0].exterior.coords)
            p = 0
            for x in range(len(poly_pts) - 1):
                p += np.linalg.norm(poly_pts[x] - poly_pts[x + 1])
            shape_area = np.count_nonzero(mask)
            circle_area = old_div(np.square(p), (4 * np.pi))
            return old_div(shape_area, circle_area) > circularity_threhold
        super(CircularityFilter, self).__init__(f)


class MergeOverlapping(PostProcessingStep):
    """Post-processing step to merge overlapping ROIs.

    Parameters
    ----------
    threshold : float
        Minimum percent of the smaller ROIs total area which must be covered
        in order for the ROIs to be evaluated as overlapping.

    """
    def __init__(self, threshold):
        if threshold < 0. or threshold > 1.:
            raise ValueError('percent_overlap must be in the interval [0,1]')
        self.percent_overlap = threshold

    def apply(self, rois, dataset=None):
        """ Remove overlapping ROIs

        Parameters
        ----------
        rois : list
            list of sima.ROI ROIs
        percent_overlap : float
            percent of the smaller ROIs total area which must be covered in
            order for the ROIs to be evaluated as overlapping

        Returns
        -------
        rois : list
            A list of sima.ROI ROI objects with the overlapping ROIs combined
        """

        for roi in rois:
            roi.mask = roi.mask

        for i in range(len(rois)):  # TODO: more efficient strategy
            for j in [j for j in range(len(rois)) if j != i]:
                if rois[i] is not None and rois[j] is not None:
                    overlap = np.logical_and(rois[i], rois[j])
                    small_area = min(np.size(rois[i]), np.size(rois[j]))

                    if len(np.where(overlap)[0]) > \
                            self.percent_overlap * small_area:
                        new_shape = np.logical_or(rois[i], rois[j])

                        rois[i] = ROI(mask=new_shape.astype('bool'))
                        rois[j] = None
        return ROIList(roi for roi in rois if roi is not None)


class _FilterParallel(object):
    """
    Helper class to parallelize ROI smoothing

    Parameters
    ----------
    threshold : float
        threshold on gradient measures to cut off
    x_smoothing : int
        number of times to apply Gaussian blur smoothing process to
        each component.
    """
    def __init__(self, threshold, x_smoothing):
        self.threshold = threshold
        self.x_smoothing = x_smoothing

    def __call__(self, roi):

        roi_static = []
        roi_cpy = np.array(roi)
        for frame in roi_cpy:
            # copy the component, remove pixels with low weights
            frame[frame < 2 * np.std(frame)] = 0

            # smooth the component via static removal and gaussian blur
            for _ in range(self.x_smoothing):
                check = frame[1:-1, :-2] + frame[1:-1, 2:] + \
                    frame[:-2, 1:-1] + frame[2, 1:-1]
                z = np.zeros(frame.shape)
                z[1:-1, 1:-1] = check
                frame[np.logical_not(z)] = 0

                blurred = ndimage.gaussian_filter(frame, sigma=1)
                frame = blurred + frame

                frame = frame / np.max(frame)
                frame[frame < 2 * np.std(frame)] = 0

            # calculate the remaining static in the component
            static = np.sum(np.abs(frame[1:-1, 1:-1] - frame[:-2, 1:-1])) + \
                np.sum(np.abs(frame[1:-1, 1:-1] - frame[2:, 1:-1])) + \
                np.sum(np.abs(frame[1:-1, 1:-1] - frame[1:-1, :-2])) + \
                np.sum(np.abs(frame[1:-1, 1:-1] - frame[1:-1, 2:])) + \
                np.sum(np.abs(frame[1:-1, 1:-1] - frame[2:, 2:])) + \
                np.sum(np.abs(frame[1:-1, 1:-1] - frame[:-2, 2:])) + \
                np.sum(np.abs(frame[1:-1, 1:-1] - frame[2:, :-2])) + \
                np.sum(np.abs(frame[1:-1, 1:-1] - frame[:-2, :-2]))

            static = np.sum(static) * 2.0 / (frame.shape[0] * frame.shape[1])
            roi_static.append(static)

        # decide if the component should be accepted or rejected
        if np.mean(static) < self.threshold:
            return ROI(mask=roi_cpy), ROI(mask=roi), None
        else:
            return None, None, ROI(mask=roi)


class SparseROIsFromMasks(PostProcessingStep):
    """PostProcessingStep to extract sparse ROIs from masks.

    Parameters
    ----------
    min_size : int, optional
        The minimum size in number of pixels that an ROI can be. Default: 50.
    static_threshold : float, optional
        threshold on the static allowable in an ICA components, eliminating
        high scoring components speeds the ROI extraction and may improve
        the results. Default: 0.5
    smooth_size : int, optional
        number of iterations of static removal and Gaussian blur to
        perform on each stICA component. 0 provides no Gaussian blur,
        larger values produce stICA components with less static but the
        ROIs lose definition. Default: 4
    sign_split : bool, optional
        Whether to split each mask into its positive and negative components.
    n_processes : int, optional
        Number of processes to farm out the roi smoothing across.
        Should be at least 1 and at most one less than the number
        of CPUs in the computer. Defaults to 1.

    """
    def __init__(self, min_size=50, static_threshold=0.5, smooth_size=4,
                 sign_split=True, n_processes=1):
        self.min_size = min_size
        self.static_threshold = static_threshold
        self.smooth_size = smooth_size
        self.sign_split = sign_split
        self.n_processes = n_processes

    def apply(self, rois, dataset=None):
        smoothed, _, _ = SparseROIsFromMasks._find_and_smooth(
            rois, self.static_threshold, self.smooth_size, self.sign_split,
            self.n_processes)

        return ROIList(SparseROIsFromMasks._extract_st_rois(smoothed,
                                                            self.min_size))

    @staticmethod
    def _extract_st_rois(input_rois, min_area):
        """ Extract ROIs from the spatio-temporal components

        Parameters
        ----------
        input_rois : ROIList
            list of ROIs
        min_area : int
            The minimum size in number of pixels that an ROI can be.

        Returns
        -------
        rois : list
            A list of sima.ROI ROI objects
        """
        rois = []
        for frame in input_rois:
            img = np.array(frame)
            img[np.where(img > 0)] = 1
            img, seg_count = ndimage.measurements.label(img)
            for i in range(seg_count):
                segment = np.where(img == i + 1)
                if segment[0].size >= min_area:
                    thisroi = np.zeros(img.shape, 'bool')
                    thisroi[segment] = True
                    rois.append(ROI(mask=thisroi, im_shape=thisroi.shape))
        return rois

    @staticmethod
    def _find_and_smooth(rois, threshold, x_smoothing=4, sign_split=True,
                         n_processes=1):
        """ finds ICA components with axons and brings them to the foreground

        Parameters
        ----------
        rois : ROIList
        threshold : float
            threshold on gradient measures to cut off
        x_smoothing : int
            number of times to apply Gaussian blur smoothing process to
            each component. Default: 4
        sign_split : bool, optional
            Whether to split each mask into its positive and negative
            components.
        n_processes : int, optional
            Number of processes to farm out the roi smoothing across.
            Should be at least 1 and at most one less than the number
            of CPUs in the computer. Defaults to 1.

        Returns
        -------
        accepted : list
            ROIs which contain axons have been processed.
        accepted_components : list
            ROIs found to contain axons but without image processing applied
        rejected : list
            ROIs that are determined to have no axon information in them
        """
        def split_by_sign(roi):
            arr = np.array(roi)
            return [np.maximum(arr, 0), np.maximum(-arr, 0)]

        if sign_split:
            rois = sum((split_by_sign(r) for r in rois), [])
        else:
            rois = [np.abs(r) for r in rois]

        FilterFunc = _FilterParallel(threshold, x_smoothing)
        if n_processes > 1:
            pool = Pool(processes=n_processes)
            smooth_results = pool.map(FilterFunc, rois)
            pool.close()
        else:
            smooth_results = map(FilterFunc, rois)

        accepted = [res[0] for res in smooth_results if res[0] is not None]
        accepted_components = [res[1] for res in smooth_results
                               if res[1] is not None]
        rejected = [res[2] for res in smooth_results if res[2] is not None]

        return accepted, accepted_components, rejected


class _SmoothBoundariesParallel(object):
    """ Smooth out the ROI boundaries and reduce the number of points in
    the ROI polygons. Helper class for parallelization.

    Parameters
    ----------
    tolerance : float
        Maximum distance from original points of polygon to approximated
        polygonal chain. If tolerance is 0, the original roi coordinates array
        are returned. (See skimage.measure.approximate_polygon).
    min_verts : int
        Minimum number of verticies an ROI polygon can have prior to attemting
        smoothing


    Returns
    -------
    roi : sima.ROI
        An ROI object which has been smoothed.
    """
    def __init__(self, tolerance, min_verts):
        self.tolerance = tolerance
        self.min_verts = min_verts

    def __call__(self, roi):
        smoothed_polygons = []
        coords = roi.coords
        for polygon in coords:
            if polygon.shape[0] > self.min_verts:
                plane = polygon[0, -1]
                smoothed_coords = approximate_polygon(polygon[:, :2],
                                                      self.tolerance)
                smoothed_coords = np.hstack(
                    (smoothed_coords, plane * np.ones(
                        (smoothed_coords.shape[0], 1))))

                if smoothed_coords.shape[0] < self.min_verts:
                    smoothed_coords = polygon

            else:
                smoothed_coords = polygon

            smoothed_polygons += [smoothed_coords]

        return ROI(polygons=smoothed_polygons, im_shape=roi.im_shape)


class SmoothROIBoundaries(PostProcessingStep):
    """
    Post-processing step to smooth ROI boundaries.

    Reduces the number of points in the ROI polygons.

    Parameters
    ----------
    tolerance : float
        Maximum distance from original points of polygon to approximated
        polygonal chain. If tolerance is 0, the original roi coordinates array
        are returned. (See skimage.measure.approximate_polygon).
        Default is 0.5.
    min_verts : int
        Minimum number of verticies an ROI polygon can have prior to attemting
        smoothing. Defaults is 8.
    n_processes : int, optional
        Number of processes to farm out the roi smoothing across.
        Should be at least 1 and at most one less than the number
        of CPUs in the computer. Defaults to 1.

    """

    def __init__(self, tolerance=0.5, min_verts=8, n_processes=1):
        self.tolerance = tolerance
        self.min_verts = min_verts
        self.n_processes = n_processes

    def apply(self, rois, dataset=None):
        SmoothFunc = _SmoothBoundariesParallel(self.tolerance, self.min_verts)
        if self.n_processes > 1:
            pool = Pool(processes=self.n_processes)
            smooth_rois = pool.map(SmoothFunc, rois)
            pool.close()
        else:
            smooth_rois = map(SmoothFunc, rois)

        return ROIList(smooth_rois)
