import abc

import numpy as np
from scipy import sparse, ndimage

from sima.ROI import ROI, ROIList, mask2poly


class SegmentationStrategy(object):
    """Abstract segmentation method."""
    __metaclass__ = abc.ABCMeta

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
        return


class PlaneSegmentationStrategy(SegmentationStrategy):
    __metaclass__ = abc.ABCMeta

    def _segment(self, dataset):
        if dataset.frame_shape[0] is not 1:
            raise ValueError('This segmentation strategy requires a '
                             'dataset with exactly one plane.')
        return self._segment_plane(dataset)

    @abc.abstractmethod
    def _segment_plane(self, dataset):
        pass


class PlaneWiseSegmentationStrategy(SegmentationStrategy):
    """Segmentation approach with each plane segmented separately.

    Parameters
    ----------
    plane_strategy : PlaneSegmentationStrategy
        The strategy to be applied to each plane.
    """

    def __init__(self, plane_strategy):
        super(PlaneWiseSegmentationStrategy, self).__init__()
        self.strategy = plane_strategy

    def _segment(self, dataset):
        def set_z(roi, z):
            old_mask = roi.mask
            return ROI(
                mask=[sparse.lil_matrix(old_mask[0].shape, old_mask[0].dtype)
                      for _ in range(z - 1)] + [old_mask[0]])

        rois = ROIList([])
        for plane in range(dataset.frame_shape[0]):
            plane_rois = self.strategy.segment(dataset[:, :, plane])
            for roi in plane_rois:
                set_z(roi, plane)
            rois.extend(plane_rois)
        return rois


def _remove_overlapping(rois, percent_overlap=0.9):
    """ Remove overlapping ROIs

    Parameters
    ----------
    rois : list
        list of sima.ROI ROIs
    percent_overlap : float
        percent of the smaller ROIs total area which must be covered in order
        for the ROIs to be evaluated as overlapping

    Returns
    -------
    rois : list
        A list of sima.ROI ROI objects with the overlapping ROIs combined
    """

    if percent_overlap > 0 and percent_overlap <= 1:
        for roi in rois:
            roi.mask = roi.mask

        for i in xrange(len(rois)):
            for j in [j for j in xrange(len(rois)) if j != i]:
                if rois[i] is not None and rois[j] is not None:
                    overlap = np.logical_and(rois[i].mask.toarray(),
                                             rois[j].mask.toarray())
                    small_area = np.min(
                        (rois[i].mask.size, rois[j].mask.size))

                    if len(np.where(overlap)[0]) > \
                            percent_overlap * small_area:
                        new_shape = np.logical_or(rois[i].mask.toarray(),
                                                  rois[j].mask.toarray())

                        rois[i] = ROI(mask=new_shape.astype('bool'),
                                      im_shape=rois[i].mask.shape)
                        rois[j] = None
    return ROIList(roi for roi in rois if roi is not None)


class PostProcessingStep(object):
    """Post processing step applied to segmented ROIs."""
    __metaclass__ = abc.ABCMeta
    # TODO: method for clearing memory after the step is applied

    @abc.abstractmethod
    def apply(rois, dataset=None):
        """Apply the post-processing step to rois from a dataset.

        Parameters
        ----------
        rois : sima.ROI.ROIList
            The ROIs to be post-processed.
        dataset : sima.ImagingDataset
            The dataset from which the ROIs were segmented.

        Returns
        -------
        sima.ROI.ROIList
            The post-processed ROIs.
        """
        return


class ROIFilter(PostProcessingStep):
    """Filter a set of ROIs.

    Parameters
    ----------
    func : function
        A boolean-valued function taking arguments (rois, dataset)
        that is used to filter the ROIs.
    setup_func : function, optional
        A function with parameters rois, dataset that is called
        to set parameters depending on the dataset.
    """

    def __init__(self, func, setup_func=None):
        self._valid = func
        self._setup = setup_func

    def apply(self, rois, dataset=None):
        if self._setup is not None:
            self._setup(rois, dataset)
        return ROIList([r for r in rois if self._valid(r)])


class ROISizeFilter(ROIFilter):
    """Filter that accepts ROIs based on size.

    Parameters
    ----------
    min_size : int
        The minimum ROI size in pixels.
    max_size : int, optional
        The maximum ROI size in pixels.
    """

    def __init__(self, min_size, max_size=None):
        if min_size is None:
            min_size = 0
        if max_size is None:
            max_size = np.Inf

        def f(roi, dataset=None):
            size = sum(np.count_nonzero(plane.todense()) for plane in roi.mask)
            return not (size > max_size or size < min_size)

        super(ROISizeFilter, self).__init__(f)


class CircularityFilter(ROIFilter):
    """Filter based on circularity of the ROIs.

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
            circle_area = np.square(p) / (4 * np.pi)
            return shape_area / circle_area > circularity_threhold
        super(CircularityFilter, self).__init__(f)


def _smooth_roi(roi, radius=3):
    """ Smooth out the ROI boundaries and reduce the number of points in the
    ROI polygons.

    Parameters
    ----------
    roi : sima.ROI
        ROI object to be smoothed
    radius : initial radius of the smoothing

    Returns
    -------
    roi : sima.ROI
        If successful, an ROI object which have been smoothed, otherwise, the
        original ROI is returned
    success : bool
        True if the smoothing has been successful, False otherwise
    """

    frame = roi.mask[0].todense().copy()

    frame[frame > 0] = 1
    check = frame[:-2, :-2] + frame[1:-1, :-2] + frame[2:, :-2] + \
        frame[:-2, 1:-1] + frame[2:, 1:-1] + frame[:-2:, 2:] + \
        frame[1:-1, 2:] + frame[2:, 2:]
    z = np.zeros(frame.shape)
    z[1:-1, 1:-1] = check

    # initialize and array to hold the new polygon and find the first point
    b = []
    rows, cols = np.where(z > 0)
    p = [cols[0], rows[0]]
    base = p

    # establish an iteration limit to ensue the loop terminates if smoothing
    # is unsuccessful
    limit = 1500

    # store wether the radius of search is increased aboved the initial value
    tmp_rad = False
    for i in range(limit - 1):
        b.append(p)
        # find the ist of all points at the given radius and adjust to be lined
        # up for clockwise traversal
        x = np.roll(np.array(list(p[0] + range(-radius, radius)) +
                             [p[0] + radius] * (2 * radius + 1) +
                             list(p[0] + range(-radius, radius)[::-1]) +
                             [p[0] - (radius + 1)] * (2 * radius + 1)), -2)
        y = np.roll(np.array([p[1] - radius] * (2 * radius) +
                             list(p[1] + range(-radius, radius)) +
                             [p[1] + radius] * (2 * radius + 1) +
                             list(p[1] + range(-radius, (radius + 1))[::-1])),
                    -radius)

        # insure that the x and y points are within the image
        x[x < 0] = 0
        y[y < 0] = 0
        x[x >= z.shape[1]] = z.shape[1] - 1
        y[y >= z.shape[0]] = z.shape[0] - 1

        vals = z[y, x]

        # ensure that the vals array has a valid transition from 0 to 1
        # otherwise the algorithm has failed
        if len(np.where(np.roll(vals, 1) == 0)[0]) == 0 or \
                len(np.where(vals > 0)[0]) == 0:
            return roi, False

        idx = np.intersect1d(np.where(vals > 0)[0],
                             np.where(np.roll(vals, 1) == 0)[0])[0]
        p = [x[idx], y[idx]]

        # check if the traveral is near to the starting point indicating that
        # the algirthm has completed. If less then 3 points are found this is
        # not yet a valid ROI
        if ((p[0] - base[0]) ** 2 + (p[1] - base[1]) ** 2) ** 0.5 < \
                1.5 * radius and len(b) > 3:
            new_roi = ROI(polygons=[b], im_shape=roi.im_shape)
            if new_roi.mask[0].size != 0:
                # "well formed ROI"
                return new_roi, True

        # if p is already in the list of polygon points, increase the radius of
        # search. if radius is already larger then 6, blur the mask and try
        # again
        if p in b:
            if radius > 6:
                radius = 3
                z = ndimage.gaussian_filter(z, sigma=1)

                b = []
                rows, cols = np.where(z > 0)
                p = [cols[0], rows[0]]
                base = p
                tmp_rad = False

            else:
                radius = radius + 1
                tmp_rad = True
                if len(b) > 3:
                    p = b[-3]
                    del b[-3:]

        elif tmp_rad:
            tmp_rad = False
            radius = 3

    # The maximum number of cycles has completed and no suitable smoothed ROI
    # has been determined
    return roi, False


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
