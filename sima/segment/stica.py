import os

import numpy as np
from scipy import ndimage
from scipy.ndimage import measurements
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean

import sima.misc
from .segment import (
    SegmentationStrategy,
    _remove_overlapping,
    _smooth_roi,
    _check_single_plane,
)
from .normcut import _OPCA
from sklearn.decomposition import FastICA
from sima.ROI import ROI, ROIList


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def _stica(space_pcs, time_pcs, mu=0.01, n_components=30, path=None):
    """Perform spatio-temporal ICA given spatial and temporal Principal
    Components

    Parameters
    ----------
    space_pcs : array
        The spatial representations of the PCs.
        Shape: (num_rows, num_columns, num_pcs).
    time_pcs : array
        The temporal representations of the PCs.
        Shape: (num_times, num_pcs).
    mu : float
        Weighting parameter for the trade off between spatial and temporal
        information. Must be between 0 and 1. Low values give higher weight
        to temporal information. Default: 0.01
    n_components : int
        The maximum number of ICA components to generate. Default: 30
    path : str
        Directory for saving or loading stICA results.

    Returns
    -------
    st_components : array
        stICA components
        Shape: (num_rows, num_columns, n_components)
    """

    # attempt to retrive the stICA data from a save file
    ret = None
    if path is not None:
        try:
            data = np.load(path)
        except IOError:
            pass
        else:
            if data['st_components'].shape[2] == n_components and \
                    data['mu'].item() == mu and \
                    data['num_pcs'] == time_pcs.shape[1]:
                ret = data['st_components']
            data.close()

    if ret is not None:
        return ret

    # preprocess the PCA data
    for i in range(space_pcs.shape[2]):
        space_pcs[:, :, i] = mu * \
            (space_pcs[:, :, i] -
             nanmean(space_pcs[:, :, i])) / np.max(space_pcs)
    for i in range(time_pcs.shape[1]):
        time_pcs[:, i] = (1 - mu) * \
            (time_pcs[:, i] - nanmean(time_pcs[:, i])) / np.max(time_pcs)

    # concatenate the space and time PCs
    y = np.concatenate((space_pcs.reshape(
        space_pcs.shape[0] * space_pcs.shape[1],
        space_pcs.shape[2]), time_pcs))

    # execute the FastICA algorithm
    ica = FastICA(n_components=n_components, max_iter=1500)
    st_components = np.real(np.array(ica.fit_transform(y)))

    # pull out the spacial portion of the st_components
    st_components = \
        st_components[:(space_pcs.shape[0] * space_pcs.shape[1]), :]
    st_components = st_components.reshape(space_pcs.shape[0],
                                          space_pcs.shape[1],
                                          st_components.shape[1])

    # normalize the ica results
    for i in range(st_components.shape[2]):
        st_component = st_components[:, :, i]
        st_component = abs(st_component - np.mean(st_component))
        st_component = st_component / np.max(st_component)
        st_components[:, :, i] = st_component

    # save the ica components if a path has been provided
    if path is not None:
        np.savez(path, st_components=st_components, mu=mu,
                 num_pcs=time_pcs.shape[1])

    return st_components


def _find_useful_components(st_components, threshold, x_smoothing=4):
    """ finds ICA components with axons and brings them to the foreground

    Parameters
    ----------
    st_components : array
        stICA components
        Shape: (num_rows, num_columns, n_components)
    threshold : float
        threshold on gradient measures to cut off
    x_smoothing : int
        number of times to apply gaussiian blur smoothing process to
        each component. Default: 4

    Returns
    -------
    accepted : list
        stICA components which contain axons have been processed
        Shape: n_components
    accepted_components : list
        stICA components which are found to contain axons but without image
        processing applied
    rejected : list
        stICA components that are determined to have no axon information
        in them
    """

    accepted = []
    accepted_components = []
    rejected = []
    for i in xrange(st_components.shape[2]):

        # copy the component, remove pixels with low weights
        frame = st_components[:, :, i].copy()
        frame[frame < 2 * np.std(frame)] = 0

        # smooth the component via static removal and gaussian blur
        for n in xrange(x_smoothing):
            check = frame[1:-1, :-2] + frame[1:-1, 2:] + frame[:-2, 1:-1] + \
                frame[2, 1:-1]
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

        static = static * 2.0 / (frame.shape[0] * frame.shape[1])

        # decide if the component should be accepted or rejected
        if np.sum(static) < threshold:
            accepted.append(frame)
            accepted_components.append(st_components[:, :, i])
        else:
            rejected.append(frame)
    return accepted, accepted_components, rejected


def _extract_st_rois(frames, min_area=50, spatial_sep=True):
    """ Extract ROIs from the spatio-temporal components

    Parameters
    ----------
    frames : list
        list of arrays containing stICA components
    min_area : int
        The minimum size in number of pixels that an ROI can be. Default: 50
    spatial_sep : bool
        If True, the stICA components will be segmented spatially and
        non-contiguous poitns will be made into sparate ROIs. Default: True

    Returns
    -------
    rois : list
        A list of sima.ROI ROI objects
    """

    rois = []
    for frame_no in range(len(frames)):
        img = np.array(frames[frame_no])

        img[np.where(img > 0)] = 1
        img, seg_count = measurements.label(img)
        component_mask = np.zeros(img.shape, 'bool')

        for i in xrange(seg_count):
            segment = np.where(img == i + 1)
            if segment[0].size >= min_area:
                if spatial_sep:
                    thisroi = np.zeros(img.shape, 'bool')
                    thisroi[segment] = True
                    rois.append(ROI(mask=thisroi, im_shape=thisroi.shape))
                else:
                    component_mask[segment] = True
        if not spatial_sep and np.any(component_mask):
            rois.append(ROI(mask=component_mask, im_shape=thisroi.shape))

        frame_no = frame_no + 1

    return rois


class STICA(SegmentationStrategy):
    """
    Segmentation using spatiotemporial indepenent component analysis (stICA).

    Parameters
    ----------
    channel : int, optional
        The index of the channel to be used. Default: 0
    mu : float, optional
        Weighting parameter for the trade off between spatial and temporal
        information. Must be between 0 and 1. Low values give higher
        weight to temporal information. Default: 0.01
    components : int or list, optional
        Number of principal components to use. If list is given, then use
        only the principcal componenets indexed by the list Default: 75
    static_threshold : float, optional
        threhold on the static allowable in an ICA components, eliminating
        high scoring components speeds the ROI extraction and may improve
        the results. Default: 0.5
    min_area : int, optional
        minimum ROI size in number of pixels
    x_smoothing : int, optional
        number of itereations of static removial and gaussian blur to
        perform on each stICA component. 0 provides no gaussian blur,
        larger values produce stICA components with less static but the
        ROIs loose defination. Default: 5
    overlap_per : float, optional
        percentage of an ROI that must be covered in order to combine the
        two segments. Values outside of (0,1] will result in no removal of
        overlapping ROIs. Requires x_smoothing to be > 0. Default: 0
    smooth_rois : bool, optional
        Set to True in order to translate the ROIs into polygons and
        execute smoothing algorithm. Requires x_smoothing to be > 0.
        Default: True
    spatial_sep : bool, optional
        If True, the stICA components will be segmented spatially and
        non-contiguous points will be made into sparate ROIs. Requires
        x_smoothing to be > 0. Default: True
    verbose : bool, optional
        Whether to print progress updates.

    Notes
    -----
    Spatiotemporal (stICA) [1]_ is a procedure which applys ICA to
    extracted PCA components in a process that takes into consideration
    both the spatial and temporal character of these components. This
    method has been used to segment calcium imaging data [2]_, and can be
    used to segment cell bodies, dendrites, and axons.

    In order to implement spatio and temporal ICA, temporal components
    from PCA are concatenated to the spatial ones.  The following
    spatiotemporal variable :math:`y_i` and the resulting ICA components
    :math:`z_i` are defined by:

    .. math::

        y_i &= \\begin{cases} \\mu U_{ki}      & i \\leq N_x \\\\
                (1-\\mu)V_{ki} & N_x < i \\leq (N_x+N_t)
                \\end{cases} \\\\
        z_i^{k} &= \\sum_j \\mathbf{W}_{i,j}^{(n)} y^{(j)},

    where :math:`U` corresponds to the spatio PCA component matrix with
    dimensions :math:`N_x`, pixels, by :math:`k` principal components and
    :math:`V` corresponds to the :math:`N_t`, time frames, by :math:`k`
    temporal PCA component matrix. :math:`\\mu` is a weighting parameter
    to balance the tradeoff between the spatio and temporal information
    with low values of :math:`\\mu` giving higher weight to the signals
    temporal components. ICA is performed on :math:`y_i` to extract the
    independent components :math:`z_i`.

    References
    ----------
    .. [1] Stone JV, Porrill J, Porter NR, Wilkinson ID.  Spatiotemporal
       independent component analysis of event-related fMRI data using
       skewed probability density functions. Neuroimage. 2002
       Feb;15(2):407-21.

    .. [2] Mukamel EA, Nimmerjahn A, Schnitzer MJ. Automated analysis of
       cellular signals from large-scale calcium imaging data. Neuron.
       2009 Sep 24;63(6):747-60.

    Warning
    -------
    In version 1.0, this method currently only works on datasets with a
    single plane, or in conjunction with
    :class:`sima.segment.PlaneWiseSegmentation`.

    """

    def __init__(
            self, channel=0, mu=0.01, components=75, static_threshold=0.5,
            min_area=50, x_smoothing=4, overlap_per=0, smooth_rois=True,
            spatial_sep=True, verbose=False):
        super(STICA, self).__init__()
        d = locals()
        d.pop('self')
        self._params = Struct(**d)

    @_check_single_plane
    def _segment(self, dataset):

        channel = sima.misc.resolve_channels(self._params.channel,
                                             dataset.channel_names)
        if dataset.savedir is not None:
            pca_path = os.path.join(dataset.savedir,
                                    'opca_' + str(channel) + '.npz')
        else:
            pca_path = None

        if dataset.savedir is not None:
            ica_path = os.path.join(dataset.savedir,
                                    'ica_' + str(channel) + '.npz')
        else:
            ica_path = None

        if self._params.verbose:
            print 'performing PCA...'
        if isinstance(self._params.components, int):
            self._params.components = range(self._params.components)
        _, space_pcs, time_pcs = _OPCA(
            dataset, channel, self._params.components[-1] + 1, path=pca_path)
        space_pcs = np.real(space_pcs.reshape(
            dataset.frame_shape[1:3] + (space_pcs.shape[2],)))
        space_pcs = np.array(
            [space_pcs[:, :, i] for i in self._params.components]
        ).transpose((1, 2, 0))
        time_pcs = np.array(
            [time_pcs[:, i] for i in self._params.components]
        ).transpose((1, 0))

        if self._params.verbose:
            print 'performing ICA...'
        st_components = _stica(
            space_pcs, time_pcs, mu=self._params.mu, path=ica_path,
            n_components=space_pcs.shape[2])

        if self._params.x_smoothing > 0 or self._params.static_threshold > 0:
            accepted, _, _ = _find_useful_components(
                st_components, self._params.static_threshold,
                x_smoothing=self._params.x_smoothing)

            if self._params.min_area > 0 or self._params.spatial_sep:
                rois = _extract_st_rois(
                    accepted, min_area=self._params.min_area,
                    spatial_sep=self._params.spatial_sep)

            if self._params.smooth_rois:
                if self._params.verbose:
                    print 'smoothing ROIs...'
                rois = [_smooth_roi(roi)[0] for roi in rois]

            if self._params.verbose:
                print 'removing overlapping ROIs...'
            rois = _remove_overlapping(
                rois, percent_overlap=self._params.overlap_per)
        else:
            rois = [ROI(st_components[:, :, i]) for i in
                    xrange(st_components.shape[2])]

        return ROIList(rois)
