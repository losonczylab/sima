import os
import abc
import itertools as it
from distutils.version import LooseVersion

import numpy as np
from scipy import sparse, ndimage
from scipy.ndimage.measurements import label
from skimage.filter import threshold_otsu
try:
    import cv2
except ImportError:
    cv2_available = False
else:
    cv2_available = LooseVersion(cv2.__version__) >= LooseVersion('2.4.8')

try:
    from bottleneck import nanmean
except ImportError:
    from scipy import nanmean

try:
    from sklearn.decomposition import FastICA
except ImportError:
    SKLEARN_AVAILABLE = False
else:
    SKLEARN_AVAILABLE = True

import sima.misc
from sima.normcut import itercut
from sima.ROI import ROI, ROIList, mask2poly
import sima.oPCA as oPCA
from scipy.ndimage import measurements

try:
    import sima._opca as _opca
except ImportError:
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()},
                      reload_support=True)
    import sima._opca as _opca


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
        return self._segment(dataset)


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
                      for _ in range(z-1)] + [old_mask[0]])

        rois = ROIList([])
        for plane in range(dataset.frame_shape[0]):
            plane_rois = self.strategy.segment(dataset[:, :, plane])
            for roi in plane_rois:
                set_z(roi, plane)
            rois.extend(plane_rois)
        return rois


def _offset_corrs(dataset, pixel_pairs, channel=0, method='EM',
                  num_pcs=75, verbose=False):
    """
    Calculate the offset correlation for specified pixel pairs.

    Parameters
    -----------
    dataset : sima.ImagingDataset
        The dataset to be used.
    pixel_pairs : ndarray of int
        The pairs of pixels, indexed ((y0, x0), (y1, x1)) for
        which the correlation is to be calculated.
    channel : int, optional
        The channel to be used for estimating the pixel correlations.
        Defaults to 0.
    method : {'EM', 'fast'}, optional
        The method for estimating the correlations. EM uses the EM
        algorithm to perform OPCA. Fast calculates the offset correlations
        directly, but is more noisy since all PCs are used. Default: EM.
    num_pcs : int, optional
        The number of principal components to be used in the estimated
        correlations with the EM method. Default: 75.
    verbose : bool, optional
        Whether to print progress status. Default: False.

    Returns
    -------
    correlations: dict
        A dictionary whose keys are the elements of the pixel_pairs
        input list, and whose values are the calculated offset
        correlations.
    """
    if method == 'EM':
        if dataset.savedir is not None:
            path = os.path.join(
                dataset.savedir, 'opca_' + str(channel) + '.npz')
        else:
            path = None
        oPC_vars, oPCs, _ = _OPCA(dataset, channel, num_pcs, path,
                                  verbose=verbose)
        weights = np.sqrt(np.maximum(0, oPC_vars))
        D = _direction(oPCs, weights)
        return {
            ((u, v), (w, x)): np.dot(D[u, v, :], D[w, x, :])
            for u, v, w, x in pixel_pairs
        }
    elif method == 'fast':
        ostdevs, correlations, pixels = _opca._fast_ocorr(
            dataset, pixel_pairs, channel)
        ostdevs /= dataset.num_frames - 1.
        correlations /= 2. * (dataset.num_frames - 1)
        ostdevs = np.sqrt(np.maximum(0., ostdevs))
        for pair_idx, pair in enumerate(pixel_pairs):
            denom = ostdevs[pair[0], pair[1]] * ostdevs[pair[2], pair[3]]
            if denom <= 0:
                correlations[pair_idx] = 0.
            else:
                correlations[pair_idx] = max(
                    -1., min(1., correlations[pair_idx] / denom))
        return {
            ((PAIR[0], PAIR[1]), (PAIR[2], PAIR[3])): correlations[pair_idx]
            for pair_idx, PAIR in enumerate(pixel_pairs)}


class DatasetIterable():

    def __init__(self, dataset, channel):
        self.dataset = dataset
        self.channel = channel
        self.means = dataset.time_averages[..., self.channel].reshape(-1)

    def __iter__(self):
        for cycle in self.dataset:
            for frame in cycle:
                yield np.nan_to_num(
                    frame[..., self.channel].reshape(-1) - self.means)
        raise StopIteration


def _unsharp_mask(image, mask_weight, image_weight=1.35, sigma_x=10,
                  sigma_y=10):
    """Perform unsharp masking on an image.."""
    if not cv2_available:
        raise ImportError('OpenCV >= 2.4.8 required')
    return cv2.addWeighted(
        sima.misc.to8bit(image), image_weight,
        cv2.GaussianBlur(sima.misc.to8bit(image), (0, 0), sigma_x, sigma_y),
        -mask_weight, 0)


def _clahe(image, x_tile_size=10, y_tile_size=10, clip_limit=20):
    """Perform contrast limited adaptive histogram equalization (CLAHE)."""
    if not cv2_available:
        raise ImportError('OpenCV >= 2.4.8 required')
    transform = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=(
                                    int(image.shape[1] / float(x_tile_size)),
                                    int(image.shape[0] / float(y_tile_size))))
    return transform.apply(sima.misc.to8bit(image))


def _processed_image_ca1pc(dataset, channel_idx=-1, x_diameter=10,
                           y_diameter=10):
    """Create a processed image for identifying CA1 pyramidal cell ROIs."""
    unsharp_mask_mask_weight = 0.5
    result = []
    for plane_idx in np.arange(dataset.frame_shape[0]):
        im = dataset.time_averages[plane_idx, :, :, channel_idx]
        result.append(_unsharp_mask(_clahe(im, x_diameter, y_diameter),
                                    unsharp_mask_mask_weight,
                                    1 + unsharp_mask_mask_weight,
                                    x_diameter, y_diameter))
    return np.array(result)


def _OPCA(dataset, ch=0, num_pcs=75, path=None, verbose=False):
    """Perform offset principal component analysis on the dataset.

    Parameters
    ----------
    dataset : ImagingDataset
        The dataset to which the offset PCA will be applied.
    channel : int, optional
        The index of the channel whose signals are used. Defaults
        to using the first channel.
    num_pcs : int, optional
        The number of PCs to calculate. Default is 75.
    path : str
        Directory for saving or loading OPCA results.

    Returns
    -------
    oPC_vars : array
        The offset variance accounted for by each oPC. Shape: num_pcs.
    oPCs : array
        The spatial representations of the oPCs.
        Shape: (num_rows, num_columns, num_pcs).
    oPC_signals : array
        The temporal representations of the oPCs.
        Shape: (num_times, num_pcs).
    """
    ret = None
    if path is not None:
        try:
            data = np.load(path)
        except IOError:
            pass
        else:
            if data['oPC_signals'].shape[1] >= num_pcs:
                ret = (
                    data['oPC_vars'][:num_pcs],
                    data['oPCs'][:, :, :num_pcs],
                    data['oPC_signals'][:, :num_pcs]
                )
            data.close()
    if ret is not None:
        return ret
    shape = dataset.frame_shape[1:3]
    oPC_vars, oPCs, oPC_signals = oPCA.EM_oPCA(
        DatasetIterable(dataset, ch), num_pcs=num_pcs, verbose=verbose)
    oPCs = oPCs.reshape(shape + (-1,))
    if path is not None:
        np.savez(path, oPCs=oPCs, oPC_vars=oPC_vars, oPC_signals=oPC_signals)
    return oPC_vars, oPCs, oPC_signals


def _direction(vects, weights=None):
    if weights is None:
        vects_ = vects
    else:
        vects_ = vects * weights
    return (vects_.T / np.sqrt((vects_ ** 2).sum(axis=2).T)).T


class AffinityMatrixMethod(object):
    """Method for calculating the affinity matrix"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def calculate(self, dataset):
        """Calculate the afinity matrix for a dataset.

        Parameters
        ----------
        dataset : sima.ImagingDataset
            The dataset for which the affinity matrix is to be calculated.


        Returns
        -------
        affinities : scipy.sparse.coo_matrix
            The affinities between the image pixels.
        """
        return


class BasicAffinityMatrix(AffinityMatrixMethod):

    """Return a sparse affinity matrix for use with normalized cuts.

    The affinity :math:`A_{ij}` between each pair of pixels :math:`i,j` is a
    function of the correlation :math:`c_{i,j}` of the pixel-intensity time
    series, and the relative locations (:math:`\\mathbf X_i,\mathbf X_j`) of
    the pixels:

    .. math::

        A_{ij} = e^{k_cc_{ij}} \cdot
        e^{-\\frac{|\mathbf X_i-\mathbf X_j|_2^2}{\\sigma_{\\mathbf X}^2}},

    with :math:`k_c` and :math:`\\sigma_{\mathbf X}^2` being automatically
    determined constants.

    Parameters
    ----------
    channel : int, optional
        The channel whose signals will be used in the calculations.
    max_dist : tuple of int, optional
        Defaults to (2, 2).
    spatial_decay : tuple of int, optional
        Defaults to (2, 2).
    num_pcs : int, optional
        The number of principal component to use. Default: 75.
    verbose : bool, optional
        Whether to print progress status. Default: False.
    """
    def __init__(self, channel=0, max_dist=None, spatial_decay=None,
                 num_pcs=75, verbose=False):
        if max_dist is None:
            max_dist = (2, 2)
        if spatial_decay is None:
            spatial_decay = (2, 2)
        d = locals()
        d.pop('self')
        self._params = Struct(**d)

    def _calculate_correlations(self, dataset):
        shape = dataset.frame_shape[1:3]
        max_dist = self._params.max_dist
        pairs = []
        for y, x in it.product(xrange(shape[0]), xrange(shape[1])):
            for dx in range(max_dist[1] + 1):
                if dx == 0:
                    yrange = range(1, max_dist[0] + 1)
                else:
                    yrange = range(-max_dist[0], max_dist[0] + 1)
                for dy in yrange:
                    if (x + dx < shape[1]) and (y + dy >= 0) and \
                            (y + dy < shape[0]):
                        pairs.append(
                            np.reshape([y, x, y + dy, x + dx], (1, 4)))
        return _offset_corrs(
            dataset, np.concatenate(pairs, 0), self._params.channel,
            num_pcs=self._params.num_pcs, verbose=self._params.verbose)

    def _weight(self, r0, r1):
        Y, X = self._params.spatial_decay
        dy = r1[0] - r0[0]
        dx = r1[1] - r0[1]
        return np.exp(9. * self._correlations[(r0, r1)]) * np.exp(
            -0.5 * ((float(dx) / X) ** 2 + (float(dy) / Y) ** 2))

    def _setup(self, dataset):
        self._correlations = self._calculate_correlations(dataset)

    def calculate(self, dataset):
        self._setup(dataset)
        max_dist = self._params.max_dist
        shape = dataset.frame_shape[1:3]
        A = sparse.dok_matrix((shape[0] * shape[1], shape[0] * shape[1]))
        for y, x in it.product(xrange(shape[0]), xrange(shape[1])):
            for dx in range(max_dist[1] + 1):
                if dx == 0:
                    yrange = range(1, max_dist[0] + 1)
                else:
                    yrange = range(-max_dist[0], max_dist[0] + 1)
                for dy in yrange:
                    r0 = (y, x)
                    r1 = (y+dy, x+dx)
                    if (x + dx < shape[1]) and (y + dy >= 0) and \
                            (y + dy < shape[0]):
                        w = self._weight(r0, r1)
                        assert np.isfinite(w)
                        a = x + y * shape[1]
                        b = a + dx + dy * shape[1]
                        A[a, b] = w
                        A[b, a] = w  # TODO: Use symmetric matrix structure
        return sparse.csr_matrix(sparse.coo_matrix(A), dtype=float)


class AffinityMatrixCA1PC(BasicAffinityMatrix):

    def __init__(self, channel=0, max_dist=None, spatial_decay=None,
                 num_pcs=75, x_diameter=10, y_diameter=10, verbose=False):
        super(AffinityMatrixCA1PC, self).__init__(
            channel, max_dist, spatial_decay, num_pcs, verbose)
        self._params.x_diameter = x_diameter
        self._params.y_diameter = y_diameter

    def _setup(self, dataset):
        super(AffinityMatrixCA1PC, self)._setup(dataset)
        processed_image = _processed_image_ca1pc(
            dataset, self._params.channel, self._params.x_diameter,
            self._params.y_diameter)
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


class PlaneNormalizedCuts(PlaneSegmentationStrategy):

    """Segment image by iteratively performing normalized cuts.

    Parameters
    ----------
    affinity_method : AffinityMatrixMethod
        The method used to calculate the affinity matrix.
    max_pen : float
        Iterative cutting will continue as long as the cut cost is less than
        max_pen.
    cut_min_size, cut_max_size : int
        Regardless of the cut cost, iterative cutting will not be performed on
        regions with fewer pixels than min_size and will always be performed
        on regions larger than max_size.

    Notes
    -----
    The normalized cut procedure [3]_ is iteratively applied, first to the
    entire image, and then to each cut made from the previous application of
    the procedure.

    References
    ----------
    .. [3] Jianbo Shi and Jitendra Malik. Normalized Cuts and Image
       Segmentation.  IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE
       INTELLIGENCE, VOL. 22, NO. 8, AUGUST 2000.

    """
    def __init__(self, affinty_method=None, cut_max_pen=0.01,
                 cut_min_size=40, cut_max_size=200):
        super(PlaneNormalizedCuts, self).__init__()
        if affinty_method is None:
            affinty_method = BasicAffinityMatrix(channel=0, num_pcs=75)
        d = locals()
        d.pop('self')
        self._params = Struct(**d)

    @classmethod
    def _rois_from_cuts(cls, cuts):
        """Return ROI structures each containing the full extent of a cut.

        Parameters
        ----------
        cuts : list of sima.normcut.CutRegion
            The segmented regions identified by normalized cuts.

        Returns
        -------
        sima.ROI.ROIList
            ROI structures corresponding to each cut.
        """
        ROIs = ROIList([])
        for cut in cuts:
            if len(cut.indices):
                mask = np.zeros(cut.shape)
                for x in cut.indices:
                    mask[np.unravel_index(x, cut.shape)] = 1
                ROIs.append(ROI(mask=mask))
        return ROIs

    def _segment(self, dataset):
        params = self._params
        affinity = params.affinty_method.calculate(dataset)
        shape = dataset.frame_shape[1:3]
        cuts = itercut(affinity, shape, params.cut_max_pen,
                       params.cut_min_size, params.cut_max_size)
        return self._rois_from_cuts(cuts)


class PlaneCA1PC(PlaneSegmentationStrategy):
    """Segmentation method designed for finding CA1 pyramidal cell somata.

    Parameters
    ----------
    channel : int, optional
        The channel whose signals will be used in the calculations.
    max_dist : tuple of int, optional
        Defaults to (2, 2).
    spatial_decay : tuple of int, optional
        Defaults to (2, 2).
    max_pen : float
        Iterative cutting will continue as long as the cut cost is less than
        max_pen.
    cut_min_size, cut_max_size : int
        Regardless of the cut cost, iterative cutting will not be performed on
        regions with fewer pixels than min_size and will always be performed
        on regions larger than max_size.
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
        self._normcut_method.append(ROISizeFilter(min_cut_size))
        self._normcut_method.append(
            CA1PCNucleus(channel, x_diameter, y_diameter))
        self._normcut_method.append(ROISizeFilter(min_roi_size))
        self._normcut_method.append(CircularityFilter(circularity_threhold))

    def _segment(self, dataset):
        return self._normcut_method.segment(dataset)


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
        space_pcs[:, :, i] = mu*(space_pcs[:, :, i] -
                                 nanmean(space_pcs[:, :, i]))/np.max(space_pcs)
    for i in range(time_pcs.shape[1]):
        time_pcs[:, i] = (1-mu)*(time_pcs[:, i]-nanmean(time_pcs[:, i])) / \
            np.max(time_pcs)

    # concatenate the space and time PCs
    y = np.concatenate((space_pcs.reshape(
        space_pcs.shape[0]*space_pcs.shape[1],
        space_pcs.shape[2]), time_pcs))

    # execute the FastICA algorithm
    ica = FastICA(n_components=n_components, max_iter=1500)
    st_components = np.real(np.array(ica.fit_transform(y)))

    # pull out the spacial portion of the st_components
    st_components = \
        st_components[:(space_pcs.shape[0]*space_pcs.shape[1]), :]
    st_components = st_components.reshape(space_pcs.shape[0],
                                          space_pcs.shape[1],
                                          st_components.shape[1])

    # normalize the ica results
    for i in range(st_components.shape[2]):
        st_component = st_components[:, :, i]
        st_component = abs(st_component-np.mean(st_component))
        st_component = st_component/np.max(st_component)
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
        frame[frame < 2*np.std(frame)] = 0

        # smooth the component via static removal and gaussian blur
        for n in xrange(x_smoothing):
            check = frame[1:-1, :-2]+frame[1:-1, 2:]+frame[:-2, 1:-1] + \
                frame[2, 1:-1]
            z = np.zeros(frame.shape)
            z[1:-1, 1:-1] = check
            frame[np.logical_not(z)] = 0

            blurred = ndimage.gaussian_filter(frame, sigma=1)
            frame = blurred+frame

            frame = frame/np.max(frame)
            frame[frame < 2*np.std(frame)] = 0

        # calculate the remaining static in the component
        static = np.sum(np.abs(frame[1:-1, 1:-1]-frame[:-2, 1:-1])) + \
            np.sum(np.abs(frame[1:-1, 1:-1]-frame[2:, 1:-1])) + \
            np.sum(np.abs(frame[1:-1, 1:-1]-frame[1:-1, :-2])) + \
            np.sum(np.abs(frame[1:-1, 1:-1]-frame[1:-1, 2:])) + \
            np.sum(np.abs(frame[1:-1, 1:-1]-frame[2:, 2:])) + \
            np.sum(np.abs(frame[1:-1, 1:-1]-frame[:-2, 2:])) + \
            np.sum(np.abs(frame[1:-1, 1:-1]-frame[2:, :-2])) + \
            np.sum(np.abs(frame[1:-1, 1:-1]-frame[:-2, :-2]))

        static = static*2.0/(frame.shape[0]*frame.shape[1])

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
            segment = np.where(img == i+1)
            if segment[0].size >= min_area:
                if spatial_sep:
                    thisroi = np.zeros(img.shape, 'bool')
                    thisroi[segment] = True
                    rois.append(ROI(mask=thisroi, im_shape=thisroi.shape))
                else:
                    component_mask[segment] = True
        if not spatial_sep and np.any(component_mask):
            rois.append(ROI(mask=component_mask, im_shape=thisroi.shape))

        frame_no = frame_no+1

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

                    if len(np.where(overlap)[0]) > percent_overlap*small_area:
                        new_shape = np.logical_or(rois[i].mask.toarray(),
                                                  rois[j].mask.toarray())

                        rois[i] = ROI(mask=new_shape.astype('bool'),
                                      im_shape=rois[i].mask.shape)
                        rois[j] = None
    return [roi for roi in rois if roi is not None]


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
        processed_im = _processed_image_ca1pc(
            dataset, self._channel, self._x_diameter, self._y_diameter)
        shape = processed_im.shape[:2]
        ROIs = ROIList([])
        for roi in rois:
            roi_indices = np.nonzero(roi.mask[0])[0]
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
            labeled_array, num_features = label(mask)
            for feat in range(num_features):
                blob_inds = np.where(labeled_array.flat == feat + 1)[0]

                twoD_indices = [np.unravel_index(x, shape) for x in blob_inds]
                mask = np.zeros(shape)
                for x in twoD_indices:
                    mask[x] = 1

                ROIs.append(ROI(mask=mask))

        return ROIs


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
    check = frame[:-2, :-2]+frame[1:-1, :-2]+frame[2:, :-2] + \
        frame[:-2, 1:-1]+frame[2:, 1:-1]+frame[:-2:, 2:] + \
        frame[1:-1, 2:]+frame[2:, 2:]
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
    for i in range(limit-1):
        b.append(p)
        # find the ist of all points at the given radius and adjust to be lined
        # up for clockwise traversal
        x = np.roll(np.array(list(p[0]+range(-radius, radius)) +
                             [p[0]+radius]*(2*radius+1) +
                             list(p[0]+range(-radius, radius)[::-1]) +
                             [p[0]-(radius+1)]*(2*radius+1)), -2)
        y = np.roll(np.array([p[1]-radius]*(2*radius) +
                             list(p[1] + range(-radius, radius)) +
                             [p[1] + radius] * (2*radius+1) +
                             list(p[1] + range(-radius, (radius + 1))[::-1])),
                    -radius)

        # insure that the x and y points are within the image
        x[x < 0] = 0
        y[y < 0] = 0
        x[x >= z.shape[1]] = z.shape[1]-1
        y[y >= z.shape[0]] = z.shape[0]-1

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
        if ((p[0]-base[0])**2+(p[1]-base[1])**2)**0.5 < 1.5*radius and \
                len(b) > 3:
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
                radius = radius+1
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


class PlaneSTICA(PlaneSegmentationStrategy):
    """
    Segmentation using spatiotemporial indepenent component analysis (stICA).

    Parameters
    ----------
    channel : int, optional
        The index of the channel to be used. Default: 0
    mu : float, optional
        Weighting parameter for the trade off between spatial and temporal
        information. Must be between 0 and 1. Low values give higher weight
        to temporal information. Default: 0.01
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
        number of itereations of static removial and gaussian blur to perform
        on each stICA component. 0 provides no gaussian blur, larger values
        produce stICA components with less static but the ROIs loose
        defination. Default: 5
    overlap_per : float, optional
        percentage of an ROI that must be covered in order to combine the two
        segments. Values outside of (0,1] will result in no removal of
        overlapping ROIs. Requires x_smoothing to be > 0. Default: 0
    smooth_rois : bool, optional
        Set to True in order to translate the ROIs into polygons and execute
        smoothing algorithm. Requires x_smoothing to be > 0. Default: True
    spatial_sep : bool, optional
        If True, the stICA components will be segmented spatially and
        non-contiguous points will be made into sparate ROIs. Requires
        x_smoothing to be > 0. Default: True
    verbose : bool, optional
        Whether to print progress updates.

    Notes
    -----
    Spatiotemporal (stICA) [1]_ is a procedure which applys ICA to extracted
    PCA components in a process that takes into consideration both the spatial
    and temporal character of these components. This method has been used to
    segment calcium imaging data [2]_, and can be used to segment cell bodies,
    dendrites, and axons.

    In order to implement spatio and temporal ICA, temporal components from PCA
    are concatenated to the spatial ones.  The following spatiotemporal
    variable :math:`y_i` and the resulting ICA components :math:`z_i` are
    defined by:

    .. math::

        y_i &= \\begin{cases} \\mu U_{ki}      & i \\leq N_x \\\\
                (1-\\mu)V_{ki} & N_x < i \\leq (N_x+N_t)
                \\end{cases} \\\\
        z_i^{k} &= \\sum_j \\mathbf{W}_{i,j}^{(n)} y^{(j)},

    where :math:`U` corresponds to the spatio PCA component matrix with
    dimensions :math:`N_x`, pixels, by :math:`k` principal components and
    :math:`V` corresponds to the :math:`N_t`, time frames, by :math:`k`
    temporal PCA component matrix. :math:`\\mu` is a weighting parameter to
    balance the tradeoff between the spatio and temporal information with low
    values of :math:`\\mu` giving higher weight to the signals temporal
    components. ICA is performed on :math:`y_i` to extract the independent
    components :math:`z_i`.

    References
    ----------
    .. [1] Stone JV, Porrill J, Porter NR, Wilkinson ID.  Spatiotemporal
       independent component analysis of event-related fMRI data using skewed
       probability density functions. Neuroimage. 2002 Feb;15(2):407-21.

    .. [2] Mukamel EA, Nimmerjahn A, Schnitzer MJ. Automated analysis of
       cellular signals from large-scale calcium imaging data.  Neuron. 2009
       Sep 24;63(6):747-60.
    """

    def __init__(self, channel=0, mu=0.01, components=75, static_threshold=0.5,
                 min_area=50, x_smoothing=4, overlap_per=0, smooth_rois=True,
                 spatial_sep=True, verbose=False):
        super(PlaneSTICA, self).__init__()
        d = locals()
        d.pop('self')
        self._params = Struct(**d)

    def _segment(self, dataset):

        if not SKLEARN_AVAILABLE:
            raise ImportError('scikit-learn >= 0.11 required')

        if dataset.savedir is not None:
            pca_path = os.path.join(
                dataset.savedir, 'opca_' + str(self._params.channel) + '.npz')
        else:
            pca_path = None

        if dataset.savedir is not None:
            ica_path = os.path.join(
                dataset.savedir, 'ica_' + str(self._params.channel) + '.npz')
        else:
            ica_path = None

        if self._params.verbose:
            print 'performing PCA...'
        if isinstance(self._params.components, int):
            self._params.components = range(self._params.components)
        _, space_pcs, time_pcs = _OPCA(
            dataset, self._params.channel, self._params.components[-1]+1,
            path=pca_path)
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
