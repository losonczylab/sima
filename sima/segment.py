import os
import itertools as it
from distutils.version import StrictVersion

import numpy as np
from scipy import sparse, ndimage
from scipy.ndimage.measurements import label
from skimage.filter import threshold_otsu
try:
    import cv2
except ImportError:
    cv2_available = False
else:
    cv2_available = StrictVersion(cv2.__version__) >= StrictVersion('2.4.8')

from scipy import nanmean
try:
    from sklearn.decomposition import FastICA
except ImportError:
    SKLEARN_AVAILABLE = False
else:
    SKLEARN_AVAILABLE = True

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


def _rois_from_cuts_full(cuts):
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


def _rois_from_cuts_ca1pc(cuts, im_set, circularity_threhold=0.5,
                          min_roi_size=20, min_cut_size=30,
                          channel=0, x_diameter=8, y_diameter=8):
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

    processed_im = _processed_image_ca1pc(im_set, channel, x_diameter,
                                          y_diameter)
    shape = processed_im.shape[:2]
    ROIs = ROIList([])
    for cut in cuts:
        if len(cut.indices) > min_cut_size:
            # pixel values in the cut
            vals = processed_im.flat[cut.indices]

            # indices of those values below the otsu threshold
            # if all values are identical, continue without adding an ROI
            try:
                roi_indices = cut.indices[vals < threshold_otsu(vals)]
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

                # Apply min ROI size threshold
                if len(blob_inds) > min_roi_size:
                    twoD_indices = [np.unravel_index(x, shape)
                                    for x in blob_inds]
                    mask = np.zeros(shape)
                    for x in twoD_indices:
                        mask[x] = 1

                    # APPLY CIRCULARITY THRESHOLD
                    poly_pts = np.array(mask2poly(mask)[0].exterior.coords)
                    p = 0
                    for x in range(len(poly_pts) - 1):
                        p += np.linalg.norm(poly_pts[x] - poly_pts[x + 1])

                    shape_area = len(roi_indices)
                    circle_area = np.square(p) / (4 * np.pi)
                    if shape_area / circle_area > circularity_threhold:
                        ROIs.append(ROI(mask=mask))

    return ROIs


def _rois_from_cuts(cuts, method, *args, **kwargs):
    """Generate ROIs from the normalized cuts.

    Parameters
    ----------
    cuts : list of CutRegion
        The normalized cuts from which ROIs are to be made.
    method : {'FULL', 'CA1PC'}
        The method to be called.
    Additional parameters args and kwargs are passed onto
    the method specific function.
    """
    if method == 'full':
        return _rois_from_cuts_full(cuts)
    elif method == 'ca1pc':
        return _rois_from_cuts_ca1pc(cuts, *args, **kwargs)
    else:
        raise ValueError('Unrecognized method')


def _affinity_matrix(dataset, channel, max_dist=None, spatial_decay=None,
                     variant=None, num_pcs=75, processed_image=None,
                     verbose=False):
    """Return a sparse affinity matrix for use with normalized cuts.

    .. math::

        A_{ij} = e^{k_cc_{ij}} \cdot
        e^{-\\frac{|\mathbf X_i-\mathbf X_j|_2^2}{\\sigma_{\\mathbf X}^2}}


    Parameters
    ----------
    dataset : sima.ImagingDataset
        The dataset for which the affinity matrix is being calculated.
    channel : int, optional
        The channel whose signals will be used in the calculations.
    max_dist : tuple of int, optional
        Defaults to (2, 2).
    spatial_decay : tuple of int, optional
        Defaults to (2, 2).
    variant : str, optional
        Specifies a modification to the affinity matrix calculation.
        Extra kwargs can be used with the variant.
    processed_image : numpy.ndarry, optional
        See _processed_image_ca1pc. Only used if variant='ca1pc'.

    Returns
    -------
    affinities : scipy.sparse.coo_matrix
        The affinities between the image pixels.
    """
    if max_dist is None:
        max_dist = (2, 2)
    if spatial_decay is None:
        spatial_decay = (2, 2)
    if variant == 'ca1pc':
        time_avg = processed_image
        std = np.std(time_avg)
        time_avg = np.minimum(time_avg, 2 * std)
        time_avg = np.maximum(time_avg, -2 * std)
        dm = time_avg.max() - time_avg.min()

    Y, X = spatial_decay
    shape = (dataset.num_rows, dataset.num_columns)
    A = sparse.dok_matrix((shape[0] * shape[1], shape[0] * shape[1]))

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
                    pairs.append(np.reshape([y, x, y + dy, x + dx], (1, 4)))
    correlations = _offset_corrs(dataset, np.concatenate(pairs, 0), channel,
                                 num_pcs=num_pcs, verbose=verbose)
    for y, x in it.product(xrange(shape[0]), xrange(shape[1])):
        for dx in range(max_dist[1] + 1):
            if dx == 0:
                yrange = range(1, max_dist[0] + 1)
            else:
                yrange = range(-max_dist[0], max_dist[0] + 1)
            for dy in yrange:
                if (x + dx < shape[1]) and (y + dy >= 0) and \
                        (y + dy < shape[0]):
                    a = x + y * shape[1]
                    b = a + dx + dy * shape[1]
                    w = np.exp(
                        9. * correlations[((y, x), (y + dy, x + dx))]
                    ) * np.exp(
                        -0.5 * ((float(dx) / X) ** 2 + (float(dy) / Y) ** 2)
                    )
                    if variant == 'ca1pc':
                        m = -np.Inf
                        for xt in range(dx + 1):
                            if dx != 0:
                                ya = y + 0.5 + max(0., xt - 0.5) * \
                                    float(dy) / float(dx)
                                yb = y + 0.5 + min(xt + 0.5, dx) * \
                                    float(dy) / float(dx)
                            else:
                                ya = y
                                yb = y + dy
                            ym = int(min(ya, yb))
                            yM = int(max(ya, yb))
                            for yt in range(ym, yM + 1):
                                m = max(m, time_avg[yt, x + xt])
                        w *= np.exp(-3. * m / dm)
                    # TODO: Use symmetric matrix structure
                    assert np.isfinite(w)
                    A[a, b] = w
                    A[b, a] = w
    return sparse.csr_matrix(sparse.coo_matrix(A), dtype=float)


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
    method : {'EM', 'fast'}
        The method for estimating the correlations. EM uses the EM
        algorithm to perform OPCA. Fast calculates the offset correlations
        directly, but is more noisy since all PCs are used.

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


class dataset_iterable():

    def __init__(self, dataset, channel):
        self.dataset = dataset
        self.channel = channel
        self.means = dataset.time_averages[self.channel].reshape(-1)

    def __iter__(self):
        for cycle in self.dataset:
            for frame in cycle:
                yield np.nan_to_num(
                    frame[self.channel].reshape(-1) - self.means)
        raise StopIteration


def _unsharp_mask(image, mask_weight, image_weight=1.35, sigma_x=10,
                  sigma_y=10):
    """Perform unsharp masking on an image.."""
    if not cv2_available:
        raise ImportError('OpenCV >= 2.4.8 required')
    return cv2.addWeighted(_to8bit(image), image_weight,
                           cv2.GaussianBlur(_to8bit(image), (0, 0), sigma_x,
                                            sigma_y),
                           -mask_weight, 0)


def _clahe(image, x_tile_size=10, y_tile_size=10, clip_limit=20):
    """Perform contrast limited adaptive histogram equalization (CLAHE)."""
    if not cv2_available:
        raise ImportError('OpenCV >= 2.4.8 required')
    transform = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=(
                                    int(image.shape[1] / float(x_tile_size)),
                                    int(image.shape[0] / float(y_tile_size))))
    return transform.apply(_to8bit(image))


def _to8bit(array):
    """Convert an arry to 8 bit."""
    return ((255. * array) / array.max()).astype('uint8')


def _processed_image_ca1pc(dataset, channel_idx=-1, x_diameter=10,
                           y_diameter=10):
    """Create a processed image for identifying CA1 pyramidal cell ROIs."""
    unsharp_mask_mask_weight = 0.5
    im = dataset.time_averages[channel_idx]
    return _unsharp_mask(_clahe(im, x_diameter, y_diameter),
                         unsharp_mask_mask_weight,
                         1 + unsharp_mask_mask_weight,
                         x_diameter, y_diameter)


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
    shape = (dataset.num_rows, dataset.num_columns)
    oPC_vars, oPCs, oPC_signals = oPCA.EM_oPCA(
        dataset_iterable(dataset, ch), num_pcs=num_pcs, verbose=verbose)
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


def _normcut(dataset, channel=0, num_pcs=75, pc_list=None,
             max_dist=None, spatial_decay=None,
             cut_max_pen=0.01, cut_min_size=40, cut_max_size=200, variant=None,
             **kwargs):
    affinity = _affinity_matrix(dataset, channel, max_dist, spatial_decay,
                                variant, num_pcs, **kwargs)
    shape = (dataset.num_rows, dataset.num_columns)
    cuts = itercut(affinity, shape, cut_max_pen, cut_min_size, cut_max_size)
    return cuts


def normcut(
        dataset, channel=0, num_pcs=75, max_dist=None,
        spatial_decay=None, cut_max_pen=0.01, cut_min_size=40,
        cut_max_size=200):
    """Segment image by iteratively performing normalized cuts.

    Parameters
    ----------
    dataset : ImagingDataset
        The dataset whose affinity matrix is being calculated.
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

    Returns
    -------
    list of sima.ROI.ROI
        Segmented ROI structures.

    Notes
    -----
    The normalized cut procedure [3]_ is iteratively applied, first to the
    entire image, and then to each cut made from the previous application of
    the procedure.

    The affinity :math:`A_{ij}` between each pair of pixels :math:`i,j` is a
    function of the correlation :math:`c_{i,j}` of the pixel-intensity time
    series, and the relative locations (:math:`\\mathbf X_i,\mathbf X_j`) of
    the pixels:

    .. math::

        A_{ij} = e^{k_cc_{ij}} \cdot
        e^{-\\frac{|\mathbf X_i-\mathbf X_j|_2^2}{\\sigma_{\\mathbf X}^2}},

    with :math:`k_c` and :math:`\\sigma_{\mathbf X}^2` being automatically
    determined constants.


    References
    ----------
    .. [3] Jianbo Shi and Jitendra Malik. Normalized Cuts and Image
       Segmentation.  IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE
       INTELLIGENCE, VOL. 22, NO. 8, AUGUST 2000.

    """
    cuts = _normcut(
        dataset, channel, num_pcs, None, max_dist, spatial_decay,
        cut_max_pen, cut_min_size, cut_max_size)
    return _rois_from_cuts(cuts, 'full')


def ca1pc(
        dataset, channel=0, num_pcs=75, max_dist=None,
        spatial_decay=None, cut_max_pen=0.01, cut_min_size=40,
        cut_max_size=200, x_diameter=10, y_diameter=10,
        circularity_threhold=.5, min_roi_size=20, min_cut_size=30,
        verbose=False):
    """Segmentation method designed for finding CA1 pyramidal cell somata.

    Parameters
    ----------
    dataset : ImagingDataset
        The dataset whose affinity matrix is being calculated.
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

    Returns
    -------
    list of sima.ROI.ROI
        Segmented ROI structures.

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
    processed_image = _processed_image_ca1pc(dataset, channel, x_diameter,
                                             y_diameter)
    cuts = _normcut(
        dataset, channel, num_pcs, None, max_dist, spatial_decay,
        cut_max_pen, cut_min_size, cut_max_size, 'ca1pc',
        processed_image=processed_image, verbose=verbose)
    return _rois_from_cuts(cuts, 'ca1pc', dataset, circularity_threhold,
                           min_roi_size, min_cut_size, channel, x_diameter,
                           y_diameter)


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

    frame = roi.mask.todense().copy()

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
        y = np.roll(np.array([p[1]-radius]*(2*radius)+list(p[1] +
                             range(-radius, radius)) + [p[1] + radius] *
                             (2*radius+1)+list(p[1] +
                             range(-radius, (radius + 1))[::-1])), -radius)

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
            if new_roi.mask.size != 0:
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


def stica(dataset, channel=0, mu=0.01, components=75,
          static_threshold=0.5, min_area=50, x_smoothing=4, overlap_per=0,
          smooth_rois=True, spatial_sep=True):
    """
    Segmentation using spatiotemporial indepenent component analysis (stICA).

    Parameters
    ----------
    dataset : sima.ImagingDataset
        dataset to be segmented
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

    Returns
    -------
    rois : list
        A list of sima.ROI ROI objects

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
    if not SKLEARN_AVAILABLE:
        raise ImportError('scikit-learn >= 0.11 required')

    if dataset.savedir is not None:
        pca_path = os.path.join(
            dataset.savedir, 'opca_' + str(channel) + '.npz')
    else:
        pca_path = None

    if dataset.savedir is not None:
        ica_path = os.path.join(
            dataset.savedir, 'ica_' + str(channel) + '.npz')
    else:
        ica_path = None

    print 'performing PCA...'
    if isinstance(components, int):
        components = range(components)
    _, space_pcs, time_pcs = _OPCA(dataset, channel, components[-1]+1,
                                   path=pca_path)
    space_pcs = np.real(space_pcs.reshape(dataset.num_rows,
                        dataset.num_columns, space_pcs.shape[2]))
    space_pcs = np.array([space_pcs[:, :, i]
                          for i in components]).transpose((1, 2, 0))
    time_pcs = np.array([time_pcs[:, i] for i in components]).transpose((1, 0))

    print 'performing ICA...'
    st_components = _stica(space_pcs, time_pcs, mu=mu, path=ica_path,
                           n_components=space_pcs.shape[2])

    if x_smoothing > 0 or static_threshold > 0:
        accepted, _, _ = _find_useful_components(
            st_components, static_threshold, x_smoothing=x_smoothing)

        if min_area > 0 or spatial_sep:
            rois = _extract_st_rois(accepted, min_area=min_area,
                                    spatial_sep=spatial_sep)

        if smooth_rois:
            print 'smoothing ROIs...'
            rois = [_smooth_roi(roi)[0] for roi in rois]

        print 'removing overlapping ROIs...'
        rois = _remove_overlapping(rois, percent_overlap=overlap_per)
    else:
        rois = [ROI(st_components[:, :, i]) for i in
                xrange(st_components.shape[2])]

    return rois
