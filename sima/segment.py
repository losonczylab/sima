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

import matplotlib.pyplot as plt
from scipy import nanmean
from sklearn.decomposition import FastICA
import sys
from descartes import PolygonPatch
import scipy.stats as stats


from sima.normcut import itercut
from sima.ROI import ROI, ROIList, mask2poly
import sima.oPCA as oPCA

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
    The normalized cut procedure [1]_ is iteratively applied, first to the
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
    .. [1] Jianbo Shi and Jitendra Malik. Normalized Cuts and Image
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


def _stICA(space_pcs,time_pcs,mu=0.01,n_components=30,path=None):
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

    ret = None
    
    if path is not None:
        try:
            data = np.load(path)
        except IOError:
            pass
        else:
            print 'loaded ica components from savefile'
            if data['st_components'].shape[2] == n_components and \
                    data['mu'].item() == mu and data['num_pcs'] == time_pcs.shape[1]:
                ret = data['st_components']
            data.close()

    if ret is not None:
        return ret
    
    print "calculating stICA data..."
    
    for i in range(space_pcs.shape[2]):
        space_pcs[:,:,i] = mu*(space_pcs[:,:,i]-nanmean(space_pcs[:,:,i])) \
                                /np.max(space_pcs)
    for i in range(time_pcs.shape[1]):
        time_pcs[:,i] = (1-mu)*(time_pcs[:,i]-nanmean(time_pcs[:,i]))/ \
                                np.max(time_pcs)

    y = np.concatenate((space_pcs.reshape(
            space_pcs.shape[0]*space_pcs.shape[1],
            space_pcs.shape[2]),time_pcs))
        
    print "using Fast ICA..."
    ica = FastICA(n_components=n_components,max_iter=1500)
    st_components = np.real(np.array(ica.fit_transform(y)))

    st_components = \
        st_components[:(space_pcs.shape[0]*space_pcs.shape[1]),:]
    st_components = st_components.reshape(space_pcs.shape[0],
                                              space_pcs.shape[1],
                                              st_components.shape[1])

    n_components = np.zeros(st_components.shape)
    for i in range(st_components.shape[2]):
        st_component = st_components[:,:,i]
        st_component = abs(st_component-np.mean(st_component))
        st_component[np.where(st_component<0)]=0
        st_components[:,:,i] = st_component

    if path is not None:
        print 'saving ica'
        np.savez(path, st_components=st_components, mu=mu, 
                 num_pcs=time_pcs.shape[1])

    return st_components


def _findUsefulComponents(st_components,threshold,x_smoothing=4):
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
        stICA components that are determined to have no axon information in them
    """

    accepted = []
    accepted_components = []
    rejected = []
    for i in xrange(st_components.shape[2]):
        frame = st_components[:,:,i].copy()
        frame[frame<2*np.std(frame)] = 0

        for n in range(x_smoothing):
            check = frame[1:-1,:-2]+frame[1:-1,2:]+frame[:-2,1:-1]+frame[2,1:-1]
            z = np.zeros(frame.shape)
            z[1:-1,1:-1] = check;
            frame[np.logical_not(z)] = 0

            blurred = ndimage.gaussian_filter(frame, sigma=1)
            frame = blurred+frame

            frame = frame/np.max(frame)
            frame[frame<2*np.std(frame)] = 0

        static = np.sum(np.abs(frame[1:-1,1:-1]-frame[:-2,1:-1])) + \
                            np.sum(np.abs(frame[1:-1,1:-1]-frame[2:,1:-1])) + \
                            np.sum(np.abs(frame[1:-1,1:-1]-frame[1:-1,:-2])) + \
                            np.sum(np.abs(frame[1:-1,1:-1]-frame[1:-1,2:])) + \
                            np.sum(np.abs(frame[1:-1,1:-1]-frame[2:,2:])) + \
                            np.sum(np.abs(frame[1:-1,1:-1]-frame[:-2,2:])) + \
                            np.sum(np.abs(frame[1:-1,1:-1]-frame[2:,:-2])) + \
                            np.sum(np.abs(frame[1:-1,1:-1]-frame[:-2,:-2]))
        
        static = static*2.0/(frame.shape[0]*frame.shape[1])
        print static
        if np.sum(static) < threshold:
            accepted.append(frame)
            accepted_components.append(st_components[:,:,i])
        else:
            print '*removed'
            rejected.append(frame)
    return accepted,accepted_components,rejected


def fndpts(p,frame,arr=[],i=0,recursion_limit=5000):
    """ recursivly search of adjacent points

    Parameters
    ----------
    p : list
        A list of length 2, [x,y], of the coordinates to the current point
    frame : array
        Single stICA component, to have points removed from and put into the 
        return array, arr
        Shape: (num_rows,num_columns)

    Returns
    -------
    arr : list
        A list of adjacent point length 2 lists that have been removed from the
        frame. Shape: (n,2)

    Notes
    _______
    This function requires the system recursive depth to be increased:
    >>> import sys
    >>> sys.setrecursionlimit(10500)
    """

    if p[0]<0 or p[0]>=frame.shape[0] or p[1]<0 or p[1]>=frame.shape[1] or \
            frame[p[0],p[1]] == 0:
        return []

    if i == recursion_limit:
        print 'recurr limit'
        return []
    
    frame[p[0],p[1]]=0

    arr = fndpts([p[0]-1,p[1]-1],frame,arr=arr,i=i+1)+ \
            fndpts([p[0]-1,p[1]],frame,arr=arr,i=i+1)+ \
            fndpts([p[0]-1,p[1]+1],frame,arr=arr,i=i+1)+ \
            fndpts([p[0],p[1]-1],frame,arr=arr,i=i+1)+ \
            fndpts([p[0],p[1]+1],frame,arr=arr,i=i+1)+ \
            fndpts([p[0]+1,p[1]-1],frame,arr=arr,i=i+1)+ \
            fndpts([p[0]+1,p[1]],frame,arr=arr,i=i+1)+ \
            fndpts([p[0]+1,p[1]+1],frame,arr=arr,i=i+1)
    arr.append(p)
    return arr


def _extractStRois(frames,min_area=50,spatial_sep=True):
    """ Extrace ROIs from the spatio-temporal components

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

    sys.setrecursionlimit(10500)

    rois = []
    for frame_no in range(len(frames)):
        print frame_no
        image_index = frame_no
        img = np.array(frames[frame_no])
        component_mask = np.zeros(img.shape)
        pts = np.where(img>0)

        while pts[0].shape[0]>0:
            p = [pts[0][0],pts[1][0]]
            arr = []
            arr = fndpts(p,img)
            thisroi = np.zeros(img.shape,'bool')
            
            for p in arr:
                thisroi[p[0],p[1]] = True

            img[thisroi>0]=0

            thisarea = len(np.where(thisroi>0)[0])
            if thisarea > min_area:
                if spatial_sep:
                    rois.append(ROI(mask=thisroi,im_shape=thisroi.shape))
                component_mask[np.where(thisroi)] = True
            
            pts = np.where(img>0)

        if not spatial_sep and np.any(component_mask):
            rois.append(ROI(mask=component_mask,im_shape=thisroi.shape))

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

    if percent_overlap > 0 and percent_overlap < 1:
        print 'removing overlapping rois'
        for roi in rois:
            roi.mask = roi.mask
        
        for i in xrange(len(rois)):
            for j in [j for j in xrange(len(rois)) if j != i]:
                if rois[i] is not None and rois[j] is not None:
                    overlap = np.logical_and(rois[i].mask.toarray(),
                                             rois[j].mask.toarray())
                    small_area = np.min(
                        (rois[i].mask.size,rois[j].mask.size))

                    if len(np.where(overlap)[0]) > percent_overlap*small_area:
                        new_shape = np.logical_or(rois[i].mask.toarray(),
                                                  rois[j].mask.toarray())

                        rois[i] = ROI(mask=new_shape.astype('bool'),
                                      im_shape=rois[i].mask.shape)
                        rois[j] = None
    return [roi for roi in rois if roi is not None]



def _smoothROI(roi):
    """ Smooth out the ROI boundaries and reduce the number of points in the 
    ROI polygons

    Parameters
    ----------
    rois : list
        list of sima.ROI ROIs

    Returns
    -------
    rois : list
        A list of sima.ROI ROI objects which have been smoothed
    """

    frame = roi.mask.todense().copy()
          
    frame[frame>0]=1
    check = frame[:-2,:-2]+frame[1:-1,:-2]+frame[2:,:-2]+frame[:-2,1:-1]+ \
                    frame[2:,1:-1]+frame[:-2:,2:]+frame[1:-1,2:]+frame[2:,2:]
    z = np.zeros(frame.shape)
    z[1:-1,1:-1] = check;

    b = []
    rows,cols = np.where(z>0)
    p = [cols[0],rows[0]]
    base = p

    radius = 3
    x=np.roll(np.array(list(p[0]+range(-3,3))+[p[0]+3]*(2*3+1)+list(p[0]+ \
                range(-3,3)[::-1]) +[p[0]-(3+1)]*(2*3+1)),-2)
    y=np.roll(np.array([p[1]-3]*(2*3)+list(p[1]+range(-3,3))+[p[1]+3]*(2*3+1)+ \
                list(p[1]+range(-3,(3+1))[::-1])),-3)
            
    limit = 1500
    tmpRad = False
    for i in range(limit-1):
        b.append(p)
        x=np.roll(np.array(
                list(p[0]+range(-radius,radius)) + \
                [p[0]+radius]*(2*radius+1) + \
                list(p[0]+range(-radius,radius)[::-1]) + \
                [p[0]-(radius+1)]*(2*radius+1)),-2)
        y=np.roll(np.array(
                [p[1]-radius]*(2*radius)+list(p[1] + range(-radius,radius))+ \
                [p[1]+radius]*(2*radius+1)+list(p[1] + \
                range(-radius,(radius+1))[::-1])),-radius)
    
        x[x<0]=0
        y[y<0]=0
        x[x>=z.shape[1]] = z.shape[1]-1
        y[y>=z.shape[0]] = z.shape[0]-1
            
        vals = z[y,x]

        if len(np.where(np.roll(vals,1) == 0)[0]) == 0 or \
                len(np.where(vals>0)[0]) == 0:
            # "confusion failure"
            print 'failure - 1 '
            return roi,False

        idx = np.intersect1d(np.where(vals>0)[0], 
                             np.where(np.roll(vals,1) == 0)[0])[0]
        p = [x[idx],y[idx]]
        
        if ((p[0]-base[0])**2+(p[1]-base[1])**2)**0.5 < 2*radius and len(b) > 3:
            newRoi = ROI(polygons=[np.array(b)],im_shape=roi.im_shape)
            if newRoi.mask.size != 0:
                # "well formed ROI"
                return newRoi,True

        if p in b:
            if radius > 6:
                radius = 3
                z = ndimage.gaussian_filter(z, sigma=1)

                b = []
                rows,cols = np.where(z>0)
                p = [cols[0],rows[0]]
                base = p
                tmpRad = False

            else:
                radius = radius+1
                tmpRad = True
                if len(b) > 3:
                    p = b[-3]
                    del b[-3:]
                
        elif tmpRad:
            tmpRad = False
            radius = 3

    # "terminal failure"
    print 'terminal failure'
    print ((p[0]-base[0])**2+(p[1]-base[1])**2)**0.5 
    print ((p[0]-base[0])**2+(p[1]-base[1])**2)**0.5 < radius
    return roi, False


def patchGradient(img,n=5):
    bstICA = np.array(img)
    #bstICA = bstICA-np.min(bstICA)
    #bstICA = bstICA/np.max(bstICA)
    bstICA = np.abs(bstICA-np.mean(bstICA))

    bstICA[bstICA<np.std(bstICA)] = 0
    bstICA[bstICA>np.std(bstICA)] = 1

    patch_static = np.zeros(bstICA.shape)
    for y in range(bstICA.shape[0]):
        for x in range(bstICA.shape[1]):
            rx = (max((0,x-n)),min(x+n,bstICA.shape[1]))
            ry = (max(0,y-n),min(y+n,bstICA.shape[0]))
        
            patch = bstICA[ry[0]:ry[1],rx[0]:rx[1]]
            patch_val = np.sum(np.abs(patch[1:-1,1:-1]-patch[:-2,1:-1])) + \
                            np.sum(np.abs(patch[1:-1,1:-1]-patch[2:,1:-1])) + \
                            np.sum(np.abs(patch[1:-1,1:-1]-patch[1:-1,:-2])) + \
                            np.sum(np.abs(patch[1:-1,1:-1]-patch[1:-1,2:])) + \
                            np.sum(np.abs(patch[1:-1,1:-1]-patch[2:,2:])) + \
                            np.sum(np.abs(patch[1:-1,1:-1]-patch[:-2,2:])) + \
                            np.sum(np.abs(patch[1:-1,1:-1]-patch[2:,:-2])) + \
                            np.sum(np.abs(patch[1:-1,1:-1]-patch[:-2,:-2]))
            
            patch_static[y,x] = (patch_val*(1.0/(patch.size**(np.sqrt(2)))))

    patch_static = patch_static-np.min(patch_static)
    patch_static = 1-patch_static/np.max(patch_static)
    
    """
    plt.imshow(patch_static)
    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(bstICA)
    plt.show()
    """

    return patch_static,bstICA


def classify(accepted, st_components,storeFile=None,frames=None,force=False):
    print "calculating pixel classifications"

    x_vals=np.linspace(-1,1,4000)

    patch_roi_params = []
    patch_static_params = []

    roi_params = []
    static_params = []

    frame_rois = []
    if frames is None:
        frames = range(len(accepted))
    
    

    for frame_no in frames:
        print frame_no
        if frame_no >= len(accepted):
            print "index excedes available frames"
            continue
   
        rois = _extractStRois([accepted[frame_no]])
        
        frame_rois.append(rois)

        if False:
            fig = plt.figure(figsize=(12,6))
            BLUE = '#32cd32'
            ax = plt.subplot(121)
            plt.imshow(st_components[frame_no])
            for roi in rois:
                for poly in roi.polygons:
                    patch = PolygonPatch(poly, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
                    ax.add_patch(patch)
        
            plt.subplot(122)
            plt.imshow(accepted[frame_no])
            plt.show()

        patch_static,bstICA = patchGradient(st_components[frame_no])

        allmasks = np.sum(np.array([roi.mask.todense() for roi in rois]),axis=0)
    
        patch_vals1 = patch_static[np.where(allmasks)]
        patch_roi_params.append(stats.rayleigh.fit(patch_vals1))

        patch_vals2 = patch_static[np.where(np.logical_not(allmasks))]
        patch_static_params.append(stats.norm.fit(patch_vals2))
    
        stICA_img = np.array(st_components[frame_no])/np.max(st_components[frame_no])
        vals1 = stICA_img[np.where(np.logical_and(allmasks,stICA_img>0))]
        roi_params.append(stats.rayleigh.fit(vals1))
    
        vals2 = stICA_img[np.where(np.logical_and(np.logical_not(allmasks),stICA_img>0))]
        vals2 = np.concatenate((vals2,-vals2))
        static_params.append(stats.norm.fit(vals2))

        
        if frame_no in range(10) and False:

            fig = plt.figure(figsize=(12,6))
            plt.title("Training ROIs Frame")
            BLUE = '#32cd32'
            ax = plt.subplot(121)
            plt.imshow(st_components[frame_no])
            for roi in rois:
                for poly in roi.polygons:
                    patch = PolygonPatch(poly, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
                    ax.add_patch(patch)
            plt.axis([0,rois[0].im_shape[1],0,rois[0].im_shape[0]])

            ####
            """
            fig = plt.figure()

            plt.imshow(patch_static)
            plt.title("Patch Gradients")
            for roi in rois:
                roi.plotROI(fig,linewidth=2,color='k')
            plt.axis([0,rois[0]._imsize[1],0,rois[0]._imsize[0]])

            plt.figure(figsize=(16,8))
            plt.subplot(121)
            patch_roi_p = patch_roi_params[-1]
            plt.hist(patch_vals1,bins=100,normed=1)
            patch_roi_fitted = stats.rayleigh.pdf(x_vals,loc=patch_roi_p[0],scale=patch_roi_p[1])
            plt.plot(x_vals,patch_roi_fitted,linewidth=2.0,color='r')
            plt.title("Patch Gradient ROI Histogram and fitted rayleigh")
            plt.xlim((0,1))

            plt.subplot(122)
            patch_static_p = patch_static_params[-1]
            plt.hist(patch_vals2,bins=100,normed=1)
            patch_static_fitted = stats.norm.pdf(x_vals,loc=patch_static_p[0],scale=patch_static_p[1])
            plt.plot(x_vals,patch_static_fitted,linewidth=2.0,color='r')
            plt.title("Patch Gradient Static Histogram and fitted normal")
            plt.xlim((0,1))
            
            fig = plt.figure(figsize=(16,8))
            plt.subplot(121)
            plt.hist(vals1,bins=100,normed=1)
            fitted1 = stats.rayleigh.pdf(x_vals,loc=roi_params[-1][0],scale=roi_params[-1][1])
            plt.plot(x_vals,fitted1,linewidth=3.0,color='r')
            plt.xlim((0,1))
            plt.title("ROI pixel value histogram and fitted rayleigh")

            plt.subplot(122)
            plt.hist(vals2,bins=150,normed=1)
            fitted2 = stats.norm.pdf(x_vals,loc=static_params[-1][0],scale=static_params[-1][1])
            plt.plot(x_vals,fitted2,linewidth=3.0,color='r')
            plt.xlim((0,1))
            plt.title("non-ROI pixel value histogram and fitted normal")

            """
            ###


            fig = plt.figure(figsize=(12,10))
            plt.subplot(221)
            plt.imshow(patch_static)
            plt.title("Sum of gradients r=4")
            plt.subplot(222)
            plt.imshow(bstICA)
            plt.title("stICA component, converted to binary")

            plt.subplot(223)
            patch_roi_p = patch_roi_params[-1]
            plt.hist(patch_vals1,bins=100,normed=1)
            patch_roi_fitted = stats.rayleigh.pdf(x_vals,loc=patch_roi_p[0],scale=patch_roi_p[1])
            plt.plot(x_vals,patch_roi_fitted,linewidth=2.0,color='r')
            plt.title("Patch Gradient ROI Histogram and fitted rayleigh")
            plt.xlim((0,1))

            plt.subplot(224)
            patch_static_p = patch_static_params[-1]
            plt.hist(patch_vals2,bins=100,normed=1)
            patch_static_fitted = stats.norm.pdf(x_vals,loc=patch_static_p[0],scale=patch_static_p[1])
            plt.plot(x_vals,patch_static_fitted,linewidth=2.0,color='r')
            plt.title("Patch Gradient Static Histogram and fitted normal")
            plt.xlim((0,1))


            fig = plt.figure(figsize=(16,8))
            plt.subplot(121)
            plt.hist(vals1,bins=100,normed=1)
            fitted1 = stats.rayleigh.pdf(x_vals,loc=roi_params[-1][0],scale=roi_params[-1][1])
            plt.plot(x_vals,fitted1,linewidth=3.0,color='r')
            plt.xlim((0,1))
            plt.title("ROI pixel value histogram and fitted rayleigh")

            plt.subplot(122)
            plt.hist(vals2,bins=150,normed=1)
            fitted2 = stats.norm.pdf(x_vals,loc=static_params[-1][0],scale=static_params[-1][1])
            plt.plot(x_vals,fitted2,linewidth=3.0,color='r')
            plt.xlim((0,1))
            plt.title("non-ROI pixel value histogram and fitted normal")

            plt.show()

    static_avg = np.mean(static_params,axis=0)
    roi_avg = np.mean(roi_params,axis=0)
    
    patch_roi_avg = np.mean(patch_roi_params,axis=0)
    patch_static_avg = np.mean(patch_static_params,axis=0)

    if False:
        plt.figure()
        x=np.linspace(-1,1,1000)
        fitted1 = stats.rayleigh.pdf(x,loc=roi_avg[0],scale=roi_avg[1])
        fitted2 = stats.norm.pdf(x,loc=static_avg[0],scale=static_avg[1])
        plt.plot(x,fitted1,color="green",linewidth=2)
        plt.plot(x,fitted2,color="red",linewidth=2)
        plt.xlim((0,1))
        plt.title("Average ROI and static pixel intensity distributions")


        plt.figure()
        x=np.linspace(-1,1,1000)

        fitted1 = stats.rayleigh.pdf(x,loc=patch_roi_avg[0],scale=patch_roi_avg[1])
        fitted2 = stats.norm.pdf(x,loc=patch_static_avg[0],scale=patch_static_avg[1])
        plt.plot(x,fitted1,color="green",linewidth=2)
        plt.plot(x,fitted2,color="red",linewidth=2)
        plt.xlim((0,1))
        plt.title("Average ROI and static patch gradient distributions")

        plt.show()
    
    return static_avg,roi_avg,patch_static_avg,patch_roi_avg,frame_rois

def evalPixels(accepted,st_components,static_params,roi_params,patch_static_params,
               patch_roi_params,frame_rois,storeFile=None,force=False):

    print "evaluating pixels"
    
    x=np.linspace(-1,1,1000)

    fitted1 = stats.rayleigh.pdf(x,loc=roi_params[0],scale=roi_params[1])

    fitted2 = stats.norm.pdf(x,loc=static_params[0],scale=static_params[1])

    x=np.linspace(-1,1,1000)
    fitted1 = stats.rayleigh.pdf(x,loc=patch_roi_params[0],scale=patch_roi_params[1])

    fitted2 = stats.norm.pdf(x,loc=patch_static_params[0],scale=patch_static_params[1])

    results = []
    for i in range(st_components.shape[0]):
        print i
        stFrame = st_components[i]
        #plt.imshow(stFrame)
        #plt.show()
        #stFrame = stFrame[::-1,:]
        #bstICA = np.array(stFrame)

        patch_static,bstICA = patchGradient(stFrame)
        #plt.figure(figsize=(12,6))
        #plt.subplot(121)
        #plt.imshow(patch_static)
        #plt.subplot(122)
        #plt.imshow(bstICA)


        #stFrame = stFrame - np.min(stFrame)
        static_prob = stats.norm.pdf(accepted[i],loc=static_params[0],scale=static_params[1])
        roi_probs = stats.rayleigh.pdf(accepted[i],loc=roi_params[0],scale=roi_params[1])

        patch_static_prob = stats.norm.pdf(patch_static,loc=patch_static_params[0],scale=patch_static_params[1])
        patch_roi_probs = stats.rayleigh.pdf(patch_static,loc=patch_roi_params[0],scale=patch_roi_params[1])

        sp = static_prob+patch_static_prob
        rp = roi_probs+patch_roi_probs
        print "sp: %f, rp %f" %(np.max(sp),np.max(rp))

        result = rp>sp

        frame= result
        check = frame[1:-1,:-2]+frame[1:-1,2:]+frame[:-2,1:-1]+frame[2,1:-1]
        z = np.zeros(frame.shape)
        z[1:-1,1:-1] = check;
        frame[np.logical_not(z)] = 0

        results.append(frame)

        if i in range(10) and False:
            plt.figure()
            plt.imshow(stFrame)
            plt.title("Current Frame")

            plt.figure(figsize=(12,6))
            plt.subplot(121)
            plt.imshow(static_prob)
            plt.title("Fluorescence static prob")
            plt.subplot(122)
            plt.imshow(roi_probs)
            plt.title("Fluorescence ROI prob")

            plt.figure(figsize=(12,6))
            plt.subplot(121)
            plt.imshow(patch_static_prob)
            plt.title("patch gradient static prob")
            plt.subplot(122)
            plt.imshow(patch_roi_probs)
            plt.title("patch gradient roi prob")

            plt.figure()
            plt.imshow(frame)
            plt.title("prob roi > prob static")
            plt.show()

    return results


def stica(dataset,channel=0,mu=0.01,num_components=30,
          static_threshold=0.1,min_area=75,x_smoothing=4,overlap_per=0,smooth_rois=True):
    """ Segmentation for axon/dendrite ROIs based on spatio-temporial ICA

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

    
    Returns
    -------
    rois : list
        A list of sima.ROI ROI objects which have been smoothed
    """

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

    _, space_pcs, time_pcs = _OPCA(dataset, channel, num_components, path=pca_path)
    space_pcs = np.real(space_pcs.reshape(dataset.num_rows,dataset.num_columns,
                                          num_components))

    st_components = _stICA(space_pcs,time_pcs,mu=mu,path=ica_path,
                           n_components=num_components)
    accepted,accepted_components,_ = _findUsefulComponents(
            st_components,static_threshold,x_smoothing=x_smoothing)
    rois = _extractStRois(accepted,min_area=min_area)

    print 'smoothing ROIs'
    if smooth_rois:
        rois = [_smoothROI(roi)[0] for roi in rois]

    rois = _remove_overlapping(rois,percent_overlap=overlap_per)

    return rois


def stica_class(dataset,channel=0,mu=0.01,num_components=30,
                static_threshold=0.1,min_area=75,x_smoothing=4,overlap_per=0,smooth_rois=True):
    """ Segmentation for axon/dendrite ROIs based on spatio-temporial ICA

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

    
    Returns
    -------
    rois : list
        A list of sima.ROI ROI objects which have been smoothed
    """

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
    
    if dataset.savedir is not None:
        res_path = os.path.join(
            dataset.savedir, 'res_' + str(channel) + '.npz')
    else:
        res_path = None
    
    _, space_pcs, time_pcs = _OPCA(dataset, channel, num_components, path=pca_path)
    space_pcs = np.real(space_pcs.reshape(dataset.num_rows,dataset.num_columns,
                                          num_components))

    st_components = _stICA(space_pcs,time_pcs,mu=mu,path=ica_path,
                           n_components=num_components)

    accepted,accepted_components,_ = _findUsefulComponents(
            st_components,0.12,x_smoothing=5)
    
    static_params,roi_params,patch_static_params,patch_roi_params,frame_rois = classify(accepted[:5],accepted_components[:5])

    accepted,accepted_components,_ = _findUsefulComponents(
            st_components,static_threshold,x_smoothing=x_smoothing)

    results = evalPixels(accepted,np.array(accepted_components),static_params,roi_params,patch_static_params,patch_roi_params,frame_rois)
    
    np.savez(res_path,results=results)
    rois = _extractStRois(results,min_area=min_area)
   
    print 'smoothing ROIs'
    if smooth_rois:
        rois = [_smoothROI(roi)[0] for roi in rois]

    #rois = _remove_overlapping(rois,percent_overlap=0.75)

    return rois,accepted
