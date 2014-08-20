import os
import itertools as it

import numpy as np
from scipy import sparse, ndimage
from scipy.ndimage.measurements import label
from skimage.filter import threshold_otsu
import cv2

####
import matplotlib.pyplot as plt
from scipy import nanmean
from sklearn.decomposition import FastICA
from axonROI import AxonROI
from descartes import PolygonPatch
####

from sima.normcut import itercut
from sima.ROI import ROI, ROIList, mask2poly
import sima.oPCA as oPCA

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
            #pixel values in the cut
            vals = processed_im.flat[cut.indices]

            # indices of those values below the otsu threshold
            # if all values are identical, continue without adding an ROI
            try:
                roi_indices = cut.indices[vals < threshold_otsu(vals)]
            except ValueError:
                continue

            #apply binary opening and closing to the surviving pixels
            twoD_indices = [np.unravel_index(x, shape) for x in roi_indices]
            mask = np.zeros(shape)
            for x in twoD_indices:
                mask[x] = 1
            mask = ndimage.binary_closing(ndimage.binary_opening(mask))
            roi_indices = np.where(mask.flat)[0]

            #label blobs in each cut
            labeled_array, num_features = label(mask)
            for feat in range(num_features):
                blob_inds = np.where(labeled_array.flat == feat + 1)[0]

                #Apply min ROI size threshold
                if len(blob_inds) > min_roi_size:
                    twoD_indices = [np.unravel_index(x, shape)
                                    for x in blob_inds]
                    mask = np.zeros(shape)
                    for x in twoD_indices:
                        mask[x] = 1

                    #APPLY CIRCULARITY THRESHOLD
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
    return cv2.addWeighted(_to8bit(image), image_weight,
                           cv2.GaussianBlur(_to8bit(image), (0, 0), sigma_x,
                                            sigma_y),
                           -mask_weight, 0)


def _clahe(image, x_tile_size=10, y_tile_size=10, clip_limit=20):
    """Perform contrast limited adaptive histogram equalization (CLAHE)."""

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


def findStComponents(space_pcs,time_pcs,mu=0.01,storeFile=None,force=False):
    data=None
    """
    if os.path.isfile(storeFile+'.npz') and not force:
        try:
            print "recovering stICA data from %s" %storeFile
            data = np.load(storeFile+'.npz')['attrs'].item()
            st_components = data['st_components']
            return st_components
        except:
            print "failed to recover saved data"
    """

    if data is None or 'st_components' not in data.keys():
        print "calculating stICA data..."

        for i in range(space_pcs.shape[2]):
            space_pcs[:,:,i] = mu*(space_pcs[:,:,i]-nanmean(space_pcs[:,:,i]))/np.max(space_pcs)
        for i in range(time_pcs.shape[1]):
            time_pcs[:,i] = (1-mu)*(time_pcs[:,i]-nanmean(time_pcs[:,i]))/np.max(time_pcs)

        y = np.concatenate((space_pcs.reshape(space_pcs.shape[0]*space_pcs.shape[1],space_pcs.shape[2]),time_pcs))
        
        print "using Fast ICA..."
        ica = FastICA(n_components=50)
        st_components = np.real(np.array(ica.fit_transform(y)))

        st_components = st_components[:(space_pcs.shape[0]*space_pcs.shape[1]),:]
        st_components = st_components.reshape(space_pcs.shape[0],space_pcs.shape[1],st_components.shape[1])

        n_components = np.zeros(st_components.shape)
        for i in range(st_components.shape[2]):
            st_component = st_components[:,:,i]
            st_component = abs(st_component-np.mean(st_component))
            st_component[np.where(st_component<0)]=0
            st_components[:,:,i] = st_component

        if False:
            plt.imshow(st_components[:,:,0])
            plt.title("Example stICA Component")
            plt.show()

        if storeFile is not None:

            try:
                data = np.load(storeFile+'.npz')['attrs'].item()
            except:
                data = {}

            data['st_components'] = st_components
            np.savez(storeFile,attrs=data)

    return st_components


def findUsefulFrames(st_components,threshold,x_smoothing=4):
    
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
        
        static = static*0.8/(frame.shape[0]*frame.shape[1])
        
        print static
        print "\n"

        if np.sum(static) < threshold:
            accepted.append(frame)
            accepted_components.append(st_components[:,:,i])

            if False:
                plt.figure(figsize=(6,6))
                plt.text(20,20,static,fontsize=12)
                plt.imshow(frame);
                plt.title("Sum(  | Pixel Gradients |  ) = %f" %static)
                plt.show();
        else:
            rejected.append(frame)

    return accepted,accepted_components,rejected


def fndpts(p,arr,frame,i=0,recursion_limit=5000):
    
    if p[0]<0 or p[0]>=frame.shape[0] or p[1]<0 or p[1]>=frame.shape[1] or frame[p[0],p[1]] == 0:
        return []

    if i == -1:
        return []
    
    frame[p[0],p[1]]=0

    arr = fndpts([p[0]-1,p[1]-1],arr,frame,i=i+1)+ \
            fndpts([p[0]-1,p[1]],arr,frame,i=i+1)+ \
            fndpts([p[0]-1,p[1]+1],arr,frame,i=i+1)+ \
            fndpts([p[0],p[1]-1],arr,frame,i=i+1)+ \
            fndpts([p[0],p[1]+1],arr,frame,i=i+1)+ \
            fndpts([p[0]+1,p[1]-1],arr,frame,i=i+1)+ \
            fndpts([p[0]+1,p[1]],arr,frame,i=i+1)+ \
            fndpts([p[0]+1,p[1]+1],arr,frame,i=i+1)
    arr.append(p)
    return arr


def extractRois(frames,storeFile=None,min_area=50,force=False,spatial_sep=True,overlap_per=0):
    
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
            arr = fndpts(p,arr,img)
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

    if overlap_per > 0 and overlap_per < 1:
        print 'removing duplicates'
        for i in xrange(len(rois)):
            for j in [j for j in xrange(len(rois)) if j != i]:
                if rois[i] is not None and rois[j] is not None:
                    overlap = np.logical_and(rois[i].mask.toarray(),rois[j].mask.toarray())
                    small_area = np.min((len(np.where(rois[i].mask.toarray())[0]),len(np.where(rois[j].mask.toarray())[0])))

                    if len(np.where(overlap)[0]) > overlap_per*small_area:
                        new_shape = np.logical_or(rois[i].mask.toarray(),rois[j].mask.toarray())

                        rois[i] = ROI(mask=new_shape.astype('bool'),im_shape=rois[i].mask.shape)
                        rois[j] = None
        rois = [roi for roi in rois if roi is not None]

    return rois


def stica(dataset,channel=0,num_ica_components=50,num_pcs=75,static_threshold=0.1,min_area=75, force_pca=False):
    if not force_pca and dataset.savedir is not None:
        path = os.path.join(
            dataset.savedir, 'opca_' + str(channel) + '.npz')
    else:
        path = None
    
    _, space_pcs, time_pcs = _OPCA(dataset, channel, num_pcs, path=path)
    space_pcs = np.real(space_pcs.reshape(dataset.num_rows,dataset.num_columns,num_pcs))

    st_components = findStComponents(space_pcs,time_pcs)
    accepted,accepted_components,_ = findUsefulFrames(st_components,static_threshold)
    rois = extractRois(accepted,min_area=min_area,overlap_per=0)

    BLUE = '#6699cc'
    plt.figure()
    ax = plt.subplot(111)
    for roi in rois:
        polys = roi.polygons
            
        for poly in polys:
            patch = PolygonPatch(poly, fc=BLUE, ec=BLUE, alpha=0.5, zorder=1)
            ax.add_patch(patch)
        plt.axis([0,st_components.shape[1],0,st_components.shape[0]])
    plt.show()
