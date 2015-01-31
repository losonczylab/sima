"""
Implementation of normalized cut segmentation methods.

Reference
---------
    Jianbo Shi and Jitendra Malik. Normalized Cuts and Image Segmentation.
    IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE,
    VOL. 22, NO. 8, AUGUST 2000.
"""
import os
from distutils.version import LooseVersion
import itertools as it
import abc

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from scipy import sparse, ndimage

import sima.misc
from sima.ROI import ROI, ROIList
from .segment import Struct, SegmentationStrategy, _check_single_plane
from . import oPCA
from . import _opca

try:
    import cv2
except ImportError:
    cv2_available = False
else:
    cv2_available = LooseVersion(cv2.__version__) >= LooseVersion('2.4.8')


def normcut_vectors(affinity_matrix, k):
    """Return the normalized cut vectors.

    These vectors satisfy are the largest eigenvalues of $D^{-1/2} W D^{-1/2}$,
    or equivalently the smallest of $D^{-1/2} (D - W) D^{-1/2}$. The first
    eigenvalue should be constant in all entries, but the second eigenvalue
    can be used to determine the normalized cut. See Shi & Malik, 2000.

    Parameters
    ----------
    A : array
        The affinity matrix containing the weight between graph nodes.
    k : int
        The number of cut eigenvectors to return.

    Returns
    -------
    array
        The normcut vectors.  Shape (num_nodes, k).
    """
    node_degrees = np.array(affinity_matrix.sum(axis=0)).flatten()
    transformation_matrix = diags(np.sqrt(1. / node_degrees), 0)
    normalized_affinity_matrix = transformation_matrix * affinity_matrix * \
        transformation_matrix
    _, vects = eigsh(normalized_affinity_matrix, k + 1, sigma=1.001,
                     which='LM')  # Get the largest eigenvalues.
    cuts = transformation_matrix * vects
    return cuts


class CutRegion():

    """A subgraph of an affinity matrix used for iteratively cutting with the
    normalized cut procedure.

    Parameters
    ----------
    affinity_matrix : (sparse) array
        The affinities between the nodes of the graph.
    indices : array-like
        Indices of the nodes in the original graph that are contained in the
        CutRegion.
    shape : tuple
        The shape of the image represented by the graph.
    """

    def __init__(self, affinity_matrix, indices, shape):
        self.affinity_matrix = affinity_matrix
        self.indices = indices
        self.shape = shape
        assert self.affinity_matrix.shape[0] == self.affinity_matrix.shape[1]
        assert self.affinity_matrix.shape[0] == len(self.indices)
        self._discard_isolated_nodes()
        assert self.affinity_matrix.shape[0] == self.affinity_matrix.shape[1]
        assert self.affinity_matrix.shape[0] == len(self.indices)

    def _discard_isolated_nodes(self):
        """Removes any nodes with degree 0 from the region."""
        node_degrees = np.array(self.affinity_matrix.sum(axis=0)).flatten()
        ix = np.nonzero(node_degrees)[0]
        if len(ix):
            self.affinity_matrix = self.affinity_matrix[ix, :][:, ix]
            self.indices = self.indices[ix]
        else:
            self.affinity_matrix = np.zeros((0, 0))
            self.indices = self.indices[ix]

    def _normalized_cut_cost(self, cut):
        """Return the normalized cut cost for a given cut.

        Parameters
        ----------
        cut : ndarray of bool
            True/False indicating one of the segments.
        """
        node_degrees = self.affinity_matrix.sum(axis=0)
        k = node_degrees[:, cut].sum() / node_degrees.sum()
        node_degrees = diags(np.array(node_degrees).flatten(), 0)
        b = k / (1 - k)
        y = np.matrix(cut - b * np.logical_not(cut)).T
        return float(y.T * (node_degrees - self.affinity_matrix) * y) / (
            y.T * node_degrees * y)

    def split(self):
        """Split the region according to the normalized cut criterion.

        Returns
        -------
        list of CutRegion
            The regions created by splitting.
        float
            The normalized cut cost.
        """
        if not cv2_available:
            raise ImportError('OpenCV >= 2.4.8 required')
        tmp_im = np.zeros(self.shape[0] * self.shape[1])
        tmp_im[self.indices] = 1
        labeled_array, num_features = ndimage.label(tmp_im.reshape(self.shape))
        if num_features > 1:
            labeled_array = labeled_array.reshape(-1)[self.indices]
            segments = []
            for i in range(1, num_features + 1):
                idx = np.nonzero(labeled_array == i)[0]
                segments.append(CutRegion(self.affinity_matrix[idx, :][:, idx],
                                          self.indices[idx], self.shape))
            return segments, 0.0

        C = normcut_vectors(self.affinity_matrix, 1)
        im = C[:, -2]
        im -= im.min()
        im /= im.max()
        markers = -np.ones(self.shape[0] * self.shape[1]).astype('uint16')
        markers[self.indices] = 0
        markers[self.indices[im < 0.02]] = 1
        markers[self.indices[im > 0.98]] = 2
        markers = markers.reshape(self.shape)
        vis2 = 0.5 * np.ones(self.shape[0] * self.shape[1])
        vis2[self.indices] = im
        vis2 *= (2 ** 8 - 1)
        vis2 = cv2.cvtColor(np.uint8(vis2.reshape(self.shape)),
                            cv2.COLOR_GRAY2BGR)
        markers = np.int32(markers)
        cv2.watershed(vis2, markers)
        cut = ndimage.morphology.binary_dilation(markers == 2).reshape(-1)[
            self.indices]
        cut[im > 0.98] = True
        cost = self._normalized_cut_cost(cut)
        for thresh in np.linspace(0, 1, 20)[1:-1]:
            tmp_cut = im > thresh
            tmp_cost = self._normalized_cut_cost(tmp_cut)
            if tmp_cost < cost:
                cost = tmp_cost
                cut = tmp_cut
        a = cut.nonzero()[0]
        a = cut.nonzero()[0]
        b = np.logical_not(cut).nonzero()[0]

        return (
            [CutRegion(self.affinity_matrix[x, :][:, x], self.indices[x],
                       self.shape) for x in [a, b] if len(x)],
            self._normalized_cut_cost(cut)
        )


def itercut(affinity_matrix, shape, max_pen=0.01, min_size=40, max_size=200):
    """Iteratively cut the graph represented by the affinity matrix.

    Parameters
    ----------
    affinity_matrix : (sparse) array
        The affinities between the nodes of the graph.
    shape : tuple
        The shape of the image represented by the graph.
    max_pen : float
        Iterative cutting will continue as long as the cut cost is less than
        max_pen.
    min_size, max_size : int
        Regardless of the cut cost, iterative cutting will not be performed on
        regions with fewer pixels than min_size and will always be performed
        on regions larger than max_size.

    Returns
    -------
    list of CutRegion
        The regions produced by the iterative cutting procedure.
    """
    cut_cue = [CutRegion(affinity_matrix, np.arange(affinity_matrix.shape[0]),
                         shape)]
    region_list = []

    while len(cut_cue):
        cut = cut_cue.pop()
        if len(cut.indices) < min_size:
            region_list.append(cut)
        else:
            cuts, penalty = cut.split()
            # assert set(cut.indices) == set.union(
            #     *[set(c.indices) for c in cuts])
            if penalty < max_pen or len(cut.indices) > max_size:
                cut_cue.extend(cuts)
            else:
                region_list.append(cut)
    return region_list


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


class DatasetIterable():

    def __init__(self, dataset, channel):
        self.dataset = dataset
        self.channel = sima.misc.resolve_channels(
            channel, dataset.channel_names)
        self.means = dataset.time_averages[..., self.channel].reshape(-1)

    def __iter__(self):
        for cycle in self.dataset:
            for frame in cycle:
                yield np.nan_to_num(
                    frame[..., self.channel].reshape(-1) - self.means)
        raise StopIteration


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
        channel = sima.misc.resolve_channels(self._params.channel,
                                             dataset.channel_names)
        return _offset_corrs(
            dataset, np.concatenate(pairs, 0), channel,
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
                    r1 = (y + dy, x + dx)
                    if (x + dx < shape[1]) and (y + dy >= 0) and \
                            (y + dy < shape[0]):
                        w = self._weight(r0, r1)
                        assert np.isfinite(w)
                        a = x + y * shape[1]
                        b = a + dx + dy * shape[1]
                        A[a, b] = w
                        A[b, a] = w  # TODO: Use symmetric matrix structure
        return sparse.csr_matrix(sparse.coo_matrix(A), dtype=float)


class PlaneNormalizedCuts(SegmentationStrategy):

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

    Warning
    -------
    In version 1.0, this method currently only works on datasets with a
    single plane, or in conjunction with
    :class:`sima.segment.PlaneWiseSegmentation`.

    """
    def __init__(self, affinity_method=None, cut_max_pen=0.01,
                 cut_min_size=40, cut_max_size=200):
        super(PlaneNormalizedCuts, self).__init__()
        if affinity_method is None:
            affinity_method = BasicAffinityMatrix(channel=0, num_pcs=75)
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

    @_check_single_plane
    def _segment(self, dataset):
        params = self._params
        affinity = params.affinity_method.calculate(dataset)
        shape = dataset.frame_shape[1:3]
        cuts = itercut(affinity, shape, params.cut_max_pen,
                       params.cut_min_size, params.cut_max_size)
        return self._rois_from_cuts(cuts)
