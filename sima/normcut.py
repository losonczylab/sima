"""
Implementation of normalized cut segmentation methods.

Reference
---------
    Jianbo Shi and Jitendra Malik. Normalized Cuts and Image Segmentation.
    IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE,
    VOL. 22, NO. 8, AUGUST 2000.
"""
from distutils.version import StrictVersion

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from scipy import ndimage

try:
    import cv2
except ImportError:
    cv2_available = False
else:
    cv2_available = StrictVersion(cv2.__version__) >= StrictVersion('2.4.8')


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
    transformation_matrix = diags(np.sqrt(1./node_degrees), 0)
    normalized_affinity_matrix = transformation_matrix * affinity_matrix * \
        transformation_matrix
    _, vects = eigsh(normalized_affinity_matrix, k+1, sigma=1.001,
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
        b = k / (1-k)
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
        tmp_im = np.zeros(self.shape[0]*self.shape[1])
        tmp_im[self.indices] = 1
        labeled_array, num_features = ndimage.label(tmp_im.reshape(self.shape))
        if num_features > 1:
            labeled_array = labeled_array.reshape(-1)[self.indices]
            segments = []
            for i in range(1, num_features + 1):
                x = np.nonzero(labeled_array == i)[0]
                segments.append(CutRegion(self.affinity_matrix[x, :][:, x],
                                          self.indices[x], self.shape))
            return segments, 0.0

        C = normcut_vectors(self.affinity_matrix, 1)
        im = C[:, -2]
        im -= im.min()
        im /= im.max()
        markers = -np.ones(self.shape[0]*self.shape[1]).astype('uint16')
        markers[self.indices] = 0
        markers[self.indices[im < 0.02]] = 1
        markers[self.indices[im > 0.98]] = 2
        markers = markers.reshape(self.shape)
        vis2 = 0.5 * np.ones(self.shape[0]*self.shape[1])
        vis2[self.indices] = im
        vis2 *= (2**8 - 1)
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

        return [CutRegion(self.affinity_matrix[x, :][:, x], self.indices[x],
                          self.shape) for x in [a, b] if len(x)], \
            self._normalized_cut_cost(cut)


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
