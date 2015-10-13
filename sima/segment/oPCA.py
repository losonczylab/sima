"""
Offset principal component analysis functions.
"""
from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import numpy as np
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean
from scipy.linalg import eig, eigh, inv, norm
from scipy.sparse.linalg import eigsh, eigs
import warnings

import sima.misc
from . import _opca


def _method_1(data, num_pcs=None):
    """Compute OPCA when num_observations > num_dimensions."""
    data = np.nan_to_num(data - nanmean(data, axis=0))
    T = data.shape[0]
    corr_offset = np.dot(data[1:].T, data[:-1])
    corr_offset += corr_offset.T
    if num_pcs is None:
        eivals, eivects = eigh(corr_offset)
    else:
        eivals, eivects = eigsh(corr_offset, num_pcs, which='LA')
    eivals = np.real(eivals)
    eivects = np.real(eivects)
    idx = np.argsort(-eivals)  # sort the eigenvectors and eigenvalues
    eivals = old_div(eivals[idx], (2. * (T - 1)))
    eivects = eivects[:, idx]
    return eivals, eivects, np.dot(data, eivects)


def _method_2(data, num_pcs=None):
    """Compute OPCA when num_observations <= num_dimensions."""
    data = np.nan_to_num(data - nanmean(data, axis=0))
    T = data.shape[0]
    tmp = np.dot(data, data.T)
    corr_offset = np.zeros(tmp.shape)
    corr_offset[1:] = tmp[:-1]
    corr_offset[:-1] += tmp[1:]
    if num_pcs is None:
        eivals, eivects = eig(corr_offset)
    else:
        eivals, eivects = eigs(corr_offset, num_pcs, which='LR')
    eivals = np.real(eivals)
    eivects = np.real(eivects)
    idx = np.argsort(-eivals)  # sort the eigenvectors and eigenvalues
    eivals = old_div(eivals[idx], (2. * (T - 1)))
    eivects = eivects[:, idx]
    transformed_eivects = np.dot(data.T, eivects)
    for i in range(transformed_eivects.shape[1]):  # normalize the eigenvectors
        transformed_eivects[:, i] /= np.linalg.norm(transformed_eivects[:, i])
    return eivals, transformed_eivects, np.dot(data, transformed_eivects)


def EM_oPCA(data, num_pcs, tolerance=0.01, max_iter=1000, verbose=False):
    """Calculate the leading magnitude PCs with the EM algorithm.

    Adapted from the EM algorithm for PCA [1].

    Parameters
    ----------
    data : array
        see offsetPCA
    num_pcs : int
        Then number of oPCs to be computed.
    tolerance : float, optional
        The criterion for fractional difference between subsequent estimates
        used to determine when to terminate the algorithm. Default: 0.01.
    max_iter : int, optional
        The maximum number of iterations. Default: 1000.

    Returns
    -------
    see offsetPCA

    References
    -----
    .. [1] S. Roweis. EM Algorithms for PCA and SPCA. Advances in Neural
       Information Processing Systems. 1998.

    """

    for X in (data):
        p = X.size
        break
    # initialize by calling oPCA on first 2 * num_pcs frames
    first_frames = np.zeros((2 * num_pcs, p))
    for i, X in enumerate(data):
        if i == 2 * num_pcs:
            break
        first_frames[i] = X
    U = offsetPCA(first_frames[:i])[1].T[:num_pcs]
    Z = np.zeros((num_pcs, p))
    U = np.linalg.qr(U.T)[0].T
    iter_count = 0
    eivals_old = np.zeros(num_pcs)
    while True:
        iter_count += 1
        if iter_count > max_iter:
            warnings.warn("max_iter reached by EM_oPCA")
            break
        _opca._Z_update(Z, U, data)
        ZUT = np.dot(Z, U.T)
        eivals = np.diag(ZUT)
        order = np.argsort(abs(eivals))[::-1]
        error = old_div(np.sum(abs(eivals - eivals_old)), np.sum(abs(eivals)))
        eivals_old = eivals[order]
        # Although part of the original OPCA algorithm, the line below
        # causes problems with convergence for large numbers of PCs,
        # presumably because of inaccuracies in the inverse. Therefore
        # we have replaced it with QR decomposition to obtain orthogonal
        # eigenvectors via Gram-Schmidt.
        # U = np.dot(np.dot(np.dot(U, U.T), inv(ZUT)), Z)[order]
        U = np.linalg.qr(Z[order].T)[0].T
        if verbose:
            print("iter_count:", iter_count, "\t\terror:", error,
                  '\t\teivals', eivals)
        if error < tolerance:
            break
    # project data with U
    reduced_data = _project(data, U.T)
    eivals, eivects, coeffs = offsetPCA(reduced_data)
    eivects = np.dot(U.T, eivects)
    return eivals, eivects, coeffs


def _project(iterator, matrix):
    return np.concatenate(
        [np.dot(x, matrix).reshape(1, -1) for x in iterator])


def power_iteration_oPCA(data, num_pcs, tolerance=0.01, max_iter=1000):
    """Compute OPCs recursively with the power iteration method.

    Parameters
    ----------
    data : array
        see offsetPCA
    num_pcs : int
        Then number of oPCs to be computed.
    tolerance : float, optional
        The criterion for fractional difference between subsequent estimates
        used to determine when to terminate the algorithm. Default: 0.01.
    max_iter : int, optional
        The maximum number of iterations. Default: 1000.

    Returns
    -------
    see offsetPCA


    WARNING: INCOMPLETE!!!!!!!!!!!

    """

    for X in (data):
        U0 = X.reshape(1, -1)
        p = X.shape
        break
    eivects, eivals = [], []
    Z = np.zeros((1, p))
    for pc_idx in range(num_pcs):
        U = old_div(U0, norm(U0))  # np.random.randn(num_pcs, p)
        iter_count = 0
        while True:
            print(iter_count)
            iter_count += 1
            if iter_count > max_iter:
                warnings.warn("max_iter reached by power_iteration_oPCA")
                break
            _opca._Z_update(Z, U, data)
            U_new = np.dot(np.dot(np.dot(U, U.T), inv(np.dot(Z, U.T))), Z)
            error = old_div(norm(U_new - U), norm(U))
            U = old_div(U_new, norm(U_new))
            if error < tolerance:
                break
        eivects.append(U.T)
        eivals.append(float(np.dot(Z, U.T) / np.dot(U, U.T)) /
                      (2. * (X.shape[0] - 1.)))
        """
        XUtU = np.dot(np.dot(X, U.T), U)
        X -= XUtU
        OX[1:] -= XUtU[:-1]
        OX[:-1] -= XUtU[1:]  # TODO: isn't this wrong???
        """
        print('Eigenvalue', pc_idx + 1, 'found')
    eivects = np.concatenate(eivects, axis=1)
    for i in range(eivects.shape[1]):
        eivects[:, i] /= norm(eivects[:, i])
    eivals = np.array(eivals)
    idx = np.argsort(-eivals)
    return eivals[idx], eivects[:, idx], np.dot(X, eivects[:, idx])


def offsetPCA(data, num_pcs=None):
    """Perform offsetPCA on the data.

    Parameters
    ----------
    data : array
        Data array of shape (num_observations, num_dimensions)
    num_pcs : int, optional
        The number of offset principal components (OPCs) to be computed.
        Default (None) results in all possible OPCs being computed.

    Returns
    -------
    variances : array
        The offset variances of each offset principal component (oPC).
    OPCs : array
        The calculated OPCs.  Shape: (num_dimensions, num_pcs).
    signals : array
        The signals for each OPC. Shape: (num_observations, num_pcs).

    """

    # Determine most efficient method of computation.
    if data.shape[0] > data.shape[1]:
        return _method_1(data, num_pcs)
    else:
        return _method_2(data, num_pcs)


"""
BELOW IS CODE FOR RUNNING OPCA ON A SIMA DATASET
"""


class DatasetIterable(object):

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


def dataset_opca(dataset, ch=0, num_pcs=75, path=None, verbose=False):
    """Perform offset principal component analysis on the dataset.

    Parameters
    ----------
    dataset : ImagingDataset
        The dataset to which the offset PCA will be applied.
    ch : string or int, optional
        Channel containing the signal to be used, either an integer
        index or a channel name. Default: 0.
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
        Shape: (num_planes, num_rows, num_columns, num_pcs).
    oPC_signals : array
        The temporal representations of the oPCs.
        Shape: (num_times, num_pcs).

    """

    ch = sima.misc.resolve_channels(ch, dataset.channel_names)

    num_pcs = min(num_pcs, dataset.num_frames - 1,
                  np.prod(dataset.frame_shape[:3]))
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
                    data['oPCs'][..., :num_pcs],
                    data['oPC_signals'][:, :num_pcs]
                )
            data.close()
    if ret is not None:
        return ret
    oPC_vars, oPCs, oPC_signals = EM_oPCA(
        DatasetIterable(dataset, ch), num_pcs=num_pcs, verbose=verbose)
    oPCs = oPCs.reshape(dataset.frame_shape[:3] + (-1,))
    if path is not None:
        np.savez(path, oPCs=oPCs, oPC_vars=oPC_vars, oPC_signals=oPC_signals)
    return oPC_vars, oPCs, oPC_signals
