"""
Offset principal component analysis functions.
"""
import numpy as np
from scipy.stats import nanmean
from scipy.linalg import eig, eigh, inv, norm, cholesky
from scipy.sparse.linalg import eigsh, eigs
import warnings


def _method_1(data, num_pcs=None):
    """Compute OPCA when num_observations > num_dimensions."""
    data = data - nanmean(data, axis=0)
    data[np.logical_not(np.isfinite(data))] = 0.
    T = data.shape[0]
    corr_offset = np.dot(data[1:].T, data[0:-1])
    corr_offset += corr_offset.T
    if num_pcs is None:
        eivals, eivects = eigh(corr_offset)
    else:
        eivals, eivects = eigsh(corr_offset, num_pcs, which='LA')
    idx = np.argsort(-eivals)  # sort the eigenvectors and eigenvalues
    eivals = eivals[idx] / (2. * (T-1))
    eivects = eivects[:, idx]
    return eivals, eivects, np.dot(data, eivects)


def _method_2(data, num_pcs):
    """Compute OPCA when num_observations <= num_dimensions."""
    data = data - nanmean(data, axis=0)
    data[np.logical_not(np.isfinite(data))] = 0.
    T = data.shape[0]
    tmp = np.dot(data, data.T)
    corr_offset = np.zeros(tmp.shape)
    corr_offset[:, 1:] = tmp[:, 0:-1]
    corr_offset[:, 0:-1] += tmp[:, 1:]
    if num_pcs is None:
        eivals, eivects = eig(corr_offset)
    else:
        eivals, eivects = eigs(corr_offset, num_pcs, which='LR')
    eivals = np.real(eivals)
    idx = np.argsort(-eivals)  # sort the eigenvectors and eigenvalues
    eivals = eivals[idx] / (2. * (T-1))
    eivects = eivects[:, idx]
    transformed_eivects = np.zeros(eivects.shape)  # trasform to signal space
    transformed_eivects[1:, :] = eivects[0:-1, :]
    transformed_eivects[0:-1, :] += eivects[1:, :]
    transformed_eivects = np.dot(data.T, transformed_eivects)
    for i in range(transformed_eivects.shape[1]):  # normalize the eigenvectors
        transformed_eivects[:, i] /= np.linalg.norm(transformed_eivects[:, i])
    return eivals, transformed_eivects, np.dot(data, transformed_eivects)


def EM_oPCA(data, num_pcs, tolerance=0.01, max_iter=1000):
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
    X = data - nanmean(data, axis=0)
    X[np.logical_not(np.isfinite(X))] = 0.
    T = X.shape[0]

    OX = np.zeros_like(X)
    OX[1:] += X[:-1]
    OX[:-1] += X[1:]

    DT = int(np.floor(float(T)/num_pcs))
    U = np.concatenate([X[i*DT:(i+1)*DT].mean(axis=0).reshape(1, -1)
                        for i in range(num_pcs)], axis=0)
    assert np.all(np.isfinite(U))
    for i in range(num_pcs):
        U[i] /= norm(U[i])
    iter_count = 0
    while True:
        iter_count += 1
        if iter_count > max_iter:
            warnings.warn("max_iter reached by EM_oPCA")
            break
        Z = np.dot(np.dot(U, X.T), OX)  # TODO: parallelize
        U_new = np.dot(np.dot(np.dot(U, U.T), inv(np.dot(Z, U.T))), Z)
        error = max(norm(U_new[i] - U[i]) / norm(U[i]) for i in range(num_pcs))
        print "iter_count:", iter_count, "\t\terror:", error, \
            '\t\tnorm', norm(U), norm(U_new)
        U = U_new / norm(U_new)
        if error < tolerance:
            break
    # project data with U
    orthonormal_basis = np.dot(U.T, inv(cholesky(np.dot(U, U.T))))
    reduced_data = np.dot(data,  orthonormal_basis)
    eivals, eivects, coeffs = offsetPCA(reduced_data)
    eivects = np.dot(orthonormal_basis, eivects)
    return eivals, eivects, coeffs


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
    """
    X = data - nanmean(data, axis=0)
    X[np.logical_not(np.isfinite(X))] = 0.
    OX = np.zeros_like(X)
    OX[1:] += X[:-1]
    OX[:-1] += X[1:]

    eivects, eivals = [], []
    for pc_idx in range(num_pcs):
        U = X[0].reshape(1, -1)  # np.random.randn(num_pcs, p)
        U /= norm(U)
        iter_count = 0
        while True:
            iter_count += 1
            if iter_count > max_iter:
                warnings.warn("max_iter reached by power_iteration_oPCA")
                break
            Z = np.dot(np.dot(U, X.T), OX)  # TODO: parallelize
            U_new = np.dot(np.dot(np.dot(U, U.T), inv(np.dot(Z, U.T))), Z)
            error = norm(U_new - U) / norm(U)
            U = U_new / norm(U_new)
            if error < tolerance:
                break
        eivects.append(U.T)
        eivals.append(float(np.dot(np.dot(U, X.T), np.dot(OX, U.T)) /
                            np.dot(U, U.T)) / (2. * (X.shape[0] - 1.)))
        XUtU = np.dot(np.dot(X, U.T), U)
        X -= XUtU
        OX[1:] -= XUtU[:-1]
        OX[:-1] -= XUtU[1:]
        print 'Eigenvalue', pc_idx + 1, 'found'
    eivects = np.concatenate(eivects, axis=1)
    for i in range(eivects.shape[1]):
        eivects[:, i] /= norm(eivects[:, i])
    Z = np.dot(np.dot(eivects.T, X.T), OX)
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
    data = data - nanmean(data, axis=0)
    data = np.nan_to_num(data)
    # Determine most efficient method of computation.
    if data.shape[0] > data.shape[1]:
        return _method_1(data, num_pcs)
    else:
        return _method_2(data, num_pcs)
