from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
import os

import numpy as np
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean

import sima.misc
from .segment import SegmentationStrategy
from . import oPCA
from sklearn.decomposition import FastICA
from sima.ROI import ROI, ROIList


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
    if time_pcs.shape[-1] != space_pcs.shape[-1]:
        raise ValueError('Different number of time and space PCs.')

    # attempt to retrive the stICA data from a save file
    ret = None
    if path is not None:
        try:
            data = np.load(path)
        except IOError:
            pass
        else:
            if data['st_components'].shape[-1] == n_components and \
                    data['mu'].item() == mu and \
                    data['num_pcs'] == time_pcs.shape[1]:
                ret = data['st_components']
            data.close()

    if ret is not None:
        return ret

    # pre-process the PCA data
    space_factor = mu / np.max(space_pcs)
    time_factor = (1 - mu) / np.max(time_pcs)
    for i in range(space_pcs.shape[-1]):
        space_pcs[..., i] = space_factor * (
            space_pcs[..., i] - nanmean(space_pcs[..., i]))
    for i in range(time_pcs.shape[1]):
        time_pcs[:, i] = time_factor * (
            time_pcs[:, i] - nanmean(time_pcs[:, i]))

    # concatenate the space and time PCs
    y = np.concatenate((
        space_pcs.reshape(
            np.prod(space_pcs.shape[:-1]), space_pcs.shape[-1]),
        time_pcs))

    # execute the FastICA algorithm
    ica = FastICA(n_components=n_components, max_iter=1500)
    st_components = np.real(np.array(ica.fit_transform(y)))

    # pull out the spacial portion of the st_components
    st_components = st_components[:np.prod(space_pcs.shape[:-1]), :]
    st_components = st_components.reshape(
        space_pcs.shape[:-1] + (st_components.shape[-1],))

    # save the ica components if a path has been provided
    if path is not None:
        np.savez(path, st_components=st_components, mu=mu,
                 num_pcs=time_pcs.shape[-1])

    return st_components


class STICA(SegmentationStrategy):

    """
    Segmentation using spatiotemporal independent component analysis (stICA).

    Parameters
    ----------
    channel : string or int, optional
        Channel containing the signal to be segmented, either an integer
        index or a channel name. Default: 0.
    mu : float, optional
        Weighting parameter for the trade off between spatial and temporal
        information. Must be between 0 and 1. Low values give higher
        weight to temporal information. Default: 0.01
    components : int or list, optional
        Number of principal components to use. If list is given, then use
        only the principal components indexed by the list Default: 75
    verbose : bool, optional
        Whether to print progress updates.

    Notes
    -----
    Spatiotemporal (stICA) [1]_ is a procedure which applies ICA to
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
    to balance the trade-off between the spatial and temporal information
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
            self, channel=0, mu=0.01, components=75, verbose=False):
        super(STICA, self).__init__()
        self._params = dict(locals())
        self._params.pop('self')

    def _segment(self, dataset):

        channel = sima.misc.resolve_channels(self._params['channel'],
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

        if self._params['verbose']:
            print('performing PCA...')
        components = self._params['components']
        if isinstance(components, int):
            components = list(range(components))
        _, space_pcs, time_pcs = oPCA.dataset_opca(
            dataset, channel, components[-1] + 1, path=pca_path)
        space_pcs = np.real(space_pcs)

        # Remove components greater than the number of PCs returned
        # in case more components were asked for than the number of
        # independent dimensions in the dataset.
        components = [c for c in components if c < time_pcs.shape[1]]

        if self._params['verbose']:
            print('performing ICA...')
        st_components = _stica(
            space_pcs, time_pcs, mu=self._params['mu'], path=ica_path,
            n_components=space_pcs.shape[-1])

        return ROIList([ROI(st_components[..., i]) for i in
                        range(st_components.shape[-1])])
