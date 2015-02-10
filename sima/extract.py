"""Methods used to extract signals from an ImagingDataset."""

import os
from datetime import datetime
import cPickle as pickle
import itertools as it
from multiprocessing import Pool
import warnings

import numpy as np
from scipy.sparse import hstack, vstack, diags, csc_matrix
from scipy.sparse.linalg import inv

# import multiprocessing.util as util
# util.log_to_stderr(util.SUBDEBUG)


def _demixing_matrix(dataset):
    """Calculate the linear transformation to demix two channels.

    Parameters
    ----------
    dataset : ImagingDataset
        The dataset which is to be demixed. This must have two channels.

    Returns
    -------
    array
        Matrix by which the data can be (left) multiplied to be demixed.
    """
    from mdp.nodes import FastICANode

    # Make matrix of the time averaged channels.
    time_avgs = np.concatenate(
        [im.reshape(-1, 1) for im in dataset.time_averages], axis=1)

    # Perform ICA on the time averaged data.
    node = FastICANode()  # TODO: switch to ICA from scikit-learn ?
    node(time_avgs)
    W = np.dot(node.white.v, node.filters).T

    # Reorder and normalize the rows so that the diagonal coefficients
    # are 1 and the off diagonals have minimal magnitude.
    if abs(W[0, 0] / W[0, 1]) < abs(W[1, 0] / W[1, 1]):
        W = W[::-1]
    W[0] /= W[0, 0]
    W[1] /= W[1, 1]
    assert np.allclose(np.diag(W), 1.)

    return W


def _roi_extract(inputs):
    """ROI extract code, intended to be used by extract_rois.
    Needs to be a top-level function to allow it to be used with Pools.

    Parameters - a single two-element tuple, 'inputs'
    ----------
    frame : array
        An individual aligned frame from the imaging session.
    constants : dict
        Variables that do not change each loop and are pre-calculated to speed
        up extraction. Includes demixer, mask_stack, A, masked_pixels, and
        is_overlap.

    Returns - a single three-element tuple
    -------
    values : array
        n_rois length array of average pixel intensity for all pixels
        in each ROI
    demixed_values : array
        If demixer is None, the second returned value will also be None.
        Same format as values, but calculated from the demixed raw signal.
    frame : array
        Return back the frame as it was passed in, used for calculating a mean
        image.

    """
    frame, frame_idx, constants = inputs

    def put_back_nans(values, imaged_rois, n_rois):
        """Puts NaNs back in output arrays for ROIs that were not imaged this
        frame.
        """
        value_idx = 0
        roi_idx = 0
        final_values = np.empty(n_rois)
        while roi_idx < n_rois:
            if roi_idx in imaged_rois:
                final_values[roi_idx] = values[value_idx]
                value_idx += 1
            else:
                final_values[roi_idx] = np.nan
            roi_idx += 1
        return final_values

    n_rois = constants['A'].shape[1]
    masked_frame = frame[constants['masked_pixels']]

    # Determine which pixels and ROIs were imaged this frame
    imaged_pixels = np.isfinite(masked_frame)

    # If there is overlapping pixels between the ROIs calculate the full
    # pseudoinverse of A, if not use a shortcut
    if constants['is_overlap']:
        A = constants['A'][imaged_pixels, :]
        # Identify all of the rois that were imaged this frame
        imaged_rois = np.unique(
            constants['mask_stack'][:, imaged_pixels].nonzero()[0])
        if len(imaged_rois) < n_rois:
            A = A.tocsc()[:, imaged_rois].tocsr()
        # First assume ROIs are independent, if not fallback to full pseudo-inv
        try:
            weights = inv(A.T * A) * A.T
        except RuntimeError:
            weights = csc_matrix(np.linalg.pinv(A.todense()))
    else:
        orig_masks = constants['mask_stack'].copy()
        imaged_masks = orig_masks[:, imaged_pixels]
        imaged_rois = np.unique(imaged_masks.nonzero()[0])
        if len(imaged_rois) < n_rois:
            orig_masks = orig_masks.tocsr()[imaged_rois, :].tocsc()
            imaged_masks = imaged_masks.tocsr()[imaged_rois, :].tocsc()
        orig_masks.data **= 2
        imaged_masks.data **= 2
        scale_factor = orig_masks.sum(axis=1) / imaged_masks.sum(axis=1)
        scale_factor = np.array(scale_factor).flatten()
        weights = diags(scale_factor, 0) \
            * constants['mask_stack'][imaged_rois][:, imaged_pixels]

    # Extract signals
    values = weights * masked_frame[imaged_pixels, np.newaxis]
    weights_sums = weights.sum(axis=1)
    result = values + weights_sums

    if len(imaged_rois) < n_rois:
        result = put_back_nans(result, imaged_rois, n_rois)

    if constants['demixer'] is None:
        return (frame_idx, result, None)

    # Same as 'values' but with the demixed frame data
    demixed_frame = masked_frame + constants['demixer']
    demixed_values = weights * demixed_frame[imaged_pixels, np.newaxis]
    demixed_result = demixed_values + weights_sums

    if len(imaged_rois) < n_rois:
        demixed_result = put_back_nans(demixed_result, imaged_rois, n_rois)
    return (frame_idx, result, demixed_result)


def _save_extract_summary(signals, save_directory, rois):
    """Used to save an extract summary prototype image"""
    import matplotlib as mpl
    mpl.use('pdf')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from sima.ROI import NonBooleanMask

    mean_frame = signals['mean_frame']

    figs = []
    for plane_idx in xrange(mean_frame.shape[0]):
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, rasterized=False)

        ax.imshow(
            mean_frame[plane_idx, :, :], cmap='gray', interpolation='none')

        for mask in signals['_masks']:
            m = mask.toarray().reshape((mean_frame.shape))[plane_idx, :, :]
            if not np.all(m == 0):
                ax.spy(m, marker='.', markersize=2, aspect='auto',
                       color='cyan')

        for roi in rois:
            try:
                for poly in roi.coords:
                    if poly[0][2] == plane_idx:
                        poly -= 0.5  # Shift the polygons to line up with masks
                        ax.plot(
                            poly[:, 0], poly[:, 1], linestyle='-', color='b')
            except NonBooleanMask:
                pass

        if 'overlap' in signals:
            # 'overlap' was calculated on a flat array, so overlap[0] is all
            # '0's and overlap[1] is the actual indices
            overlap_pix = np.unravel_index(
                signals['overlap'][1], mean_frame.shape)
            if len(overlap_pix[0]) and overlap_pix[0][0] == plane_idx:
                ax.plot(overlap_pix[2], overlap_pix[1], 'r.', markersize=2)

        ax.tick_params(bottom=False, top=False, left=False,
                       right=False, labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)

        ax.set_title('Extraction summary: {}\n{}\nPlane {}'.format(
            signals['timestamp'], save_directory, str(plane_idx)))
        figs.append(fig)

    pp = PdfPages(os.path.join(save_directory, 'extractSummary_{}.pdf'.format(
        signals['timestamp'])))
    for fig in figs:
        pp.savefig(fig)
    pp.close()
    plt.close('all')


def _identify_overlapping_pixels(masks):
    """Identify any pixel that is nonzero in more than 1 mask

    Parameters
    ----------
    masks : list of arrays
        A list of boolean mask arrays

    Returns
    -------
    overlap : list of arrays
        Returns the points that were overlapping

    """
    master_mask = np.zeros(masks[0].shape, dtype='uint16')
    for mask in masks:
        master_mask += (mask.todense() != 0).astype('uint16')
    overlap = np.nonzero(master_mask > 1)
    return overlap


def _remove_pixels(masks, pixels_to_remove):
    """Remove pixels from all masks

    Parameters
    ----------
    masks : list of arrays
        List of masks
    pixels_to_remove : list of arrays
        List as [[rows], [cols]] that expands with zip(*pixels_to_remove) to
        iterate over pairs of (row, col) points to remove from all masks.

    Returns
    -------
    new_masks : list of arrays
        Returns original masks with overlapping pixels removed

    """
    new_masks = []
    for mask in masks:
        new_mask = mask.copy().todok()
        for row, col in zip(*pixels_to_remove):
            new_mask[row, col] = 0
        new_masks.append(new_mask.tocoo())
    return new_masks


def extract_rois(dataset, rois, signal_channel=0, remove_overlap=True,
                 n_processes=1, demix_channel=None):
    """Extracts imaging data from the current dataset using the
    supplied ROIs file.

    Parameters
    ----------
    dataset : ImagingDataset
        The dataset from which signals are to be extracted.
    rois : ROIList
        ROIList of rois to extract
    signal_channel : int
        Index of the channel containing the signal to be extracted.
    remove_overlap : bool, optional
        If True, remove any pixels that overlap between masks.
    n_processes : int, optional
        Number of processes to farm out the extraction across. Should be
        at least 1 and at most one less then the number of CPUs in the
        computer. Defaults to 1.
    demix_channel : int, optional
        Index of channel to demix from the signal channel. If None, do not
        demix signals.

    Output - dictionary of arrays
    ------
    raw : array
    demixed_raw : array
    _masks : array
    mean_frame : array
    overlap : array
    signal_channel : int
    rois : list of roi dictionaries
    timestamp : string

    """

    if n_processes > 1:
        pool = Pool(processes=n_processes)

    num_sequences = dataset.num_sequences
    num_planes, num_rows, num_columns, num_channels = dataset.frame_shape

    for roi in rois:
        roi.im_shape = (num_planes, num_rows, num_columns)
    masks = [hstack([mask.reshape((1, num_rows * num_columns))
             for mask in roi.mask]) for roi in rois]

    # Find overlapping pixels
    overlap = _identify_overlapping_pixels(masks)

    # Remove pixels that overlap between ROIs
    if remove_overlap:
        masks = _remove_pixels(masks, overlap)

    # If mask is boolean convert to float and normalize values such that
    # the sum of the weights in each ROI is 1
    for mask_idx, mask in it.izip(it.count(), masks):
        if mask.dtype == bool and mask.nnz:
            masks[mask_idx] = mask.astype('float') / mask.nnz

    # Identify non-empty ROIs
    original_n_rois = len(masks)
    rois_to_include = np.array(
        [idx for idx, mask in enumerate(masks) if mask.nnz > 0])
    n_rois = len(rois_to_include)
    if n_rois != original_n_rois:
        warnings.warn("Empty ROIs will return all NaN values: "
                      + "{} empty ROIs found".format(original_n_rois - n_rois))

    # Stack masks to a 2-d array
    mask_stack = vstack([masks[idx] for idx in rois_to_include]).tocsc()

    # Only include pixels that are included in a ROI
    masked_pixels = np.unique(mask_stack.nonzero()[1])
    mask_stack = mask_stack[:, masked_pixels]

    # A is defined as the pseudoinverse of the mask weights
    if n_rois != 1:
        try:
            A = mask_stack.T * inv(mask_stack * mask_stack.T).tocsc()
        except RuntimeError:
            A = csc_matrix(np.linalg.pinv(mask_stack.todense()))
    else:
        mask_mask_t = mask_stack * mask_stack.T
        mask_mask_t.data = 1 / mask_mask_t.data
        A = mask_stack.T * mask_mask_t.tocsc()

    demixer = None
    if demix_channel is not None:
        demixed_signal = [None] * num_sequences
        demix_matrix = _demixing_matrix(dataset)
        demixer = demix_matrix[signal_channel, demix_channel] * \
            dataset.time_averages[demix_channel]
        demixer = demixer.flatten().astype('float32')[masked_pixels]

    raw_signal = [None] * num_sequences

    def _data_chunker(cycle, time_averages, channel=0):
        """Takes an aligned_data generator for a single cycle
        and returns df/f of each pixel formatted correctly for extraction"""
        while True:
            df_frame = (
                next(cycle)[..., channel] - time_averages[..., channel]) \
                / time_averages[..., channel]
            yield df_frame.flatten()

    for cycle_idx, sequence in it.izip(it.count(), dataset):

        signal = np.empty((n_rois, len(sequence)), dtype='float32')
        if demixer is not None:
            demix = np.empty((n_rois, len(sequence)), dtype='float32')

        constants = {}
        constants['demixer'] = demixer
        constants['mask_stack'] = mask_stack
        constants['A'] = A
        constants['masked_pixels'] = masked_pixels
        constants['is_overlap'] = len(overlap[0]) > 0 and not remove_overlap

        # Determine chunksize and limit to prevent pools from hanging
        chunksize = min(1 + len(sequence) / n_processes, 200)

        # This will farm out signal extraction across 'n_processes' CPUs
        # The actual extraction is in _roi_extract, it's a separate
        # top-level function due to Pool constraints.
        if n_processes > 1:
            map_generator = pool.imap_unordered(_roi_extract, it.izip(
                _data_chunker(
                    iter(sequence), dataset.time_averages, signal_channel),
                it.count(), it.repeat(constants)), chunksize=chunksize)
        else:
            map_generator = it.imap(_roi_extract, it.izip(
                _data_chunker(
                    iter(sequence), dataset.time_averages, signal_channel),
                it.count(), it.repeat(constants)))

        # Loop over generator and extract signals
        while True:
            try:
                frame_idx, raw_result, demix_result = next(map_generator)
            except StopIteration:
                break

            signal[:, frame_idx] = np.array(raw_result).flatten()
            if demixer is not None:
                demix[:, frame_idx] = np.array(demix_result).flatten()

        raw_signal[cycle_idx] = signal
        if demixer is not None:
            demix[np.isinf(demix)] = np.nan
            demixed_signal[cycle_idx] = demix

    if n_processes > 1:
        pool.close()
        pool.join()

    def put_back_nan_rois(signals, included_rois, n_rois):
        """Put NaN rows back in the signals file for ROIs that were never
        imaged or entirely overlapped with other ROIs and were removed.
        """
        final_signals = []
        for cycle_signals in signals:
            signals_idx = 0
            roi_idx = 0
            final_cycle_signals = np.empty((n_rois, cycle_signals.shape[1]))
            nan_row = np.empty((1, cycle_signals.shape[1]))
            nan_row.fill(np.nan)
            while roi_idx < n_rois:
                if roi_idx in included_rois:
                    final_cycle_signals[roi_idx] = cycle_signals[signals_idx]
                    signals_idx += 1
                else:
                    final_cycle_signals[roi_idx] = nan_row
                roi_idx += 1
            final_signals.append(final_cycle_signals)
        return final_signals

    if original_n_rois > n_rois:
        raw_signal = put_back_nan_rois(
            raw_signal, rois_to_include, original_n_rois)
        if demixer is not None:
            demixed_signal = put_back_nan_rois(
                demixed_signal, rois_to_include, original_n_rois)

    signals = {'raw': raw_signal}
    if demixer is not None:
        signals['demixed_raw'] = demixed_signal
    signals['_masks'] = [masks[idx].tolil() for idx in rois_to_include]
    signals['mean_frame'] = dataset.time_averages[..., signal_channel]
    if remove_overlap:
        signals['overlap'] = overlap
    signals['signal_channel'] = signal_channel
    if demix_channel is not None:
        signals['demix_channel'] = demix_channel
    signals['rois'] = [roi.todict() for roi in rois]
    timestamp = datetime.strftime(datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')
    signals['timestamp'] = timestamp

    return signals


def save_extracted_signals(dataset, rois, save_path=None, label=None,
                           metadata=None, signal_channel=0,
                           save_summary=True, **kwargs):
    """Save extracted signals

    Parameters
    ----------
    dataset : ImagingDataset
        ImagingDataset from which to extract signals
    rois : ROIList
        An ROIList of rois to extract signals for
    save_path : string
        Directory in which to store saved signals
    label : string or None
        Text label to describe this extraction, if None defaults to a
        timestamp.
    metadata : dict, optional
        Additional data to save in the corrected signals file. Should be a
        dictionary that will be EXTENDED on to the signals dict, so keys in
        metadata will be keys in signals.pkl. No checking is done, so keys
        should not match any generated during extraction, i.e. 'raw', 'rois'
    signal_channel : int, optional
        Index of channel to extract, defaults to the first channel.
    save_summary : boolean
        If True, additionally save a summary of the extracted ROIs.
    kwargs : dict, optional
        Additional keyword arguments will be pass directly to extract_rois.

    """
    if save_path is None:
        save_path = dataset.savedir
    if save_path is None:
        raise Exception('Cannot save extraction data without a savepath.')

    signals = extract_rois(dataset=dataset, rois=rois,
                           signal_channel=signal_channel, **kwargs)

    if save_summary:
        try:
            _save_extract_summary(signals, save_path, rois)
        except ImportError:
            warnings.warn('Failed to import matplotlib. No extraction '
                          'summary could be saved.')

    signals.pop('_masks')

    if metadata is not None:
        signals.update(metadata)
    if label is None:
        label = signals['timestamp']

    signals_filename = os.path.join(
        save_path, 'signals_{}.pkl'.format(signals['signal_channel']))
    try:
        with open(signals_filename, 'rb') as f:
            sig_data = pickle.load(f)
    except (IOError, pickle.UnpicklingError):
        sig_data = {}
    sig_data[label] = signals
    pickle.dump(sig_data,
                open(signals_filename, 'wb'), pickle.HIGHEST_PROTOCOL)

    return sig_data[label]
