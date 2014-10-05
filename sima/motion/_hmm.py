import itertools as it
import warnings
try:
    from future_builtins import zip
except ImportError:  # Python 3.x
    pass

import numpy as np
from numpy.linalg import det, svd, pinv
from scipy.special import gammaln
try:
    from bottleneck import nansum, nanmedian
except ImportError:
    from numpy import nansum
    from scipy.stats import nanmedian
from scipy.stats.mstats import mquantiles
# from scipy.ndimage.filters import gaussian_filter

try:
    import _motion as mc
except ImportError as error:
    # Attempt auto-compilation.
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()},
                      reload_support=True)
    import _motion as mc
from sima.imaging import ImagingDataset
import sima.motion.frame_align
import sima.misc
from misc import trim_coords as _trim_coords

np.seterr(invalid='ignore', divide='ignore')


def _discrete_transition_prob(r, r0, transition_probs, n):
    """Calculate the transition probability between two discrete position
    states.

    Parameters
    ----------
    r : array
        The location being transitioned to.
    r0 : array
        The location being transitioned from.
    transition_probs : function
        The continuous transition probability function.
    n : int
        The number of partitions along each axis.

    Returns
    -------
    float
        The discrete transition probability between the two states.
    """
    p = 0.
    for x in (r[0] + np.linspace(-0.5, 0.5, n + 2))[1:-1]:
        for x0 in (r0[0] + np.linspace(-0.5, 0.5, n + 2))[1:-1]:
            for y in (r[1] + np.linspace(-0.5, 0.5, n + 2))[1:-1]:
                for y0 in (r0[1] + np.linspace(-0.5, 0.5, n + 2))[1:-1]:
                    p += transition_probs(np.array([x, y]), np.array([x0, y0]))
    return p / (n ** 4)


def _estimate_movement_model(shifts, num_rows):
    """Estimate the HMM motion parameters by fitting to an AR(1) model

    .. math:: D_t \sim N(A D_{t-1}, C)

    Parameters
    ----------
    shifts : list of array
        The estimated displacement of each frame.  Size: (2, T).
    num_rows : int
        The number of rows in each image frame.

    Returns
    -------
    cov_matrix : array
        Estimated covariance matrix for the
    decay_matrix : array

    log_transition_matrix : array
        The log transition probabilities for nearest neighbor discrete jumps.
    """
    shifts = np.array(list(it.chain(*it.chain(*shifts))))
    assert shifts.shape[1] == 2
    diffs = np.diff(shifts, axis=0)
    num_frames = len(shifts)
    cov_matrix = np.cov(diffs[np.isfinite(diffs).all(axis=1)].T) / num_rows
    # don't allow singular covariance matrix
    cov_matrix[0, 0] = max(cov_matrix[0, 0], 1. / (num_frames * num_rows))
    cov_matrix[1, 1] = max(cov_matrix[1, 1], 1. / (num_frames * num_rows))
    assert det(cov_matrix) > 0

    mean_shift = np.nanmean(shifts, axis=0)
    assert len(mean_shift) == 2
    centered_shifts = np.nan_to_num(shifts - mean_shift)
    # fit to autoregressive AR(1) model
    A = np.dot(pinv(centered_shifts[:-1]), centered_shifts[1:])
    # symmetrize A, assume independent motion on orthogonal axes
    A = 0.5 * (A + A.T)
    U, s, V = svd(A)
    s **= 1. / num_rows
    decay_matrix = np.dot(U, np.dot(np.diag(s), U))  # take A^(1/num_rows)
    # NOTE: U == V for positive definite A

    # Gaussian Transition Probabilities
    transition_probs = lambda x, x0: 1. / np.sqrt(
        2 * np.pi * det(cov_matrix)) * np.exp(
        -np.dot(x - x0, np.linalg.solve(cov_matrix, x - x0)) / 2.)
    max_jump_size = 1  # Only allow nearest-neighbor shifts (per line)
    log_transition_matrix = -np.inf * np.ones([
        max_jump_size + 1, max_jump_size + 1])
    for i in range(max_jump_size + 1):
        for j in range(max_jump_size + 1):
            log_transition_matrix[i, j] = np.log(
                _discrete_transition_prob(
                    np.array([i, j]), np.array([0., 0.]), transition_probs, 8))
    assert np.all(np.isfinite(cov_matrix)) and \
        np.all(np.isfinite(decay_matrix)) and \
        np.all(np.isfinite(log_transition_matrix))
    return cov_matrix, decay_matrix, log_transition_matrix, mean_shift


def _threshold_gradient(im):
    """Indicate pixel locations with gradient below the bottom 10th percentile

    Parameters
    ----------
    im : array
        The mean intensity images for each channel.
        Size: (num_channels, num_rows, num_columns).

    Returns
    -------
    array
        Binary values indicating whether the magnitude of the gradient is below
        the 10th percentile.  Same size as im.
    """
    if im.shape[0] > 1:
        # Calculate directional relative derivatives
        _, g_x, g_y = np.gradient(np.log(im))
    else:
        # Calculate directional relative derivatives
        g_x, g_y = np.gradient(np.log(im[0]))
        g_x = g_x.reshape([1, g_x.shape[0], g_x.shape[1]])
        g_y = g_y.reshape([1, g_y.shape[0], g_y.shape[1]])
    gradient_magnitudes = np.sqrt((g_x ** 2) + (g_y ** 2))
    below_threshold = []
    for chan in gradient_magnitudes:
        threshold = mquantiles(chan[np.isfinite(chan)].flatten(), [0.1])[0]
        below_threshold.append(chan < threshold)
    return np.array(below_threshold)


def _initial_distribution(decay, noise_cov, mean_shift):
    """Get the initial distribution of the displacements."""
    initial_cov = np.linalg.solve(np.diag([1, 1]) - decay * decay.T,
                                  noise_cov.newbyteorder('>').byteswap())
    for _ in range(1000):
        initial_cov = decay * initial_cov * decay.T + noise_cov
    # don't let C be singular
    initial_cov[0, 0] = max(initial_cov[0, 0], 0.1)
    initial_cov[1, 1] = max(initial_cov[1, 1], 0.1)

    return lambda x: 1.0 / np.sqrt(2.0 * np.pi * det(initial_cov)) * np.exp(
        -np.dot(x - mean_shift, np.linalg.solve(initial_cov, x - mean_shift)
                ) / 2.0)


def _lookup_tables(d_min, d_max, log_markov_matrix,
                   num_columns, references, offset):
    """Generate lookup tables to speed up the algorithm performance.

    Parameters
    ----------
    d_min : int
        The minimum allowable displacement.
    d_max : int
        The maximum allowable displacement.
    log_markov_matrix :
        The log transition probabilities.
    num_columns : int
        The number of columns in the 2-photon images.
    references : list of array
        The reference images for each channel.
    offset : array
        The displacement to add to each shift to align the minimal shift with
        the edge of the corrected image.

    Returns
    -------
    position_tbl : array
        Lookup table used to index each possible displacement.
    transition_tbl : array
        Lookup table used to find the indices of displacements to which
        transitions can occur from the position.
    log_markov_matrix_tbl : array
        Lookup table used to find the transition probability of the transitions
        from transition_tbl.
    slice_tbl : array
        Lookup table for the indices to use for slicing image arrays.
    """
    position_tbl = []
    for j in range(d_min[1], d_max[1] + 1):
        for i in range(d_min[0], d_max[0] + 1):
            position_tbl.append([i, j])
    position_tbl = np.array(position_tbl, dtype=int)

    # create transition lookup and create lookup for transition probability
    transition_tbl = []
    log_markov_matrix_tbl = []
    for k in range(9):
        stp = np.array([(k % 3) - 1, (k / 3) - 1], dtype=int)
        tmp_tbl = []
        for i in range(np.prod(d_max - d_min + 1)):
            position = position_tbl[i] + stp
            if np.all(position >= d_min) and np.all(position <= d_max):
                # get index of position
                idx = (position[1] - d_min[1]) * (d_max[0] - d_min[0] + 1)\
                    + position[0] - d_min[0]
                tmp_tbl.append(idx)
                assert np.array_equal(position_tbl[idx], position)
            else:
                tmp_tbl.append(-1)
        transition_tbl.append(tmp_tbl)
        log_markov_matrix_tbl.append(
            log_markov_matrix[abs(stp[0]), abs(stp[1])])
    transition_tbl = np.array(transition_tbl, dtype=int)
    log_markov_matrix_tbl = np.fromiter(log_markov_matrix_tbl, dtype=float)
    slice_tbls = [mc.slice_lookup(r, position_tbl, num_columns, offset)
                  for r in references]
    assert position_tbl.dtype == int
    return position_tbl, transition_tbl, log_markov_matrix_tbl, slice_tbls


def _backtrace(start_idx, backpointer, states, position_tbl):
    """Perform the backtracing stop of the Viterbi algorithm.

    Parameters
    ----------
    start_idx : int
        ...

    Returns:
    --------
    trajectory : array
        The maximum aposteriori trajectory of displacements.
        Shape: (2, len(states))
    """
    T = len(states)
    i = start_idx
    trajectory = np.zeros([T, 2], dtype=int)
    trajectory[-1] = position_tbl[states[-1][i]]
    for t in xrange(T - 2, -1, -1):
        # NOTE: backpointer index 0 corresponds to second timestep
        i = backpointer[t][i]
        trajectory[t] = position_tbl[states[t][i]]
    return trajectory


class _MCImagingDataset(object):

    """ImagingDataset sub-classed with motion correction functionality"""

    def __init__(self, sequences):
        self.dataset = ImagingDataset(sequences, None)

    def _neighbor_viterbi(
            self, log_transition_matrix, references, gains, decay_matrix,
            cov_matrix_est, mean_shift, offset, min_displacements,
            max_displacements, pixel_means, pixel_variances,
            num_states_retained, verbose=True):
        """Estimate the MAP trajectory with the Viterbi Algorithm.

        See _MCCycle.neighbor_viterbi for details."""
        displacements = []
        for i, cycle in enumerate(self.dataset):
            if verbose:
                print 'Estimating displacements for cycle ', i
            displacements.append(
                SolverHMM(cycle).neighbor_viterbi(
                    log_transition_matrix, references, gains, decay_matrix,
                    cov_matrix_est, mean_shift, offset, min_displacements,
                    max_displacements, pixel_means, pixel_variances,
                    num_states_retained, verbose=verbose
                )
            )
        return displacements

    def estimate_displacements(
            self, num_states_retained=50, max_displacement=None,
            verbose=True, path=None, n_processes=None):
        """Estimate and save the displacements for the time series.

        Parameters
        ----------
        num_states_retained : int
            Number of states to retain at each time step of the HMM.
        max_displacement : array of int
            The maximum allowed displacement magnitudes in [y,x].
        path : str, optional
            Path for saving a record of the displacement estimation.
            If there is already a .pkl file, the data will be added
            to this file.

        Returns
        -------
        dict
            The estimated displacements and partial results of motion
            correction.
        """
        if verbose:
            print 'Estimating model parameters.'
        if max_displacement is not None:
            max_displacement = np.array(max_displacement)
        else:
            max_displacement = np.array([-1, -1])

        shifts, correlations = self._correlation_based_correction(
            max_displacement=max_displacement, n_processes=n_processes)
        references, variances, offset = self._whole_frame_shifting(
            shifts, correlations)
        gains = nanmedian(
            (variances / references).reshape(-1, references.shape[-1]))
        # gains = self._estimate_gains(references, offset,
        #                              shifts.astype(int), correlations)
        assert np.all(np.isfinite(gains)) and np.all(gains > 0)
        pixel_means, pixel_variances = self._pixel_distribution()
        cov_matrix_est, decay_matrix, log_transition_matrix, mean_shift = \
            _estimate_movement_model(shifts, self.dataset.frame_shape[1])

        # add a bit of extra room to move around
        min_shifts = np.nanmin(list(it.chain(*it.chain(*shifts))), 0)
        max_shifts = np.nanmax(list(it.chain(*it.chain(*shifts))), 0)
        extra_buffer = (
            (max_displacement - max_shifts + min_shifts) / 2).astype(int)
        extra_buffer[max_displacement < 0] = 5
        min_displacements = (min_shifts - extra_buffer)
        max_displacements = (max_shifts + extra_buffer)

        displacements = self._neighbor_viterbi(
            log_transition_matrix, references, gains, decay_matrix,
            cov_matrix_est, mean_shift, offset, min_displacements,
            max_displacements, pixel_means, pixel_variances,
            num_states_retained, verbose)
        """
        if path is not None:
            d = np.zeros(len(displacements), dtype=np.object)
            for i, disp in enumerate(displacements):
                d[i] = np.array(disp)
            r = np.zeros(len(references), dtype=np.object)
            for i, ref in enumerate(references):
                r[i] = np.array(ref)
            record = {'displacements': d,
                      'shifts': shifts,
                      'references': r,
                      'gains': gains,
                      'cov_matrix_est': cov_matrix_est,
                      'max_displacements': max_displacements}
            try:
                with open(path, 'wb'):
                    data = pickle.load(path)
            except IOError:
                data = {}
            data['motion_correction'] = record
            mkdir_p(os.path.dirname(path))
            with open(path, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        """

        # make all the displacements non-negative
        min_disp = np.min(
            list(it.chain(*it.chain(*it.chain(*displacements)))), axis=0)
        return [disp - min_disp for disp in displacements]

    def _pixel_distribution(self, tolerance=0.001, min_frames=1000):
        """Estimate the distribution of pixel intensities for each channel.

        Parameters
        ----------
        tolerance : float
            The maximum relative error in the estimates that must be
            achieved for termination.
        min_frames: int
            The minimum number of frames that must be evaluated before
            termination.

        Returns
        -------
        mean_est : array
            Mean intensities of each channel.
        var_est :
            Variances of the intensity of each channel.
        """
        # TODO: separate distributions for each plane
        sums = np.zeros(self.dataset.frame_shape[-1]).astype(float)
        sum_squares = np.zeros_like(sums)
        counts = np.zeros_like(sums)
        t = 0
        for plane in it.chain(*it.chain(*self.dataset)):
            if t > 0:
                mean_est = sums / counts
                var_est = (sum_squares / counts) - (mean_est ** 2)
            if t > min_frames and np.all(
                    np.sqrt(var_est / counts) / mean_est < tolerance):
                break
            # im = np.concatenate(
            #     [np.expand_dims(x, 0) for x in plane],
            #     axis=0).astype(float)  # NOTE: integers overflow
            # sums += im.sum(axis=0).sum(axis=0)
            # sum_squares += (im ** 2).sum(axis=0).sum(axis=0)
            # cnt += np.prod(im.shape[0] * im.shape[1])
            sums += nansum(nansum(plane, axis=0), axis=0)
            sum_squares += nansum(nansum(plane ** 2, axis=0), axis=0)
            counts += np.isfinite(plane).sum(axis=0).sum(axis=0)
            t += 1
        assert np.all(mean_est > 0)
        assert np.all(var_est > 0)
        return mean_est, var_est

    def _correlation_based_correction(
            self, max_displacement=None, n_processes=None):
        return sima.motion.frame_align.estimate(
            self.dataset.sequences, max_displacement, n_processes=n_processes)

    def _whole_frame_shifting(self, shifts, correlations):
        """Line up the data by the frame-shift estimates

        Parameters
        ----------
        shifts : array
            2xT array with the estimated shifts for each frame.
        correlations : array
            Intensity correlations between shifted frames and the reference.

        Returns
        -------
        reference : array
            Time average of each channel after frame-by-frame alignment.
            Size: (num_channels, num_rows, num_columns).
        variances : array
            Variance of each channel after frame-by-frame alignment.
            Size: (num_channels, num_rows, num_columns)
        offset : array
            The displacement to add to each shift to align the minimal shift
            with the edge of the corrected image.
        """
        # Calculate a correlation threshold for each plane of the frame
        thresh = \
            np.nanmean(list(it.chain(*correlations)), axis=0) - \
            2 * np.nanstd(list(it.chain(*correlations)), axis=0)
        # only include image frames with sufficiently high correlation
        min_shifts = np.min(np.concatenate(
            [s[c > thresh] for s, c in zip(shifts, correlations)]
        ), axis=0)
        max_shifts = np.max(np.concatenate(
            [s[c > thresh] for s, c in zip(shifts, correlations)]
        ), axis=0)
        out_shape = list(self.dataset.frame_shape)
        out_shape[1] += max_shifts[0] - min_shifts[0]
        out_shape[2] += max_shifts[1] - min_shifts[1]
        reference = np.zeros(out_shape)
        sum_squares = np.zeros_like(reference)
        count = np.zeros_like(reference)
        for frame, shift, corr in it.izip(
                it.chain(*self.dataset), it.chain(*shifts),
                it.chain(*correlations)):
            for plane, p_shifts, p_corr, th, ref, ssq, cnt in it.izip(
                    frame, shift, corr, thresh, reference, sum_squares, count):
                if p_corr > th:
                    low = (p_shifts - min_shifts)  # TOOD: NaN considerations
                    high = low + plane.shape[:-1]
                    ref[low[0]:high[0], low[1]:high[1]] += plane
                    ssq[low[0]:high[0], low[1]:high[1]] += plane ** 2
                    cnt[low[0]:high[0], low[1]:high[1]] += np.isfinite(plane)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference /= count
            assert np.all(np.isnan(reference[np.equal(count, 0)]))
            variances = (sum_squares / count) - reference ** 2
            assert not np.any(variances < 0)
        offset = - min_shifts
        return reference, variances, offset


class SolverHMM(object):

    """_ImagingCycle sub-classed with motion correction methods.

    Parameters
    ----------
    channels : list of iterable
        A list of iterable objects, one for each channel.
        Each iterable should yield 2D numpy arrays.

    Attributes
    ----------
    num_frames, num_channels, num_rows, num_columns : int
    """

    def __init__(self, sequence):
        self.sequence = sequence

    def _iter_processed(self, gains, pixel_means, pixel_variances):
        """Generator of preprocessed frames for efficient computation.

        Parameters
        ----------
        gains : array
            The photon-to-intensity gains for each channel.
        pixel_means : array
            The mean pixel intensities for each channel.
        pixel_variances : array
            The pixel intensity variance for each channel.

        Yields
        ------
        im : list of array
            The estimated photon counts for each channel.
        log_im_fac : list of array
            The logarithm of the factorial of the photon counts in im.
        log_im_p: list of array
            The log likelihood of observing each pixel intensity (without
            spatial information).
        """
        means = pixel_means / gains
        variances = pixel_variances / gains ** 2
        for frame in self.sequence:
            im = frame / gains
            # replace NaN pixels with the mean value
            for ch_idx, ch_mean in enumerate(means):
                im_nans = np.isnan(im[..., ch_idx])
                im[..., ch_idx][im_nans] = ch_mean
            assert(np.all(np.isfinite(im)))
            # take the log of the factorial of each pixel
            log_im_fac = gammaln(im + 1)
            # probability of observing the pixels (ignoring reference)
            log_im_p = -(im - means) ** 2 / (2 * variances) \
                - 0.5 * np.log(2. * np.pi * variances)
            # inf_indices = np.logical_not(np.isfinite(log_im_fac))
            # log_im_fac[inf_indices] = im[inf_indices] * (
            #     np.log(im[inf_indices]) - 1)
            assert(np.all(np.isfinite(log_im_fac)))
            assert(np.all(np.isfinite(log_im_p)))
            yield im, log_im_fac, log_im_p

    def neighbor_viterbi(
            self, log_markov_matrix, references, gains, mov_decay, mov_cov,
            mean_shift, offset, min_displacements, max_displacements,
            pixel_means, pixel_variances, num_retained, verbose=False):
        """Apply Viterbi algorithm to estimate the MAP displacement trajectory.

        Parameters
        ----------
        log_markov_matrix : array
            The log transition probabilities.
        references : array
            Time average of each channel after frame-by-frame alignment.
            Shape: (num_channels, num_rows, num_columns)
        gains : array
            The photon-to-intensity gains for each channel.
        mov_decay : array
            The per-line decay-term in the AR(1) motion model
        mov_cov : array
            The per-line covariance-term in the AR(1) motion model
        mean_shift: the mean of the whole-frame displacement estimates
        offset : array
            The displacement to add to each shift to align the minimal shift
            with the edge of the corrected image.
        min_displacements : array
            The minimum allowable displacement
        max_displacements : array
            The maximum allowable position.
        pixel_means : array
            The mean pixel intensity for each channel.
        pixel_variances : array
            The pixel intensity variance for each channel.
        num_retained : int
            The number of states to retain at each time step of the Viterbi
            algorithm.
        verbose : bool, optional
            Whether to print progress. Defaults to True.

        Returns
        -------
        array
            The maximum aposteriori displacement trajectory.  Shape: (2, T).
        """
        offset = np.array(offset, dtype=int)  # type verification
        T = np.prod(self.sequence.shape[:3])  # determine number of timesteps
        backpointer = []
        states = []

        # store outputs of various functions applied to the reference images
        # for later use
        assert references.ndim == 4
        scaled_refs = references / gains
        log_scaled_refs = np.log(scaled_refs)
        position_tbl, transition_tbl, log_markov_matrix_tbl, slice_tbls = \
            _lookup_tables(min_displacements, max_displacements,
                           log_markov_matrix, self.sequence.shape[3],
                           references, offset)
        initial_dist = _initial_distribution(mov_decay, mov_cov, mean_shift)
        iter_processed = iter(self._iter_processed(gains, pixel_means,
                                                   pixel_variances))

        def initial_probs():
            tmp_states = []
            log_p = []
            for index, position in enumerate(position_tbl):  # TODO parallelize
                # check that the displacement is allowable
                if np.all(min_displacements <= position) and np.all(
                        position <= max_displacements):
                    tmp_states.append(index)
                    # probability of initial displacement
                    log_p.append(np.log(initial_dist(position)))
            return np.array(tmp_states, dtype='int'), np.array(log_p)

        t = 0
        log_p_old = None  # for flaking purposes; does nothing
        for frame in iter_processed:
            for (plane, log_plane_fac, log_plane_p, plane_ref, log_plane_ref,
                 stbl) in zip(
                    *(frame + (scaled_refs, log_scaled_refs, slice_tbls))):
                for row_idx, row in enumerate(plane):
                    if t == 0:
                        tmp_states, log_p = initial_probs()
                    else:
                        tmp_states, log_p, tmp_backpointer = mc.transitions(
                            states[-1], log_markov_matrix_tbl, log_p_old,
                            position_tbl, transition_tbl)
                    mc.log_observation_probabilities(
                        log_p, tmp_states, plane, log_plane_p,
                        log_plane_fac, plane_ref, log_plane_ref, row_idx,
                        stbl, position_tbl, offset, references[0].shape[0])
                    if np.any(np.isfinite(log_p)):
                        # Remove nans to sort.
                        log_p[np.isnan(log_p)] = -np.Inf
                        # Keep only num_retained most likely states.
                        ix = np.argsort(-log_p)[0:num_retained]
                        states.append(tmp_states[ix])
                        log_p_old = log_p[ix] - log_p[ix[0]]
                        if t > 0:
                            backpointer.append(tmp_backpointer[ix])
                    else:
                        # If none of the observation probabilities are finite,
                        # then use states from the previous timestep.
                        states.append(states[-1])
                        backpointer.append(np.arange(num_retained))
                        warnings.warn('No finite observation probabilities.')
                    if verbose and (t * 10) % T < 10:
                        print t * 100 / T, '% done'  # report progress
                    t += 1

        assert position_tbl.dtype == int
        displacements = _backtrace(np.argmax(log_p_old), backpointer, states,
                                   position_tbl)
        assert displacements.dtype == int
        return displacements.reshape(self.sequence.shape[:3] + (2,))


def hmm(sequences, savedir, channel_names=None, info=None,
        num_states_retained=50, max_displacement=None,
        correction_channels=None, trim_criterion=None, verbose=True,
        n_processes=None):
    """
    Create a motion-corrected ImagingDataset using a row-wise hidden
    Markov model (HMM).

    Parameters
    ----------
    sequences : list of list of iterable
        Iterables yielding frames from imaging cycles and channels.
    savedir : str
        The directory used to store the dataset. If the directory
        name does not end with .sima, then this extension will
        be appended.
    channel_names : list of str, optional
        Names for the channels. Defaults to ['0', '1', '2', ...].
    metadata : dict
        Data for the order and timing of the data acquisition.
        See sima.ImagingDataset for details.
    num_states_retained : int, optional
        Number of states to retain at each time step of the HMM.
        Defaults to 50.
    max_displacement : array of int, optional
        The maximum allowed displacement magnitudes in [y,x]. By
        default, arbitrarily large displacements are allowed.

    Returns
    -------
    dataset : sima.ImagingDataset
        The motion-corrected dataset.

    Keyword Arguments
    -----------------
    correction_channels : list of int, optional
        Information from the channels corresponding to these indices
        will be used for motion correction. By default, all channels
        will be used.
    trim_criterion : float, optional
        The required fraction of frames during which a location must
        be within the field of view for it to be included in the
        motion-corrected imaging frames. By default, only locations
        that are always within the field of view are retained.
    verbose : boolean, optional
        Whether to print the progress status. Defaults to True.

    References
    ----------
    * Dombeck et al. 2007. Neuron. 56(1): 43-57.
    * Kaifosh et al. 2013. Nature Neuroscience. 16(9): 1182-4.

    """
    if correction_channels:
        correction_channels = [
            sima.misc.resolve_channels(c, channel_names, len(sequences[0]))
            for c in correction_channels]
        mc_sequences = [s[:, :, :, :, correction_channels] for s in sequences]
    else:
        mc_sequences = sequences
    displacements = _MCImagingDataset(mc_sequences).estimate_displacements(
        num_states_retained, max_displacement, verbose,
        n_processes=n_processes)
    max_disp = np.max(
        list(it.chain(*it.chain(*it.chain(*displacements)))), axis=0)
    frame_shape = np.array(sequences[0].shape)[1:]
    frame_shape[1:3] += max_disp
    corrected_sequences = [
        s.apply_displacements(d, frame_shape)
        for s, d in zip(sequences, displacements)]
    rows, columns = _trim_coords(
        trim_criterion, displacements, sequences[0].shape[1:4], frame_shape[:3]
    )
    corrected_sequences = [
        s[:, :, rows, columns] for s in corrected_sequences]
    return ImagingDataset(
        corrected_sequences, savedir, channel_names=channel_names)
