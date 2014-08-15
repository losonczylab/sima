"""
Motion correction
=================

The SIMA package can be used to motion correct sequentially
acquired images.

Currently, there is only one implemented method, which uses a
hidden Markov model (HMM) to correct for motion artifacts both
between frames and within frame.

Methods
-------

"""
from itertools import chain, izip
import warnings

import numpy as np
from numpy.linalg import det, svd, pinv
from scipy.special import gammaln
from scipy.signal import fftconvolve
from scipy.cluster.vq import kmeans2
from scipy.stats import nanstd
from scipy.stats.mstats import mquantiles
from scipy.ndimage.filters import gaussian_filter

try:
    import sima._motion as mc
except ImportError as error:
    # Attempt auto-compilation.
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()},
                      reload_support=True)
    import sima._motion as mc
from sima.imaging import ImagingDataset, _ImagingCycle
import sima.misc

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
    shifts : array
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
    diffs = np.diff(shifts)
    diffs = diffs[:, np.isfinite(diffs).all(axis=0)]
    cov_matrix = np.cov(diffs) / num_rows
    # don't allow singular covariance matrix
    cov_matrix[0, 0] = max(cov_matrix[0, 0], 1. / (
        shifts.shape[1] * num_rows))
    cov_matrix[1, 1] = max(cov_matrix[1, 1], 1. / (
        shifts.shape[1] * num_rows))
    assert det(cov_matrix) > 0

    mean_shift = np.nanmean(shifts, axis=1)
    centered_shifts = np.nan_to_num(
        shifts -
        np.dot(mean_shift.reshape([2, 1]), np.ones([1, shifts.shape[1]]))
    )
    # fit to autoregressive AR(1) model
    A = np.dot(pinv(centered_shifts[:, :-1].T), centered_shifts[:, 1:].T)
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
                    np.array([i, j]), np.array([0., 0.]),
                    transition_probs, 8))
    assert np.all(np.isfinite(cov_matrix)) and \
        np.all(np.isfinite(decay_matrix)) and \
        np.all(np.isfinite(log_transition_matrix))
    return cov_matrix, decay_matrix, log_transition_matrix


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
    ## don't let C be singular
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

    ## create transition lookup and create lookup for transition probability
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
    slice_tbl = mc.slice_lookup(references, position_tbl, num_columns, offset)

    assert position_tbl.dtype == int

    return position_tbl, transition_tbl, log_markov_matrix_tbl, slice_tbl


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


class _MCImagingDataset(ImagingDataset):
    """ImagingDataset sub-classed with motion correction functionality"""

    def __init__(self, iterables, invalid_frames=None):
        super(_MCImagingDataset, self).__init__(iterables, None)
        self.invalid_frames = invalid_frames

    def _create_cycles(self, iterables):
        """Create _MCCycle objects."""
        return [_MCCycle(iterable) for iterable in iterables]

    def _neighbor_viterbi(
            self, log_transition_matrix, references, gains, decay_matrix,
            cov_matrix_est, mean_shift, offset, min_displacements,
            max_displacements, pixel_means, pixel_variances,
            num_states_retained, valid_rows, verbose=True):
        """Estimate the MAP trajectory with the Viterbi Algorithm.

        See _MCCycle.neighbor_viterbi for details."""
        displacements = []
        for i, cycle in enumerate(self):
            if verbose:
                print 'Estimating displacements for cycle ', i
            displacements.append(
                cycle.neighbor_viterbi(
                    log_transition_matrix, references, gains, decay_matrix,
                    cov_matrix_est, mean_shift, offset, min_displacements,
                    max_displacements, pixel_means, pixel_variances,
                    num_states_retained,
                    {k: v[i] for k, v in valid_rows.iteritems()},
                    invalid_frames=set(self.invalid_frames[i]),
                    verbose=verbose
                )
            )
        return displacements

    def estimate_displacements(
            self, num_states_retained=50, max_displacement=None,
            artifact_channels=None, verbose=True, path=None):
        """Estimate and save the displacements for the time series.

        Parameters
        ----------
        num_states_retained : int
            Number of states to retain at each time step of the HMM.
        max_displacement : array of int
            The maximum allowed displacement magnitudes in [y,x].
        artifact_channels : list of int
            Channels for which artifact light should be checked.
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

        valid_rows = self._detect_artifact(artifact_channels)
        shifts, correlations = self._correlation_based_correction(
            valid_rows, max_displacement=max_displacement)
        references, _, offset = self._whole_frame_shifting(
            shifts, correlations)
        gains = self._estimate_gains(references, offset,
                                     shifts.astype(int), correlations)
        assert np.all(np.isfinite(gains)) and np.all(gains > 0)
        pixel_means, pixel_variances = self._pixel_distribution()
        cov_matrix_est, decay_matrix, log_transition_matrix = \
            _estimate_movement_model(shifts, self.num_rows)
        mean_shift = np.nanmean(shifts, axis=1)

        # add a bit of extra room to move around
        extra_buffer = ((max_displacement - np.nanmax(shifts, 1) +
                         np.nanmin(shifts, 1)) / 2).astype(int)
        if max_displacement[0] < 0:
            extra_buffer[0] = 5
        if max_displacement[1] < 0:
            extra_buffer[1] = 5
        min_displacements = (
            np.nanmin(shifts, 1) - extra_buffer).astype(int)
        max_displacements = (
            np.nanmax(shifts, 1) + extra_buffer).astype(int)

        displacements = self._neighbor_viterbi(
            log_transition_matrix, references, gains, decay_matrix,
            cov_matrix_est, mean_shift, offset, min_displacements,
            max_displacements, pixel_means, pixel_variances,
            num_states_retained, valid_rows, verbose)
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
        return displacements

    def _detect_artifact(self, channels=None):
        """Detect pixels that have been saturated by an external source

        NOTE: this is written to deal with an artifact specific to our lab.

        Parameters
        ----------
        channels : list of int
            The channels in which artifact light is to be detected.

        Returns
        -------
        dict of (int, array)
            Channel indices index boolean arrays indicating whether the rows
            have valid (i.e. not saturated) data.
            Array shape: (num_cycles, num_rows*num_timepoints).
        """
        channels = [] if channels is None else channels
        ret = {}
        for channel in channels:
            row_intensities = []
            for frame in chain(*self):
                im = frame[channel].astype('float')
                row_intensities.append(im.mean(axis=1))
            row_intensities = np.array(row_intensities)
            for i in range(row_intensities.shape[1]):
                row_intensities[:, i] += -row_intensities[:, i].mean() + \
                    row_intensities.mean()  # remove periodic component
            row_intensities = row_intensities.reshape(-1)
            # Separate row means into 2 clusters
            [centroid, labels] = kmeans2(row_intensities, 2)
            #only discard rows if clusters are substantially separated
            if max(centroid) / min(centroid) > 3 and \
                    max(centroid) - min(centroid) > 2 * np.sqrt(sum([np.var(
                    row_intensities[np.equal(labels, i)]) for i in [0, 1]])):
                # row intensities in the lower cluster are valid
                valid_rows = np.equal(labels, np.argmin(centroid))
                # also exclude rows prior to those in the higher cluster
                valid_rows[:-1] *= valid_rows[1:].copy()
                # also exclude rows following those in the higher cluster
                valid_rows[1:] *= valid_rows[:-1].copy()
            else:
                valid_rows = np.ones(labels.shape).astype('bool')
            # Reshape back to a list of arrays, one per cycle
            row_start = 0
            valid_rows_by_cycle = []
            for cycle in self:
                valid_rows_by_cycle.append(
                    valid_rows[row_start:
                               row_start + cycle.num_frames * cycle.num_rows])
                row_start += cycle.num_frames * cycle.num_rows
            ret[channel] = valid_rows_by_cycle
        return ret

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
        sums = np.zeros(self.num_channels).astype(float)
        sum_squares = np.zeros(self.num_channels).astype(float)
        count = 0
        t = 0
        for frame in chain(*self):
            if t > 0:
                mean_est = sums / count
                var_est = (sum_squares / count) - (mean_est ** 2)
            if t > min_frames and np.all(
                    np.sqrt(var_est / count) / mean_est < tolerance):
                break
            im = np.concatenate(
                [np.expand_dims(x, 0) for x in frame],
                axis=0).astype(float)  # NOTE: integers overflow
            sums += im.sum(axis=1).sum(axis=1)
            sum_squares += (im ** 2).sum(axis=1).sum(axis=1)
            count += im.shape[1] * im.shape[2]
            t += 1
        assert np.all(mean_est > 0)
        assert np.all(var_est > 0)
        return mean_est, var_est

    def _correlation_based_correction(self, valid_rows,
                                      max_displacement=None):
        """Estimate whole-frame displacements based on pixel correlations.

        Parameters
        ----------
        valid_rows : dict of (int, array)
            Channel indices index boolean arrays indicating whether the rows
            have valid (i.e. not saturated) data.
            Array shape: (num_cycles, num_rows*num_timepoints).
        max_displacement : array
            see estimate_displacements

        Returns
        -------
        shifts : array
            (2, num_frames*num_cycles)-array of integers giving the
            estimated displacement of each frame
        correlations : array
            (num_frames*num_cycles)-array giving the correlation of
            each shifted frame with the reference
        """
        shifts = np.zeros([2, self.num_frames], dtype='float')
        correlations = np.empty(self.num_frames, dtype='float')
        offset = np.zeros(2, dtype='int')
        search_space = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0],
                                [0, 1], [1, -1], [1, 0], [1, 1]])
        i = 0
        started = False
        for cyc_idx, cycle in enumerate(self):
            invalid_frames = set(self._invalid_frames[cyc_idx])
            for frame_idx, frame in enumerate(cycle):
                if frame_idx in invalid_frames:
                    correlations[i] = np.nan
                    shifts[:, i] = np.nan
                elif not started:
                    # NOTE: float64 gives nan when divided by 0
                    pixel_sums = np.concatenate(
                        [np.expand_dims(x, 0) for x in frame], axis=0
                    ).astype('float64')
                    pixel_counts = np.ones(pixel_sums.shape)
                    correlations[i] = 1.
                    shifts[:, i] = 0
                    started = True
                else:
                    im = np.concatenate(
                        [np.expand_dims(x, 0) for x in frame], axis=0
                    ).astype('float64')
                    for channel, rows in valid_rows.iteritems():
                        # ignore frames in which there are invalid rows
                        if not all(
                                rows[cyc_idx][frame_idx * self.num_rows:
                                              (frame_idx + 1) * self.num_rows]
                        ):
                            im[channel] = 0
                    # recompute reference using all aligned images
                    with warnings.catch_warnings():  # ignore divide by 0
                        warnings.simplefilter("ignore")
                        reference = pixel_sums / pixel_counts
                    reference[np.equal(0, pixel_counts)] = np.nan
                    # coarse alignment
                    small_ref = 0.25 * (
                        reference[:, :-1:2, :-1:2] +
                        reference[:, :-1:2, 1::2] + reference[:, 1::2, :-1:2] +
                        reference[:, 1::2, 1::2])
                    small_im = 0.25 * (
                        im[:, :-1:2, 0:-1:2] + im[:, :-1:2, 1::2] +
                        im[:, 1::2, :-1:2] + im[:, 1::2, 1::2])
                    for c in range(self.num_channels):
                        small_im[c] -= small_im[c].mean()
                        small_ref[c] -= small_ref[c][np.isfinite(small_ref[c])
                                                     ].mean()
                    # zeroing nans is equivalent to nancorr
                    small_ref[np.logical_not(np.isfinite(small_ref))] = 0
                    corrs = np.array([
                        fftconvolve(small_ref[ch], small_im[ch][::-1, ::-1])
                        for ch in range(self.num_channels)]).sum(axis=0)
                    if max_displacement is None:
                        max_idx = np.array(np.unravel_index(np.argmax(corrs),
                                                            corrs.shape))
                    else:
                        min_i = np.maximum(
                            (np.nanmax(shifts, 1) - max_displacement + offset) /
                            2 + np.array(small_im[0].shape) - 1, 0)
                        max_i = (
                            np.ceil(
                                (np.nanmin(shifts, 1) + max_displacement +
                                 offset).astype(float) / 2. +
                                np.array(small_im[0].shape) - 1)
                        ).astype(int)
                        if max_displacement[0] < 0:
                            min_i[0] = 0
                        else:
                            #restrict to allowable displacements
                            corrs = corrs[min_i[0]:(max_i[0] + 1), :]
                        if max_displacement[1] < 0:
                            min_i[1] = 0
                        else:
                            #restrict to allowable displacements
                            corrs = corrs[:, min_i[1]:(max_i[1] + 1)]
                        max_idx = min_i + np.array(
                            np.unravel_index(np.argmax(corrs), corrs.shape))
                    #shift that maximizes correlation
                    shift = 2 * (max_idx - small_im[0].shape + 1) - offset

                    #refine using the full image
                    max_corr = 0.
                    for small_shift in search_space:
                        ref_indices = [
                            np.maximum(0, offset + shift + small_shift),
                            np.minimum(reference[0].shape, offset +
                                       shift + small_shift + im[0].shape)
                        ]
                        im_min_indices = np.maximum(
                            0, - offset - shift - small_shift)
                        im_max_indices = im_min_indices + ref_indices[1] - \
                            ref_indices[0]
                        tmp_ref = reference[
                            :, ref_indices[0][0]:ref_indices[1][0],
                            ref_indices[0][1]:ref_indices[1][1]
                        ].copy()
                        tmp_im = im[:, im_min_indices[0]:im_max_indices[0],
                                    im_min_indices[1]:im_max_indices[1]].copy()
                        for channel in range(self.num_channels):
                            tmp_im[channel] -= tmp_im[channel].mean()
                            tmp_ref[channel] -= tmp_ref[channel][
                                np.isfinite(tmp_ref[channel])].mean()
                        tmp_ref[np.logical_not(np.isfinite(
                                tmp_ref))] = 0  # equivalent to nancorr
                        corr = sum([(tmp_im[c] * tmp_ref[c]).sum() / np.sqrt((
                            tmp_ref[c] ** 2).sum() * (tmp_im[c] ** 2).sum())
                            for c in range(self.num_channels)]) / 2.
                        if corr > max_corr:
                            max_corr = corr
                            refinement = small_shift
                    shifts[:, i] = shift + refinement
                    correlations[i] = max_corr

                    #enlarge storage arrays if necessary
                    l = - np.minimum(0, shifts[:, i] + offset).astype(int)
                    r = np.maximum(0, shifts[:, i] + offset + im[0].shape -
                                   reference[0].shape).astype(int)
                    if np.any(l > 0) or np.any(r > 0):
                        pixel_sums = np.concatenate([np.zeros([
                            pixel_sums.shape[0], l[0], pixel_sums.shape[2]]),
                            pixel_sums, np.zeros([pixel_sums.shape[0], r[0],
                                                 pixel_sums.shape[2]])],
                            axis=1)
                        pixel_sums = np.concatenate([np.zeros([
                            pixel_sums.shape[0], pixel_sums.shape[1], l[1]]),
                            pixel_sums, np.zeros([pixel_sums.shape[0],
                                                 pixel_sums.shape[1], r[1]])],
                            axis=2)
                        pixel_counts = np.concatenate([np.zeros([
                            pixel_counts.shape[0], l[0],
                            pixel_counts.shape[2]]), pixel_counts,
                            np.zeros([pixel_counts.shape[0], r[0],
                                      pixel_counts.shape[2]])], axis=1)
                        pixel_counts = np.concatenate([np.zeros([
                            pixel_counts.shape[0], pixel_counts.shape[1],
                            l[1]]), pixel_counts,
                            np.zeros([pixel_counts.shape[0],
                                     pixel_counts.shape[1], r[1]])], axis=2)
                        offset += l
                    ref_indices = [offset + shifts[:, i],
                                   offset + shifts[:, i] + im[0].shape]
                    pixel_counts[:, ref_indices[0][0]:ref_indices[1][0],
                                 ref_indices[0][1]:ref_indices[1][1]] += 1
                    pixel_sums[:, ref_indices[0][0]:ref_indices[1][0],
                               ref_indices[0][1]:ref_indices[1][1]] += im
                    assert np.prod(pixel_sums[0].shape) < 4 * np.prod(im[0].shape)
                i += 1
        return shifts.astype('float'), correlations.astype('float')

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
        good_corr = correlations > np.nanmean(correlations) - \
            2 * nanstd(correlations)
        # only include image frames with sufficiently high correlation
        min_shifts = np.nanmin(shifts[:, good_corr], axis=1).astype(int)
        max_shifts = np.nanmax(shifts[:, good_corr], axis=1).astype(int)
        reference = np.zeros([
            self.num_channels,
            self.num_rows + max_shifts[0] - min_shifts[0],
            self.num_columns + max_shifts[1] - min_shifts[1]])
        sum_squares = np.zeros_like(reference)
        count = np.zeros_like(reference)
        for frame, shift, gc in izip(chain(*self), shifts.T, good_corr):
            if gc:
                im = np.concatenate([np.expand_dims(x, 0) for x in frame],
                                    axis=0).astype(float)
                # indices for shifted image
                low_idx = shift - min_shifts
                high_idx = low_idx + im.shape[1:]
                reference[:, low_idx[0]:high_idx[0],
                          low_idx[1]:high_idx[1]] += im
                sum_squares[:, low_idx[0]:high_idx[0],
                            low_idx[1]:high_idx[1]] += im ** 2
                count[:, low_idx[0]:high_idx[0], low_idx[1]:high_idx[1]] += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference /= count
        reference[np.equal(count, 0)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            variances = (sum_squares / count) - reference ** 2
        offset = - min_shifts
        return reference, variances, offset

    def _estimate_gains(self, references, offset, shifts, correlations):
        """Estimate the photon-pixel transformation gain for each channel

        Parameters
        ----------
        references : array
            Time average of each channel after frame-by-frame alignment.
            Shape: (num_channels, num_rows, num_columns)
        offset : array
            The displacement to add to each shift to align the minimal shift
            with the edge of the corrected image.
        shifts : array
            The estimated shifts for each frame.  Shape: (2, T).
        correlations : array
            Intensity correlations between shifted frames and reference.

        Returns
        -------
        array
            The photon-to-intensity gains for each channel.
        """
        corr_mean = np.nanmean(correlations)
        corr_stdev = nanstd(correlations)
        # Calculate displacements between consecutive images
        diffs = np.diff(shifts, axis=1)
        # Threshold the gradient of the smoothed reference image to minimize
        # estimation errors due to uncorrected movements
        smooth_references = np.array([gaussian_filter(chan, sigma=1)
                                      for chan in references])
        grad_below_threshold = _threshold_gradient(smooth_references)
        # Initialize variables for the loop
        count = np.zeros(self.num_channels)
        sum_estimates = np.zeros(self.num_channels)
        sum_square_estimates = np.zeros(self.num_channels)
        im2 = None
        t = - 1
        for cycle in self:
            for frame_idx, frame in enumerate(cycle):
                im1 = im2
                im2 = np.concatenate([np.expand_dims(x, 0) for x in frame],
                                     axis=0).astype(float)
                if frame_idx > 0 and (
                        correlations[t + 1] > corr_mean - 2 * corr_stdev and
                        correlations[t] > corr_mean - 2 * corr_stdev):
                    # calculate the coordinates of the overlap of the images
                    im1_min_coords = np.maximum(diffs[:, t], 0)
                    im1_max_coords = im1.shape[1:] + np.minimum(diffs[:, t], 0)
                    im2_min_coords = np.maximum(-diffs[:, t], 0)
                    im2_max_coords = im1.shape[1:] + np.minimum(
                        -diffs[:, t], 0)
                    # select the overlap regions
                    x0 = im1[:, im1_min_coords[0]:im1_max_coords[0],
                             im1_min_coords[1]:im1_max_coords[1]]
                    x1 = im2[:, im2_min_coords[0]:im2_max_coords[0],
                             im2_min_coords[1]:im2_max_coords[1]]
                    # Include only values where x0 > x1 because the calcium
                    # signal increases more quickly than it decays)
                    scaling_estimates = ((x1 - x0) ** 2) / (x1 + x0)
                    scaling_estimates = scaling_estimates * (x0 > x1)
                    shifted_thresholds = grad_below_threshold[
                        :,
                        (im1_min_coords[0] + shifts[0, t] + offset[0]):
                        (im1_max_coords[0] + shifts[0, t] + offset[0]),
                        (im1_min_coords[1] + shifts[1, t] + offset[1]):
                        (im1_max_coords[1] + shifts[1, t] + offset[1])]
                    # Discard if grad > threshold
                    scaling_estimates = scaling_estimates * shifted_thresholds
                    sum_estimates += np.nansum(
                        np.nansum(scaling_estimates, axis=2), axis=1)
                    sum_square_estimates += np.nansum(np.nansum(
                        scaling_estimates ** 2, axis=2), axis=1)
                    #Keep track of which indices provide valid estimates
                    valid_matrix = np.isfinite(scaling_estimates) * (
                        x0 > x1) * shifted_thresholds
                    # update the count of the number of estimates
                    count += valid_matrix.sum(axis=2).sum(axis=1)
                    mean_est = sum_estimates / count
                    var_est = ((sum_square_estimates / count) - (mean_est ** 2)
                               ) / count
                    # Stop if rel error below threshold and >10 frames examined
                    if np.all(np.sqrt(var_est) / mean_est < 0.01) and t > 10:
                        break
                t += 1  # increment time
        return sum_estimates / count


class _MCCycle(_ImagingCycle):
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
        for frame in self:
            im = np.concatenate([np.expand_dims(x, 0) for x in frame],
                                axis=0).astype(float)
            for i in range(self.num_channels):
                im[i] /= gains[i]  # scale by inverse of the gain factor
            # take the log of the factorial of each pixel
            log_im_fac = gammaln(im + 1)
            # probability of observing the pixels (ignoring reference)
            log_im_p = np.zeros_like(im, dtype=float)
            for i in range(self.num_channels):
                log_im_p[i, :, :] = -((im[i] - means[i]) ** 2 / (
                    2 * variances[i])) \
                    - 0.5 * np.log(2. * np.pi * variances[i])
            assert(np.all(np.isfinite(im)))
            inf_indices = np.logical_not(np.isfinite(log_im_fac))
            log_im_fac[inf_indices] = im[inf_indices] * (
                np.log(im[inf_indices]) - 1)
            assert(np.all(np.isfinite(log_im_fac)))
            assert(np.all(np.isfinite(log_im_p)))
            yield im, log_im_fac, log_im_p

    def neighbor_viterbi(
            self, log_markov_matrix, references, gains, mov_decay, mov_cov,
            mean_shift, offset, min_displacements, max_displacements,
            pixel_means, pixel_variances, num_retained, valid_rows,
            invalid_frames, verbose=False):
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
        valid_rows : dict of (int, array)
            Channel indices index boolean arrays indicating whether the rows
            have valid (i.e. not saturated) data.
            Array shape: (num_cycles, num_rows*num_timepoints).
        verbose : bool, optional
            Whether to print progress. Defaults to True.

        Returns
        -------
        array
            The maximum aposteriori displacement trajectory.  Shape: (2, T).
        """
        offset = np.array(offset, dtype=int)  # type verification
        T = self.num_rows * self.num_frames  # determine number of timesteps
        backpointer = []
        states = []

        # store outputs of various functions applied to the reference images
        # for later use
        references = np.array([ref for ref in references])
        assert len(references.shape) == 3
        scaled_refs = np.array(
            [ref / gains[i] for i, ref in enumerate(references)])
        log_scaled_refs = np.log(scaled_refs)
        position_tbl, transition_tbl, log_markov_matrix_tbl, slice_tbl = \
            _lookup_tables(min_displacements, max_displacements,
                           log_markov_matrix, self.num_columns, references,
                           offset)
        initial_dist = _initial_distribution(mov_decay, mov_cov, mean_shift)

        iter_processed = iter(self._iter_processed(gains, pixel_means,
                                                   pixel_variances))
        # Initial timestep
        im, log_im_fac, log_im_p = next(iter_processed)
        frame_number = 0
        frame_row = 0
        tmp_states = []
        tmp_log_p = []
        for index, position in enumerate(position_tbl):  # TODO parallelize
            # check that the displacement is allowable
            if np.all(min_displacements <= position) and np.all(
                    position <= max_displacements):
                tmp_states.append(index)
                # probability of initial displacement
                tmp_log_p.append(np.log(initial_dist(position)))
        tmp_log_p = np.array(tmp_log_p)
        tmp_states = np.array(tmp_states, dtype='int')
        mc.log_observation_probabilities(
            tmp_log_p, tmp_states, im, log_im_p, log_im_fac, scaled_refs,
            log_scaled_refs, frame_row, slice_tbl, position_tbl, offset,
            references[0].shape[0])
        tmp_log_p[np.isnan(tmp_log_p)] = -np.Inf  # get rid of NaNs for sorting
        ix = np.argsort(-tmp_log_p)[0:num_retained]  # keep most likely states
        states.append(np.array(tmp_states)[ix])
        log_p_old = np.array(tmp_log_p)[ix] - tmp_log_p[ix[0]]

        # subsequent time steps
        for t in range(1, T):
            assert(np.any(np.isfinite(log_p_old)))
            frame_row = t % self.num_rows
            if frame_row == 0:  # load new image data if frame time has changed
                im, log_im_fac, log_im_p = next(iter_processed)
                frame_number += 1
            tmp_states, tmp_log_p, tmp_backpointer = mc.transitions(
                states[t - 1], log_markov_matrix_tbl, log_p_old,
                position_tbl, transition_tbl)
            #observation probabilities
            if frame_number not in invalid_frames:
                if all(x[t] for x in valid_rows.itervalues()):
                    mc.log_observation_probabilities(
                        tmp_log_p, tmp_states, im, log_im_p, log_im_fac,
                        scaled_refs, log_scaled_refs, frame_row, slice_tbl,
                        position_tbl, offset, references[0].shape[0])
                else:
                    mc.log_observation_probabilities(
                        tmp_log_p, tmp_states, im[1:, :, :],
                        log_im_p[1:, :, :], log_im_fac[1:, :, :],
                        scaled_refs[1:, :, :], log_scaled_refs[1:, :, :],
                        frame_row, slice_tbl, position_tbl, offset,
                        references[0].shape[0])
            if np.any(np.isfinite(tmp_log_p)):
                # assert not any(np.isnan(tmp_log_p))
                tmp_log_p[np.isnan(tmp_log_p)] = -np.Inf  # remove nans to sort
                # Keep only num_retained most likely states
                ix = np.argsort(-tmp_log_p)[0:num_retained]
                states.append(tmp_states[ix])
                log_p_old = tmp_log_p[ix] - tmp_log_p[ix[0]]
                backpointer.append(tmp_backpointer[ix])
            else:
                # if no finite observation probabilities, then use previous
                # timesteps states
                states.append(states[t - 1])
                backpointer.append(np.arange(num_retained))
                warnings.warn('No finite observation probabilities.')
            if verbose and (t * 10) % T < 10:
                print t * 100 / T, '% done'  # report progress

        assert position_tbl.dtype == int
        displacements = _backtrace(np.argmax(log_p_old), backpointer, states,
                                   position_tbl)
        assert displacements.dtype == int
        return displacements


def hmm(iterables, savedir, channel_names=None, metadata=None,
        num_states_retained=50, max_displacement=None,
        correction_channels=None, artifact_channels=None,
        trim_criterion=None, invalid_frames=None, verbose=True):
    """
    Create a motion-corrected ImagingDataset using a row-wise hidden
    Markov model (HMM).

    Parameters
    ----------
    iterables : list of list of iterable
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
    artifact_channels : list of int, optional
        Channels for which artifact light should be checked.
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
            sima.misc.resolve_channels(c, channel_names, len(iterables[0]))
            for c in correction_channels]
    if artifact_channels:
        artifact_channels = [
            sima.misc.resolve_channels(c, channel_names, len(iterables[0]))
            for c in artifact_channels]
    if correction_channels:
        mc_iterables = [[cycle[i] for i in correction_channels]
                        for cycle in iterables]
        if artifact_channels is not None:
            artifact_channels = [i for i, c in enumerate(correction_channels)
                                 if c in artifact_channels]
    else:
        mc_iterables = iterables
    displacements = _MCImagingDataset(
        mc_iterables, invalid_frames=invalid_frames
    ).estimate_displacements(
        num_states_retained, max_displacement, artifact_channels, verbose
    )

    return ImagingDataset(iterables, savedir,
                          channel_names=channel_names,
                          displacements=displacements,
                          trim_criterion=trim_criterion,
                          invalid_frames=invalid_frames)
