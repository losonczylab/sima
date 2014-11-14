import itertools as it
import warnings
try:
    from future_builtins import zip
except ImportError:  # Python 3.x
    pass

import numpy as np
from numpy.linalg import det, svd
from scipy.special import gammaln
try:
    from bottleneck import nansum, nanmedian
except ImportError:
    from numpy import nansum
    from scipy.stats import nanmedian
from scipy.stats.mstats import mquantiles

import _motion as mc
import sima.motion.frame_align
import sima.misc
from sima.motion import MotionEstimationStrategy

np.seterr(invalid='ignore', divide='ignore')


def _pixel_distribution(dataset, tolerance=0.001, min_frames=1000):
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
    sums = np.zeros(dataset.frame_shape[-1]).astype(float)
    sum_squares = np.zeros_like(sums)
    counts = np.zeros_like(sums)
    t = 0
    for frame in it.chain(*dataset):
        for plane in frame:
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
            sums += np.nan_to_num(nansum(nansum(plane, axis=0), axis=0))
            sum_squares += np.nan_to_num(
                nansum(nansum(plane ** 2, axis=0), axis=0))
            counts += np.isfinite(plane).sum(axis=0).sum(axis=0)
            t += 1
    assert np.all(mean_est > 0)
    assert np.all(var_est > 0)
    return mean_est, var_est


def _whole_frame_shifting(dataset, shifts):
    """Line up the data by the frame-shift estimates

    Parameters
    ----------
    shifts : array
        DxT or DxTxP array with the estimated shifts for each frame/plane.

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
    # # Calculate a correlation threshold for each plane of the frame
    # thresh = \
    #     np.nanmean(list(it.chain(*correlations)), axis=0) - \
    #     2 * np.nanstd(list(it.chain(*correlations)), axis=0)
    # # only include image frames with sufficiently high correlation
    min_shifts = np.nanmin(
        list(it.chain(*[s.reshape(-1, s.shape[-1]) for s in shifts])), 0)
    max_shifts = np.nanmax(
        list(it.chain(*[s.reshape(-1, s.shape[-1]) for s in shifts])), 0)
    out_shape = list(dataset.frame_shape)
    if len(min_shifts) == 2:
        out_shape[1] += max_shifts[0] - min_shifts[0]
        out_shape[2] += max_shifts[1] - min_shifts[1]
    elif len(min_shifts) == 3:
        for i in range(3):
            out_shape[i] += max_shifts[i] - min_shifts[i]
    else:
        raise Exception
    reference = np.zeros(out_shape)
    sum_squares = np.zeros_like(reference)
    count = np.zeros_like(reference)
    for frame, shift in zip(it.chain(*dataset), it.chain(*shifts)):
        if shift.ndim == 1:  # single shift for the whole volume
            l = shift - min_shifts
            h = shift + frame.shape[:-1]
            reference[l[0]:h[0], l[1]:h[1], l[2]:h[2]] += frame
            sum_squares[l[0]:h[0], l[1]:h[1], l[2]:h[2]] += frame ** 2
            count[l[0]:h[0], l[1]:h[1], l[2]:h[2]] += np.isfinite(frame)
        else:  # plane-specific shifts
            for plane, p_shifts, ref, ssq, cnt in zip(
                    frame, shift, reference, sum_squares, count):
                low = p_shifts - min_shifts  # TOOD: NaN considerations
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


def _lookup_tables(position_bounds, log_markov_matrix):
    """Generate lookup tables to speed up the algorithm performance.

    Parameters
    ----------
    position_bounds : array of int
        The minimum and maximum (+1) allowable coordinates.
    step_bounds : array of int
        The minimum and maximum (+1) allowable steps.
    log_markov_matrix :
        The log transition probabilities.

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
    """
    position_tbl = np.array(
        list(it.product(*[range(m, M) for m, M in zip(*position_bounds)])),
        dtype=int)
    position_dict = {tuple(position): i
                     for i, position in enumerate(position_tbl)}
    # create transition lookup and create lookup for transition probability
    transition_tbl = []
    log_markov_matrix_tbl = []
    for step in it.product(
            *[range(-s + 1, s) for s in log_markov_matrix.shape]):
        if len(step) is 2:
            step = (0,) + step
        tmp_tbl = []
        for pos in position_tbl:
            new_position = tuple(pos + np.array(step))
            try:
                tmp_tbl.append(position_dict[new_position])
            except KeyError:
                tmp_tbl.append(-1)
        transition_tbl.append(tmp_tbl)
        log_markov_matrix_tbl.append(
            log_markov_matrix[tuple(abs(s) for s in step)])
    transition_tbl = np.array(transition_tbl, dtype=int)
    log_markov_matrix_tbl = np.fromiter(log_markov_matrix_tbl, dtype=float)
    return position_tbl, transition_tbl, log_markov_matrix_tbl


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
    dim = len(position_tbl[0])
    i = start_idx
    trajectory = np.zeros([T, dim], dtype=int)
    trajectory[-1] = position_tbl[states[-1][i]]
    for t in xrange(T - 2, -1, -1):
        # NOTE: backpointer index 0 corresponds to second timestep
        i = backpointer[t][i]
        trajectory[t] = position_tbl[states[t][i]]
    return trajectory


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class _HiddenMarkov(MotionEstimationStrategy):

    def __init__(self, granularity=3, num_states_retained=50,
                 max_displacement=None, n_processes=None, verbose=True):

        d = locals()
        del d['self']
        self._params = Struct(**d)

    def _neighbor_viterbi(
            self, dataset, references, gains, movement_model,
            min_displacements, max_displacements, pixel_means, pixel_variances,
            max_step=1):
        """Estimate the MAP trajectory with the Viterbi Algorithm.

        """
        assert references.ndim == 4
        granularity = self._params.granularity
        scaled_refs = references / gains
        displacement_tbl, transition_tbl, log_markov_tbl, = _lookup_tables(
            [min_displacements, max_displacements + 1],
            movement_model.log_transition_matrix(
                max_distance=max_step,
                dt=1./np.prod(references.shape[:(granularity-1)])))
        assert displacement_tbl.dtype == int
        tmp_states, log_p = movement_model.initial_probs(
            displacement_tbl, min_displacements, max_displacements)
        displacements = []
        for i, sequence in enumerate(dataset):
            if self._params.verbose:
                print 'Estimating displacements for cycle ', i
            imdata = NormalizedIterator(sequence, gains, pixel_means,
                                        pixel_variances, granularity)
            positions = PositionIterator(sequence.shape[:-1], granularity)
            disp = _beam_search(
                imdata, positions,
                it.repeat((transition_tbl, log_markov_tbl)), scaled_refs,
                displacement_tbl, (tmp_states, log_p),
                self._params.num_states_retained)
            displacements.append(disp.reshape(
                sequence.shape[:granularity] + (disp.shape[-1],)))
        return displacements

    def _estimate(self, dataset):
        """Estimate and save the displacements for the time series.

        Parameters
        ----------
        num_states_retained : int
            Number of states to retain at each time step of the HMM.
        max_displacement : array of int
            The maximum allowed displacement magnitudes in [y,x].

        Returns
        -------
        dict
            The estimated displacements and partial results of motion
            correction.
        """
        params = self._params
        if params.verbose:
            print 'Estimating model parameters.'
        shifts = self._estimate_shifts(dataset)
        references, variances, offset = _whole_frame_shifting(dataset, shifts)
        if params.max_displacement is None:
            params.max_displacement = np.array(dataset.frame_shape[:3]) / 2
        else:
            params.max_displacement = np.array(params.max_displacement)
        assert np.all(offset == 0)
        gains = nanmedian(
            (variances / references).reshape(-1, references.shape[-1]))
        if not (np.all(np.isfinite(gains)) and np.all(gains > 0)):
            raise Exception('Failed to estimate positive gains')
        pixel_means, pixel_variances = _pixel_distribution(dataset)
        movement_model = MovementModel.estimate(shifts)
        if shifts[0].shape[-1] == 2:
            shifts = [np.concatenate([np.zeros(s.shape[:-1] + (1,), dtype=int),
                                      s], axis=-1) for s in shifts]

        # TODO: detect unreasonable shifts before doing this calculation
        min_shifts = np.nanmin(list(it.chain(*it.chain(*shifts))), 0)
        max_shifts = np.nanmax(list(it.chain(*it.chain(*shifts))), 0)

        # add a bit of extra room to move around
        if params.max_displacement.size == 2:
            params.max_displacement = np.hstack(([0], params.max_displacement))
        extra_buffer = ((params.max_displacement - max_shifts + min_shifts) / 2
                        ).astype(int)
        min_displacements = (min_shifts - extra_buffer)
        max_displacements = (max_shifts + extra_buffer)

        displacements = self._neighbor_viterbi(
            dataset, references, gains, movement_model, min_displacements,
            max_displacements, pixel_means, pixel_variances)

        return self._post_process(displacements)

    def _post_process(self, displacements):
        return displacements


class HiddenMarkov2D(_HiddenMarkov):
    """
    Row-wise hidden Markov model (HMM).

    Parameters
    ----------
    num_states_retained : int, optional
        Number of states to retain at each time step of the HMM.
        Defaults to 50.
    max_displacement : array of int, optional
        The maximum allowed displacement magnitudes in [y,x]. By
        default, arbitrarily large displacements are allowed.

    References
    ----------
    * Dombeck et al. 2007. Neuron. 56(1): 43-57.
    * Kaifosh et al. 2013. Nature Neuroscience. 16(9): 1182-4.

    """
    def _estimate_shifts(self, dataset):
        return sima.motion.frame_align.PlaneTranslation2D(
            self._params.max_displacement,
            n_processes=self._params.n_processes).estimate(dataset)

    def _post_process(self, displacements):
        return [d[..., 1:] for d in displacements]


class MovementModel(object):
    """

    Attributes
    ----------
    mean_shift : array of int
        The mean of the whole-frame displacement estimates

    """

    def __init__(self, cov_matrix, U, s, mean_shift):
        if not np.all(np.isfinite(cov_matrix)):
            raise ValueError
        self._cov_matrix = cov_matrix
        self._U = U
        self._s = s
        self.mean_shift = mean_shift

    @classmethod
    def estimate(cls, shifts, times=None):
        """Estimate the movement model from displacements.

        Parameters
        ----------
        shifts : list of ndarray
            The shape of the ndarray may vary depending on whether
            displacements are estimated per volume, per plane, per row, etc.

        """
        shifts = np.array(list(it.chain(
            *[s.reshape(-1, s.shape[-1]) for s in shifts])))
        if not shifts.shape[1] in (2, 3):
            raise ValueError
        # var_model = VAR(shifts)
        # results = var_model.fit(1)  # fit VAR(1) modoel
        # A = results.coefs[0]
        # cov_matrix = results.sigma_u
        # diffs = np.diff(shifts, axis=0)
        # cov_matrix = np.cov(diffs[np.isfinite(diffs).all(axis=1)].T)
        # # don't allow singular covariance matrix
        # for i in range(len(cov_matrix)):
        #     cov_matrix[i, i] = max(cov_matrix[i, i], 1. / len(shifts))
        # assert det(cov_matrix) > 0

        mean_shift = np.nanmean(shifts, axis=0)
        assert len(mean_shift) == shifts.shape[1]
        centered_shifts = np.nan_to_num(shifts - mean_shift)
        past = centered_shifts[:-1]
        future = centered_shifts[1:]
        past_future = np.dot(past.T, future)
        past_past = np.dot(past.T, past)
        idx = 0
        D = shifts.shape[1]
        n = D * (D+1) / 2
        y = np.zeros(n)
        M = np.zeros((n, n))
        for i in range(D):
            for j in range(i + 1):
                y[idx] = past_future[i, j] + past_future[j, i]
                idx_2 = 0
                for k in range(D):
                    for l in range(k + 1):
                        if k == i:
                            M[idx, idx_2] += past_past[j, l]
                        elif l == i:
                            M[idx, idx_2] += past_past[j, k]
                        if k == j:
                            M[idx, idx_2] += past_past[i, l]
                        elif l == j:
                            M[idx, idx_2] += past_past[i, k]
                        idx_2 += 1
                idx += 1
        coefficients = np.linalg.solve(M, y)
        if D == 2:
            A = np.array([[coefficients[0], coefficients[1]],
                          [coefficients[1], coefficients[2]]])
        if D == 3:
            A = np.array([[coefficients[0], coefficients[1], coefficients[3]],
                          [coefficients[1], coefficients[2], coefficients[4]],
                          [coefficients[3], coefficients[4], coefficients[5]]])
        cov_matrix = np.cov(future.T - np.dot(A, past.T))
        U, s, _ = svd(A)  # NOTE: U == V for positive definite A
        assert np.max(s) < 1
        return cls(cov_matrix, U, s, mean_shift)

    def decay_matrix(self, dt=1.):
        """

        Paramters
        ---------
        dt : float

        Returns
        -------
        mov_decay : array
            The per-line decay-term in the AR(1) motion model
        """
        decay_matrix = np.dot(self._U, np.dot(self._s ** dt, self._U))
        if not np.all(np.isfinite(decay_matrix)):
            raise Exception
        return decay_matrix

    def cov_matrix(self, dt=1.):
        """

        Paramters
        ---------
        dt : float

        Returns
        -------
        mov_cov : array
            The per-line covariance-term in the AR(1) motion model
        """
        return self._cov_matrix * dt

    def log_transition_matrix(self, max_distance=1, dt=1.):
        """
        Gaussian Transition Probabilities

        Parameters
        ----------
        max_distance : int
        dt : float

        """
        cov_matrix = self.cov_matrix(dt)
        log_transition_probs = lambda x, x0: -0.5 * (
            np.log(2 * np.pi * det(cov_matrix)) +
            np.dot(x - x0, np.linalg.solve(cov_matrix, x - x0)))
        log_transition_matrix = -np.inf * np.ones(
            [max_distance + 1] * len(cov_matrix))
        for disp in it.product(*([range(max_distance + 1)] * len(cov_matrix))):
            log_transition_matrix[disp] = log_transition_probs(
                np.array(disp), np.zeros(len(cov_matrix)))
        assert np.all(np.isfinite(log_transition_matrix))
        if log_transition_matrix.ndim is 2:
            log_transition_matrix = np.expand_dims(log_transition_matrix, 0)
        return log_transition_matrix

    def _initial_distribution(self):
        """Get the initial distribution of the displacements."""
        decay = self.decay_matrix()
        noise_cov = self.cov_matrix()
        initial_cov = np.linalg.solve(
            np.diag(np.ones(len(decay))) - decay * decay.T,
            noise_cov.newbyteorder('>').byteswap())
        for _ in range(1000):
            initial_cov = decay * initial_cov * decay.T + noise_cov
        # don't let C be singular
        for i in range(len(initial_cov)):
            initial_cov[i, i] = max(initial_cov[i, i], 0.1)

        def f(x):
            if len(x) is 3 and len(initial_cov) is 2:
                x = x[1:]
            return np.exp(
                -0.5 * np.dot(x - self.mean_shift,
                              np.linalg.solve(initial_cov, x - self.mean_shift)
                              )
            ) / np.sqrt(2.0 * np.pi * det(initial_cov))
        assert np.isfinite(f(self.mean_shift))
        return f

    def initial_probs(self, displacement_tbl, min_displacements,
                      max_displacements):
        """Give the initial probabilites for a displacement table"""
        initial_dist = self._initial_distribution()
        states = []
        log_p = []
        for index, position in enumerate(displacement_tbl):  # TODO parallelize
            # check that the displacement is allowable
            if np.all(min_displacements <= position) and np.all(
                    position <= max_displacements):
                states.append(index)
                # probability of initial displacement
                log_p.append(np.log(initial_dist(position)))
        if not np.any(np.isfinite(log_p)):
            raise Exception
        return np.array(states, dtype='int'), np.array(log_p)


class PositionIterator(object):
    """Position iterator

    Parameters
    ----------
    shape : tuple of int
        (times, planes, rows, columns)

    Examples
    --------

    >>> from sima.motion.hmm import PositionIterator
    >>> pi = PositionIterator((100, 5, 128, 256), 'frame')
    >>> positions = next(iter(pi))

    >>> pi = PositionIterator((100, 5, 128, 256), 'plane')
    >>> positions = next(iter(pi))

    >>> pi = PositionIterator((100, 5, 128, 256), 'row', [10, 12])
    >>> positions = next(iter(pi))

    >>> pi = PositionIterator((100, 5, 128, 256), 'column', [3, 10, 12])
    >>> positions = next(iter(pi))

    """

    def __init__(self, shape, granularity, offset=None):
        try:
            self.granularity = int(granularity)
        except ValueError:
            self.granularity = {'frame': 1,
                                'plane': 2,
                                'row': 3,
                                'column': 4}[granularity]
        self.shape = shape
        if offset is None:
            self.offset = [0, 0, 0, 0]
        else:
            self.offset = ([0, 0, 0, 0] + list(offset))[-4:]

    def __iter__(self):
        for base in it.product(*[range(o, x + o) for x, o in
                                 zip(self.shape[:self.granularity],
                                     self.offset[:self.granularity])]):
            l = [[x] for x in base[1:]] + [
                xrange(x) for x in self.shape[self.granularity:]]
            yield np.concatenate(
                [a.reshape(-1, 1) for a in np.meshgrid(*l)],
                axis=1)


def _beam_search(imdata, positions, transitions, references, state_table,
                 initial_dist, num_retained=50):
    """Perform a beam search (modified Viterbi algorithm).

    Parameters
    ----------
    imdata : iterator of ndarray
        The imaging data for each time step.
    positions : iterator
        The acquisition positions (e.g. position of scan-head) corresponding
        to the imdata.
    transitions : iterator of tuple ()
    references : ndarray
    state_table : ndarray
    initial_dist : tuple
    num_retained : int

    """
    if state_table.shape[1] is not 3:
        raise ValueError
    log_references = np.log(references)
    backpointer = []
    states = []
    states.append(initial_dist[0])
    log_p_old = initial_dist[1]
    assert np.any(np.isfinite(log_p_old))
    for data, pos, trans in zip(imdata, positions, transitions):
        transition_table, log_transition_probs = trans
        tmp_states, log_p, tmp_backpointer = mc.transitions(
            states[-1], log_transition_probs, log_p_old, state_table,
            transition_table)
        obs, log_obs_fac, log_obs_p = data
        assert len(obs) == len(pos)
        mc.log_observation_probabilities_generalized(
            log_p, tmp_states, obs, log_obs_p, log_obs_fac,
            references, log_references, pos, state_table)
        if np.any(np.isfinite(log_p)):
            log_p[np.isnan(log_p)] = -np.Inf  # Remove nans to sort.
            ix = np.argsort(-log_p)[0:num_retained]  # Keep likely states.
            states.append(tmp_states[ix])
            log_p_old = log_p[ix] - log_p[ix[0]]
            backpointer.append(tmp_backpointer[ix])
        else:
            # If none of the observation probabilities are finite,
            # then use states from the previous timestep.
            warnings.warn('No finite observation probabilities.')
            states.append(states[-1])
            backpointer.append(np.arange(num_retained))
    end_state_idx = np.argmax(log_p_old)
    return _backtrace(end_state_idx, backpointer[1:], states[1:], state_table)


class HiddenMarkov3D(_HiddenMarkov):
    """
    Row-wise hidden Markov model (HMM).

    Parameters
    ----------
    num_states_retained : int, optional
        Number of states to retain at each time step of the HMM.
        Defaults to 50.
    max_displacement : array of int, optional
        The maximum allowed displacement magnitudes in [y,x]. By
        default, arbitrarily large displacements are allowed.

    References
    ----------
    * Dombeck et al. 2007. Neuron. 56(1): 43-57.
    * Kaifosh et al. 2013. Nature Neuroscience. 16(9): 1182-4.

    """
    def _estimate_shifts(self, dataset):
        return sima.motion.frame_align.VolumeTranslation(
            self._params.max_displacement).estimate(dataset)


class NormalizedIterator(object):
    """Generator of preprocessed frames for efficient computation.

    Parameters
    ----------
    sequence : sima.Sequence
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

    Examples
    --------

    Plane-wise iteration

    >>> from sima.motion.hmm import NormalizedIterator
    >>> it = NormalizedIterator(
    ...         np.ones((100, 10, 6, 5, 2)), np.ones(2), np.ones(2),
    ...         np.ones(2), 'plane')
    >>> next(iter(it))[0].shape
    (30, 2)

    Row-wise iteration:

    >>> it = NormalizedIterator(
    ...         np.ones((100, 10, 6, 5, 2)), np.ones(2), np.ones(2),
    ...         np.ones(2), 'row')
    >>> next(iter(it))[0].shape
    (5, 2)

    """
    def __init__(self, sequence, gains, pixel_means, pixel_variances,
                 granularity):
        self.sequence = sequence
        self.gains = gains
        self.pixel_means = pixel_means
        self.pixel_variances = pixel_variances
        try:
            self.granularity = int(granularity)
        except ValueError:
            self.granularity = {'frame': 1,
                                'plane': 2,
                                'row': 3,
                                'column': 4}[granularity]

    def __iter__(self):
        means = self.pixel_means / self.gains
        variances = self.pixel_variances / self.gains ** 2
        for frame in self.sequence:
            frame = frame.reshape(
                int(np.prod(frame.shape[:(self.granularity-1)])),
                -1, frame.shape[-1])
            for chunk in frame:
                im = chunk / self.gains
                # replace NaN pixels with the mean value for the channel
                for ch_idx, ch_mean in enumerate(means):
                    im_nans = np.isnan(im[..., ch_idx])
                    im[..., ch_idx][im_nans] = ch_mean
                assert(np.all(np.isfinite(im)))
                log_im_fac = gammaln(im + 1)  # take the log of the factorial
                # probability of observing the pixels (ignoring reference)
                log_im_p = -(im - means) ** 2 / (2 * variances) \
                    - 0.5 * np.log(2. * np.pi * variances)
                # inf_indices = np.logical_not(np.isfinite(log_im_fac))
                # log_im_fac[inf_indices] = im[inf_indices] * (
                #     np.log(im[inf_indices]) - 1)
                assert(np.all(np.isfinite(log_im_fac)))
                assert(np.all(np.isfinite(log_im_p)))
                yield im, log_im_fac, log_im_p
