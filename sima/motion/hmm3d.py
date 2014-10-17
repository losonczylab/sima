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
    from bottleneck import nanmedian
except ImportError:
    from scipy.stats import nanmedian

import _motion as mc
from sima.motion import MotionEstimationStrategy
from align3d import VolumeTranslation
from _hmm import (
    _backtrace, _lookup_tables, _whole_frame_shifting, _pixel_distribution)

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


class MovementModel(object):
    """

    Attributes
    ----------
    mean_shift : array of int
        The mean of the whole-frame displacement estimates

    Example
    -------

    >>> from sima.motion import MovementModel
    >>> import numpy as np
    >>> shifts = [np.cumsum(np.random.randint(-1, 2, (100, 3)), axis=0)]
    >>> mm = MovementModel.estimate(shifts)
    >>> cov = mm.cov_matrix()
    >>> decay = mm.decay_matrix()
    >>> log_trans = mm.log_transition_matrix

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
        diffs = np.diff(shifts, axis=0)
        cov_matrix = np.cov(diffs[np.isfinite(diffs).all(axis=1)].T)
        # don't allow singular covariance matrix
        for i in range(len(cov_matrix)):
            cov_matrix[i, i] = max(cov_matrix[i, i], 1. / len(shifts))
        assert det(cov_matrix) > 0

        mean_shift = np.nanmean(shifts, axis=0)
        assert len(mean_shift) == shifts.shape[1]
        centered_shifts = np.nan_to_num(shifts - mean_shift)
        # fit to autoregressive AR(1) model
        A = np.dot(pinv(centered_shifts[:-1]), centered_shifts[1:])
        # symmetrize A, assume independent motion on orthogonal axes
        A = 0.5 * (A + A.T)
        U, s, _ = svd(A)  # NOTE: U == V for positive definite A
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
                        transition_probs, 8
                    )
                )
        assert np.all(np.isfinite(log_transition_matrix))
        return log_transition_matrix

    def _initial_distribution(self, decay, noise_cov, mean_shift):
        """Get the initial distribution of the displacements."""
        initial_cov = np.linalg.solve(np.diag([1, 1]) - decay * decay.T,
                                      noise_cov.newbyteorder('>').byteswap())
        for _ in range(1000):
            initial_cov = decay * initial_cov * decay.T + noise_cov
        # don't let C be singular
        initial_cov[0, 0] = max(initial_cov[0, 0], 0.1)
        initial_cov[1, 1] = max(initial_cov[1, 1], 0.1)

        return lambda x:  np.exp(
            - 0.5 * np.dot(x - mean_shift,
                           np.linalg.solve(initial_cov, x - mean_shift))
            ) / np.sqrt(2.0 * np.pi * det(initial_cov))

    def initial_probs(self, displacement_tbl, min_displacements,
                      max_displacements, movement_model):
        """Give the initial probabilites for a displacement table"""
        initial_dist = self._initial_distribution(
            self.decay_matrix(), self.cov_matrix(), self.mean_shift)
        states = []
        log_p = []
        for index, position in enumerate(displacement_tbl):  # TODO parallelize
            # check that the displacement is allowable
            if np.all(min_displacements <= position) and np.all(
                    position <= max_displacements):
                states.append(index)
                # probability of initial displacement
                log_p.append(np.log(initial_dist(position)))
        return np.array(states, dtype='int'), np.array(log_p)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class HiddenMarkov3D(MotionEstimationStrategy):
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

    def __init__(self, num_states_retained=50, max_displacement=None,
                 n_processes=None, verbose=True):

        d = locals()
        del d['self']
        self._params = Struct(**d)

    def _neighbor_viterbi(
            self, dataset, references, gains, movement_model, offset,
            min_displacements, max_displacements, pixel_means, pixel_variances,
            num_states_retained, verbose=True):
        """Estimate the MAP trajectory with the Viterbi Algorithm.

        See _MCCycle.neighbor_viterbi for details."""
        offset = np.array(offset, dtype=int)  # type verification
        assert references.ndim == 4
        scaled_refs = references / self.gains

        displacement_tbl, transition_tbl, log_markov_tbl, = _lookup_tables(
            [min_displacements, max_displacements + 1], [[-1] * 3, [2] * 3],
            movement_model.log_transition_matrix(
                max_distance=1, dt=1./len(references))
        )
        assert displacement_tbl.dtype == int
        tmp_states, log_p = movement_model.initial_probs(
            displacement_tbl, min_displacements, max_displacements)

        displacements = []
        for i, sequence in enumerate(dataset):
            if verbose:
                print 'Estimating displacements for cycle ', i
            imdata = NormalizedIterator(sequence, gains, pixel_means,
                                        pixel_variances)
            positions = PositionIterator(sequence.shape[:-1])
            displacements.append(
                _beam_search(
                    imdata, positions,
                    it.repeat((transition_tbl, log_markov_tbl)), scaled_refs,
                    displacement_tbl, (tmp_states, log_p), num_states_retained
                ).reshape(self.sequence.shape[:3] + (2,))
            )
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
        if params.max_displacement is not None:
            params.max_displacement = np.array(params.max_displacement)
        shifts = VolumeTranslation(params.max_displacement).estimate(dataset)
        references, variances, offset = _whole_frame_shifting(dataset, shifts)
        gains = nanmedian(
            (variances / references).reshape(-1, references.shape[-1]))
        if not (np.all(np.isfinite(gains)) and np.all(gains > 0)):
            raise Exception('Failed to estimate positive gains')
        pixel_means, pixel_variances = _pixel_distribution(dataset)
        movement_model = MovementModel.estimate(shifts)

        # TODO: detect unreasonable shifts before doing this calculation
        min_shifts = np.nanmin(list(it.chain(*it.chain(*shifts))), 0)
        max_shifts = np.nanmax(list(it.chain(*it.chain(*shifts))), 0)

        # add a bit of extra room to move around
        extra_buffer = ((params.max_displacement - max_shifts + min_shifts) / 2
                        ).astype(int)
        extra_buffer[params.max_displacement < 0] = 5
        min_displacements = (min_shifts - extra_buffer)
        max_displacements = (max_shifts + extra_buffer)

        return self._neighbor_viterbi(
            dataset, references, gains, movement_model, offset,
            min_displacements, max_displacements, pixel_means,
            pixel_variances, params.num_states_retained, params.verbose)


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
    """
    def __init__(self, sequence, gains, pixel_means, pixel_variances):
        self.sequence = sequence
        self.gains = gains
        self.pixel_means = pixel_means
        self.pixel_variances = pixel_variances

    def __iter__(self):
        means = self.pixel_means / self.gains
        variances = self.pixel_variances / self.gains ** 2
        for data in self.sequence:
            im = data / self.gains
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


class PositionIterator(object):
    """Position iterator

    Parameters
    ----------
    shape : tuple of int
        (times, planes, rows, columns)

    Examples
    --------

    >>> from sima.motion.hmm3d import PositionIterator
    >>> pi = PositionIterator((100, 5, 128, 256), 3)
    >>> positions = next(iter(pi))

    """

    def __init__(self, shape, granularity=1):
        self.shape = shape
        self.granularity = granularity

    def __iter__(self):
        for base in it.product(*[range(x) for x in
                                 self.shape[:self.granularity]]):
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
    log_references = np.log(references)
    backpointer = []
    states = []
    states.append(initial_dist[0])
    log_p_old = initial_dist[1]
    for data, pos, trans in zip(imdata, positions, transitions):
        transition_table, log_transition_probs = trans
        tmp_states, log_p, tmp_backpointer = mc.transitions(
            states[-1], log_transition_probs, log_p_old, state_table,
            transition_table)
        obs, log_obs_fac, log_obs_p = data
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
            states.append(states[-1])
            backpointer.append(np.arange(num_retained))
            warnings.warn('No finite observation probabilities.')
    end_state_idx = np.argmax(log_p_old)
    return _backtrace(end_state_idx, backpointer[1:], states[1:], state_table)
