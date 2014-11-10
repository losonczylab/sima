import itertools as it
import warnings
try:
    from future_builtins import zip
except ImportError:  # Python 3.x
    pass

import numpy as np
from scipy.special import gammaln
try:
    from bottleneck import nanmedian
except ImportError:
    from scipy.stats import nanmedian
# from statsmodels.tsa.vector_ar.var_model import VAR

import _motion as mc
from sima.motion import MotionEstimationStrategy
from frame_align import VolumeTranslation
from _hmm import (
    _backtrace, _lookup_tables, _whole_frame_shifting, _pixel_distribution,
    MovementModel)

np.seterr(invalid='ignore', divide='ignore')


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
            self, dataset, references, gains, movement_model,
            min_displacements, max_displacements, pixel_means, pixel_variances,
            num_states_retained, max_step=1, verbose=True):
        """Estimate the MAP trajectory with the Viterbi Algorithm.

        See _MCCycle.neighbor_viterbi for details."""
        assert references.ndim == 4
        granularity = 2
        scaled_refs = references / gains
        displacement_tbl, transition_tbl, log_markov_tbl, = _lookup_tables(
            [min_displacements, max_displacements + 1],
            movement_model.log_transition_matrix(
                max_distance=max_step,
                dt=1./np.prod(references.shape[:(granularity-1)])
            )
        )
        assert displacement_tbl.dtype == int
        tmp_states, log_p = movement_model.initial_probs(
            displacement_tbl, min_displacements, max_displacements)
        displacements = []
        for i, sequence in enumerate(dataset):
            if verbose:
                print 'Estimating displacements for cycle ', i
            imdata = NormalizedIterator(sequence, gains, pixel_means,
                                        pixel_variances, granularity)
            positions = PositionIterator(sequence.shape[:-1], granularity)
            disp = _beam_search(
                imdata, positions,
                it.repeat((transition_tbl, log_markov_tbl)), scaled_refs,
                displacement_tbl, (tmp_states, log_p), num_states_retained)
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
        if params.max_displacement is not None:
            params.max_displacement = np.array(params.max_displacement)
        shifts = VolumeTranslation(params.max_displacement).estimate(dataset)
        references, variances, offset = _whole_frame_shifting(dataset, shifts)
        assert np.all(offset == 0)
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
        if params.max_displacement is None:
            extra_buffer = 5
        else:
            extra_buffer = (
                (params.max_displacement - max_shifts + min_shifts) / 2
            ).astype(int)
        min_displacements = (min_shifts - extra_buffer)
        max_displacements = (max_shifts + extra_buffer)

        return self._neighbor_viterbi(
            dataset, references, gains, movement_model, min_displacements,
            max_displacements, pixel_means,
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

    Examples
    --------

    Plane-wise iteration

    >>> from sima.motion.hmm3d import NormalizedIterator
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


class PositionIterator(object):
    """Position iterator

    Parameters
    ----------
    shape : tuple of int
        (times, planes, rows, columns)

    Examples
    --------

    >>> from sima.motion.hmm3d import PositionIterator
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
            raise Exception('No finite observation probabilities.')
            states.append(states[-1])
            backpointer.append(np.arange(num_retained))
    end_state_idx = np.argmax(log_p_old)
    return _backtrace(end_state_idx, backpointer[1:], states[1:], state_table)
