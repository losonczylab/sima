import os
import sys
from pickle import Unpickler as _Unpickler
import cPickle as pkl
from itertools import chain
try:
    from future_builtins import zip
except ImportError:  # Python 3.x
    pass

import numpy as np

from sima import ImagingDataset, Sequence
from sima.sequence import _resolve_paths
from sima.motion import _MotionCorrectedSequence


class Unpickler(_Unpickler):
    """A modified Unpickler for loading classes that are not present."""

    def find_class(self, module, name):
        # Subclasses may override this
        try:
            __import__(module)
        except ImportError as err:
            if err.args[0] == 'No module named iterables':
                return module + '.' + name
            else:
                raise
        else:
            mod = sys.modules[module]
            klass = getattr(mod, name)
            return klass


def _load_version0(path):
    """Load a SIMA 0.x dataset

    Parameters
    ----------
    path : str
        The path to the original saved dataset, ending in .sima

    Examples
    --------

    >>> from sima.misc import example_data
    >>> from sima.misc.convert import _load_version0
    >>> ds = _load_version0(example_data())

    """

    def parse_channel(channel):
        """Parse an old format channel stored a dictionary

        Parameters
        ----------
        channel : dict

        Returns
        -------
        result : sima.Sequence
            A sequence equivalent to the old format channel.
        """
        _resolve_paths(channel, path)
        klass = channel.pop('__class__')
        if klass == 'sima.iterables.MultiPageTIFF':
            result = Sequence.create('TIFF', channel['path'])
            try:
                clip = channel['clip']
            except KeyError:
                pass
            else:
                if clip is not None:
                    s = (slice(None), slice(None)) + tuple(
                        slice(*[None if x is 0 else x for x in dim])
                        for dim in clip)
                    result = result[s]
            return result

        elif klass == 'sima.iterables.HDF5':
            raise Exception('TODO')
        else:
            raise Exception('Format not recognized.')

    def parse_sequence(sequence):
        channels = [parse_channel(c) for c in sequence]
        return Sequence.join(channels)

    with open(os.path.join(path, 'dataset.pkl'), 'rb') as f:
        unpickler = Unpickler(f)
        dataset_dict = unpickler.load()
    iterables = dataset_dict.pop('iterables')
    sequences = [parse_sequence(seq) for seq in iterables]

    # Apply displacements if they exist
    try:
        with open(os.path.join(path, 'displacements.pkl'), 'rb') as f:
            displacements = pkl.load(f)
    except IOError:
        pass
    else:
        assert all(np.all(d >= 0) for d in displacements)
        max_disp = np.max(list(chain(*displacements)), axis=0)
        frame_shape = np.array(sequences[0].shape)[1:]
        frame_shape[1:3] += max_disp
        sequences = [
            _MotionCorrectedSequence(
                s, d.reshape(s.shape[:3] + (2,)), frame_shape)
            for s, d in zip(sequences, displacements)]
        try:
            trim_coords = dataset_dict.pop('_lazy__trim_coords')
        except KeyError:
            try:
                trim_criterion = dataset_dict.pop('trim_criterion')
            except KeyError:
                pass
            else:
                raise Exception(
                    'Parsing of trim_criterion ' + str(trim_criterion) +
                    ' not yet implemented')
        else:
            sequences = [s[:, :, trim_coords[0][0]:trim_coords[1][0],
                           trim_coords[0][1]:trim_coords[1][1]]
                         for s in sequences]
    ds = ImagingDataset(sequences, None)
    ds.savedir = path
    return ds


def _0_to_1(source, target):
    """Convert a version 0.x dataset to a version 1.x dataset.

    Parameters
    ----------
    source : str
        The path (ending in .sima) of the version 0.x dataset.
    target : str
        The path (ending in .sima) for saving the version 0.x dataset.

    Examples
    --------

    >>> from sima import ImagingDataset
    >>> from sima.misc import example_data
    >>> from sima.misc.convert import _0_to_1
    >>> _0_to_1(example_data(), '0_to_1.sima')
    >>> ds = ImagingDataset.load('0_to_1.sima')

    """
    ds0 = _load_version0(source)
    ds0.save(target)
