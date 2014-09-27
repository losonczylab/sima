import os
import sys
from pickle import Unpickler as _Unpickler

from sima import ImagingDataset, Sequence
from sima.sequence import _resolve_paths


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
