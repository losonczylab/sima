import os
import sys
from pickle import Unpickler as _Unpickler
import cPickle as pkl
try:
    from future_builtins import zip
except ImportError:  # Python 3.x
    pass
from distutils.util import strtobool

import numpy as np

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
    """Returns a v1 dataset converted from a v0 dataset

    Parameters
    ----------
    path : str
        The path (ending in .sima) of the version 0.x dataset.

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
                clip = None
            if clip is not None:
                s = (slice(None), slice(None)) + tuple(
                    slice(*[None if x is 0 else x for x in dim])
                    for dim in clip)
                result = result[s]
        elif klass == 'sima.iterables.HDF5':
            result = Sequence.create(
                'HDF5', channel['path'], channel['dim_order'],
                channel['group'], channel['key'])
            c = channel['dim_order'].index('c')
            chan = channel['channel']
            s = tuple([slice(None) if x != c else slice(chan, chan + 1)
                       for x in range(len(channel['dim_order']))])
            result = result[s]
            try:
                clip = channel['clip']
            except KeyError:
                clip = None
            if clip is not None:
                s = (slice(None), slice(None)) + tuple(
                    slice(*[None if x is 0 else x for x in dim])
                    for dim in clip) + (slice(None),)
                result = result[s]
        else:
            raise Exception('Format not recognized.')
        return result

    def parse_sequence(sequence):
        channels = [parse_channel(c) for c in sequence]
        return Sequence.join(*channels)

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
        max_disp = np.nanmax([np.nanmax(d.reshape(-1, d.shape[-1]), 0)
                              for d in displacements], 0)
        frame_shape = np.array(sequences[0].shape)[1:-1]  # z, y, x
        frame_shape[1:3] += max_disp
        sequences = [
            s.apply_displacements(d.reshape(s.shape[:3] + (2,)), frame_shape)
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
    # Not making it read-only. If you set a savedir, you'll be asked about
    # overwriting it then
    ds._channel_names = [str(n) for n in dataset_dict.pop('channel_names')]
    ds._savedir = path
    return ds


def _0_to_1(source, target=None):
    """Convert a version 0.x dataset to a version 1.x dataset.

    Parameters
    ----------
    path : str
        The path (ending in .sima) of the version 0.x dataset.
    target : str, optional
        The path (ending in .sima) for saving the version 1.x dataset. Defaults
        to None, resulting in overwrite of the existing dataset.pkl file.  to
        avoid the overwrite permission prompt, explicitly set this argument

    Examples
    --------

    >>> from sima import ImagingDataset
    >>> from sima.misc import example_data
    >>> from sima.misc.convert import _0_to_1
    >>> _0_to_1(example_data(), 'v1_dataset.sima')
    >>> ds = ImagingDataset.load('v1_dataset.sima')
    """
    if target is None:
        overwrite = strtobool(raw_input("Source dataset path = target path. " +
                                        "Overwrite existing?"))
        if not overwrite:
            return
        target = source

    ds = _load_version0(source)
    ds.save(target)
