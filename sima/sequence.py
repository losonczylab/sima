# ImagingDataset objects must be initialized with a list of
# `iterable <http://docs.python.org/2/glossary.html#term-iterable>`_
# objects that satisfy the following properties:
#
# * The iterable should not be its own iterator, i.e. it should be able to
#   spawn multiple iterators that can be iterated over independently.
# * Each iterator spawned from the iterable must yield image frames in the form
#   of numpy arrays with shape (num_rows, num_columns).
# * Iterables must survive pickling and unpickling.
#
# Examples of valid sequence include:
#
# * numpy arrays of shape (num_frames, num_rows, num_columns)
#
#   >>> import sima
#   >>> from numpy import ones
#   >>> frames = ones((100, 128, 128))
#   >>> sima.ImagingDataset([[frames]], None)
#   <ImagingDataset: num_channels=1, num_cycles=1, frame_size=128x128,
#   num_frames=100>
#
# * lists of numpy arrays of shape (num_rows, num_columns)
#
#   >>> frames = [ones((128, 128)) for _ in range(100)]
#   >>> sima.ImagingDataset([[frames]], None)
#   <ImagingDataset: num_channels=1, num_cycles=1, frame_size=128x128,
#   num_frames=100>
#
# For convenience, we have created iterable objects that can be used with
# common data formats.


import itertools as it
import warnings
import collections
from distutils.version import StrictVersion
from os.path import (abspath, dirname, join, normpath, normcase, isfile,
                     relpath)
from abc import ABCMeta, abstractmethod
import numpy as np

try:
    from os.path import samefile
except ImportError:
    # Windows does not have the samefile function
    from os import stat

    def samefile(file1, file2):
        return stat(file1) == stat(file2)

try:
    from libtiff import TIFF
    libtiff_available = True
except ImportError:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sima.misc.tifffile import TiffFile
    libtiff_available = False
try:
    import h5py
except ImportError:
    h5py_available = False
else:
    h5py_available = StrictVersion(h5py.__version__) >= StrictVersion('2.3.1')

import sima.misc
from sima.motion._motion import _align_frame
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sima.misc.tifffile import TiffFileWriter


class Sequence(object):

    """Object containing data from sequentially acquired imaging data.

    Sequences are created with a call to the create method.

    >>> from sima import Sequence
    >>> from sima.misc import example_hdf5
    >>> path = example_hdf5()
    >>> seq = Sequence.create('HDF5', path, 'yxt')

    For numpy 1.9 or higher, Sequences are array like, and can be converted to
    numpy arrays or passed as arguments into numpy functions that take arrays.

    >>> import numpy as np
    >>> arr = np.array(seq)
    >>> time_avg = np.mean(seq, axis=0)
    >>> np.shape(seq) == seq.shape
    True

    Note, however, that the application of numpy functions to Sequence objects
    may force the entire sequence to be loaded into memory at once.  Depending
    on the size of the data and the available memory, this may result in memory
    errors. If possible, consider slicing the array prior prior to applying the
    numpy function.


    Attributes
    ----------
    shape : tuple
        (num_frames, num_planes, num_rows, num_columns, num_channels)

    """
    __metaclass__ = ABCMeta

    def __getitem__(self, indices):
        """Create a new Sequence by slicing this Sequence."""
        return _IndexedSequence(self, indices)

    @abstractmethod
    def __iter__(self):
        """Iterate over the frames of the Sequence.

        The yielded structures are numpy arrays of the shape (num_planes,
        num_rows, num_columns, num_channels).
        """
        raise NotImplementedError

    def _get_frame(self, t):
        raise NotImplementedError

    @abstractmethod
    def _todict(self, savedir=None):
        raise NotImplementedError

    @classmethod
    def _from_dict(cls, d, savedir=None):
        """Create a Sequence instance from a dictionary."""
        if savedir is not None:
            _resolve_paths(d, savedir)
        return cls(**d)

    def __len__(self):
        return sum(1 for _ in self)

    @property
    def shape(self):
        return (len(self),) + iter(self).next().shape

    def apply_displacements(self, displacements, frame_shape=None):
        return _MotionCorrectedSequence(self, displacements, frame_shape)

    def mask(self, masks):
        """Apply a mask to the sequence.

        Masked values will be represented as numpy.NaN.

        Parameters
        ----------
        masks : list of tuple
            Each element of the list is a tuple describing a mask.  Each mask
            tuple can have the form (frames, zyx-mask, channels) or (frames,
            planes, yx-mask, channels). The frames and channels elements of
            the tuple are lists of the frames and channels to which the mask
            is to be applied. They yx-mask element is a binary array whose
            True values indicate the pixels to be masked. If any of the
            entries is set to None, than the mask will be fully applied along
            that dimension.

        Examples
        --------

        Mask out frame 3 entirely:

        >>> from sima import Sequence
        >>> from sima.misc import example_hdf5
        >>> path = example_hdf5()
        >>> seq = Sequence.create('HDF5', path, 'yxt')
        >>> masked_seq = seq.mask([(3, None, None)])
        >>> [np.all(np.isnan(frame))
        ...  for i, frame in enumerate(masked_seq) if i < 5]
        [False, False, False, True, False]

        Mask out plane 0 of frame 3:

        >>> masked_seq = seq.mask([(3, 0, None, None)])

        Mask out certain pixels at all times in channel 0.

        >>> mask = np.random.binomial(1, 0.5, seq.shape[1:-1])
        >>> masked_seq = seq.mask([(None, mask, 0)])

        """
        return _MaskedSequence(self, masks)

    @staticmethod
    def join(*sequences):
        """Join together sequences representing different channels.

        Parameters
        ----------
        sequences : sima.Sequence
            Each argument is a Sequence representing a different channel.

        Returns
        -------
        joined_sequence : sima.Sequence
            A single sequence with multiple channels.

        Examples
        --------

        >>> from sima import Sequence
        >>> from sima.misc import example_hdf5
        >>> path = example_hdf5()
        >>> seq = Sequence.create('HDF5', path, 'yxt')
        >>> joined = Sequence.join(seq, seq)
        >>> joined.shape[:4] == seq.shape[:4]
        True
        >>> joined.shape[4] == 2 * seq.shape[4]
        True

        """
        return _Joined_Sequence(sequences)

    @classmethod
    def create(cls, fmt, *args, **kwargs):
        """Create a Sequence object.

        Parameters
        ----------
        fmt : {'HDF5', 'TIFF', 'ndarray'}
            The format of the data used to create the Sequence.
        *args, **kwargs
            Additional arguments depending on the data format.

        Notes
        -----

        Below are explanations of the arguments for each format.

        **HDF5**

        path : str
            The HDF5 filename, typicaly with .h5 extension.
        dim_order : str
            Specification of the order of the dimensions. This
            string can contain the letters 't', 'x', 'y', 'z',
            and 'c', representing time, column, row, plane,
            and channel, respectively.
            For example, 'tzyxc' indicates that the HDF5 data
            dimensions represent time (t), plane (z), row (y),
            column(x), and channel (c), respectively.
            The string 'tyx' indicates data that data for a single
            imaging plane and single channel has been stored in a
            HDF5 dataset with three dimensions representing time (t),
            column (y), and row (x) respectively.
            Note that SIMA 0.1.x does not support multiple z-planes,
            although these will be supported in future versions.
        group : str, optional
            The HDF5 group containing the imaging data.
            Defaults to using the root group '/'
        key : str, optional
            The key for indexing the the HDF5 dataset containing
            the imaging data. This can be omitted if the HDF5
            group contains only a single key.

        >>> from sima import Sequence
        >>> from sima.misc import example_hdf5
        >>> path = example_hdf5()
        >>> seq = Sequence.create('HDF5', path, 'yxt')
        >>> seq.shape
        (20, 1, 128, 256, 1)

        Warning
        -------
        Moving the HDF5 file may make this Sequence unusable
        when the ImagingDataset is reloaded. The HDF5 file can
        only be moved if the ImagingDataset path is also moved
        such that they retain the same relative position.


        **TIFF**

        path : str
            The path to the file storing the imaging data.
        num_planes : int, optional
            The number of interleaved planes. Default: 1.
        num_channels : int, optional
            The number of interleaved channels. Default: 1.

        Warning
        -------
        Moving the TIFF file may make this Sequence unusable
        when the ImagingDataset is reloaded. The TIFF file can
        only be moved if the ImagingDataset path is also moved
        such that they retain the same relative position.


        **ndarray**

        array : numpy.ndarray
            A numpy array of shape (num_frames, num_planes, num_rows,
            num_columns, num_channels)

        """
        if fmt == 'HDF5':
            return _Sequence_HDF5(*args, **kwargs)
        elif fmt == 'TIFF':
            return _Sequence_TIFF_Interleaved(*args, **kwargs)
        elif fmt == 'ndarray':
            return _Sequence_ndarray(*args, **kwargs)
        else:
            raise ValueError('Unrecognized format')

    def export(self, filenames, fmt='TIFF16', fill_gaps=False,
               channel_names=None):
        """Save frames to the indicated filenames.

        This function stores a multipage tiff file for each channel.

        Paramters
        ---------
        filenames : str or list of list str
            The names of the output files. For HDF5 files, this must be a
            single string. For TIFF formats, this should be a list of list
            of strings, such that filenames[i][j] corresponds to the ith
            plane and the jth channel.
        fmt : {'HDF5', 'TIFF16', 'TIFF8'}
            The output file format.
        fill_gaps : bool, optional
            Whether to fill in missing data with pixel intensities from
            adjacent frames. Default: False.
        channel_names : list of str, optional
            List of labels for the channels to be saved if using HDF5 format.
        """
        depth = lambda L: \
            isinstance(L, collections.Sequence) and \
            (not isinstance(L, str)) and max(map(depth, L)) + 1
        if fmt not in ['TIFF8', 'TIFF16', 'HDF5']:
            raise ValueError('Unrecognized output format.')
        if (fmt in ['TIFF16', 'TIFF8']) and not depth(filenames) == 2:
            raise ValueError

        # Make directories necessary for saving the files.
        try:  # HDF5 case
            out_dirs = [[dirname(filenames)]]
        except AttributeError:  # TIFF case
            out_dirs = [[dirname(f) for f in plane] for plane in filenames]
        for d in filter(None, it.chain.from_iterable(out_dirs)):
            sima.misc.mkdir_p(d)

        if 'TIFF' in fmt:
            output_files = [[TiffFileWriter(fn) for fn in plane]
                            for plane in filenames]
        elif fmt == 'HDF5':
            if not h5py_available:
                raise ImportError('h5py >= 2.3.1 required')
            f = h5py.File(filenames, 'w')
            output_array = np.empty(self.shape, dtype='uint16')
            # TODO: change dtype?

        if fill_gaps:
            save_frames = _fill_gaps(iter(self), iter(self))
        else:
            save_frames = iter(self)
        for f_idx, frame in enumerate(save_frames):
            if fmt == 'HDF5':
                output_array[f_idx] = frame
            else:
                for plane_idx, plane in enumerate(frame):
                    for ch_idx, channel in enumerate(np.rollaxis(plane, -1)):
                        f = output_files[plane_idx][ch_idx]
                        if fmt == 'TIFF16':
                            f.write_page(channel.astype('uint16'))
                        elif fmt == 'TIFF8':
                            f.write_page(channel.astype('uint8'))
                        else:
                            raise ValueError('Unrecognized output format.')
        if 'TIFF' in fmt:
            for f in it.chain.from_iterable(output_files):
                f.close()
        elif fmt == 'HDF5':
            f.create_dataset(name='imaging', data=output_array)
            for idx, label in enumerate(['t', 'z', 'y', 'x', 'c']):
                f['imaging'].dims[idx].label = label
            if channel_names is not None:
                f['imaging'].attrs['channel_names'] = np.array(channel_names)
            f.close()


class _Sequence_TIFF_Interleaved(Sequence):
    """

    Parameters
    ----------

    Warning
    -------
    Moving the TIFF files may make this iterable unusable
    when the ImagingDataset is reloaded. The TIFF file can
    only be moved if the ImagingDataset path is also moved
    such that they retain the same relative position.

    """
    def __init__(self, path, num_planes=1, num_channels=1, len_=None):
        self._num_planes = num_planes
        self._num_channels = num_channels
        self._path = abspath(path)
        self._len = len_
        if not libtiff_available:
            self.stack = TiffFile(self._path)

    def __iter__(self):
        base_iter = self._iter_pages()
        while True:
            yield np.concatenate(
                [np.expand_dims(
                    np.concatenate(
                        [np.expand_dims(next(base_iter), 2).astype(float)
                         for _ in range(self._num_channels)],
                        axis=2), 0)
                 for _ in range(self._num_planes)], 0)

    def _iter_pages(self):
        if libtiff_available:
            tiff = TIFF.open(self._path, 'r')
            for frame in tiff.iter_images():
                yield frame.astype(float)
        else:
            for frame in self.stack.pages:
                yield frame.asarray(colormapped=False)
        if libtiff_available:
            tiff.close()

    def _todict(self, savedir=None):
        d = {'__class__': self.__class__,
             'num_planes': self._num_planes,
             'num_channels': self._num_channels,
             'len_': self._len}
        if savedir is None:
            d.update({'path': abspath(self._path)})
        else:
            d.update({'_abspath': abspath(self._path),
                      '_relpath': relpath(self._path, savedir)})
        return d

    def __len__(self):
        if self._len is None:
            self._len = sum(1 for _ in self)
        return self._len


class _IndexableSequence(Sequence):

    """Iterable whose underlying structure supports indexing."""
    __metaclass__ = ABCMeta

    def __iter__(self):
        for t in xrange(len(self)):
            yield self._get_frame(t)

    # @abstractmethod
    # def _get_frame(self, t):
    #     """Return frame with index t."""
    #     pass


class _Sequence_ndarray(_IndexableSequence):
    def __init__(self, array):
        self._array = array

    def __len__(self):
        return len(self._array)

    def _get_frame(self, t):
        return self._array[t].astype(float)

    def _todict(self, savedir=None):
        return {'__class__': self.__class__, 'array': self._array}


class _Sequence_HDF5(_IndexableSequence):

    """
    Iterable for an HDF5 file containing imaging data.

    See sima.Sequence.create() for details.
    """

    def __init__(self, path, dim_order, group=None, key=None):
        if not h5py_available:
            raise ImportError('h5py >= 2.3.1 required')
        self._path = abspath(path)
        self._file = h5py.File(path, 'r')
        if group is None:
            group = '/'
        self._group = self._file[group]
        if key is None:
            if len(self._group.keys()) != 1:
                raise ValueError(
                    'key must be provided to resolve ambiguity.')
            key = self._group.keys()[0]
        self._key = key
        self._dataset = self._group[key]
        if len(dim_order) != len(self._dataset.shape):
            raise ValueError(
                'dim_order must have same length as the number of ' +
                'dimensions in the HDF5 dataset.')
        self._T_DIM = dim_order.find('t')
        self._Z_DIM = dim_order.find('z')
        self._Y_DIM = dim_order.find('y')
        self._X_DIM = dim_order.find('x')
        self._C_DIM = dim_order.find('c')
        self._dim_order = dim_order

    def __len__(self):
        return self._dataset.shape[self._T_DIM]
        # indices = self._time_slice.indices(self._dataset.shape[self._T_DIM])
        # return (indices[1] - indices[0] + indices[2] - 1) // indices[2]

    def _get_frame(self, t):
        """Get the frame at time t, but not clipped"""
        slices = tuple(slice(None) for _ in range(self._T_DIM)) + (t,)
        frame = self._dataset[slices]
        swapper = [None for _ in range(frame.ndim)]
        for i, v in [(self._Z_DIM, 0), (self._Y_DIM, 1),
                     (self._X_DIM, 2), (self._C_DIM, 3)]:
            if i >= 0:
                j = i if self._T_DIM > i else i - 1
                swapper[j] = v
            else:
                swapper.append(v)
                frame = np.expand_dims(frame, -1)
        assert not any(s is None for s in swapper)
        for i in range(frame.ndim):
            idx = swapper.index(i)
            if idx != i:
                swapper[i], swapper[idx] = swapper[idx], swapper[i]
                frame = frame.swapaxes(i, idx)
        assert swapper == [0, 1, 2, 3]
        assert frame.ndim == 4
        return frame.astype(float)

    def _todict(self, savedir=None):
        d = {'__class__': self.__class__,
             'dim_order': self._dim_order,
             'group': self._group.name,
             'key': self._key}
        if savedir is None:
            d.update({'path': abspath(self._path)})
        else:
            d.update({'_abspath': abspath(self._path),
                      '_relpath': relpath(self._path, savedir)})
        return d


class _Joined_Sequence(Sequence):

    def __init__(self, sequences):
        shape = None
        num_channels = 0
        self._sequences = sequences
        for seq in sequences:
            if shape is None:
                shape = seq.shape[:-1]
            if not shape == seq.shape[:-1]:
                raise ValueError(
                    'Sequences being joined must have the same number '
                    'of frames, planes, rows, and columns.')
            num_channels += seq.shape[-1]
        self._shape = shape + (num_channels,)

    def __len__(self):
        return self._shape[0]

    @property
    def shape(self):
        return self._shape

    def __iter__(self):
        for frames in it.izip(*self._sequences):
            yield np.concatenate(frames, axis=3)

    def _get_frame(self, t):
        return np.concatenate([seq._get_frame(t) for seq in self._sequences],
                              axis=3)

    def _todict(self, savedir=None):
        return {
            '__class__': self.__class__,
            'sequences': [s._todict(savedir) for s in self._sequences],
        }

    @classmethod
    def _from_dict(cls, d, savedir=None):
        sequences = []
        for s in d.pop('sequences'):
            seq_class = s.pop('__class__')
            sequences.append(seq_class._from_dict(s, savedir))
        return cls(sequences)


class _WrapperSequence(Sequence):
    "Abstract class for wrapping a Sequence to modify its functionality"""
    __metaclass__ = ABCMeta

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        try:
            getattr(super(_WrapperSequence, self), name)
        except AttributeError as err:
            if err.args[0] == \
                    "'super' object has no attribute '" + name + "'":
                return getattr(self._base, name)
            else:
                raise err

    def _todict(self, savedir=None):
        raise NotImplementedError

    @classmethod
    def _from_dict(cls, d, savedir=None):
        base_dict = d.pop('base')
        base_class = base_dict.pop('__class__')
        base = base_class._from_dict(base_dict, savedir)
        return cls(base, **d)


class _MotionCorrectedSequence(_WrapperSequence):

    """Wraps any other sequence to apply motion correction.

    Parameters
    ----------
    base : Sequence
    extent : tuple
        (num_planes, num_rows, num_columns)

    displacements : array
        The _D displacement of each row in the image cycle.
        Shape: (num_frames, num_planes, num_rows, 2).

    This object has the same attributes and methods as the class it wraps."""
    # TODO: check clipping and output frame size

    def __init__(self, base, displacements, extent=None):
        super(_MotionCorrectedSequence, self).__init__(base)
        if np.min(displacements) < 0:
            raise ValueError("All displacements must be non-negative")
        self.displacements = displacements.astype('int')
        if extent is None:
            max_disp = np.nanmax([np.nanmax(d.reshape(-1, d.shape[-1]), 0)
                                  for d in displacements], 0)
            extent = np.array(base._sequences[0].shape)[1:-1]
            extent[1:3] += max_disp
        assert len(extent) == 3
        self._frame_shape_zyx = tuple(extent)   # (planes, rows, columns)

    @ property
    def _frame_shape(self):
        return self._frame_shape_zyx + (self._base.shape[4],)

    def __len__(self):
        return len(self._base)  # Faster to calculate len without aligning

    def _align(self, frame, displacement):
        if displacement.ndim == 3:
            return _align_frame(frame.astype(float), displacement.astype(int),
                                self._frame_shape)
        elif displacement.ndim == 2:  # plane-wise displacement
            out = np.nan * np.ones(self._frame_shape)
            s = frame.shape
            for p, (plane, disp) in enumerate(it.izip(frame, displacement)):
                if len(disp) == 2:
                    disp = [0] + list(disp)
                out[p + disp[0],
                    disp[1]:(disp[1] + s[1]),
                    disp[2]:(disp[2] + s[2])] = plane
            return out
        elif displacement.ndim == 1:  # frame-wise displacement
            out = np.nan * np.ones(self._frame_shape)
            s = frame.shape
            out[displacement[0]:(displacement[0] + s[0]),
                displacement[1]:(displacement[1] + s[1]),
                displacement[2]:(displacement[2] + s[2])] = frame
            return out

    @property
    def shape(self):
        # Avoid aligning image
        return (len(self),) + self._frame_shape

    def __iter__(self):
        for frame, displacement in it.izip(self._base, self.displacements):
            yield self._align(frame, displacement)

    def _get_frame(self, t):
        return self._align(self._base._get_frame(t), self.displacements[t])

    def __getitem__(self, indices):
        if len(indices) > 5:
            raise ValueError
        indices = indices if isinstance(indices, tuple) else (indices,)
        times = indices[0]
        if indices[0] not in (None, slice(None)):
            new_indices = (None,) + indices[1:]
            return _MotionCorrectedSequence(
                self._base[times],
                self.displacements[times],
                self._frame_shape[:-1]
            )[new_indices]
        if len(indices) == 5:
            chans = indices[4]
            return _MotionCorrectedSequence(
                self._base[:, :, :, :, chans],
                self.displacements,
                self._frame_shape[:-1]
            )[indices[:4]]
        # TODO: similar for planes ???
        return _IndexedSequence(self, indices)

    def _todict(self, savedir=None):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(savedir),
            'displacements': self.displacements.astype('int16'),
            'extent': self._frame_shape[:3],
        }


class _MaskedSequence(_WrapperSequence):
    """Sequence for masking invalid data with NaN's.

    Parameters
    ----------
    base : Sequence
    indices : list of outer product tuples:
        (frames, zyx mask, channels)
        (frames, planes, yx mask, channels)
        If frames is None, then the mask is applied to all frames.
        If the mask is None

    """
    def __init__(self, base, outers):
        super(_MaskedSequence, self).__init__(base)
        self._base_len = len(base)
        self._outers = outers
        self._mask_dict = {}
        self._static_masks = []
        for i, outer in enumerate(self._outers):
            if outer[0] is None:
                self._static_masks.append(i)
            else:
                times = [outer[0]] if isinstance(outer[0], int) else outer[0]
                for t in times:
                    try:
                        self._mask_dict[t].append(i)
                    except KeyError:
                        self._mask_dict[t] = [i]

    def _apply_masks(self, frame, t):
        masks = self._static_masks
        try:
            masks = masks + self._mask_dict[t]
        except KeyError:
            pass
        for i in masks:
            outer = self._outers[i][1:]
            if len(outer) == 2:  # (zyx, channels)
                if outer[0] is None:
                    frame[:, :, :, outer[1]] = np.nan
                else:
                    frame[:, :, :, outer[1]][outer[0]] = np.nan
            elif len(outer) == 3:  # (planes, yx, channels)
                planes = \
                    range(frame.shape[-1]) if outer[0] is None else outer[0]
                for p in planes:
                    if outer[1] is None:
                        frame[p][:, :, outer[2]] = np.nan
                    else:
                        frame[p][:, :, outer[2]][outer[1]] = np.nan
            else:
                raise Exception

    def _get_frame(self, t):
        frame = self._base._get_frame(t)
        self._apply_masks(frame, t)
        return frame

    def __iter__(self):
        for t, frame in enumerate(self._base):
            self._apply_masks(frame, t)
            yield frame

    @property
    def shape(self):
        return self._base.shape

    def __len__(self):
        return len(self._base)

    def _todict(self, savedir=None):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(savedir),
            'outers': self._outers
        }


class _IndexedSequence(_WrapperSequence):

    def __init__(self, base, indices):
        super(_IndexedSequence, self).__init__(base)
        self._base_len = len(base)
        self._indices = \
            indices if isinstance(indices, tuple) else (indices,)
        # Reformat integer slices to avoid dimension collapse
        self._indices = tuple(
            slice(i, i + 1) if isinstance(i, int) else i
            for i in self._indices)
        self._times = range(self._base_len)[self._indices[0]]
        # TODO: switch to generator/iterator if possible?

    def __iter__(self):
        try:
            for t in self._times:
                yield self._base._get_frame(t)[self._indices[1:]]
        except NotImplementedError:
            idx = 0
            for t, frame in enumerate(self._base):
                try:
                    whether_yield = t == self._times[idx]
                except IndexError:
                    raise StopIteration
                if whether_yield:
                    yield frame[self._indices[1:]]
                    idx += 1

    def _get_frame(self, t):
        return self._base._get_frame(self._times[t])[self._indices[1:]]

    def __len__(self):
        return len(range(len(self._base))[self._indices[0]])

    def _todict(self, savedir=None):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(savedir),
            'indices': self._indices
        }

    # def __dir__(self):
    #     """Customize how attributes are reported, e.g. for tab completion.

    #     This may not be necessary if we inherit an abstract class"""
    #     heritage = dir(super(self.__class__, self)) # inherited attributes
    #     return sorted(heritage + self.__class__.__dict__.keys() +
    #                   self.__dict__.keys())


def _fill_gaps(frame_iter1, frame_iter2):
    """Fill missing rows in the corrected images with data from nearby times.

    Parameters
    ----------
    frame_iter1 : iterator of list of array
        The corrected frames (one list entry per channel).
    frame_iter2 : iterator of list of array
        The corrected frames (one list entry per channel).

    Yields
    ------
    list of array
        The corrected and filled frames.
    """
    first_obs = next(frame_iter1)
    for frame in frame_iter1:
        for frame_chan, fobs_chan in zip(frame, first_obs):
            fobs_chan[np.isnan(fobs_chan)] = frame_chan[np.isnan(fobs_chan)]
        if all(np.all(np.isfinite(chan)) for chan in first_obs):
            break
    most_recent = [x * np.nan for x in first_obs]
    for frame in frame_iter2:
        for fr_chan, mr_chan in zip(frame, most_recent):
            mr_chan[np.isfinite(fr_chan)] = fr_chan[np.isfinite(fr_chan)]
        yield [np.nan_to_num(mr_ch) + np.isnan(mr_ch) * fo_ch
               for mr_ch, fo_ch in zip(most_recent, first_obs)]


def _resolve_paths(d, savedir):
    """Resolve the relative and absolute paths to the sequence data."""
    def path_compare(p1, p2):
        """Compare two file paths."""
        return samefile(normcase(abspath(normpath(p1))),
                        normcase(abspath(normpath(p2))))
    paths = set()
    try:
        paths.add(abspath(join(savedir, d.pop('_relpath'))))
    except KeyError:
        pass
    try:
        paths.add(d.pop('_abspath'))
    except KeyError:
        pass
    if len(paths):
        paths = filter(isfile, paths)
        if not len(paths):
            raise Exception('Files have been moved. Cannot locate data.')
        if len(paths) > 1:
            testfile = paths.pop()
            if not all(path_compare(testfile, p) for p in paths):
                raise Exception('Files have been moved. The path '
                                'cannot be unambiguously resolved.')
        d['path'] = paths.pop()
