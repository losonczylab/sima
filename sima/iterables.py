"""
Iterables
=========

ImagingDataset objects must be initialized with
`iterable <http://docs.python.org/2/glossary.html#term-iterable>`_
objects that satisfy the following properties:

* The iterable should not be its own iterator, i.e. it should be able to
  spawn multiple iterators that can be iterated over independently.
* Each iterator spawned from the iterable must yield image frames in the form
  of numpy arrays with shape (num_rows, num_columns).
* Iterables must survive pickling and unpickling.

Examples of valid iterables include:

* numpy arrays of shape (num_frames, num_rows, num_columns)

  >>> import sima
  >>> from numpy import ones
  >>> frames = ones((100, 128, 128))
  >>> sima.ImagingDataset([[frames]], None)
  <ImagingDataset: num_channels=1, num_cycles=1, frame_size=128x128,
  num_frames=100>

* lists of numpy arrays of shape (num_rows, num_columns)

  >>> frames = [ones((128, 128)) for _ in range(100)]
  >>> sima.ImagingDataset([[frames]], None)
  <ImagingDataset: num_channels=1, num_cycles=1, frame_size=128x128,
  num_frames=100>

For convenience, we have created iterable objects that can be used with
common data formats.
"""

from os.path import abspath, dirname
import warnings
import copy
from distutils.version import StrictVersion
from abc import ABCMeta, abstractmethod

import numpy as np

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
from sima._motion import _align_frame
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sima.misc.tifffile import TiffFileWriter

class Sequence(object):
    """A sequence contains the data.

    Attributes
    ----------
    shape : tuple
        (num_frames, num_planes, num_rows, num_columns, num_channels)
    channel_names : list of str -- or just at the dataset level???
    invalid_frames/clear_frame

    """
    __metaclass__ = ABCMeta

    def __getitem__(self, indices):
        """Create a new Sequence by slicing this Sequence."""
        return _IndexedIterable(self, indices)

    @abstractmethod
    def __iter__(self):
        """Iterate over the frames of the Sequence.

        The yielded structures are numpy arrays of the shape (num_planes,
        num_rows, num_columns, num_channels).
        """
        pass

    def __len__(self):
        return sum(1 for _ in self)

    @property
    def shape(self):
        return (len(self),) + iter(self).next().shape

    def toarray(self, squeeze=False):
        """Convert to a numpy array.

        Arguments
        ---------
        squeeze : bool

        Returns
        -------
        array : numpy.ndarray
            The pixel values from the dataset as a numpy array
            with the same shape as the Sequence.
        """
        return np.concatenate(np.expand_dims(x, 0) for x in self)

    @classmethod
    def create(cls, fmt, *args, **kwargs):
        if fmt == 'HDF5':
            return _Sequence_HDF5(*args, **kwargs)


    def export(self, filenames, fmt='TIFF16', fill_gaps=True,
               scale_values=False, channel_names=None):
        """Save frames to the indicated filenames.

        This function stores a multipage tiff file for each channel.
        """
        for filename in filenames:
            if dirname(filename):
                sima.misc.mkdir_p(dirname(filename))

        if 'TIFF' in fmt:
            output_files = [TiffFileWriter(fn) for fn in filenames]
        elif fmt == 'HDF5':
            if not h5py_available:
                raise ImportError('h5py >= 2.3.1 required')
            f = h5py.File(filenames, 'w')
            output_array = np.empty((self.num_frames, 1,
                                     self.num_rows,
                                     self.num_columns,
                                     self.num_channels), dtype='uint16')
        else:
            raise('Not Implemented')

        for f_idx, frame in enumerate(self):
            for ch_idx, channel in enumerate(frame):
                if fmt == 'TIFF16':
                    f = output_files[ch_idx]
                    if scale_values:
                        f.write_page(sima.misc.to16bit(channel))
                    else:
                        f.write_page(channel.astype('uint16'))
                elif fmt == 'TIFF8':
                    f = output_files[ch_idx]
                    if scale_values:
                        f.write_page(sima.misc.to8bit(channel))
                    else:
                        f.write_page(channel.astype('uint8'))
                elif fmt == 'HDF5':
                    output_array[f_idx, 0, :, :, ch_idx] = channel
                else:
                    raise ValueError('Unrecognized output format.')

        if 'TIFF' in fmt:
            for f in output_files:
                f.close()
        elif fmt == 'HDF5':
            f.create_dataset(name='imaging', data=output_array)
            for idx, label in enumerate(['t', 'z', 'y', 'x', 'c']):
                f['imaging'].dims[idx].label = label
            if channel_names is not None:
                f['imaging'].attrs['channel_names'] = np.array(channel_names)
            f.close()


class IndexableSequence(Sequence):
    """Iterable whose underlying structure supports indexing."""
    __metaclass__ = ABCMeta

    def __iter__(self):
        for t in xrange(len(self)):
            yield self._get_frame(t)

    @abstractmethod
    def _get_frame(self, t):
        """Return frame with index t."""
        pass


# class _SequenceMultipageTIFF(_BaseSequence):
#
#     """
#     Iterable for a multi-page TIFF file in which the pages
#     correspond to sequentially acquired image frames.
#
#     Parameters
#     ----------
#     paths : list of str
#         The TIFF filenames, one per channel.
#     clip : tuple of tuple of int, optional
#         The number of rows/columns to clip from each edge
#         in order ((top, bottom), (left, right)).
#
#     Warning
#     -------
#     Moving the TIFF files may make this iterable unusable
#     when the ImagingDataset is reloaded. The TIFF file can
#     only be moved if the ImagingDataset path is also moved
#     such that they retain the same relative position.
#
#     """
#
#     def __init__(self, paths, clip=None):
#         super(MultiPageTIFF, self).__init__(clip)
#         self.path = abspath(path)
#         if not libtiff_available:
#             self.stack = TiffFile(self.path)
#
#     def __len__(self):
#         # TODO: remove this and just use
#         if libtiff_available:
#             tiff = TIFF.open(self.path, 'r')
#             l = sum(1 for _ in tiff.iter_images())
#             tiff.close()
#             return l
#         else:
#             return len(self.stack.pages)
#
#     def __iter__(self):
#         if libtiff_available:
#             tiff = TIFF.open(self.path, 'r')
#             for frame in tiff.iter_images():
#                 yield frame
#         else:
#             for frame in self.stack.pages:
#                 yield frame.asarray(colormapped=False)
#         if libtiff_available:
#             tiff.close()
#
#     def _todict(self):
#         return {'path': self.path, 'clip': self._clip}


class _Sequence_HDF5(IndexableSequence):

    """
    Iterable for an HDF5 file containing imaging data.

    Parameters
    ----------
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

    Warning
    -------
    Moving the HDF5 file may make this iterable unusable
    when the ImagingDataset is reloaded. The HDF5 file can
    only be moved if the ImagingDataset path is also moved
    such that they retain the same relative position.

    """

    def __init__(self, path, dim_order, group=None, key=None):
        if not h5py_available:
            raise ImportError('h5py >= 2.3.1 required')
        self.path = abspath(path)
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
        slices = [slice(None) for _ in range(len(self._dataset.shape))]
        swapper = [None for _ in range(len(self._dataset.shape))]
        if self._Z_DIM > -1:
            swapper[self._Z_DIM] = 0
        swapper[self._Y_DIM] = 1
        swapper[self._X_DIM] = 2
        if self._C_DIM > -1:
            swapper[self._C_DIM] = 3
        swapper = filter(lambda x: x is not None, swapper)
        slices[self._T_DIM] = t
        frame = self._dataset[tuple(slices)]
        for i in range(frame.ndim):
            idx = np.argmin(swapper[i:]) + i
            if idx != i:
                swapper[i], swapper[idx] = swapper[idx], swapper[i]
                frame.swapaxes(i, idx)
        return frame

    def _todict(self):
        return {
            'path': self.path,
            'dim_order': self._dim_order,
            'group': self._group.name,
            'key': self._key,
        }


# class _MotionSequence(Sequence):  # TODO: Use a decorator to make this appear
#                                   # as a sequence instead of inheriting??
#     """Wraps any other sequence to apply motion correction.
#
#     Parameters
#     ----------
#     base : Sequence
#
#     displacements : array
#         The _D displacement of each row in the image cycle.
#         Shape: (num_rows * num_frames, 2).
#
#     This object has the same attributes and methods as the class it wraps."""
#     # TODO: check clipping and output frame size
#     def __init__(self, base, displacements):
#         self._base = base
#         self.displacements = displacements
#
#     def __len__(self):
#         return len(self._base)  # Faster to calculate len without aligning
#
#     def __iter__(self)
#         for frame, displacement in it.izip(self._base, displacements):
#             yield _align(frame, displacement)
#
#     def __getitem__(self, *index)
#         # TODO: make this work for slicing
#         return _align(self._base[index[0]], displacements[index[0]])[index[1:]
#
#     def __getattr__(self, name):
#         return getattr(self._base, name)
#
#     def __dir__(self):
#         """Customize how attributes are reported, e.g. for tab completion"""
#         heritage = dir(super(self.__class__, self)) # inherited attributes
#         return sorted(heritage + self.__class__.__dict__.keys() +
#                       self.__dict__.keys())

class _IndexedIterable(Sequence):

    def __init__(self, base, indices):
        self._base = base
        self._base_len = len(base)
        self._indices = \
            indices if isinstance(indices, tuple) else (indices,)
        self._times = range(self._base_len)[self._indices[0]]
        # TODO: switch to generator/iterator if possible?

    def __iter__(self):
        try:
            for t in self._times:
                yield self._base._get_frame(t)[self._indices[1:]]
        except AttributeError:
            idx = 0
            for t, frame in enumerate(self._base):
                try:
                    whether_yield = t == self._times[idx]
                except IndexError:
                    break
                if whether_yield:
                    yield frame[self._indices[1:]]
                    idx += 1

    def _get_frame_(self, t):
        return self._base._get_frame(self._times[t])[self._indices[1:]]

    def __len__(self):
        return len(range(len(self._base))[self._indices[0]])

    def __getattr__(self, name):  # TODO: switch to get-attribute???
        try:
            getattr(super(_IndexedIterable, self), name)
        except AttributeError as err:  # TODO: more specific check
            if err.args[0] == "'super' object has no attribute '_" + name + "'":
                return getattr(self._base, name)
            else:
                raise err

    # def __dir__(self):
    #     """Customize how attributes are reported, e.g. for tab completion.

    #     This may not be necessary if we inherit an abstract class"""
    #     heritage = dir(super(self.__class__, self)) # inherited attributes
    #     return sorted(heritage + self.__class__.__dict__.keys() +
    #                   self.__dict__.keys())
