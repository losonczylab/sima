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
from distutils.version import StrictVersion
from os.path import (abspath, dirname, join, normpath, normcase, isfile,
                     samefile)
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

    """Object containing data from sequentially acquired imaging data.

    Sequences are created with a call to the create method.


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
    def _todict(self):
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

    def toarray(self, squeeze=False):
        """Convert to a numpy array.

        Parameters
        ----------
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
        """Create a Sequence object.

        Parameters
        ----------
        fmt : {'HDF5', 'TIFF'}
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

        """
        if fmt == 'HDF5':
            return _Sequence_HDF5(*args, **kwargs)
        elif fmt == 'TIFF':
            return _Sequence_TIFF_Interleaved(*args, **kwargs)
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
        if fmt not in ['TIFF8', 'TIFF16', 'HDF5']:
            raise ValueError('Unrecognized output format.')

        # Make directories necessary for saving the files.
        try:
            out_dirs = [dirname(filenames)]
        except AttributeError:
            out_dirs = [dirname(f) for f in filenames]
        for f in filter(None, out_dirs):
            sima.misc.mkdir_p(dirname(f))

        if 'TIFF' in fmt:
            output_files = [[TiffFileWriter(fn) for fn in plane]
                            for plane in filenames]
        elif fmt == 'HDF5':
            if not h5py_available:
                raise ImportError('h5py >= 2.3.1 required')
            f = h5py.File(filenames, 'w')
            output_array = np.empty(self.shape, dtype='uint16')  # TODO: change dtype?

        if fill_gaps:
            save_frames = _fill_gaps(iter(self), iter(self))
        else:
            save_frames = iter(self)
        for f_idx, frame in enumerate(save_frames):
            if fmt == 'HDF5':
                output_array[f_idx] = frame
            for ch_idx, channel in enumerate(frame):
                if fmt == 'TIFF16':
                    f = output_files[ch_idx]
                    f.write_page(channel.astype('uint16'))
                elif fmt == 'TIFF8':
                    f = output_files[ch_idx]
                    f.write_page(channel.astype('uint8'))
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
                    np.concatenate([np.expand_dims(next(base_iter), 2)
                                    for _ in range(self._num_channels)],
                                   axis=2), 0)
                 for _ in range(self._num_planes)], 0)

    def _iter_pages(self):
        if libtiff_available:
            tiff = TIFF.open(self._path, 'r')
            for frame in tiff.iter_images():
                yield frame
        else:
            for frame in self.stack.pages:
                yield frame.asarray(colormapped=False)
        if libtiff_available:
            tiff.close()

    def _todict(self):
        return {'path': self._path,
                '__class__': self.__class__,
                'num_planes': self._num_planes,
                'num_channels': self._num_channels,
                'len_': self._len}

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
# TODO: remove this and just use
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


class _Sequence_HDF5(_IndexableSequence):

    """
    Iterable for an HDF5 file containing imaging data.

    See sima.Sequence.create() for details.
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
        slices = tuple(slice(None) for _ in range(self._T_DIM)) + (t,)
        frame = self._dataset[slices]
        swapper = [None for _ in range(frame.ndim)]
        for i, v in [(self._Z_DIM, 0), (self._Y_DIM, 1),
                     (self._X_DIM, 2), (self._C_DIM, 3)]:
            if i >= 0:
                swapper[i] = v
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

    def _todict(self):
        return {
            '__class__': self.__class__,
            'path': abspath(self.path),
            'dim_order': self._dim_order,
            'group': self._group.name,
            'key': self._key,
        }


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

    def _todict(self):
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

    displacements : array
        The _D displacement of each row in the image cycle.
        Shape: (num_rows * num_frames, 2).

    This object has the same attributes and methods as the class it wraps."""
    # TODO: check clipping and output frame size

    def __init__(self, base, displacements, frame_shape):
        super(_MotionCorrectedSequence, self).__init__(base)
        self.displacements = displacements
        self._frame_shape = frame_shape  # (planes, rows, columns)

    def __len__(self):
        return len(self._base)  # Faster to calculate len without aligning

    def _align(self, frame, displacement):
        if displacement.ndim == 3:
            return _align_frame(frame.astype(float), displacement,
                                self._frame_shape)
        elif displacement.ndim == 2:
            out = np.nan * np.ones(self._frame_shape)
            s = frame.shape[1:]
            for p, (plane, disp) in enumerate(it.izip(frame, displacement)):
                out[p, disp[0]:(disp[0]+s[0]), disp[1]:(disp[1]+s[1])] = plane
            return out

    @property
    def shape(self):
        return (len(self),) + self._frame_shape  # Avoid aligning image

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
                self._frame_shape
            )[new_indices]
        if len(indices) == 5:
            chans = indices[5]
            return _MotionCorrectedSequence(
                self._base[:, :, :, :, chans],
                self.displacements[:, :, :, chans],
                self._frame_shape
            )[indices[:5]]
        # TODO: similar for planes ???
        return _IndexedSequence(self, indices)

    def _todict(self):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(),
            'displacements': self.displacements,
            'frame_shape': self._frame_shape,
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
            elif len(outer) == 3:
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

    def _todict(self):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(),
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
            slice(i, i+1) if isinstance(i, int) else i
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

    def _todict(self):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(),
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
    paths = set()
    try:
        relp = d.pop('_relpath')
    except KeyError:
        pass
    else:
        paths.add(normcase(abspath(normpath(join(savedir, relp)))))
    try:
        paths.add(normcase(abspath(normpath(d.pop('_abspath')))))
    except KeyError:
        pass
    if len(paths):
        paths = filter(isfile, paths)
        if len(paths) != 1:
            testfile = paths.pop()
            if not all(samefile(testfile, p) for p in paths):
                raise Exception(
                    'Files have been moved. The path '
                    'cannot be unambiguously resolved.'
                )
        d['path'] = paths.pop()
