"""Base classes for multiframe imaging data."""
import warnings
import itertools as it
import os
import errno
import csv
from os.path import dirname, join, normpath, normcase, isfile, relpath, \
    abspath
import cPickle as pickle
from h5py import File

import numpy as np

import sima
import sima.segment as segment
import sima.misc
from sima.misc import lazyprop, mkdir_p, most_recent_key, affine_transform
from sima.extract import extract_rois, save_extracted_signals
from sima.ROI import ROIList
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sima.misc.tifffile import imsave, TiffFileWriter
from sima._motion import _align_frame


class ImagingDataset(object):
    """A multiple cycle imaging dataset.

    Imaging data sets can be iterated over to generate cycles, which
    can in turn be iterated over to generate imaging frames.

    Examples
    --------
    >>> import sima # doctest: +ELLIPSIS
    ...
    >>> from sima.misc import example_data
    >>> dataset = sima.ImagingDataset.load(example_data())

    Datasets can be iterated over as follows:
    >>> for cycle in dataset:
    ...     for frame in cycle:
    ...         for channel in frame:
    ...             for row in channel:
    ...                 for column in row:
    ...                         pass

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
        See Notes for details.
    displacements : list of array, optional
    trim_criterion : float, optional
        The required fraction of frames during which a location must
        be within the field of view for it to be included in the
        motion-corrected imaging frames. By default, only locations
        that are always within the field of view are retained. This
        argument only has effect if the ImagingDataset object is
        initialized with displacements.

    Notes
    -----
    Keys for metadata:
        'acquisition period' :\n
        'plane order' :\n
        'plane times' :\n
        'plane heights' :\n

    Attributes
    ----------
    frame_shape : tuple of int
        The shape of the data acquired for each frame, in order
        (num_planes, num_rows, num_columns).
    num_cycles : int
        The number of cycles in the ImagingDataset.
    num_channels : int
        The number of simultaneously recorded channels in the
        ImagingDataset.
    num_columns : int
        The number of columns per image frame.
    num_frames : int
        The total number of image frames in the ImagingDataset.
    num_planes : int
        The number of imaging planes in the ImagingDataset.
    num_rows : int
        The number of rows per image frame.
    ROIs : dict of (str, ROIList)
        The sets of ROIs saved with this ImagingDataset.
    time_averages : list of ndarray
        The time-averaged intensity for each channel.
    invalid_frames : list of list of int
        The indices of the invalid frames in each cycle.

    """
    def __init__(self, iterables, savedir, channel_names=None,
                 metadata=None, displacements=None, trim_criterion=None,
                 invalid_frames=None):

        # Convert savedir into an absolute path ending with .sima
        if savedir is None:
            self.savedir = None
        else:
            self.savedir = abspath(savedir)
            if not self.savedir.endswith('.sima'):
                self.savedir += '.sima'

        self.metadata = {} if metadata is None else metadata

        if iterables is None:
            # Special case used to load an existing ImagingDataset

            if not self.savedir:
                raise Exception('Cannot initialize dataset without iterables '
                                'or a directory.')
            with open(join(savedir, 'dataset.pkl'), 'rb') as f:
                data = pickle.load(f)

            def resolve_paths(d):
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
                        if not all(os.path.samefile(testfile, p)
                                   for p in paths):
                            raise Exception(
                                'Files have been moved. The path '
                                'cannot be unambiguously resolved.'
                            )
                    d['path'] = paths.pop()

            def unpack(iterable):
                try:
                    resolve_paths(iterable)
                    return iterable.pop('__class__')(**iterable)
                except AttributeError:
                    return iterable

            iterables = [[unpack(channel) for channel in cycle]
                         for cycle in data.pop('iterables')]

            self.trim_criterion = data.pop('trim_criterion', None)
            self._channel_names = data.pop('channel_names', None)
            try:
                self._lazy__trim_coords = data.pop('_lazy__trim_coords')
            except KeyError:
                pass
            try:
                invalid_frames = data.pop('_invalid_frames')
            except KeyError:
                pass
            try:
                self.num_frames = data.pop('num_frames')
            except KeyError:
                pass
            save = False
        else:
            save = True
            if self.savedir is not None:
                try:
                    os.makedirs(self.savedir)
                except OSError as exc:
                    if exc.errno == errno.EEXIST and \
                            os.path.isdir(self.savedir):
                        raise Exception(
                            'Cannot overwrite existing ImagingDataset.'
                        )
            # initialize displacements
            if displacements is not None:
                if not all(d.shape[1] == 2 for d in displacements):
                    raise ValueError('Invalid displacements shape.')
                min_displacement = np.amin(
                    [x.min(axis=0) for x in displacements], axis=0).astype(int)
                displacements = [x - min_displacement for x in displacements]
                with open(join(self.savedir, 'displacements.pkl'), 'wb') as f:
                    pickle.dump(displacements, f, pickle.HIGHEST_PROTOCOL)

            self.trim_criterion = trim_criterion
            self._channel_names = channel_names

        # initialize cycles
        self._cycles = self._create_cycles(iterables)
        self.num_cycles = len(self._cycles)
        self.num_channels = self._cycles[0].num_channels
        if not np.all([cycle.num_channels == self.num_channels
                       for cycle in self._cycles]):
            raise ValueError(
                'All cycles must have the same number of channels.')
        self.num_rows = self._cycles[0].num_rows
        if not np.all([cycle.num_rows == self.num_rows
                      for cycle in self._cycles]):
            raise ValueError('All cycles must have images of the same size.')
        self.num_columns = self._cycles[0].num_columns
        if not np.all([cycle.num_columns == self.num_columns
                      for cycle in self._cycles]):
            raise ValueError('All cycles must have images of the same size.')
        if not hasattr(self, 'num_frames'):
            self.num_frames = sum(c.num_frames for c in self)

        if self.channel_names is None:
            self.channel_names = [str(x) for x in range(self.num_channels)]

        if invalid_frames is not None:
            self.invalid_frames = invalid_frames

        if save and self.savedir is not None:
            self._save(self.savedir)

    @lazyprop
    def _max_displacement(self):
        displacements = self._displacements
        if displacements:
            return np.amax([x.max(axis=0) for x in self._displacements],
                           axis=0).astype(int)

    @property
    def _displacements(self):
        if self.savedir is None:
            return None
        try:
            with open(join(self.savedir, 'displacements.pkl'), 'rb') as f:
                displacements = pickle.load(f)
        except IOError:
            return None
        else:
            to_fix = False
            for cycle_idx, cycle in enumerate(displacements):
                if cycle.dtype != np.dtype('int'):
                    displacements[cycle_idx] = cycle.astype('int')
                    to_fix = True
            if to_fix:
                print('Updated old displacements file')
                with open(join(self.savedir, 'displacements.pkl'), 'wb') as f:
                    pickle.dump(displacements, f, pickle.HIGHEST_PROTOCOL)
            return displacements

    @property
    def channel_names(self):
        return self._channel_names

    @channel_names.setter
    def channel_names(self, names):
        self._channel_names = [str(n) for n in names]
        if self.savedir is not None:
            self._save()

    @property
    def time_averages(self):
        if self.savedir is not None:
            try:
                with open(join(self.savedir, 'time_averages.pkl'),
                          'rb') as f:
                    return pickle.load(f)
            except IOError:
                pass
        shape = (self.num_rows, self.num_columns)
        sums = [np.zeros(shape) for _ in range(self.num_channels)]
        counts = [np.zeros(shape) for _ in range(self.num_channels)]
        for cycle in self:
            for frame in cycle:
                for f_ch, s_ch, c_ch in zip(frame, sums, counts):
                    s_ch[np.isfinite(f_ch)] += f_ch[np.isfinite(f_ch)]
                    c_ch[np.isfinite(f_ch)] += 1
        averages = [s / c for s, c in zip(sums, counts)]
        if self.savedir is not None:
            with open(join(self.savedir, 'time_averages.pkl'), 'wb') as f:
                pickle.dump(averages, f, pickle.HIGHEST_PROTOCOL)
        return averages

    @property
    def ROIs(self):
        try:
            with open(join(self.savedir, 'rois.pkl'), 'rb') as f:
                return {label: ROIList(**v)
                        for label, v in pickle.load(f).iteritems()}
        except (IOError, pickle.UnpicklingError):
            return {}

    @classmethod
    def load(cls, path):
        """Load a saved ImagingDataset object."""
        return cls(None, path)

    def _todict(self, savedir):
        """Returns the dataset as a dictionary, useful for saving"""
        def set_paths(d):
            try:
                p = d.pop('path')
            except KeyError:
                pass
            else:
                d['_abspath'] = abspath(p)
                d['_relpath'] = relpath(p, savedir)

        def pack(iterable):
            try:
                d = iterable._todict()
            #TODO: is this necessary?
            except AttributeError:
                return iterable
            else:
                set_paths(d)
                d['__class__'] = iterable.__class__
                return d

        iterables = [[pack(channel) for channel in cycle.channels]
                     for cycle in self]
        d = {
            'iterables': iterables,
            'channel_names': self.channel_names,
            'trim_criterion': self.trim_criterion,
            'num_frames': self.num_frames,
            '__version__' : sima.__version__
        }
        if hasattr(self, '_lazy__trim_coords'):
            d['_lazy__trim_coords'] = self._trim_coords
        if hasattr(self, '_invalid_frames'):
            d['_invalid_frames'] = self._invalid_frames
        return d

    def add_ROIs(self, ROIs, label=None):
        """Add a set of ROIs to the ImagingDataset.

        Parameters
        ----------
        ROIs : ROIList
            The regions of interest (ROIs) to be added.
        label : str, optional
            The label associated with the ROIList. Defaults to using
            the timestamp as a label.

        Examples
        --------
        Import an ROIList from a zip file containing ROIs created
        with NIH ImageJ.

        >>> from sima.ROI import ROIList
        >>> from sima.misc import example_imagej_rois,example_data
        >>> from sima.imaging import ImagingDataset
        >>> dataset = ImagingDataset.load(example_data())
        >>> rois = ROIList.load(example_imagej_rois(), fmt='ImageJ')
        >>> dataset.add_ROIs(rois, 'from_ImageJ')

        """
        if self.savedir is None:
            raise Exception('Cannot add ROIs unless savedir is set.')
        ROIs.save(join(self.savedir, 'rois.pkl'), label)

    def import_transformed_ROIs(self, source_dataset, source_channel=0,
                                target_channel=0, source_label=None,
                                target_label=None, copy_properties=True):
        """Calculate an affine transformation that maps the source ImagingDataset
        onto this ImagingDataset, tranform the source ROIs by this mapping, and
        then import them into this ImagingDataset.

        Parameters
        ----------
        source_dataset : ImagingDataset
            The ImagingDataset object from which ROIs are to be imported.  This
            dataset must be roughly of the same field-of-view as self in order
            to calculate an affine transformation.

        source_channel : string or int, optional
            The channel of the source image from which to calculate an affine
            transformation, either an integer index or a string in
            source_dataset.channel_names.

        target_channel : string or int, optional
            The channel of the target image from which to calculate an affine
            transformation, either an integer index or a string in
            self.channel_names.

        source_label : string, optional
            The label of the ROIList to transform

        target_label : string, optional
            The label to assign the transformed ROIList

        copy_properties : bool, optional
            Copy the label, id, tags, and im_shape properties from the source
            ROIs to the transformed ROIs
        """

        source_channel = source_dataset._resolve_channel(source_channel)
        target_channel = self._resolve_channel(target_channel)
        source = source_dataset.time_averages[source_channel]
        target = self.time_averages[target_channel]

        transform = affine_transform(source, target)

        src_rois = source_dataset.ROIs
        if source_label is None:
            source_label = most_recent_key(src_rois)
        src_rois = src_rois[source_label]
        transformed_ROIs = src_rois.transform(
            transform, copy_properties=copy_properties)
        self.add_ROIs(transformed_ROIs, label=target_label)

    def delete_ROIs(self, label):
        """Delete an ROI set from the rois.pkl file

        Removes the file if no sets left.

        Parameters
        ----------
        label : string
            The label of the ROI Set to remove from the rois.pkl file

        """

        try:
            with open(join(self.savedir, 'rois.pkl'), 'rb') as f:
                rois = pickle.load(f)
        except IOError:
            return

        try:
            rois.pop(label)
        except KeyError:
            pass
        else:
            if len(rois):
                with open(join(self.savedir, 'rois.pkl'), 'wb') as f:
                    pickle.dump(rois, f, pickle.HIGHEST_PROTOCOL)
            else:
                os.remove(join(self.savedir, 'rois.pkl'))

    def export_averages(self, filenames, fmt='TIFF16', scale_values=True):
        """Save TIFF files with the time average of each channel.

        For datasets with multiple frames, the resulting TIFF files
        have multiple pages.

        Parameters
        ----------
        filenames : list of str
            The (tif) filenames for saving the time averages.
        fmt : {'TIFF8', 'TIFF16'}, optional
            The format of the output files. Defaults to 16-bit TIFF.
        scale_values : bool, optional
            Whether to scale the values to use the full range of the
            output format. Defaults to False.
        """
        for filename, im in it.izip(filenames, self.time_averages):
            if dirname(filename):
                mkdir_p(dirname(filename))
            if fmt is 'TIFF8':
                if scale_values:
                    out = sima.misc.to8bit(im)
                else:
                    out = im.astype('uint8')
            elif fmt is 'TIFF16':
                if scale_values:
                    out = sima.misc.to16bit(im)
                else:
                    out = im.astype('uint16')
            else:
                raise ValueError('Unrecognized format.')
            imsave(filename, out)

    def export_frames(self, filenames, fmt='TIFF16', fill_gaps=True,
                      scale_values=False):
        """Save a multi-page TIFF files of the motion-corrected time series.

        # TODO: HDF5, multiple Z planes
        One TIFF file is created for each cycle and channel.
        The TIFF files have the same name as the uncorrected files, but should
        be saved in a different directory.

        Parameters
        ----------
        filenames : list of list of string or list of string
            Path to the locations where the output files will be saved.
            If fmt is TIFF, filenames[i][j] is the path to the file
            for the jth channel of the ith cycle.
            If fmt is 'HDF5', filenames[i] is the path to the file for the
            ith cycle
        fmt : {'TIFF8', 'TIFF16', 'HDF5'}, optional
            The format of the output files. Defaults to 16-bit TIFF.
        fill_gaps : bool, optional
            Whether to fill in unobserved rows with data from adjacent frames.
            Defaults to True.
        scale_values : bool, optional
            Whether to scale the values to use the full range of the
            output format. Defaults to False.
        """
        for cycle, fns in it.izip(self, filenames):
            cycle._export_frames(fns, fmt, fill_gaps, scale_values,
                                 self.channel_names)

    def export_signals(self, path, fmt='csv', channel=0, signals_label=None):
        """Export extrated signals to a file.

        Parameters
        ----------
        path : str
            The name of the file that will store the exported data.
        fmt : {'csv'}, optional
            The export format. Currently, only 'csv' export is available.
        channel : string or int
            The channel from which to export signals, either an integer
            index or a string in self.channel_names.
        signals_label : str, optional
            The label of the extracted signal set to use. By default,
            the most recently extracted signals are used.
        """
        with open(path, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')

            signals = self.signals(channel)
            if signals_label is None:
                signals_label = most_recent_key(signals)
            rois = signals[signals_label]['rois']

            writer.writerow(['cycle', 'frame'] + [r['id'] for r in rois])
            writer.writerow(['', 'label'] + [r['label'] for r in rois])
            writer.writerow(
                ['', 'tags'] +
                [''.join(t + ',' for t in sorted(r['tags']))[:-1]
                 for r in rois]
            )
            for cycle_idx, cycle in enumerate(signals[signals_label]['raw']):
                for frame_idx, frame in enumerate(cycle.T):
                    writer.writerow([cycle_idx, frame_idx] + frame.tolist())

    def extract(self, rois=None, signal_channel=0, label=None,
                remove_overlap=True, n_processes=None, demix_channel=None):
        """Extracts imaging data from the current dataset using the
        supplied ROIs file.

        Parameters
        ----------
        rois : sima.ROI.ROIList, optional
            ROIList of rois to extract
        signal_channel : string or int, optional
            Channel containing the signal to be extracted, either an integer
            index or a name in self.channel_names
        label : string or None, optional
            Text label to describe this extraction, if None defaults to a
            timestamp.
        remove_overlap : bool, optional
            If True, remove any pixels that overlap between masks.
        n_processes : int, optional
            Number of processes to farm out the extraction across. Should be
            at least 1 and at most one less then the number of CPUs in the
            computer. If None, uses half the CPUs.
        demix_channel : string or int, optional
            Channel to demix from the signal channel, either an integer index
            or a name in self.channel_names If None, do not demix signals.

        Return
        ------
        dict of arrays
            Keys: raw, demixed_raw, mean_frame, overlap, signal_channel, rois,
            timestamp

        See also
        --------
        sima.ROI.ROIList

        """

        signal_channel = self._resolve_channel(signal_channel)
        demix_channel = self._resolve_channel(demix_channel)

        if rois is None:
            rois = self.ROIs[most_recent_key(self.ROIs)]
        if rois is None:
            raise Exception('Cannot extract dataset with no ROIs.')
        if self.savedir:
            return save_extracted_signals(
                self, rois, self.savedir, label, signal_channel=signal_channel,
                remove_overlap=remove_overlap, n_processes=n_processes,
                demix_channel=demix_channel
            )
        else:
            return extract_rois(self, rois, signal_channel, remove_overlap,
                                n_processes, demix_channel)

    def _save(self, savedir=None):
        """Save the ImagingDataset to a file."""
        if savedir is None:
            savedir = self.savedir
        with open(join(savedir, 'dataset.pkl'), 'wb') as f:
            pickle.dump(self._todict(savedir), f, pickle.HIGHEST_PROTOCOL)

    @property
    def invalid_frames(self):
        try:
            return self._invalid_frames
        except AttributeError:
            return [[] for _ in self]

    @invalid_frames.setter
    def invalid_frames(self, x):
        if x is None:
            x = [[] for _ in self]
        if not len(x) == self.num_cycles:
            raise ValueError("Input must be a list with one entry per cycle.")
        if not all(all(isinstance(y, int) for y in z) for z in x):
            raise TypeError(" input must be a list of lists of integers.")
        self._invalid_frames = x
        for cycle, invalids in it.izip(self, x):
            cycle._invalid_frames = x
        if self.savedir is not None:
            self._save()

    def segment(self, method='normcut', label=None, planes=None, **kwargs):
        """Segment an ImagingDataset to generate ROIs.

        Parameters
        ----------
        method : {normcut, ca1pc}, optional
            The method for segmentation. Defaults to normcut.
        label : str, optional
            Label to be associated with the segmented set of ROIs.
        planes : list of int
            List of the planes that are to be segmented.
        kwargs : dict
            Additional keyword arguments are passed to the function
            implementing the selected segmentation method.

        Returns
        -------
        ROIs : sima.ROI.ROIList
            The segmented regions of interest.
        """
        if kwargs.has_key('channel'):
            kwargs['channel'] = self._resolve_channel(kwargs['channel'])
        if method is 'normcut':
            rois = segment.normcut(self, **kwargs)
        elif method is 'ca1pc':
            rois = segment.ca1pc(self, **kwargs)
        else:
            raise ValueError('Unrecognized segmentation method.')
        if self.savedir is not None:
            rois.save(join(self.savedir, 'rois.pkl'), label)
        return rois

    def signals(self, channel=0):
        """Return a dictionary of extracted signals

        Parameters
        ----------
        channel : string or int
            The channel to load signals for, either an integer index or a
            string in self.channel_names

        """
        channel = self._resolve_channel(channel)
        try:
            with open(join(
                    self.savedir, 'signals_{}.pkl'.format(channel))) as f:
                return pickle.load(f)
        except (IOError, pickle.UnpicklingError):
            return {}

    @lazyprop
    def _untrimmed_frame_size(self):
        """Calculate the size of motion corrected frames before trimming.

        The size of these frames is given by the size of the uncorrected
        frames plus the extent of the displacements.

        """
        return [
            x + y for x, y in zip((self._raw_num_rows, self._raw_num_columns),
                                  self._max_displacement)
        ]

    @lazyprop
    def _trim_coords(self):
        """The coordinates used to trim the corrected imaging data."""
        if self.trim_criterion is None:
            trim_coords = [
                list(self._max_displacement),
                [self._raw_num_rows, self._raw_num_columns]
            ]
        elif isinstance(self.trim_criterion, (float, int)):
            obs_counts = _observation_counts(
                it.chain(*self._displacements),
                (self._raw_num_rows, self._raw_num_columns),
                self._untrimmed_frame_size
            )
            num_frames = sum(len(x) for x in self._displacements
                             ) / self._raw_num_rows
            occupancy = obs_counts.astype(float) / num_frames
            row_occupancy = \
                occupancy.sum(axis=1) / self._raw_num_columns \
                > self.trim_criterion
            row_min = np.nonzero(row_occupancy)[0].min()
            row_max = np.nonzero(row_occupancy)[0].max()
            col_occupancy = occupancy.sum(axis=0) / self._raw_num_rows \
                > self.trim_criterion
            col_min = np.nonzero(col_occupancy)[0].min()
            col_max = np.nonzero(col_occupancy)[0].max()
            trim_coords = [[row_min, col_min], [row_max, col_max]]
        else:
            raise TypeError('Invalid type for trim_criterion')
        return trim_coords

    def __str__(self):
        return '<ImagingDataset>'

    def __repr__(self):
        return ('<ImagingDataset: ' +
                'num_channels={n_channels}, num_cycles={n_cycles}, ' +
                'frame_size={rows}x{cols}, num_frames={frames}>').format(
            n_channels=self.num_channels,
            n_cycles=self.num_cycles,
            rows=self.num_rows,
            cols=self.num_columns,
            frames=self.num_frames)

    def _create_cycles(self, iterables):
        """Create _ImagingCycle object based on the filenames."""
        if self._displacements is None:
            return [_ImagingCycle(iterable) for iterable in iterables]
        else:
            self._raw_num_rows, self._raw_num_columns = next(
                iter(iterables[0][0])).shape
            trim_coords = self._trim_coords
            return [_CorrectedCycle(iterable, d, self._untrimmed_frame_size,
                                    trim_coords)
                    for iterable, d in it.izip(iterables, self._displacements)]

    def __iter__(self):
        return self._cycles.__iter__()

    def _resolve_channel(self, chan):
        """Return the index corresponding to the channel."""
        return sima.misc.resolve_channels(chan, self.channel_names)


class _ImagingCycle(object):
    """Object for the imaging data from a continuous time interval.

    Parameters
    ----------
    channels : list of iterable
        A list of iterable objects, one for each channel.
        Each iterable should yield 2D numpy arrays.

    Attributes
    ----------
    num_frames, num_channels, num_rows, num_columns : int
    """
    def __init__(self, channels):
        self.channels = channels
        self._invalid_frames = []

    @lazyprop
    def num_frames(self):
        """The number of frames per channel."""
        try:
            return len(self.channels[0])
        except TypeError:
            return sum(1 for _ in self.channels[0])

    @property
    def num_channels(self):
        """The number of simultaneously imaged channels."""
        return len(self.channels)

    @lazyprop
    def num_planes(self):
        """The number of planes in the dataset."""
        return  # TODO

    @lazyprop
    def num_rows(self):
        """The number of rows per frame."""
        return next(iter(self))[0].shape[0]

    @lazyprop
    def num_columns(self):
        """The number of columns per frame."""
        return next(iter(self))[0].shape[1]

    def __iter__(self):
        """Iterate over the image frames.

        Each frame is returned as a list of arrays (one per channel)."""
        for frame_idx, frame in enumerate(it.izip(*self.channels)):
            if frame_idx in self._invalid_frames:
                yield [np.nan * chan for chan in frame]
            else:
                yield [chan for chan in frame]

    def _export_frames(self, filenames, fmt='TIFF16', fill_gaps=True,
                       scale_values=False, channel_names=None):
        """Save frames to the indicated filenames.

        This function stores a multipage tiff file for each channel.
        """
        for filename in filenames:
            if dirname(filename):
                mkdir_p(dirname(filename))

        if 'TIFF' in fmt:
            output_files = [TiffFileWriter(fn) for fn in filenames]
        elif fmt == 'HDF5':
            f = File(filenames, 'w')
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


class _CorrectedCycle(_ImagingCycle):
    """A multiple cycle imaging dataset that has been motion corrected.

    Parameters
    ----------
    iterables : list of iterable
    displacements : array
        The 2D displacement of each row in the image cycle.
        Shape: (num_rows * num_frames, 2).
    untrimmed_frame_size : tuple
        The shape of the aligned images before under-observed pixels
        are trimmed.
    trim_coords : list of list of int
        The top left and bottom right coordinates for trimming
        the returned frames. If set to None, no trimming occurs.

    """
    def __init__(self, iterables, displacements, untrimmed_frame_size,
                 trim_coords):
        self._raw_num_rows, self._raw_num_columns = next(
            iter(iterables[0])).shape
        self._displacements = displacements
        self._untrimmed_frame_size = untrimmed_frame_size
        self._trim_coords = trim_coords
        super(_CorrectedCycle, self).__init__(iterables)

    def observation_counts(self):
        """Return an array with the number of observations of each location."""
        return _observation_counts(
            self._displacements, (self._raw_num_rows, self._raw_num_columns),
            self._untrimmed_frame_size)

    def __iter__(self):
        """Align the imaging data using calculated displacements.

        Yields
        ------
        list of arrays
            The corrected images for each channel are yielded on frame at a
            time.
        """
        for frame_idx, frame in enumerate(
                super(_CorrectedCycle, self).__iter__()):
            out = []
            for channel in frame:
                tmp = channel.astype(float)
                im = _align_frame(tmp, self._displacements[
                    (frame_idx * channel.shape[0]):
                    ((frame_idx + 1) * channel.shape[0])],
                    self._untrimmed_frame_size)
                if self._trim_coords is not None:
                    im = im[self._trim_coords[0][0]:self._trim_coords[1][0],
                            self._trim_coords[0][1]:self._trim_coords[1][1]]
                out.append(im)
            yield out

    def _export_frames(self, filenames, fmt='TIFF16', fill_gaps=True,
                       scale_values=False, channel_names=None):
        """Save a multi-page TIFF files of the motion corrected time series.

        One TIFF file is created for each channel.
        The TIFF files have the same name as the uncorrected files, but should
        be saved in a different directory.

        Parameters
        ----------
        filenames : list of str
            The filenames used for saving each channel.
        fill_gaps : bool, optional
            Whether to fill in unobserved pixels with data from nearby frames.
        """
        if fill_gaps:
            if 'TIFF' in fmt:
                output_files = [TiffFileWriter(fn) for fn in filenames]
            elif fmt == 'HDF5':
                f = File(filenames, 'w')
                output_array = np.empty((self.num_frames, 1,
                                         self.num_rows,
                                         self.num_columns,
                                         self.num_channels), dtype='uint16')
            else:
                raise('Not Implemented')

            save_frames = _fill_gaps(iter(self), iter(self))
            for f_idx, frame in enumerate(save_frames):
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
                    f['imaging'].attrs['channel_names'] = np.array(
                        channel_names)
                f.close()
        else:
            super(_CorrectedCycle, self)._export_frames(
                filenames, fmt=fmt, scale_values=scale_values)


def _observation_counts(displacements, im_size, output_size):
    """Count the number of times that each location was observed."""
    count = np.zeros(output_size, dtype=np.int)
    for row_idx, disp in enumerate(displacements):
        i = row_idx % im_size[0]
        count[i + disp[0], disp[1]:(disp[1] + im_size[1])] += 1
    return count


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
