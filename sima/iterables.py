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

  >>> from sima import ImagingDataset
  >>> from numpy import ones
  >>> frames = ones((100, 128, 128))
  >>> ImagingDataset([[frames]], None)
  <ImagingDataset: num_channels=1, num_cycles=1, frame_size=128x128, num_frames=100>

* lists of numpy arrays of shape (num_rows, num_columns)

  >>> from sima import ImagingDataset
  >>> from numpy import ones
  >>> frames = [ones((128, 128)) for _ in range(100)]
  >>> ImagingDataset([[frames]], None)
  <ImagingDataset: num_channels=1, num_cycles=1, frame_size=128x128, num_frames=100>

For convenience, we have created iterable objects that can be used with
common data formats.
"""

from os.path import abspath
import warnings

try:
    from libtiff import TIFF
except ImportError:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sima.misc.tifffile import TiffFile
    LIBTIFF = False
else:
    LIBTIFF = True


class MultiPageTIFF(object):
    """
    Iterable for a multi-page TIFF file in which the pages
    correspond to sequentially acquired image frames.

    Parameters
    ----------
    path : str
        The TIFF filename.
    clip : tuple of tuple of int, optional
        The number of rows/columns to clip from each edge
        in order ((top, bottom), (left, right)).

    Warning
    -------
    Moving the TIFF files may make this iterable unusable
    when the ImagingDataset is reloaded. The TIFF file can
    only be moved if the ImagingDataset path is also moved
    such that they retain the same relative position.

    """
    def __init__(self, path, clip=None):
        self.path = abspath(path)
        self.clip = clip
        if not LIBTIFF:
            self.stack = TiffFile(self.path)

    @property
    def num_rows(self):
        return iter(self).next().shape[0]

    @property
    def num_columns(self):
        return iter(self).next().shape[1]

    @property
    def num_frames(self):
        return len(self)

    def __len__(self):
        if LIBTIFF:
            tiff = TIFF.open(self.path, 'r')
            l = sum(1 for _ in tiff.iter_images())
            tiff.close()
            return l
        else:
            return len(self.stack.pages)

    def __iter__(self):

        # Set up the clipping of frames
        if self.clip is None:
            s = (slice(None), slice(None))
        else:
            s = tuple(slice(*[None if x is 0 else x for x in dim])
                      for dim in self.clip)

        if LIBTIFF:
            tiff = TIFF.open(self.path, 'r')
            for frame in tiff.iter_images():
                yield frame[s]
        else:
            for frame in self.stack.pages:
                yield frame.asarray(colormapped=False)[s]
        if LIBTIFF:
            tiff.close()

    def _todict(self):
        return {'path': self.path, 'clip': self.clip}
