import os
import itertools as it
import errno
from distutils.version import LooseVersion

import numpy as np
try:
    from bottleneck import nanmax
except ImportError:
    from numpy import nanmax
try:
    import cv2
except ImportError:
    cv2_available = False
else:
    cv2_available = LooseVersion(cv2.__version__) >= LooseVersion('2.4.8')
from skimage import transform as tf


class TransformError(Exception):
    pass


def lazyprop(fn):
    """Like property, but only computes on first call."""
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop


def to8bit(array):
    """Convert an array to 8 bit."""
    return ((255. * array) / nanmax(array)).astype('uint8')


def to16bit(array):
    """Convert an array to 16 bit."""
    return ((65535. * array) / nanmax(array)).astype('uint16')


def mkdir_p(path):
    """Python equivalent of UNIX mkdir -p command."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def most_recent_key(d):
    """Return the key to the most recently timestamped entry"""
    try:
        return max(d.iterkeys(), key=lambda x: d[x]['timestamp'])
    except (TypeError, KeyError):
        return max(d.iterkeys(), key=lambda x: d[x].timestamp)


def auto_choose(d):
    """Automatically choose the most recent value in a timestamped
    dictionary."""
    try:
        return max(d.itervalues(), key=lambda x: x['timestamp'])
    except:
        return max(d.itervalues(), key=lambda x: x.timestamp)


def copy_label_to_id(rois):
    """Copies roi.label to roi.id for all rois in an ROI_Set.
    Used if the roi labels should be used for clustering ROIs across days.
    Modifies the set in place."""
    for roi in rois:
        roi.id = roi.label


def resolve_channels(chan, channel_names, num_channels=None):
    """Return the index corresponding to the channel."""
    if chan is None:
        return None
    if num_channels is None:
        num_channels = len(channel_names)
    if isinstance(chan, int):
        if chan >= num_channels:
            raise ValueError('Invalid channel index.')
        return chan
    else:
        try:
            return channel_names.index(chan)
        except ValueError:
            raise ValueError('No channel exists with the specified name.')


def pairwise(iterable):
    a, b = it.tee(iterable)
    next(b, None)
    return it.izip(a, b)


# was affine_trnasform
def estimate_array_transform(source, target, method='affine'):
    """Calculates an affine transformation from source array to target array

    Parameters
    ----------
    source : array
        The image to transform
    target : array
        The image used as the template for the transformation
    method : string, optional
        Method to use for transform estimation.

    Returns
    -------
    transform : skimage.transform._geometric.GeometricTransform
        An skimage transform object.

    See Also
    --------
    cv2.estimateRigidTransform
    skimage.transform

    """

    if method == 'affine':
        if not cv2_available:
            raise ImportError('OpenCV >= 2.4.8 required')

        slice_ = tuple(slice(0, min(source.shape[i], target.shape[i]))
                       for i in range(2))
        transform = cv2.estimateRigidTransform(
            to8bit(source[slice_]),
            to8bit(target[slice_]), True)

        if transform is None:
            raise TransformError('Cannot calculate affine transformation ' +
                                 'from source to target')
        else:
            # TODO: make sure the order is correct
            transform_matrix = np.vstack((transform, [0, 0, 1]))
            return tf.AffineTransform(matrix=transform_matrix)
    else:
        raise ValueError('Unrecognized transform method: {}'.format(method))


def estimate_coordinate_transform(source, target, method, **method_kwargs):
    """Calculates a transformation from a source list of coordinates to a
    target list of coordinates.

    Parameters
    ----------
    source : Nx2 array
        (x, y) coordinate pairs from source image.
    target : Nx2 array
        (x, y) coordinate pairs from target image. Must be same shape as
        'source'.
    method : string, optional
        Method to use for transform estimation.
    **method_kwargs : optional
        Additional arguments can be passed in specific to the particular
        method. For example, 'order' for a polynomial transform estimation.

    Returns
    -------
    transform : skimage.transform._geometric.GeometricTransform
        An skimage transform object.

    See Also
    --------
    skimage.transform.estimate_transform

    """

    return tf.estimate_transform(method, source, target, **method_kwargs)


def example_tiff():
    return os.path.join(os.path.dirname(__file__), "../tests/data/example.tif")


def example_tiffs():
    return os.path.join(os.path.dirname(__file__),
                        "../tests/data/example-tiffs/*.tif")


def example_data():
    return os.path.join(
        os.path.dirname(__file__), "../tests/data/example.sima")


def example_imagej_rois():
    return os.path.join(
        os.path.dirname(__file__), "../tests/data/imageJ_ROIs.zip")


def example_hdf5():
    return os.path.join(os.path.dirname(__file__), "../tests/data/example.h5")


def example_volume():
    return os.path.join(os.path.dirname(__file__),
                        "../tests/data/example-volume.h5")
