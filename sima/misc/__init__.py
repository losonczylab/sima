import os
import errno
from numpy import nanmax


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
