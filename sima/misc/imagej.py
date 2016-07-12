from __future__ import division
from __future__ import unicode_literals
from builtins import str
from builtins import range
# Copyright: Luis Pedro Coelho <luis@luispedro.org>, 2012
# License: MIT
#
# https://gist.github.com/luispedro/3437255
#
# Modified 2014 by Jeffrey Zaremba.

import numpy as np
from itertools import product
import zipfile


def read_imagej_roi_zip(filename):
    """Reads an ImageJ ROI zip set and parses each ROI individually

    Parameters
    ----------
    filename : string
        Path to the ImageJ ROis zip file

    Returns
    -------
    roi_list : list
        List of the parsed ImageJ ROIs

    """
    roi_list = []
    with zipfile.ZipFile(filename) as zf:
        for name in zf.namelist():
            roi = read_roi(zf.open(name))
            if roi is None:
                continue
            roi['label'] = str(name).rstrip('.roi')
            roi_list.append(roi)
        return roi_list


def read_roi(roi_obj):
    """Parses an individual ImageJ ROI

    _getX lines with no assignment are bytes within the imageJ roi file
    format that contain additional information that can be extracted if
    needed. In line comments label what they are.

    This is based on:
    http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
    http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiEncoder.java.html

    Parameters
    ----------
    roi_obj : file object
        File object containing a single ImageJ ROI

    Returns
    -------
    ROI
        Returns a parsed ROI object (dictionary)

    Raises
    ------
    IOError
        If there is an error reading the roi file object
    ValueError
        If unable to parse ROI

    """

    sub_pixel_resolution = 128

    # Other options that are not currently used
    # SPLINE_FIT = 1
    # DOUBLE_HEADED = 2
    # OUTLINE = 4
    # OVERLAY_LABELS = 8
    # OVERLAY_NAMES = 16
    # OVERLAY_BACKGROUNDS = 32
    # OVERLAY_BOLD = 64
    # DRAW_OFFSET = 256

    pos = [4]

    def _get8():
        """Read 1 byte from the roi file object"""
        pos[0] += 1
        s = roi_obj.read(1)
        if not s:
            raise IOError('read_imagej_roi: Unexpected EOF')
        return ord(s)

    def _get16():
        """Read 2 bytes from the roi file object"""
        b0 = _get8()
        b1 = _get8()
        return (b0 << 8) | b1

    def _get32():
        """Read 4 bytes from the roi file object"""
        s0 = _get16()
        s1 = _get16()
        return (s0 << 16) | s1

    def _getfloat():
        """Read a float from the roi file object"""
        v = np.int32(_get32())
        return v.view(np.float32)

    def _getcoords(z=0):
        """Get the next coordinate of an roi polygon"""
        if options & sub_pixel_resolution:
            getc = _getfloat
            points = np.empty((n_coordinates, 3), dtype=np.float32)
        else:
            getc = _get16
            points = np.empty((n_coordinates, 3), dtype=np.int16)
        points[:, 0] = [getc() for _ in range(n_coordinates)]
        points[:, 1] = [getc() for _ in range(n_coordinates)]
        points[:, 0] += left
        points[:, 1] += top
        points[:, 2] = z
        return points

    magic = roi_obj.read(4)
    if magic != b'Iout':
        raise IOError('read_imagej_roi: Magic number not found')

    _get16()  # version

    roi_type = _get8()
    # Discard extra second Byte:
    _get8()

    if not (0 <= roi_type < 11):
        raise ValueError('read_imagej_roi: \
                          ROI type {} not supported'.format(roi_type))

    top = _get16()
    left = _get16()
    bottom = _get16()
    right = _get16()
    n_coordinates = _get16()

    x1 = _getfloat()  # x1
    y1 = _getfloat()  # y1
    x2 = _getfloat()  # x2
    y2 = _getfloat()  # y2
    _get16()  # stroke width
    _get32()  # shape roi size
    _get32()  # stroke color
    _get32()  # fill color
    subtype = _get16()
    if subtype != 0:
        raise ValueError('read_imagej_roi: \
                          ROI subtype {} not supported (!= 0)'.format(subtype))
    options = _get16()
    _get8()  # arrow style
    _get8()  # arrow head size
    _get16()  # rectangle arc size
    z = _get32()  # position
    if z > 0:
        z -= 1  # Multi-plane images start indexing at 1 instead of 0
    _get32()  # header 2 offset

    if roi_type == 0:
        # Polygon
        coords = _getcoords(z)
        coords = coords.astype('float')
        return {'polygons': coords}
    elif roi_type == 1:
        # Rectangle
        coords = [[left, top, z], [right, top, z], [right, bottom, z],
                  [left, bottom, z]]
        coords = np.array(coords).astype('float')
        return {'polygons': coords}
    elif roi_type == 2:
        # Oval
        width = right - left
        height = bottom - top

        # 0.5 moves the mid point to the center of the pixel
        x_mid = (right + left) / 2.0 - 0.5
        y_mid = (top + bottom) / 2.0 - 0.5
        mask = np.zeros((z + 1, bottom, right), dtype=bool)
        for y, x in product(np.arange(top, bottom), np.arange(left, right)):
            mask[z, y, x] = ((x - x_mid) ** 2 / (width / 2.0) ** 2 +
                             (y - y_mid) ** 2 / (height / 2.0) ** 2 <= 1)
        return {'mask': mask}
    elif roi_type == 3:
        # Line
        coords = [[x1, y1, z], [x2, y2, z]]
        coords = np.array(coords).astype('float')
        return {'polygons': coords}
    elif roi_type == 7:
        # Freehand
        coords = _getcoords(z)
        coords = coords.astype('float')
        return {'polygons': coords}
    elif roi_type == 10:
        # Single Point
        # R = L+1, B = T+1
        coords = [left, top, z]
        coords = [float(c) for c in coords]
        return coords
    else:
        try:
            coords = _getcoords(z)
            coords = coords.astype('float')
            return {'polygons': coords}
        except:
            raise ValueError(
                'read_imagej_roi: ROI type {} not supported'.format(roi_type))
