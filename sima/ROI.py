"""ROI and ROIList classes for storing and manipulating
regions of interests (ROIs).

ROI.ROI objects allow for storing of ROIs as either as a boolean mask of
included pixels, or as multiple polygons. Masks need not be continuous and
an ROI can be defined by multiple non-adjacent polygons.
In addition, each ROI can be assigned any number or 'tags' used to define
features of the ROIs, as well as a 'group' which is used for clustering or
aligning ROIs across ROILists.

ROI.ROIList object are a list-like container for storing multiple ROIs and
includes methods for saving, sorting, and sub-grouping.

"""

from scipy.sparse import lil_matrix, issparse
import numpy as np
import cPickle as pickle
from itertools import product
from datetime import datetime

from shapely.geometry import MultiPolygon, Polygon, Point
from skimage.measure import find_contours

import sima.misc
import sima.misc.imagej


class NonBooleanMask(Exception):
    pass


class ROI(object):
    """Structure used to store ROIs

    Parameters
    ----------
    mask : array, optional
        A boolean mask in which all non-zero values define the region of
        interest.
    polygons: array_like, optional
        Either an Nx2 np.array (single polygon), a list of array_like objects
        (multiple polygons), or a list of shapely Polygon class instances
    label : str, optional
        A label associated with the ROI for reference
    tags : list of str, optional
        A list of tags associated with the ROI.
    id : str, optional
        A unique identifier for the ROI. By default, the ROI will not have
        a unique identifier.
    im_shape: tuple, optional
        The shape of the image on which the ROI is drawn.  If initialized with
        a mask, should be None, since im_shape will default to shape of the
        mask.

    Raises
    ------
    NonBooleanMask
        Raised when you try to get a polygon representation of a non-boolean
        mask.

    See Also
    --------
    sima.ROI.ROIList

    Notes
    -----
    ROI class instance must be initialized with either a mask or polygons (not
    both).  If initialized with a polygon, im_shape must be defined before the
    ROI can be converted to a mask.

    By convention polygon points are assumed to designate the top-left corner
    of a pixel (see example).

    Examples
    --------
    >>> from sima.ROI import ROI
    >>> roi = ROI(polygons=[[0, 0], [0, 1], [1, 1], [1, 0]], im_shape=(2, 2))
    >>> roi.coords
    [array([[ 0.,  0.],
           [ 0.,  1.],
           [ 1.,  1.],
           [ 1.,  0.],
           [ 0.,  0.]])]
    >>> roi.mask.todense()
    matrix([[ True, False],
            [False, False]], dtype=bool)

    Attributes
    ----------
    id : string
        The unique identifier for the ROI.
    tags : set of str
        The set of tags associated with the ROI.
    label : string
        A label associated with the ROI.
    mask : array
        A mask defining the region of interest.
    polygons : MultiPolygon
        A MultiPolygon representation of the ROI.
    coords : list of arrays
        Coordinates of the polygons as a list of Nx2 arrays
    im_shape : tuple
        The shape of the image associated with the ROI. Determines the shape
        of the mask.

    """
    def __init__(self, mask=None, polygons=None, label=None, tags=None,
                 id=None, im_shape=None):

        if (mask is None) == (polygons is None):
            raise TypeError('ROI: ROI must be initialized with either a mask \
                             or a polygon, not both and not neither')

        self.im_shape = im_shape

        if mask is not None:
            self.mask = mask
        else:
            self._mask = None

        if polygons is not None:
            self.polygons = polygons
        else:
            self._polys = None

        self.id = id
        self.tags = tags
        self.label = label

    def __str__(self):
        return '<ROI: label={label}>'.format(label=self.label)

    def __repr__(self):
        return '<ROI: ' + \
            'label={label}, id={id}, type={type}, im_shape={im_shape}'.format(
                label=self.label,
                id=self.id,
                type='mask' if self._mask is not None else 'poly',
                im_shape=self.im_shape)

    def todict(self):
        """Returns the data in the ROI as a dictionary.

        ROI(**roi.todict()) will return a new ROI equivalent to the
        original roi

        """
        polygons = None if self._polys is None else self.coords
        return {'mask': self._mask, 'polygons': polygons, 'id': self._id,
                'label': self._label, 'tags': self._tags,
                'im_shape': self._im_shape}

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, i):
        if i is None:
            self._id = None
        else:
            self._id = str(i)

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, t):
        if t is None:
            self._tags = set()
        else:
            self._tags = set(t)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, l):
        if l is None:
            self._label = None
        else:
            self._label = str(l)

    @property
    def polygons(self):
        if self._polys is not None:
            return self._polys
        if not np.all((np.array(self._mask.todense() == 0) |
                       np.array(self._mask.todense() == 1))):
            raise NonBooleanMask(
                'Unable to convert a non-boolean mask to polygons')
        return mask2poly(self._mask)

    @polygons.setter
    def polygons(self, polygons):
        self._polys = _reformat_polygons(polygons)
        self._mask = None

    @property
    def coords(self):
        coords = []
        for polygon in self.polygons:
            coords.append(np.array(polygon.exterior.coords))
        return coords

    @property
    def mask(self):
        if self._mask is None and self.im_shape is None:
            raise Exception('Polygon ROIs must have an im_shape set')
        if self._mask is not None:
            if self._mask.shape == self.im_shape:
                return self._mask
            else:
                mask = lil_matrix(self.im_shape, dtype=self._mask.dtype)
                values = self._mask.nonzero()
                for row, col in zip(*values):
                    if row < self.im_shape[0] and col < self.im_shape[1]:
                        mask[row, col] = self._mask[row, col]
                return mask
        return poly2mask(polygons=self.polygons,
                         im_size=self.im_shape)

    @mask.setter
    def mask(self, mask):
        if isinstance(mask, lil_matrix):
            self._mask = mask
        else:
            self._mask = lil_matrix(mask)
        self._polys = None

    @property
    def im_shape(self):
        if self._im_shape is not None:
            return self._im_shape
        if self._mask is not None:
            return self._mask.shape
        return None

    @im_shape.setter
    def im_shape(self, shape):
        if shape is None:
            self._im_shape = None
        else:
            self._im_shape = tuple(shape)


class ROIList(list):
    """A list-like container for storing multiple ROIs.

    This class retains all the functionality inherited from Python's built-in
    `list <https://docs.python.org/2/library/functions.html#list>`_ class.

    Parameters
    ----------
    rois : list of sima.ROI.ROI
        The ROIs in the set.
    timestamp : , optional
        The time at which the ROIList was created.
        Defaults to the current time.

    See also
    --------
    sima.ROI.ROI

    Attributes
    ----------
    timestamp : string
        The timestamp for when the ROIList was created.

    """
    def __init__(self, rois, timestamp=None):
        def convert(roi):
            if isinstance(roi, dict):
                return ROI(**roi)
            else:
                return roi
        list.__init__(self, [convert(roi) for roi in rois])
        self.timestamp = timestamp

    @classmethod
    def load(cls, path, label=None, fmt='pkl'):
        """Initialize an ROIList from either a saved pickle file or an
        Imagej ROI zip file.

        Parameters
        ----------
        path : string
            Path to either a pickled ROIList or an ImageJ ROI zip file.
        fmt : {'pkl', 'ImageJ'}
            The file format being imported.
        label : str, optional
            The label for selecting the ROIList if multiple ROILists
            have been saved in the same file. By default, the most
            recently saved ROIList will be selected.

        Returns
        -------
        sima.ROI.ROIList
            Returns an ROIList loaded from the passed in path.

        """
        if fmt == 'pkl':
            with open(path, 'rb') as f:
                roi_sets = pickle.load(f)
            if label is None:
                label = sima.misc.most_recent_key(roi_sets)
            try:
                rois = roi_sets[label]
            except KeyError:
                raise Exception(
                    'No ROIs with were saved with the given label.')
            return cls(**rois)
        elif fmt == 'ImageJ':
            return cls(rois=sima.misc.imagej.read_imagej_roi_zip(path))
        else:
            raise ValueError('Unrecognized file format.')

    def transform(self, transform, copy_properties=True):
        """Apply a 2x3 affine transformation to the ROIs

        Parameters
        ----------
        transform : 2x3 Numpy array
            The affine transformation to be applied to the ROIs.

        copy_properties : bool, optional
            Copy the label, id, tags, and im_shape properties from the source
            ROIs to the transformed ROIs

        Returns
        -------
        sima.ROI.ROIList
            Returns an ROIList consisting of the transformed ROI objects.
        """
        transformed_rois = []
        for roi in self:
            transformed_polygons = []
            for coords in roi.coords:
                transformed_coords = [np.dot(transform, np.hstack([vert, 1]))
                                      for vert in coords]
                transformed_polygons.append(transformed_coords)
            transformed_roi = ROI(polygons=transformed_polygons)
            if copy_properties:
                transformed_roi.label = roi.label
                transformed_roi.id = roi.id
                transformed_roi.tags = roi.tags
                transformed_roi.im_shape = roi.im_shape
            transformed_rois.append(transformed_roi)
        return ROIList(rois=transformed_rois)

    def __str__(self):
        return ("<ROI set: nROIs={nROIs}, timestamp={timestamp}>").format(
            nROIs=len(self), timestamp=self.timestamp)

    def __repr__(self):
        return super(ROIList, self).__repr__().replace(
            '\n', '\n    ')

    def save(self, path, label=None):
        """Save an ROI set to a file. The file can contain multiple
        ROIList objects with different associated labels. If the file
        already exists, the ROIList will be added without deleting the
        others.

        Parameters
        ----------
        path : str
            The name of the pkl file to which the ROIList will be saved.
        label : str, optional
            The label associated with the ROIList. Defaults to using
            the timestamp as a label.
        """

        time_fmt = '%Y-%m-%d-%Hh%Mm%Ss'
        timestamp = datetime.strftime(datetime.now(), time_fmt)

        rois = [roi.todict() for roi in self]
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except IOError:
            data = {}
        if label is None:
            label = timestamp
        data[label] = {'rois': rois, 'timestamp': timestamp}
        with open(path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def subset(self, tags=None, neg_tags=None):
        """Filter the ROIs in the set based on the ROI tags.

        Parameters
        ----------
        tags : list of strings, optional
            Only ROIs that contain all of the tags will be included.
        neg_tags : list of strings, optional
            Only ROIs that contain none of the neg_tags will be included.

        Returns
        -------
        sima.ROI.ROIList
            New ROIList with all filtered ROIs.

        """
        if tags is None:
            tags = []
        if neg_tags is None:
            neg_tags = []
        rois = [r for r in self if
                all(t in r.tags for t in tags) and
                not any(t in r.tags for t in neg_tags)]
        return ROIList(rois)


def poly2mask(polygons, im_size):
    """Converts polygons to a sparse binary mask.

    >>> from sima.ROI import poly2mask
    >>> poly1 = [[0,0], [0,1], [1,1], [1,0]]
    >>> poly2 = [[0,1], [0,2], [2,2], [2,1]]
    >>> mask = poly2mask([poly1, poly2], (3, 3))
    >>> mask.todense()
    matrix([[ True, False, False],
            [ True,  True, False],
            [False, False, False]], dtype=bool)

    Parameters
    ----------
    polygons : sequence of coordinates or sequence of Polygons
        A sequence of polygons where each is either a sequence of (x,y)
        coordinate pairs, an Nx2 numpy array, or a Polygon object.
    im_size : tuple
        Final size of the resulting mask

    Output
    ------
    mask
        A sparse binary mask of the points contained within the polygons.

    """

    polygons = _reformat_polygons(polygons)
    mask = np.zeros(im_size, dtype=bool)
    for poly in polygons:
        x_min, y_min, x_max, y_max = poly.bounds

        # Shift all points by 0.5 to move coordinates to corner of pixel
        shifted_poly = Polygon(np.array(poly.exterior.coords) - 0.5)

        points = [Point(x, y) for x, y in
                  product(np.arange(int(x_min), np.ceil(x_max)),
                          np.arange(int(y_min), np.ceil(y_max)))]
        points_in_poly = filter(shifted_poly.contains, points)
        for point in points_in_poly:
            x = int(point.x)
            y = int(point.y)
            if 0 <= y < im_size[0] and 0 <= x < im_size[1]:
                mask[y, x] = True
    return lil_matrix(mask)


def mask2poly(mask, threshold=0.5):
    """Takes a mask and returns a MultiPolygon

    Parameters
    ----------
    mask : array
        Sparse or dense array to identify polygon contours within.
    threshold : float, optional
        Threshold value used to separate points in and out of resulting
        polygons. 0.5 will partition a boolean mask, for an arbitrary value
        binary mask choose the midpoint of the low and high values.

    Output
    ------
    MultiPolygon
        Returns a MultiPolygon of all masked regions.

    """

    if issparse(mask):
        mask = np.array(mask.astype('byte').todense())

    if (mask != 0).sum() == 0:
        raise Exception('Empty mask cannot be converted to polygons.')

    # Add an empty row and column around the mask to make sure edge masks
    # are correctly determined
    expanded_dims = (mask.shape[0] + 2, mask.shape[1] + 2)
    expanded_mask = np.zeros(expanded_dims, dtype=float)
    expanded_mask[1:mask.shape[0] + 1, 1:mask.shape[1] + 1] = mask

    verts = find_contours(expanded_mask.T, threshold)

    # Subtract off 1 to shift coords back to their real space,
    # but also add 0.5 to move the coordinates back to the corners,
    # so net subtract 0.5 from every coordinate
    verts = [np.subtract(x, 0.5).tolist() for x in verts]

    return _reformat_polygons(verts)


def _reformat_polygons(polygons):
    """Convert polygons to a MulitPolygon

    Accepts one or more sequence of 2-element sequences (sequence of coords) or
    Polygon objects

    Parameters
    ----------
    polygons : sequence of coordinates or sequence of Polygons
        Polygon(s) to be converted to a MulitPolygon

    Returns
    -------
    MulitPolygon

    """
    if isinstance(polygons, MultiPolygon):
        return polygons
    if isinstance(polygons, Polygon):
        polygons = [polygons]
    elif isinstance(polygons[0], Polygon):
        pass
    else:
        # We got some sort of sequence of sequences, ensure it has the
        # correct depth and convert to Polygon objects
        try:
            Polygon(polygons[0])
        except (TypeError, AssertionError):
            polygons = [polygons]
        new_polygons = []
        for poly in polygons:
            # Polygon.simplify with tolerance=0 will return the exact same
            # polygon with co-linear points removed
            new_polygons.append(Polygon(poly).simplify(tolerance=0))
        polygons = new_polygons
    return MultiPolygon(polygons)
