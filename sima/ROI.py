"""ROI and ROIList classes for storing and manipulating
regions of interests (ROIs).

ROI.ROI objects allow for storing of ROIs as either as a boolean mask of
included pixels, or as multiple polygons. Masks need not be continuous and
an ROI can be defined by multiple non-adjacent polygons.
In addition, each ROI can be assigned any number or 'tags' used to define
features of the ROIs, as well as an 'id' which is used for clustering or
aligning ROIs across ROILists.

ROI.ROIList objects are a list-like container for storing multiple ROIs and
includes methods for saving, loading, and sub-grouping.

"""
from builtins import filter
from builtins import str
from builtins import zip
from builtins import range
from builtins import object

from scipy.sparse import lil_matrix, issparse
import numpy as np
import pickle as pickle
import itertools as it
from itertools import product
from datetime import datetime
from warnings import warn

from shapely.geometry import MultiPolygon, Polygon, Point
from skimage.measure import find_contours

import sima.misc
import sima.misc.imagej

import os
import glob
import re
import scipy.io

from future import standard_library
standard_library.install_aliases()


class NonBooleanMask(Exception):
    pass


class ROI(object):

    """Structure used to store ROIs

    Parameters
    ----------
    mask : 2D or 3D array or list of 2D arrays or of sparse matrices, optional
        A boolean mask in which all non-zero values define the region of
        interest.  Masks are assumed to follow a (z, y, x) convention,
        corresponding to (plane, row, column) in the image.
    polygons: array_like, optional
        Either an Nx2 or Nx3 np.array (single polygon), a list of array_like
        objects (multiple polygons), a single shapely Polygon class instance,
        or a list of shapely Polygon class instances.  Because polygons are
        stored internally as a shapely MultiPolygon, coordinates in this
        argument should follow an (x, y) or (x, y, z) convention.
    label : str, optional
        A label associated with the ROI for reference
    tags : list of str, optional
        A list of tags associated with the ROI.
    id : str, optional
        A unique identifier for the ROI. By default, the ROI will not have
        a unique identifier.
    im_shape: 2- or 3-tuple, optional
        The shape of the image on which the ROI is drawn.  If initialized with
        a mask, should be None, since im_shape will default to shape of the
        mask.  Elements should correspond to (z, y, x), equivalent to
        (nPlanes, nRows, nCols)

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
    >>> import numpy as np
    >>> from sima.ROI import ROI
    >>> roi = ROI(polygons=[[0, 0], [0, 1], [1, 1], [1, 0]], im_shape=(2, 2))
    >>> roi.coords
    [array([[0., 0., 0.],
           [0., 1., 0.],
           [1., 1., 0.],
           [1., 0., 0.],
           [0., 0., 0.]])]
    >>> np.array(roi)
    array([[[ True, False],
            [False, False]]])

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
        Coordinates of the polygons as a list of Nx3 arrays (x, y, z)
    im_shape : 3-tuple
        The shape of the image associated with the ROI (z, y, x). Determines
        the shape of the mask.
    size : int
        The number of non-zero pixel-weights in the ROI mask.

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

    def todict(self, type=None):
        """Returns the data in the ROI as a dictionary.

        ROI(**roi.todict()) will return a new ROI equivalent to the
        original roi

        Parameters
        ----------
        type : {'mask','polygons'}, optional
            If specified, convert the type of each ROI in the list prior to
            saving
        """

        if type == 'mask':
            self.mask = self.mask
        elif type == 'polygons':
            self.polygons = self.polygons

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
        for m in self._mask:
            if not np.all((np.array(m.todense() == 0) |
                           np.array(m.todense() == 1))):
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
            masks = []
            for z_idx, mask in enumerate(self._mask):
                if mask.shape == self.im_shape[1:]:
                    masks.append(mask)
                else:
                    m = lil_matrix(self.im_shape[1:], dtype=mask.dtype)
                    values = mask.nonzero()
                    for row, col in zip(*values):
                        if row < self.im_shape[1] and col < self.im_shape[2]:
                            m[row, col] = mask[row, col]
                    masks.append(m)
                if z_idx + 1 == self.im_shape[0]:
                    break
            # Note: length of output = self.im_shape[0]
            while len(masks) < self.im_shape[0]:
                masks.append(lil_matrix(self.im_shape[1:], dtype=mask.dtype))
            return masks
        return poly2mask(polygons=self.polygons,
                         im_size=self.im_shape)

    @mask.setter
    def mask(self, mask):
        self._mask = _reformat_mask(mask)
        self._polys = None

    def __array__(self):
        """Obtain a numpy.ndarray representation of the ROI mask.

        Returns
        -------
        mask : numpy.ndarray
            An array representation of the ROI mask.
        """
        return np.array([plane.todense() for plane in self.mask])

    @property
    def size(self):
        return sum(np.count_nonzero(plane.todense()) for plane in self.mask)

    @property
    def im_shape(self):
        if self._im_shape is not None:
            return self._im_shape
        if self._mask is not None:
            z = len(self._mask)
            y = np.amax([x.shape[0] for x in self._mask])
            x = np.amax([x.shape[1] for x in self._mask])
            return (z, y, x)
        return None

    @im_shape.setter
    def im_shape(self, shape):
        if shape is None:
            self._im_shape = None
        else:
            if len(shape) == 3:
                self._im_shape = tuple(shape)
            elif len(shape) == 2:
                self._im_shape = (1,) + tuple(shape)


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
    def load(cls, path, label=None, fmt='pkl', reassign_label=False):
        """Initialize an ROIList from either a saved pickle file or an
        Imagej ROI zip file.

        Parameters
        ----------
        path : string
            Path to either a pickled ROIList, an ImageJ ROI zip file, or the
            path to the direcotry containing the 'IC filter' .mat files for
            inscopix/mosaic data.
        label : str, optional
            The label for selecting the ROIList if multiple ROILists
            have been saved in the same file. By default, the most
            recently saved ROIList will be selected.
        fmt : {'pkl', 'ImageJ', 'inscopix'}
            The file format being imported.
        reassign_label: boolean
            If true, assign ascending integer strings as labels

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
            roi_list = cls(**rois)
        elif fmt == 'ImageJ':
            roi_list = cls(rois=sima.misc.imagej.read_imagej_roi_zip(path))
        elif fmt == 'inscopix':
            dirnames = next(os.walk(path))[1]
            # this naming convetion for ROI masks is used in Mosiac 1.0.0b
            files = [glob.glob(os.path.join(path, dirname, '*IC filter*.mat'))
                     for dirname in dirnames]
            files = filter(lambda f: len(f) > 0, files)[0]

            rois = []
            for filename in files:
                label = re.findall('\d+', filename)[-1]
                data = scipy.io.loadmat(filename)
                # this is the ROI mask index in Mosiac 1.0.0b
                mask = data['Object'][0][0][11]
                rois.append(ROI(mask=mask, id=label, im_shape=mask.shape))
            roi_list = cls(rois=rois)
        else:
            raise ValueError('Unrecognized file format.')
        if reassign_label:
            for idx, roi in zip(it.count(), roi_list):
                roi.label = str(idx)
        return roi_list

    def transform(self, transforms, im_shape=None, copy_properties=True):
        """Apply 2x3 affine transformations to the ROIs

        Parameters
        ----------
        transforms : list of GeometryTransforms or 2x3 Numpy arrays
            The affine transformations to be applied to the ROIs.  Length of
            list should equal the number of planes (im_shape[0]).

        im_shape : 3-element tuple, optional
            The (zyx) shape of the target image. If None, must be set before
            any ROI can be converted to a mask.

        copy_properties : bool, optional
            Copy the label, id, and tags properties from the source
            ROIs to the transformed ROIs.

        Returns
        -------
        sima.ROI.ROIList
            Returns an ROIList consisting of the transformed ROI objects.

        """

        transformed_rois = []
        for roi in self:
            transformed_polygons = []
            for coords in roi.coords:
                z = coords[0][2]  # assuming all coords share a z-coordinate
                if isinstance(transforms[0], np.ndarray):
                    transformed_coords = [np.dot(transforms[int(z)],
                                                 np.hstack([vert[:2], 1]))
                                          for vert in coords]
                else:
                    transformed_coords = transforms[int(z)](coords[:, :2])
                transformed_coords = [np.hstack((coords, z)) for coords in
                                      transformed_coords]
                transformed_polygons.append(transformed_coords)
            transformed_roi = ROI(
                polygons=transformed_polygons, im_shape=im_shape)
            if copy_properties:
                transformed_roi.label = roi.label
                transformed_roi.id = roi.id
                transformed_roi.tags = roi.tags
            transformed_rois.append(transformed_roi)
        return ROIList(rois=transformed_rois)

    def __str__(self):
        return ("<ROI set: nROIs={nROIs}, timestamp={timestamp}>").format(
            nROIs=len(self), timestamp=self.timestamp)

    def __repr__(self):
        return super(ROIList, self).__repr__().replace(
            '\n', '\n    ')

    def save(self, path, label=None, save_type=None):
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
        save_type : {'mask','polygons'}, optional
            If specified, convert the type of each ROI in the list prior to
            saving

        """

        time_fmt = '%Y-%m-%d-%Hh%Mm%Ss'
        timestamp = datetime.strftime(datetime.now(), time_fmt)

        rois = [roi.todict(type=save_type) for roi in self]
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
    >>> mask[0].todense()
    matrix([[ True, False, False],
            [ True,  True, False],
            [False, False, False]])

    Parameters
    ----------
    polygons : sequence of coordinates or sequence of Polygons
        A sequence of polygons where each is either a sequence of (x,y) or
        (x,y,z) coordinate pairs, an Nx2 or Nx3 numpy array, or a Polygon
        object.
    im_size : tuple
        Final size of the resulting mask

    Output
    ------
    mask
        A list of sparse binary masks of the points contained within the
        polygons, one mask per plane

    """
    if len(im_size) == 2:
        im_size = (1,) + im_size

    polygons = _reformat_polygons(polygons)

    mask = np.zeros(im_size, dtype=bool)
    for poly in polygons:
        # assuming all points in the polygon share a z-coordinate
        z = int(np.array(poly.exterior.coords)[0][2])
        if z > im_size[0]:
            warn('Polygon with zero-coordinate {} '.format(z) +
                 'cropped using im_size = {}'.format(im_size))
            continue
        x_min, y_min, x_max, y_max = poly.bounds

        # Shift all points by 0.5 to move coordinates to corner of pixel
        shifted_poly = Polygon(np.array(poly.exterior.coords)[:, :2] - 0.5)

        points = [Point(x, y) for x, y in
                  product(np.arange(int(x_min), np.ceil(x_max)),
                          np.arange(int(y_min), np.ceil(y_max)))]
        points_in_poly = list(filter(shifted_poly.contains, points))
        for point in points_in_poly:
            xx, yy = point.xy
            x = int(xx[0])
            y = int(yy[0])
            if 0 <= y < im_size[1] and 0 <= x < im_size[2]:
                mask[z, y, x] = True
    masks = []
    for z_coord in np.arange(mask.shape[0]):
        masks.append(lil_matrix(mask[z_coord, :, :]))
    return masks


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
    mask = _reformat_mask(mask)

    verts_list = []
    for z, m in enumerate(mask):
        if issparse(m):
            m = np.array(m.astype('byte').todense())

        if (m != 0).sum() == 0:
            # If the plane is empty, just skip it
            continue

        # Add an empty row and column around the mask to make sure edge masks
        # are correctly determined
        expanded_dims = (m.shape[0] + 2, m.shape[1] + 2)
        expanded_mask = np.zeros(expanded_dims, dtype=float)
        expanded_mask[1:m.shape[0] + 1, 1:m.shape[1] + 1] = m

        verts = find_contours(expanded_mask.T, threshold)

        # Subtract off 1 to shift coords back to their real space,
        # but also add 0.5 to move the coordinates back to the corners,
        # so net subtract 0.5 from every coordinate
        verts = [np.subtract(x, 0.5).tolist() for x in verts]

        v = []
        for poly in verts:
            new_poly = [point + [z] for point in poly]
            v.append(new_poly)
        verts_list.extend(v)

    return _reformat_polygons(verts_list)


def _reformat_polygons(polygons):
    """Convert polygons to a MulitPolygon

    Accepts one more more sequence of 2- or 3-element sequences or a sequence
    of shapely Polygon objects.

    Parameters
    ----------
    polygons : sequence of 2- or 3-element coordinates or sequence of Polygons
        Polygon(s) to be converted to a MulitPolygon.  Coordinates are used to
        initialize a shapely MultiPolygon, and thus should follow a (x, y, z)
        coordinate space convention.

    Returns
    -------
    MultiPolygon

    """

    if len(polygons) == 0:
        # Just return an empty MultiPolygon
        return MultiPolygon([])
    elif isinstance(polygons, Polygon):
        polygons = [polygons]
    elif isinstance(polygons[0], Polygon):
        # polygons is already a list of polygons
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

    # Polygon.exterior.coords is not settable, need to initialize new objects
    z_polygons = []
    for poly in polygons:
        if poly.has_z:
            z_polygons.append(poly)
        else:
            warn('Polygon initialized without z-coordinate. ' +
                 'Assigning to zeroth plane (z = 0)')
            z_polygons.append(
                Polygon([point + (0,) for point in poly.exterior.coords]))
    return MultiPolygon(z_polygons)


def _reformat_mask(mask):
    """Convert mask to a list of sparse matrices (scipy.sparse.lil_matrix)

    Accepts a 2 or 3D array, a list of 2D arrays, or a sequence of sparse
    matrices.

    Parameters
    ----------
    mask : a 2 or 3 dimensional numpy array, a list of 2D numpy arrays, or a
        sequence of sparse matrices.  Masks are assumed to follow a (z, y, x)
        convention.  If mask is a list of 2D arrays or of sparse matrices, each
        element is assumed to correspond to the mask for a single plane (and is
        assumed to follow a (y, x) convention)
    """
    if isinstance(mask, np.ndarray):
        # user passed in a 2D or 3D np.array
        if mask.ndim == 2:
            mask = [lil_matrix(mask, dtype=mask.dtype)]
        elif mask.ndim == 3:
            new_mask = []
            for s in range(mask.shape[0]):
                new_mask.append(lil_matrix(mask[s, :, :], dtype=mask.dtype))
            mask = new_mask
        else:
            raise ValueError('numpy ndarray must be either 2 or 3 dimensions')
    elif issparse(mask):
        # user passed in a single lil_matrix
        mask = [lil_matrix(mask)]
    else:
        new_mask = []
        for plane in mask:
            new_mask.append(lil_matrix(plane, dtype=plane.dtype))
        mask = new_mask
    return mask
