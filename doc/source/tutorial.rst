********
Tutorial
********

Here we provide some basic usage examples for the SIMA Python package.
These examples can be run in a standard Python shell, or with IPython.
Users new to Python may wish to consult the 
`Python documentation <https://docs.python.org/2.7/>`_.

For more details on the classes and methods that comprise the SIMA package,
please consult the `SIMA API <api/index.html>`_.

.. Contents::

Importing SIMA
--------------
Like all Python packages, the SIMA package must be imported prior to use.
Here we show a simple example of importing the SIMA package, which results
in printing the docstring containing view basic information about the package.

    >>> import sima

In all future examples, we assume that the SIMA package has been
imported as shown above.

Submodules of the SIMA package also need to be imported before use.
For example, the motion correction module can be imported as follows.

    >>> import sima.motion

Individual classes or functions can be imported from submodules.
For example, we can import the iterable object for use with multi-page
TIFF files with the following command:

    >>> from sima.motion import HiddenMarkov2D

For more details on importing, consult the `Python documentation
<https://docs.python.org/2.7/>`_.

Example data
------------
The SIMA package comes with a small amount of example data that, although
insufficient to allow for reasonable results from the motion correction and
segmentation algorithms, can at least be used to run the functions.  For the
purposes of this tutorial, we copy the example data into the current working
directory.

    >>> from shutil import copy, copytree
    >>> import sima.misc
    >>> copytree(sima.misc.example_data(), 'example.sima')
    >>> copy(sima.misc.example_tiff(), 'example.tif')
    >>> copy(sima.misc.example_tiff(), 'example_Ch1.tif')
    >>> copy(sima.misc.example_tiff(), 'example_Ch2.tif')
    >>> copy(sima.misc.example_hdf5(), 'example.h5')
    >>> copy(sima.misc.example_imagej_rois(), 'imageJ_ROIs.zip')

Creating an ImagingDataset object
---------------------------------
The SIMA package is centers around the :obj:`ImagingDataset` object class.  A
single :obj:`ImagingDataset` object can contain imaging data from multiple
sequences (i.e.  continuous imaging epochs/trials) acquired at the same imaging
location during the same imaging session.  The subsections below provide
examples of how to initialize :obj:`ImagingDataset` objects using raw data in a
variety of formats, including Numpy arrays, TIFF files, and HDF5 files.

The :obj:`ImagingDataset` object is permanently stored in the location (ending
with extension .sima) specified during initialization.  Results of
segmentation, signal extraction, and other alterations to the
:obj:`ImagingDataset` object are automatically saved to this location. 

Numpy arrays
............
To begin with, we create some Numpy arrays containing random data. The shape of
these arrays is (num_frames, num_planes, num_rows, num_columns, num_channels).

    >>> import numpy as np
    >>> cycle1_channel1 = np.random.rand(100, 1, 128, 128, 1)
    >>> cycle1_channel2 = np.random.rand(100, 1, 128, 128, 1)
    >>> cycle2_channel1 = np.random.rand(100, 1, 128, 128, 1)
    >>> cycle2_channel2 = np.random.rand(100, 1, 128, 128, 1)

Once we have the Numpy arrays containing the imaging data, we create the
ImagingDataset object as follows.

    >>> sequences = [sima.Sequence.join(
    ...                  sima.Sequence.create('ndarray', cycle1_channel1), 
    ...                  sima.Sequence.create('ndarray', cycle1_channel2)), 
    ...              sima.Sequence.join(
    ...                  sima.Sequence.create('ndarray', cycle2_channel1), 
    ...                  sima.Sequence.create('ndarray', cycle2_channel2))]
    >>> dataset = sima.ImagingDataset(
    ...     sequences, 'example_np.sima', channel_names=['green', 'red'])

Multipage TIFF files
....................
For simplicity, we consider the case of only a single cycle and channel.

    >>> sequences = [sima.Sequence.create('TIFF', 'example_Ch1.tif')]
    >>> dataset = sima.ImagingDataset(sequences, 'example_TIFF.sima')

HDF5 files
..........
The argument 'yxt' specifies that the first index of the HDF5 array corresponds
to the row, the second to the column, and the third to the time.

    >>> sequences = [sima.Sequence.create('HDF5', 'example.h5', 'yxt')]
    >>> dataset = sima.ImagingDataset(sequences, 'example_HDF5.sima')


Loading ImagingDataset objects
------------------------------
A dataset object can also be loaded from a saved path with the .sima extension.

    >>> dataset = sima.ImagingDataset.load('example.sima')


Motion correction 
-----------------
The SIMA package implements a variety of approaches for motion correction. To
use one of these approaches, the user first creates an object encapsulating the
approach and a choice of parameters. For example, an object encapsulating the
approach of translating imaging planes in two dimensions with a maximum
displacement of 15 rows and 30 columns can be created as follows:

    >>> import sima.motion
    >>> mc_approach = sima.motion.PlaneTranslation2D(max_displacement=[15, 30])

Once this object encapsulating the approach has been created, its
:func:`correct()` method can then be applied to create a corrected dataset from
a list of sequences.

    >>> sequences = [sima.Sequence.create('TIFF', 'example_Ch1.tif')]
    >>> dataset = mc_approach.correct(sequences, 'example_translation2D.sima')

In the following example, the SIMA package is used for motion correction based
on a hidden Markov model (HMM). The :class:`sima.motion.HiddenMarkov2D` class
takes the initialize arguments to specify its parameters (e.g.\ an optional
argument is used to indicate that the maximum possible displacement is 20 rows
and 30 columns). The :func:`correct` method takes the same arguments as are
used to initialize an imaging dataset object, as well as some additional
optional arguments. 

    >>> mc_approach = sima.motion.HiddenMarkov2D(
    ...     granularity='row', max_displacement=[20, 30], verbose=False) 
    >>> dataset = mc_approach.correct(sequences, 'example_mc.sima')

When the signal is of interest is very sparse or highly dynamic, it is
sometimes helpful to use a second static channel to estimate the displacements
for motion correction. The example below is for the case where the first
channel contains a dynamic GCaMP signal whose large variations would confuse
the motion correction alogorithm, and the second channel contains a static
tdTomato signal that provides a stable reference.

    >>> sequences = [
    ...     sima.Sequence.join(sima.Sequence.create('TIFF', 'example_Ch1.tif'), 
    ...                        sima.Sequence.create('TIFF', 'example_Ch1.tif'))
    ... ]
    >>> dataset = mc_approach.correct(
    ...     sequences, 'example_mc2.sima', channel_names=['GCaMP', 'tdTomato'],
    ...     correction_channels=['tdTomato'])

When motion correction is invoked as above, only the tdTomato channel is used
for estimating the displacements, which are then applied to both channels.

Segmentation and ROIs
---------------------

Automated segmentation
......................
The SIMA package implements a number of approaches for automated segmentation.
To use one of the approaches, the first step is to create an object
encapsulating the approach and a choice of parameters. For example, an object
representing the approach of segmenting a single-plane dataset with
spatiotemporal independent component analysis (STICA) can be created as follows:

    >>> import sima.segment
    >>> stica_approach = sima.segment.STICA(components=5)

We can also add post-processing steps to this approach, for example to convert
the STICA masks into sparse regions of interest (ROIs), smooth their boundaries, and
merge overlapping ROIs.

    >>> stica_approach.append(sima.segment.SparseROIsFromMasks())
    >>> stica_approach.append(sima.segment.SmoothROIBoundaries())
    >>> stica_approach.append(sima.segment.MergeOverlapping(threshold=0.5))

Once the approch has been created, it can be passed as an argument to the
:func:`segment` method of an An :class:`ImagingDataset`. The :func:`segment`
method can also take an optional label argument for the resulting set of ROIs. 

    >>> dataset = sima.ImagingDataset.load('example.sima')
    >>> rois = dataset.segment(stica_approach, 'auto_ROIs')

Editing, creating, and registering ROIs with ROI Buddy
......................................................
Note that the an :class:`ImagingDataset` object can be loaded with the `ROI
Buddy <roi_buddy.html>`_ graphical user interface (GUI) for manual editing of
existing the ROI lists, creation of new ROI lists, or registration of ROI lists
across multiple experiments in which the same field of view is imaged.  For
more details, consult the `ROI Buddy documentation <roi_buddy.html>`_.


Importing ROIs from ImageJ
..........................
ROIS can also be imported from ImageJ, as shown in the following example.

    >>> from sima.ROI import ROIList
    >>> dataset = sima.ImagingDataset.load('example.sima')
    >>> rois = ROIList.load('imageJ_ROIs.zip', fmt='ImageJ')
    >>> dataset.add_ROIs(rois, 'from_ImageJ')
    >>> dataset.ROIs.keys()  # view the labels of the existing ROILists
    ['from_ImageJ', 'auto_ROIs']

Mapping ROIs between datasets
.............................
Sometimes, for example when imaging the same field of view over multiple days,
one wishes to segment the same structures in separate :class:`ImagingDataset`
objects.  If all of the :class:`ImagingDataset` objects have been segmented,
then the results of the segmentations can be registered with the `ROI Buddy GUI
<roi_buddy.html>`_ as mentioned previously. If, however, only one of the
datasets has been segmented, the results of the segmentation can be applied to
the other datasets by applying to each ROI the affine transformation necessary
to map one imaged field of view onto the other.  This can be done either with
the `ROI Buddy GUI <roi_buddy.html>`_ or with a call to the
:func:`import_transformed_ROIs` method, whose arguments allow for specification
of the channels used to align the two datasets, the label of the :`obj`:ROIList
to be transformed from one dataset to the other, the label that will be applied
to the new :obj:`ROIList`, and whether to copy the properties of the ROIs as
well as their shapes.

    >>> source_dataset = sima.ImagingDataset.load('example.sima')
    >>> target_dataset = sima.ImagingDataset.load('example_mc2.sima')
    >>> target_dataset.ROIs.keys()
    []
    >>> target_dataset.import_transformed_ROIs(
    ...     source_dataset, source_channel='green', target_channel='GCaMP',
    ...     source_label='from_ImageJ', target_label='transformed',
    ...     copy_properties='True')
    >>> target_dataset.ROIs.keys()
    ['transformed']

This approach allows the user to focus on careful manual curation of the
segmentation for a single :class:`ImagingDataset`, with the results of this
segmentation then applied to all datasets acquired at the same field of view.

Accessing stored ROIs
.....................
Whenever ROIs are created or imported, they are permanently stored as part of
the :class:`ImagingDataset` object.  The ROIs can be recovered at any time
using the label specified at the time when the ROIs were created.

    >>> dataset = sima.ImagingDataset.load('example.sima')
    >>> dataset.ROIs.keys()  # view the labels of the existing ROILists
    ['from_ImageJ', 'auto_ROIs']
    >>> rois = dataset.ROIs['auto_ROIs']


Extraction
----------
Once the ROIs have been edited and registered, the dataset can be loaded, and
then dynamic fluorescence signals can be extracted from the ROIs with the
:func:`extract` method.


    >>> dataset = sima.ImagingDataset.load('example.sima')
    >>> dataset.ROIs.keys()
    ['from_ImageJ', 'auto_ROIs']
    >>> rois = dataset.ROIs['from_ImageJ']
    >>> dataset.channel_names
    ['red', 'green']
    >>> signals = dataset.extract(
    ...     rois, signal_channel='green', label='green_signal')

The extracted signals are permanently saved with the :obj:`ImagingDataset`
object and can be accessed at any time with the command :func:`signals` method.

    >>> dataset = sima.ImagingDataset.load('example.sima')
    >>> signals = dataset.signals(channel='green')['green_signal']

Exporting data
--------------
Data can be exported from the SIMA :class:`ImagingDataset` objects at various
stages of the analysis. This allows SIMA to be used for early stages of data
analysis, and then for the exported data to be analyzed with separate software.
If, however, further analysis is to be performed with Python, such exporting
may not be necessary. The subsections below contain examples showing how to
export image data and signal data.

Image data
..........
The :class:`ImagingDataset` class has two methods for exporting image data,
:func:`export_frames` and :func:`export_averages`, which export either all
the frames or the time averages of each channel, respectively. These methods
can be used to view the results of motion correction, as shown in the following
example.

    >>> import sima.motion
    >>> sequences = [sima.Sequence.create('TIFF', 'example_Ch1.tif')]
    >>> dataset = mc_approach.correct(sequences, 'example_mc3.sima')
    >>> dataset.export_averages(['avgs.tif'], fmt='TIFF16')
    >>> dataset.export_frames([[['frames.tif']]], fmt='TIFF16')

The paths to which the exported data are saved are organized as a list with one
filename per channel for the :func:`export_averages` method, or as a list of
lists (organized analogously to the sequence used to initialize an
:class:`ImagingDataset` object) for the :func:`export_frames` method. If
however, the export format is specified to HDF5, then the filenames for
:func:`export_frames` should be organized into a list with one filename per
cycle, since both channels are combined into a single HDF5 file.

    >>> dataset.export_frames(['exported_frames.h5'], fmt='HDF5')

Signal data
...........
For users wishing to analyze the extracted signals with an external program,
these signals can be exported to a CSV file.

    >>> dataset = sima.ImagingDataset.load('example.sima')
    >>> dataset.export_signals('example_signals.csv', channel='green')

The resulting CSV file contains the :obj:`id`, :obj:`label`, and :obj:`tags`
for each ROI, and the extracted signal from each ROI at each frame time.

Complete example
----------------
Below are the contents of workflow.py in the examples directory provided with
the SIMA source code. This example is also available as an interactive iPython
notebook in the same location, or as a static version here:
:download:`Example Workflow <_static/example_workflow.html>`

.. include:: ../../examples/workflow.py
    :code: python

