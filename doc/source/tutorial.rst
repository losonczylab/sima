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
Here we show a simple example of importing the SIMA package and then
printing its docstring to view basic information about the package.


    >>> import sima
    >>> print sima.__doc__
    SIMA: Python package for sequential image analysis.
    Developed by Patrick Kaifosh, Jeffrey Zaremba, Nathan Danielson.
    Copyright (C) 2014 The Trustees of Columbia University in the City of New York.
    Licensed under the GNU GPL version 2 or later.
    Documentation: http://www.losonczylab.org/sima
    Version 0.1.2-alpha

In all future examples, we assume that the SIMA package has been
imported as shown above.

Submodules of the SIMA package also need to be imported before use.
For example, the motion correction module can be imported as follows.

    >>> import sima.motion

Individual classes or functions can be imported from submodules.
For example, we can import the iterable object for use with multi-page
TIFF files with the following command:

    >>> from sima.iterables import MultiPageTIFF

For more details on importing, consult the `Python documentation
<https://docs.python.org/2.7/>`_.


Creating an ImagingDataset object
---------------------------------

The SIMA package is centers around the :obj:`ImagingDataset` object class.  A
single :obj:`ImagingDataset` object can contain imaging data from multiple
simultaneously recorded optical channels, as well as from multiple cycles (i.e.
continuous imaging epochs/trials) acquired at the same imaging location during
the same imaging session.  Accordingly, the raw imaging data used to initialize
the :obj:`ImagingDataset` object must be packaged into a list of lists, whose
first index runs over the cycles and whose second index runs over the channels.
The subsections below provide examples of how to initialize
:obj:`ImagingDataset` objects using raw data in a variety of formats, including
Numpy arrays, TIFF files, and HDF5 files.

The :obj:`ImagingDataset` object is permanently stored in the location (ending
with extension .sima) specified during initialization.  Results of
segmentation, signal extraction, and other alterations to the
:obj:`ImagingDataset` object are automatically saved to this location. 


Numpy arrays
............

To begin with, we create some Numpy arrays containing random data.

    >>> import numpy as np
    >>> cycle1_channel1 = np.random.rand(100, 128, 128)
    >>> cycle1_channel2 = np.random.rand(100, 128, 128)
    >>> cycle2_channel1 = np.random.rand(100, 128, 128)
    >>> cycle2_channel2 = np.random.rand(100, 128, 128)

Once we have the Numpy arrays containing the imaging data, we create the
ImagingDataset object as follows.

    >>> path = 'example.sima'
    >>> iterables = [
    ...     [cycle1_channel1, cycle1_channel2]
    ...     [cycle2_channel1, cycle2_channel2]
    ... ]
    >>> dataset = sima.ImagingDataset(
    ...     iterables, path, channel_names=['green', 'red'])

Multipage TIFF files
....................

For simplicity, we consider the case of only a single cycle and channel.

    >>> import sima.misc
    >>> from sima.iterables import MultiPageTIFF
    >>> tiff_path = sima.misc.example_tiff()  # path to the TIFF file
    >>> iterables = [[MultiPageTIFF(tiff_path)]]
    >>> dataset = sima.ImagingDataset(iterables, 'example_TIFF.sima')

HDF5 files
..........

The argument 'tyx' specifies that the first index of the HDF5 array corresponds
to the time, the second to the row, and the third to the column.

    >>> import sima.misc
    >>> from sima.iterables import HDF5
    >>> hdf5_path = sima.misc.example_hdf5()  # path to the HDF5 file
    >>> iterables = [[HDF5(hdf5_path, 'tyx')]]
    >>> dataset = sima.ImagingDataset(iterables, 'example_HDF5.sima')


Loading ImagingDataset objects
------------------------------

A dataset object can also be loaded from a saved path with the .sima extension.

    >>> path = sima.misc.example_data()  # path ending with .sima extension
    >>> dataset = sima.ImagingDataset.load(path)


Motion correction 
-----------------
In the following example, the SIMA package is used for motion correction based
on a hidden Markov model (HMM). The :func:`sima.motion.hmm` function takes the
same arguments as are used to initialize an imaging dataset object, as well as
some additional optional arguments. In the example below, an optional argument
is used to indicate that the maximum possible displacement is 20 rows and 30
columns.

    >>> import sima.motion
    >>> import sima.misc
    >>> from sima.iterables import MultiPageTIFF
    >>> iterables = [[MultiPageTIFF(sima.misc.example_tiff())]]
    >>> dataset = sima.motion.hmm(iterables, 'example_mc.sima',
    ...                           max_displacement=[20,30])




Segmentation and ROIs
---------------------
An :class:`ImagingDataset` object can be automatically segmented with a call to
its :func:`segment` method.  The arguments of the :func:`segment` method
specify the segmentation approach to be used, an optional label for the
resulting set of ROIs, and additional arguments specific to the particular
segmentation approach.  In the example below, an imaging dataset object is
segmented with the ``'ca1pc'`` method designed for segmenting CA1 pyramidal
cells.

    >>> dataset = sima.ImagingDataset.load('example_mc.sima')
    >>> rois = dataset.segment('ca1pc', 'auto_ROIS')

In addition to being returned by the :func:`segment` method, the resulting ROIs
are also permanently stored as part of the :class:`ImagingDataset` object. They
can be recovered at any time using the label specified at the time of
specification.

    >>> dataset = sima.ImagingDataset.load('example_mc.sima')
    >>> dataset.ROIs.keys()  # view the labels of the available ROI sets
    ['auto_ROIs', 'manually_edited_ROIS']
    >>> rois = dataset.ROIs['manually_edited_ROIS']

ROIS can also be imported from ImageJ, as shown in the following example.

    >>> import sima.misc
    >>> from sima.ROI import ROIList
    >>> roi_path = sima.misc.example_ROIs()
    >>> dataset = sima.ImagingDataset.load('example_mc.sima')
    >>> rois = ROIList.load(example_ROIs(), fmt='ImageJ')
    >>> dataset.add_ROIs(rois, 'from_ImageJ')
    >>> dataset.ROIs.keys()
    ['auto_ROIs', 'manually_edited_ROIS', 'from_ImageJ']

Note that the an :class:`ImagingDataset` object can be loaded with the `ROI
Buddy <roi_buddy.html>`_ graphical user interface (GUI) for manual editing of
existing the ROI sets, creation of new ROI sets, or registration of ROI sets
across multiple experiments in which the same field of view is imaged.

Extraction
----------
Once the ROIs have been edited and registered, the dataset can be loaded, and
then dynamic fluorescence signals can be extracted from the ROIs with the
:func:`extract` method.


    >>> import sima.misc
    >>> dataset = sima.ImagingDataset.load(sima.misc.example_data())
    >>> dataset.ROIs.keys()
    ['from_ImageJ']
    >>> rois = dataset.ROIs['from_ImageJ']
    >>> dataset.channel_names
    ['red', 'green']
    >>> signals = dataset.extract(
    ...     rois, signal_channel='green', label='green_signal')

The extracted signals are permanently saved with the :obj:`ImagingDataset`
object and can be accessed at any time with the command :func:`signals` method.

    >>> import sima.misc
    >>> dataset = sima.ImagingDataset.load(sima.misc.example_data())
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
    >>> import sima.misc
    >>> from sima.iterables import MultiPageTIFF
    >>> iterables = [[MultiPageTIFF(sima.misc.example_tiff())]]
    >>> dataset = sima.motion.hmm(iterables, 'example_mc.sima',
    ...                           max_displacement=[20,30])
    >>> dataset.export_averages(['exported_frames.tif'], fmt='TIFF16')
    >>> dataset.export_frames([['exported_frames.tif']], fmt='TIFF16')

The paths to which the exported data are saved are organized as a list with one
filename per channel for the :func:`export_averages` method, or as a list of
lists (organized analogously to the iterables used to initialize an
:class:`ImagingDataset` object) for the :func:`export_frames` method. If
however, the export format is specified to HDF5, then the filenames for
:func:`export_frames` should be organized into a list with one filename per
cycle, since both channels are combined into a single HDF5 file.

    >>> dataset.export_frames('exported_frames.h5', fmt='HDF5')

Signal data
...........
For users wishing to analyze the extracted signals with an external program,
these signals can be exported to a CSV file.

    >>> dataset = sima.ImagingDataset.load('example_mc.sima')
    >>> dataset.export_signals('example_signals.csv')

The resulting CSV file contains the :obj:`id`, :obj:`label`, and :obj:`tags`
for each ROI, and the extracted signal from each ROI at each frame time.

Complete example
----------------
Below are the contents of workflow.py in the examples directory provided with
the sima source code.

.. include:: ../../examples/workflow.py
    :code: python

