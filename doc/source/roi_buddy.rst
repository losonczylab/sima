*************
ROI Buddy GUI
*************

.. Contents::

Introduction
============

The ROI Buddy GUI can be used for viewing, creating, editing, and tagging the
regions of interest (ROIs) associated with multiple SIMA imaging datasets
simultaneously.
In addition it can be used for registering ROIs from different imaging
sessions of the same field-of-view, allowing for the longitudinal tracking of
cells in serial imaging experiments.


Installation
============

The ROI Buddy GUI is compiled as a Windows executable (.exe) file and is 
available for download `here
<https://dl.dropboxusercontent.com/u/25540135/ROI_Buddy.zip>`_.

Alternatively, the ROI Buddy GUI can be built from source.  Source code for
the ROI Buddy GUI is available for download `here
<https://dl.dropboxusercontent.com/u/25540135/ROI_Buddy_Source.zip>`_.
Dependencies for the GUI are identical to those for the SIMA module.

Note that the ROI Buddy .exe file is compiled with the tifffile module rather
than the libtiff C library.  If the ROI Buddy GUI is to be used with large
TIFF files containing many frames, we recommend running the ROI Buddy from
source after installing the libtiff C library and its associated Python
bindings, as it enables more efficient memory handling.


Basic Usage
===========

In order to add a sima.ImagingDataset object, select Add T-Series.  Choose the
.sima directory for which ROIs are to be drawn.  Alternatively choose auto-add
t-series to recursively load all .sima directories containing an arbitrary
text string below the selected directory.  If alignment functionality is
to be used, all loaded t-series should be of the same field of view.

For each t-series loaded, the ROI_List objects associated with each set are
listed in the ROI List dropdown menu.  Choose the ROI List you wish to edit.
New ROI List objects can be created, and ROI Lists can be removed from an 
ImagingDataset object here as well.

In 'edit' mode, ROIs can be created, edited, and deleted for each t-series
separately.
Additionally tags can be defined for the ROIs for use in sorting them in
subsequent analyses.
After ROIs have been created for each t-series loaded, ROIs can 
be registered across imaging sessions of the same field by entering 'align'
mode.  In 'align' mode, the ROIs from all imaging sessions are overlaid on the
same field of view.  Select Register ROIs to automatically calculate the
overlap between ROIs and assign common ID attributes to co-clustered ROIs.
Incorrect groupings can be corrected by using the merge and unmerge features.
The propagate tags feature allows the user to propagate tags across all
co-clustered ROIs.

Following registration of ROI objects, choose 'save all' in order to save the
ROI Lists associated with each t-series loaded.  ROIs are saved in the
rois.pkl file within the appropriate .sima directory.


Keyboard shortcuts
==================

:m (edit mode): merge separate ROIs into a single ROI object
:m (align mode): merge selected ROIs into the same cluster, assigning them the same ID attribute
:u (align mode): unmerge ROI from its cluster and assign it a unique ID attribute
:f (edit mode): select freeform tool
:s (edit mode): select pointer-selection tool
:d: delete
:r: randomize ROI colors

