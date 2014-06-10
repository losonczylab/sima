*************
ROI Buddy GUI
*************

.. Contents::

Introduction
============
The ROI Buddy GUI can be used for viewing, creating, and editing the regions
of interest (ROIs) associated with a SIMA imaging dataset.  In addition it can
be used for registering ROIs from different imaging sessions of the same
field-of-view, allowing for the longitudinal tracking of cells in serial
imaging experiments.


Installation
============
The ROI Buddy GUI is compiled as a Windows executable (.exe) file and is 
available for download `here
<ftp://losonczylab.org/ROI_Buddy.zip>`_.

Alternatively, the ROI Buddy GUI can be built from sources.  Source code for
the ROI Buddy GUI is available for download `here
<ftp://losonczylab.org/ROI_Buddy_Source.zip>`_. Dependencies for the ROI Buddy
GUI are identical to those for the SIMA module.

Note that the ROI Buddy .exe file is compiled with the tifffile module, rather than
the libtiff C library.  If the ROI Buddy GUI is to be used with large TIFF files
containing many frames, it is recommended to install the libtiff C library and 
its associated Python bindings and running the ROI Buddy GUI from source.


Basic Usage
===========
In order to add an 