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

.. figure:: roi_buddy/overview.jpg
   :align:  center

Typical Workflow
----------------
* Load multiple SIMA imaging datasets corresponding to different recordings of the same field of view.
* For each dataset, select the channel you wish to edit and align.
* Create or load pre-drawn ROIs for each dataset, editing the ROIs as necessary in Edit Mode.
* Tag cells in one dataset based on some criteria, e.g. morphology.
* Enter Align mode, bringing all the ROIs into alignment.
* Choose "Register ROIs" to assign a common ``id`` property to overlapping ROIs.
* Use the merge and unmerge tools to manually fix any incorrect groupings.
* Once registration is satisfactory, choose "Propagate Tags" to assign the morphology tags to all matched ROIs across imaging sessions.
* Save ROIs.

Installation
============

The ROI Buddy GUI has the following dependencies which should be installed
prior to installation of ROI Buddy:

* sima>=1.0 (https://pypi.python.org/pypi/sima)
* PyQt4 (http://www.riverbankcomputing.co.uk/software/pyqt)
* guidata (https://code.google.com/p/guidata/)
* guiqwt (https://code.google.com/p/guiqwt/)

Once these dependencies are installed, the ROI Buddy can be installed from 
the Python Package Index using pip.::

    $ pip install roibuddy --user

Alternatively, the ROI Buddy GUI can be built from source. Source code for
released versions can be downloaded from the `Python Package Index
<https://pypi.python.org/pypi/roibuddy>`__, and code under development can be
downloaded from the project's `GitHub page
<https://github.com/losonczylab/roibuddy>`__.

Installation of ROI Buddy will create an executable file that can be run by
typing "roibuddy" into your terminal. If this command does not work, check
the output of the installation command to determine the directory where the
executable was placed, and either add this directory to your path or run
the executable directly from that location.

Windows
-------
After installing `SIMA <install.html>`__, we recommend installing pre-compiled
wheels from (http://www.lfd.uci.edu/~gohlke/pythonlibs/) for
`PyQt4 <http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyqt4>`__,
`PyQwt <http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyqwt>`__,
`quidata <http://www.lfd.uci.edu/~gohlke/pythonlibs/#guiqwt>`__, and
`guiqwt <http://www.lfd.uci.edu/~gohlke/pythonlibs/#guiqwt>`__ (in that order),
before installing ROI Buddy from the
`Python Package Index <https://pypi.python.org/pypi/roibuddy>`__.


Mac OS X
--------
We recommend using MacPorts for installing the dependencies. 
After installing SIMA using MacPorts as described `here <install.html>`__,
run the following command in Terminal to install an additional dependency::

    $ sudo port install py27-pyqwt

Then download and install `guidata <https://code.google.com/p/guidata/>`__ and
`guiqwt <https://code.google.com/p/guiqwt/>`__ before installing ROI Buddy from
the `Python Package Index <https://pypi.python.org/pypi/roibuddy>`__.


User Interface
==============
File Menu
---------
.. csv-table::
    :file: roi_buddy/File_Menu.csv
    :header: "File Menu Option", "Action"
    :widths: 1, 4

Control Panel
-------------
.. figure:: roi_buddy/main_panel.jpg
   :align:  center

Toggling modes
++++++++++++++
.. csv-table::
    :file: roi_buddy/control_panel_modes.csv
    :widths: 1, 4

Initializing SIMA imaging dataset and ROI List objects
++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. csv-table::
    :file: roi_buddy/control_panel_initialize_sets.csv
    :widths: 1, 4

Registering ROI objects across multiple imaging sessions
++++++++++++++++++++++++++++++++++++++++++++++++++++++++
If multiple datasets have been loaded from the same field of view, it is possible to align the ROIs and commonly identify them
so that they can be tracked across sessions. The currently selected dataset will act as a template to which all other datasets
will be aligned. Each dataset is automatically aligned to the template dataset by calculating an affine transformation between
time averaged images that produces maximal similarity. A clustering algorithm based on the Jaccard Index is used to match cells 
between datasets, which can be manually adjusted by merging/unmerging ROIs from the automatic clusters. Once registered, ROIs that
are matched across days are assigned the same ``id`` property which is denoted visibly by giving them all the same color.

.. csv-table::
    :file: roi_buddy/control_panel_registration.csv
    :widths: 1, 4

.. warning::
    In align mode, it is necessary that all imaging datasets loaded must be
    roughly of the same field of view.  Alignment is based upon an affine
    transformation with 6 degrees of freedom.  If a transform between
    time averaged images cannot be calculated, an error message will be displayed printing
    the directories of the incompatible sets.

Toggling the visibility of ROIs
+++++++++++++++++++++++++++++++
.. csv-table::
    :file: roi_buddy/control_panel_view_rois.csv
    :widths: 1, 4


Keyboard shortcuts
==================

:m (edit mode): merge separate ROIs into a single ROI object
:m (align mode): merge selected ROIs into the same cluster, assigning them the same ID attribute
:u (align mode): unmerge ROI from its cluster and assign it a unique ID attribute
:f (edit mode): select freeform tool
:s (edit mode): select pointer-selection tool
:d: delete
:r: randomize ROI colors

