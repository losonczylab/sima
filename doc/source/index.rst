SIMA: sequential image analysis
===============================

:Release: |release|
:Date: |today|
:Email: software@losonczylab.org
:Forum: https://groups.google.com/forum/#!forum/sima-users
:GitHub: https://github.com/losonczylab/sima


Overview
--------
SIMA (Sequential IMage Analysis) is an Open Source package for 
analysis of time-series imaging data arising from fluorescence
microscopy.  The functionality of this package includes:

- correction of motion artifacts
- segmentation of imaging fields into regions of interest (ROIs)
- extraction of dynamic signals from ROIs

The included ROI Buddy software provides a graphical user interface
(GUI) supporting the following functionality:

- manual creation of ROIs
- editing of ROIs resulting from automated segmentation
- registration of ROIs across separate imaging sessions


Documentation
-------------

.. toctree::
   :maxdepth: 2

   Introduction <index>
   install
   tutorial
   api/index
   roi_buddy
   publications
   credits
   support


Citing SIMA
-----------
If you use SIMA for your research, please cite the following paper in any 
resulting publications:

  `Kaifosh P, Zaremba JD, Danielson NB, and Losonczy A. 
  SIMA: Python software for analysis of dynamic fluorescence imaging data. 
  Frontiers in Neuroinformatics. 2014 Aug 27; 8:77. 
  doi: 10.3389/fninf.2014.00080.
  <http://journal.frontiersin.org/article/10.3389/fninf.2014.00080/>`_


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
