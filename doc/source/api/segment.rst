============
Segmentation
============

.. Contents::

The SIMA package implements multiple segmentation strategies, which
share a common interface defined by the abstract class 
:class:`SegmentationStrategy`. Any strategy with this interface can
be used in conjunction with the :func:`ImagingDataset.segment()`
method to segment an :class:`ImagingDataset` object.

.. automethod:: sima.imaging.ImagingDataset.segment


Segmentation strategies
=======================

.. note:: Please consider contributing additional methods to the SIMA project.

.. autoclass:: sima.segment.SegmentationStrategy
    :members:
    :private-members:

The specific segmentation strategies that have been implemented are
documented below. Once initialized as documented, these strategies
all share the above interface.

Plane-Wise Segmentation
-----------------------

.. autoclass:: sima.segment.PlaneWiseSegmentation
    :members:
    :show-inheritance:

Spatialtemporal Independent Component Analysis
----------------------------------------------

.. autoclass:: sima.segment.STICA
    :members:
    :show-inheritance:

Normalized cuts
---------------

.. autoclass:: sima.segment.PlaneNormalizedCuts
    :members:
    :show-inheritance:


Affinity Matrix Methods
.......................

.. autoclass:: sima.segment.AffinityMatrixMethod 
    :members:
.. autoclass:: sima.segment.BasicAffinityMatrix 
    :members:
    :show-inheritance:
.. autoclass:: sima.segment.AffinityMatrixCA1PC
    :members:
    :show-inheritance:

CA1 pyramidal cells
-------------------

.. autoclass:: sima.segment.PlaneCA1PC
    :members:
    :show-inheritance:

Post-Processing Steps
=====================

Any number of post-processing steps can be added to a segmentation method using
the :func:`SegmentationStrategy.append()` method. These appended
post-processing steps must have the interface defined by
:class:`PostProcessingStep` below. The appended steps can be selected from
those documented in this section, or can be created by the user by subclassing
any of the classes listed below.

.. autoclass:: sima.segment.PostProcessingStep
    :members:
.. autoclass:: sima.segment.ROIFilter
    :members:
    :show-inheritance:
.. autoclass:: sima.segment.CircularityFilter
    :members:
    :show-inheritance:
.. autoclass:: sima.segment.MergeOverlapping
    :members:
    :show-inheritance:
.. autoclass:: sima.segment.SmoothROIBoundaries
    :members:
    :show-inheritance:
.. autoclass:: sima.segment.SparseROIsFromMasks
    :members:
    :show-inheritance:
