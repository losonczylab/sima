============
Segmentation
============

.. Contents::

.. automethod:: sima.imaging.ImagingDataset.segment

Segmentation approaches
=======================

.. note:: Please consider contributing additional methods to the SIMA project.

.. autoclass:: sima.segment.SegmentationStrategy
    :members:
.. autoclass:: sima.segment.PlaneWiseSegmentationStrategy
    :members:
    :show-inheritance:
.. autoclass:: sima.segment.PlaneSegmentationStrategy
    :members:
    :show-inheritance:

Spatialtemporal Independent Component Analysis
----------------------------------------------

.. autoclass:: sima.segment.PlaneSTICA
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

.. autoclass:: sima.segment.PostProcessingStep
    :members:
.. autoclass:: sima.segment.ROIFilter
    :members:
    :show-inheritance:
.. autoclass:: sima.segment.ROISizeFilter
    :members:
    :show-inheritance:
.. autoclass:: sima.segment.CircularityFilter
    :members:
    :show-inheritance:
.. autoclass:: sima.segment.CA1PCNucleus
    :members:
    :show-inheritance:

