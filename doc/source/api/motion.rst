Motion correction
=================

The SIMA package can be used to motion correct sequentially acquired images.
The package implements several strategies, all of which have a common interface
described below.

.. autoclass:: sima.motion.MotionEstimationStrategy
    :members:

The specific strategies for motion correction are listed below.  These
strategies must be initialized as documented below, and then can be applied to
datasets using the generic interface described above.

.. autoclass:: sima.motion.HiddenMarkov2D
.. autoclass:: sima.motion.HiddenMarkov3D
.. autoclass:: sima.motion.DiscreteFourier2D
.. autoclass:: sima.motion.PlaneTranslation2D
.. autoclass:: sima.motion.VolumeTranslation
.. autoclass:: sima.motion.ResonantCorrection
