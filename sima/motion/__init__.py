from numpy.testing import Tester
test = Tester().test

from motion import MotionEstimationStrategy
from frame_align import PlaneTranslation2D, VolumeTranslation
from hmm import HiddenMarkov2D, MovementModel, HiddenMarkov3D
