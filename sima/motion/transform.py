from builtins import object
import abc
from future.utils import with_metaclass


class Transform(with_metaclass(abc.ABCMeta, object)):

    """Abstract class for geometric transforms."""

    @abc.abstractmethod
    def apply(self, source, grid=None):
        """Apply the transform to raw source data.

        Parameters
        ----------
        source : np.ndarray
        grid :

        Returns
        -------
        transformed : np.ndarray
        """
        pass


class InvertibleTransform(with_metaclass(abc.ABCMeta, Transform)):

    @abc.abstractmethod
    def inverse(self):
        pass


class DifferentiableTransform(with_metaclass(abc.ABCMeta, Transform)):

    @abc.abstractmethod
    def jacobian(self):
        pass


class NullTransform(Transform):

    """Class to represent a null transform.

    This may be useful to indicate that a transform could not be estimated.
    It could return a grid of all NaN values when applied.
    """
    pass


class Identity(Transform):
    pass


class WithinFrameTranslation(Transform):
    pass
