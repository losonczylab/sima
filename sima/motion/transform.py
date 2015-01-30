import abc


class Transform(object):
    """Abstract class for geometric transforms."""
    __metaclass__ = abc.ABCMeta

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


class InvertibleTransform(Transform):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def inverse(self):
        pass


class DifferentiableTransform(Transform):
    __metaclass__ = abc.ABCMeta

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
