from abc import abstractmethod

import numpy as np
from scipy.sparse.linalg import aslinearoperator

from .extmath import standard_normal


class MeasurementMap(object):
    """Docstring for MeasurementMap. """

    def __init__(self, nr_measurements, shape):
        """@todo: to be defined1.

        :param nr_measurements: @todo
        :param shape: @todo

        """
        self.nr_measurements = nr_measurements
        self.shape = shape

    @abstractmethod
    def __call__(self, u, v):
        """Computes the application of the measurement map on the matrix `u * v`

        :param u: @todo
        :param v: @todo
        :returns: @todo

        """
        pass


    @abstractmethod
    def H(self, z):
        """Computes the action of the adjoint of the measurement map on the
        vector `z`

        :param z: @todo
        :returns: @todo

        """
        pass


class Rank1MeasurmentMap(MeasurementMap):
    """Docstring for Rank1MeasurmentMap.
    Note that this is just for testing purposes and not memory efficient!
    """

    def __init__(self, nr_measurements, shape, rgen=np.random, dtype=np.float_):
        """@todo: to be defined1. """
        super().__init__(nr_measurements, shape)
        self.left_measurements = standard_normal((nr_measurements, shape[0]),
                                                 rgen=rgen, dtype=dtype)
        self.right_measurements = standard_normal((nr_measurements, shape[1]),
                                                  rgen=rgen, dtype=dtype)

    def __call__(self, u, v):
        """@todo: Docstring for call.

        :param u: @todo
        :param v: @todo
        :returns: @todo

        """
        assert u.shape == (self.shape[0], 1)
        assert v.shape == (1, self.shape[1])

        y_left = self.left_measurements.dot(u.ravel())
        y_right = self.right_measurements.dot(v.ravel())
        return y_left * y_right / np.sqrt(self.nr_measurements)

    def H(self, z):
        """@todo: Docstring for H.

        :param z: @todo
        :returns: @todo

        """
        assert len(z) == self.nr_measurements
        X = self.left_measurements.T.dot(z[:, None] * self.right_measurements)
        return aslinearoperator(X / np.sqrt(self.nr_measurements))
