import numpy as np
from . import extmath as em
from .sketches import LRSketch


class CGMSketch(LRSketch):
    """Docstring for CGMSketch. """

    def cgm_update(self, eta, u, v, alpha):
        """Performs an in-place update of the sketched matrix `X` equivalent to

            X <- (1 - eta) X + eta * alpha (u * v)

        `u` and `v` are expected in the same format as the come out of
        :func:`linalg.svds`.

        :param float eta: Update weight
        :param u: Vector of length `X.shape[0]`
        :param v: Vector of length `X.shape[1]`
        :param float alpha:
        :returns: `self`

        """
        self.Y = (1 - eta) * self.Y + eta * u.dot(v.dot(self.Omega))
        self.W = (1 - eta) * self.W + eta * (self.Psi.dot(u)).dot(v)


def scgm_estimator(A, y):

