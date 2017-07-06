import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from copy import copy

from . import extmath as em


class LRSketch(LinearOperator):
    """Sketch of a low-rank hermitian matrix"""
    __array_priority__ = 100

    def __init__(self, shape, rank, rgen=np.random, dtype=np.float_):
        """@todo: to be defined1.

        """
        self.shape = shape
        self.dtype = dtype
        self.rank = rank

        # given by Eq. (2.1) in [3]
        alpha = 1 if dtype == np.float_ else 0
        # given by Eq. (4.6) in [3]
        k = 2 * rank + alpha
        l = 2 * k + alpha
        self.Omega = em.standard_normal((shape[1], k), rgen=rgen, dtype=dtype)
        self.Psi = em.standard_normal((l, shape[0]), rgen=rgen, dtype=dtype)
        self.Y = np.zeros((shape[0], k), dtype=dtype)
        self.W = np.zeros((l, shape[1]), dtype=dtype)

    @classmethod
    def from_full(cls, A, rank, rgen=np.random):
        """@todo: Docstring for from_full.

        """
        sketch = cls(A.shape, rank, rgen=rgen, dtype=A.dtype)
        sketch.Y[:] = A.dot(sketch.Omega)
        sketch.W[:] = sketch.Psi.dot(A)
        return sketch

    @property
    def factorization(self):
        Q, _ = np.linalg.qr(self.Y)
        X = em._llsq_solve_fast(self.Psi.dot(Q), self.W)
        return Q, X

    def to_full(self):
        """@todo: Docstring for recons.

        """
        Q, X = self.factorization
        return Q.dot(X)

    def _matmat(self, A):
        """Computes the matrix-matrix product product self * A

        :param A: @todo
        :returns: @todo

        """
        Q, X = self.factorization
        return Q.dot(X.dot(A))

    def __rmul__(self, A):
        if np.isscalar(A):
            result = copy(self)
            result.W = A * result.W
            return result
        else:
            Q, X = self.factorization
            return A.dot(Q).dot(X)

    def __imul__(self, c):
        assert np.isscalar(c)
        self.W *= c
        return self

