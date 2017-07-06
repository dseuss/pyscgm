import numpy as np

from . import extmath as em


class LRSketch(object):
    """Sketch of a low-rank hermitian matrix"""

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

    def to_full(self):
        """@todo: Docstring for recons.

        """
        Q, X = self.factorization
        return Q.dot(X)

    @property
    def factorization(self):
        Q, _ = np.linalg.qr(self.Y)
        X = em._llsq_solve_fast(self.Psi.dot(Q), self.W)
        return Q, X
