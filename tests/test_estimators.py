import numpy as np
from pyscgm import estimators as est
import pytest as pt
from numpy.testing import assert_allclose

from . import _utils as u


def generate_setting(dim, rank, rgen):
    X = u.random_lowrankh(dim, rank, rgen=rgen, psd=True)
    A = rgen.randn(4 * dim * rank, dim, dim)
    y = A.reshape((len(A), -1)) @ X.ravel()
    return X, A, y


@pt.mark.skip
def test_scgm_estimator(estimator, rgen):
    X, A, y = generate_setting(30, 3, rgen)
    X_sharp = est.LRSketch(A, y)
    assert_allclose(np.linalg.norm(X - X_sharp), 0, atol=1e-2)


@pt.mark.parametrize('rows, cols', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', pt.TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.DTYPES)
@pt.mark.parametrize('eta', [0, 1, 0.5])
def test_cgmsketch_update(rows, cols, rank, dtype, eta, rgen):
    # -2 due to the limiations of scipy.sparse.linalg.svds
    rank = min(rows, cols) - 2 if rank is 'fullrank' else rank
    A = u.random_lowrank(rows, cols, rank=rank, rgen=rgen, dtype=dtype)
    B = u.random_lowrank(rows, cols, rank=1, dtype=dtype, rgen=rgen)

    A_sketch = est.CGMSketch.from_full(A, rank + 1, rgen=rgen)
    U, alpha, V = u.svds(B, 1)
    A_sketch.cgm_update(eta, U, V, alpha)
    A_ref = (1 - eta) * A + eta * B

    assert (np.linalg.norm(A_ref - A_sketch.to_full()) <= 1e-4)
