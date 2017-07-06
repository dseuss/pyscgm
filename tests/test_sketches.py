import functools as ft

import numpy as np
import pytest as pt
from numpy.testing import assert_allclose
from pyscgm.sketches import LRSketch

from . import _utils as u


@pt.mark.parametrize('rows, cols', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', pt.TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.DTYPES)
@pt.mark.parametrize('target_gen', [u.random_lowrank, u.random_fullrank])
def test_sketch_lowrank(rows, cols, rank, dtype, target_gen, rgen):
    # -2 due to the limiations of scipy.sparse.linalg.svds
    rank = min(rows, cols) - 2 if rank is 'fullrank' else rank
    A = target_gen(rows, cols, rank=rank, rgen=rgen, dtype=dtype)
    A_r = u.lowrank_approx(A, rank)

    # compute the sketch with given rank approximation
    A_sketch = LRSketch.from_full(A, rank)

    # should be true in expectation according to Theorem 4.1 of [3] with the
    # choice of Eq. (4.6)
    # Also + 1e-10 for cases where the r.h.s. should be 0
    norm = ft.partial(np.linalg.norm, ord='fro')
    assert (norm(A - A_sketch.to_full()) <= 2 * norm(A - A_r) + 1e-10)


@pt.mark.parametrize('rows, cols', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', pt.TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.DTYPES)
def test_sketch_lowrank_matmul(rows, cols, rank, dtype, rgen):
    rank = min(rows, cols) if rank is 'fullrank' else rank
    A = u.random_lowrank(rows, cols, rank=rank, rgen=rgen, dtype=dtype)
    B = u.random_fullrank(cols, 1, rgen=rgen, dtype=dtype).ravel()

    # compute the sketch with given rank approximation
    A_sketch = LRSketch.from_full(A, rank)

    assert_allclose(A_sketch * B, A.dot(B))
