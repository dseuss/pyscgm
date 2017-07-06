import functools as ft

import numpy as np
import pytest as pt
from numpy.testing import assert_allclose
from pyscgm.sketches import LRSketch
from pyscgm.extmath import standard_normal

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

    # should be true in expectation with constant 3 -> 2 according to
    # Theorem 4.1 of [3] with the choice of Eq. (4.6)
    # Also + 1e-10 for cases where the r.h.s. should be 0
    norm = ft.partial(np.linalg.norm, ord='fro')
    assert (norm(A - A_sketch.to_full()) <= 3 * norm(A - A_r) + 1e-10)


@pt.mark.parametrize('rows, cols', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', pt.TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.DTYPES)
def test_sketch_lowrank_from_fulls(rows, cols, rank, dtype, rgen):
    rank = min(rows, cols) if rank is 'fullrank' else rank
    A = u.random_lowrank(rows, cols, rank=rank, rgen=rgen, dtype=dtype)
    B = u.random_lowrank(rows, cols, rank=rank, rgen=rgen, dtype=dtype)

    A_sketch, B_sketch = LRSketch.from_fulls((A, B), rank)
    assert (np.linalg.norm(A - A_sketch.to_full())  <= 1e-5)
    assert (np.linalg.norm(B - B_sketch.to_full())  <= 1e-5)
    assert A_sketch.Omega is B_sketch.Omega
    assert A_sketch.Psi is B_sketch.Psi


@pt.mark.parametrize('rows, cols', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', pt.TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.DTYPES)
def test_sketch_lowrank_matmul(rows, cols, rank, dtype, rgen):
    rank = min(rows, cols) if rank is 'fullrank' else rank
    A = u.random_lowrank(rows, cols, rank=rank, rgen=rgen, dtype=dtype)
    A_sketch = LRSketch.from_full(A, rank)

    B = u.random_fullrank(cols, 1, rgen=rgen, dtype=dtype).ravel()
    assert_allclose(A_sketch * B, A.dot(B))

    B = u.random_fullrank(rows, 1, rgen=rgen, dtype=dtype).ravel()
    assert_allclose(B * A_sketch, B.dot(A))


@pt.mark.parametrize('rows, cols', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', pt.TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.DTYPES)
def test_sketch_lowrank_scalarmul(rows, cols, rank, dtype, rgen):
    rank = min(rows, cols) if rank is 'fullrank' else rank
    A = u.random_lowrank(rows, cols, rank=rank, rgen=rgen, dtype=dtype)
    c = standard_normal((1,), rgen=rgen, dtype=dtype)[0]
    A_sketch = LRSketch.from_full(A, rank)

    assert_allclose((c * A_sketch).to_full(), c * A, atol=1e-5)
    assert (np.linalg.norm(A - A_sketch.to_full()) <= 1e-5)
    A_sketch *= c
    assert_allclose(A_sketch.to_full(), c * A, atol=1e-5)


@pt.mark.parametrize('rows, cols', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', pt.TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.DTYPES)
def test_sketch_lowrank_add(rows, cols, rank, dtype, rgen):
    rank = min(rows, cols) if rank is 'fullrank' else rank
    A = u.random_lowrank(rows, cols, rank=rank, rgen=rgen, dtype=dtype)
    c = standard_normal((1,), rgen=rgen, dtype=dtype)[0]
    A_sketch = LRSketch.from_full(A, rank)

    assert_allclose((c * A_sketch).to_full(), c * A, atol=1e-5)
    assert (np.linalg.norm(A - A_sketch.to_full()) <= 1e-5)
    A_sketch *= c
    assert_allclose(A_sketch.to_full(), c * A, atol=1e-5)
