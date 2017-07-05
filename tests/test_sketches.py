import functools as ft

import numpy as np
import pyscgm.cgm as m
import pytest as pt

from . import _utils as u


@pt.mark.parametrize('rows, cols', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', pt.TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.DTYPES)
@pt.mark.parametrize('target_gen', [u.random_lowrank, u.random_fullrank])
def test_sketch_lowrank(rows, cols, rank, dtype, target_gen, rgen):
    # -2 due to the limiations of scipy.sparse.linalg.svds
    rank = min(rows, cols) - 2 if rank is 'fullrank' else rank
    A = target_gen(rows, cols, rank=rank, rgen=rgen, dtype=dtype)

    # compute the sketch with given rank approximation
    A_sketch = m.LRSketch(A, rank)

    # compute the best rank r approximation to A
    U, sigma, V = u.svds(A, k=rank)
    A_r = (U * sigma).dot(V.conj().T)

    # should be true in expectation according to Theorem 4.1 of [3] with the
    # choice of Eq. (4.6)
    norm = ft.partial(np.linalg.norm, ord='fro')
    assert (norm(A - A_sketch.recons) <= 2 * norm(A - A_r))
