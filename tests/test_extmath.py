import numpy as np
import pytest as pt
from numpy.testing import assert_allclose

from ._utils import random_lowrank, truncated_svd, normalize_svec
import pyscgm.extmath as m


@pt.mark.parametrize('rows, cols', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', pt.TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.DTYPES)
@pt.mark.parametrize('piter_normalizer', pt.PITER_NORMALIZERS)
def test_approximate_range_finder(rows, cols, rank, dtype, piter_normalizer, rgen):
    # only guaranteed to work for low-rank matrices
    if rank is 'fullrank':
        return

    rf_size = rank + 10
    assert min(rows, cols) > rf_size

    A = random_lowrank(rows, cols, rank, rgen=rgen, dtype=dtype)
    A /= np.linalg.norm(A, ord='fro')
    Q = m.approx_range_finder(A, rf_size, 7, rgen=rgen,
                              piter_normalizer=piter_normalizer)

    assert Q.shape == (rows, rf_size)
    normdist = np.linalg.norm(A - Q * (Q.H * A), ord='fro')
    assert normdist < 1e-7


@pt.mark.parametrize('rows, cols', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', pt.TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.DTYPES)
@pt.mark.parametrize('transpose', [False, True, 'auto'])
def test_randomized_svd(rows, cols, rank, dtype, transpose, rgen):
    rank = min(rows, cols) if rank is 'fullrank' else rank
    A = random_lowrank(rows, cols, rank, rgen, dtype)
    U_ref, s_ref, V_ref = truncated_svd(A, rank)
    U, s, V = m.randomized_svd(A, rank, transpose=transpose, rgen=rgen,
                               n_iter=10)
    # since singular vectors are only determined up to a phase
    U, U_ref, V, V_ref = map(normalize_svec, (U, U_ref, V, V_ref))

    assert_allclose(np.linalg.norm(U - U_ref, axis=0), 0, atol=1e-3)
    assert_allclose(np.linalg.norm(V - V_ref, axis=0), 0, atol=1e-3)
    assert_allclose(s.ravel() - s_ref, 0, atol=1e-3)
