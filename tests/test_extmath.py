import numpy as np
import pytest as pt
from numpy.testing import assert_allclose, assert_array_equal

from . import _utils as u
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

    A = u.random_lowrank(rows, cols, rank, rgen=rgen, dtype=dtype)
    A /= np.linalg.norm(A, ord='fro')
    Q = m.approx_range_finder(A, rf_size, 7, rgen=rgen,
                              piter_normalizer=piter_normalizer)

    Q = np.asmatrix(Q)
    assert Q.shape == (rows, rf_size)
    normdist = np.linalg.norm(A - Q * (Q.H * A), ord='fro')
    assert normdist < 1e-7


@pt.mark.parametrize('rows, cols', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', pt.TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.DTYPES)
@pt.mark.parametrize('transpose', [False, True, 'auto'])
def test_randomized_svd(rows, cols, rank, dtype, transpose, rgen):
    rank = min(rows, cols) if rank is 'fullrank' else rank
    A = u.random_lowrank(rows, cols, rank, rgen, dtype)
    U_ref, s_ref, V_ref = u.truncated_svd(A, rank)
    U, s, V = m.randomized_svd(A, rank, transpose=transpose, rgen=rgen)
    # since singular vectors are only determined up to a phase
    U, U_ref, V, V_ref = map(u.normalize_svec, (U, U_ref, V, V_ref))

    assert_allclose(np.linalg.norm(U - U_ref, axis=0), 0, atol=1e-3)
    assert_allclose(np.linalg.norm(V - V_ref, axis=0), 0, atol=1e-3)
    assert_allclose(s.ravel() - s_ref, 0, atol=1e-3)
    # Check that singular values are returned in descending order
    assert_array_equal(s, np.sort(s)[::-1])



@pt.mark.parametrize('rows, _', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', pt.TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.DTYPES)
@pt.mark.parametrize('psd', [True, False])
@pt.mark.parametrize('method', ['direct'])
def test_randomized_eigh(rows, _, rank, dtype, psd, method, rgen):
    # Nystrom is only expected to work for psd matrices
    #  if method is 'nystrom' and (not psd or rank is not 'fullrank'):
    #      return

    rank = rows if rank is 'fullrank' else rank
    A = u.random_lowrankh(rows, rank, rgen, dtype, psd=psd)
    vals_ref, vecs_ref = u.truncated_eigh(A, rank)
    vals, vecs = m.randomized_eigh(A, rank, method=method, rgen=rgen)

    vecs_ref, vecs = map(u.normalize_svec, (vecs_ref, vecs))

    assert_allclose(np.linalg.norm(vecs - vecs_ref, axis=0), 0, atol=1e-3)
    assert_allclose(vals.ravel() - vals_ref, 0, atol=1e-3)
    # Check that eigenvalues are returned in ascening order
    assert_array_equal(vals, np.sort(vals))
