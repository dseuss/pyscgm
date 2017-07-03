from functools import partial

import numpy as np
import pyscgm.extmath as m
import pytest as pt
from numpy.testing import assert_allclose, assert_array_equal

from . import _utils as u


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
@pt.mark.parametrize('n_iter, target_gen', [(7, u.random_lowrank),
                                            (20, u.random_fullrank)])
def test_randomized_svd(rows, cols, rank, dtype, transpose, n_iter, target_gen,
                        rgen):
    # -2 due to the limiations of scipy.sparse.linalg.svds
    rank = min(rows, cols) - 2 if rank is 'fullrank' else rank
    A = target_gen(rows, cols, rank=rank, rgen=rgen, dtype=dtype)

    U_ref, s_ref, V_ref = u.svds(A, k=rank, which='LM')
    U, s, V = m.svds(A, rank, transpose=transpose, rgen=rgen, n_iter=n_iter)
    # since singular vectors are only determined up to a phase
    U, U_ref, Vt, V_reft = map(u.normalize_svec, (U, U_ref, V.T, V_ref.T))

    assert_allclose(np.linalg.norm(U - U_ref, axis=0), 0, atol=1e-3)
    assert_allclose(np.linalg.norm(Vt.T - V_reft.T, axis=0), 0, atol=1e-3)
    assert_allclose(s.ravel() - s_ref, 0, atol=1e-3)
    # Check that singular values are returned in ascending order
    assert_array_equal(s, np.sort(s))


EIGSH_GENERATORS = [(7, partial(u.random_lowrankh, psd=True)),
                    (7, partial(u.random_lowrankh, psd=False)),
                    (20, partial(u.random_fullrankh, psd=True)),
                    (20, partial(u.random_fullrankh, psd=False))]


@pt.mark.parametrize('rows, _', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', pt.TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.DTYPES)
@pt.mark.parametrize('n_iter, target_gen', EIGSH_GENERATORS)
def test_randomized_eigsh_direct(rows, _, rank, dtype, n_iter, target_gen, rgen):
    # -2 due to the limiations of scipy.sparse.linalg.eigsh
    rank = rows - 2 if rank is 'fullrank' else rank
    A = target_gen(rows, rank=rank, rgen=rgen, dtype=dtype)
    vals_ref, vecs_ref = u.eigsh(A, k=rank, which='LM')
    vals, vecs = m.eigsh(A, rank, method='direct', n_iter=n_iter, rgen=rgen)

    vecs_ref, vecs = map(u.normalize_svec, (vecs_ref, vecs))

    assert_allclose(np.linalg.norm(vecs - vecs_ref, axis=0), 0, atol=1e-3)
    assert_allclose(vals.ravel() - vals_ref, 0, atol=1e-3)
    # Check that eigenvalues are returned in ascening order
    assert_array_equal(vals, np.sort(vals))
