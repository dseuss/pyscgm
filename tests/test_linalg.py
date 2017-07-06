from functools import partial

import numpy as np
import pyscgm.extmath as m
import pytest as pt
from numpy.testing import assert_allclose, assert_array_equal

from . import _utils as u


@pt.mark.parametrize('rows, cols', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('rank', pt.TESTARGS_RANKS)
@pt.mark.parametrize('dtype', pt.DTYPES)
@pt.mark.parametrize('transpose', [False, True, 'auto'])
@pt.mark.parametrize('n_iter, target_gen', [(7, u.random_lowrank),
                                            (20, u.random_fullrank),
                                            (20, u.random_sparse)])
def test_randomized_svd(rows, cols, rank, dtype, transpose, n_iter, target_gen,
                        rgen):
    # -2 due to the limiations of scipy.sparse.linalg.svds
    rank = min(rows, cols) - 2 if rank is 'fullrank' else rank
    A = target_gen(rows, cols, rank=rank, rgen=rgen, dtype=dtype)

    U_ref, s_ref, V_ref = u.svds(A, k=rank, which='LM')
    U, s, V = m.svds(A, rank, transpose=transpose, rgen=rgen, n_iter=n_iter)

    error_U = np.abs(U.conj().T.dot(U_ref)) - np.eye(rank)
    assert_allclose(np.linalg.norm(error_U), 0, atol=1e-3)
    error_V = np.abs(V.dot(V_ref.conj().T)) - np.eye(rank)
    assert_allclose(np.linalg.norm(error_V), 0, atol=1e-3)
    assert_allclose(s.ravel() - s_ref, 0, atol=1e-3)
    # Check that singular values are returned in ascending order
    assert_array_equal(s, np.sort(s))


EIGSH_GENERATORS = [(7, partial(u.random_lowrankh, psd=True)),
                    (7, partial(u.random_lowrankh, psd=False)),
                    (20, partial(u.random_fullrankh, psd=True)),
                    (20, partial(u.random_fullrankh, psd=False)),
                    (20, u.random_sparseh)]


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

    error_vecs = np.abs(vecs.conj().T.dot(vecs_ref)) - np.eye(rank)
    assert_allclose(np.linalg.norm(error_vecs), 0, atol=1e-3)
    assert_allclose(vals.ravel() - vals_ref, 0, atol=1e-3)
    # Check that eigenvalues are returned in ascening order
    assert_array_equal(vals, np.sort(vals))
