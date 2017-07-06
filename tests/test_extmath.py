import numpy as np
import pyscgm.extmath as em
import pytest as pt
from numpy.testing import assert_allclose


@pt.mark.parametrize('rows, cols', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('n_vecs', [1, 5])
@pt.mark.parametrize('dtype', pt.DTYPES)
@pt.mark.parametrize('llsq_solve', [em._llsq_solve_fast, em._llsq_solve_pinv])
def test_llsq_solve(rows, cols, n_vecs, dtype, llsq_solve, rgen):
    A = em.standard_normal((rows, cols), rgen=rgen, dtype=dtype)
    x_shape = (cols, n_vecs) if n_vecs > 1 else (cols,)
    x = em.standard_normal(x_shape, rgen=rgen, dtype=dtype)
    x_hat = llsq_solve(A, A.dot(x))

    if rows < cols:
        # underdetermined sysyem -> ill-posed -> check that observations match
        assert_allclose(A.dot(x), A.dot(x_hat), atol=1e-6)
    else:
        assert_allclose(x, x_hat, atol=1e-6)
