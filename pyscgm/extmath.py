from warnings import warn

import numpy as np
from scipy import linalg
from scipy.linalg.blas import dgemm
from scipy.sparse.linalg import aslinearoperator


def standard_normal(shape, rgen=np.random, dtype=np.float_):
    """Generates a standard normal numpy array of given shape and dtype, i.e.
    this function is equivalent to `rgen.randn(*shape)` for real dtype and
    `rgen.randn(*shape) + 1.j * rgen.randn(shape)` for complex dtype.

    Parameters
    ----------

    shape: tuple

    rgen: RandomState (default is `np.random`))

    dtype: `np.float_` (default) or `np.complex_`

    Returns
    -------

    A: An array of given shape and dtype with standard normal entries

    """
    if dtype == np.float_:
        return rgen.randn(*shape)
    elif dtype == np.complex_:
        return rgen.randn(*shape) + 1.j * rgen.randn(*shape)
    else:
        raise ValueError('{} is not a valid dtype.'.format(dtype))


def _llsq_solve_fast(A, y):
    """ Return the least-squares solution to a linear matrix equation.
    Solves the equation `A x = y` by computing a vector `x` that
    minimizes the Euclidean 2-norm `|| b - a x ||^2`.  The equation may
    be under-, well-, or over- determined (i.e., the number of
    linearly independent rows of `A` can be less than, equal to, or
    greater than its number of linearly independent columns).  If `A`
    is square and of full rank, then `x` (but for round-off error) is
    the "exact" solution of the equation.

    However, if A is rank-deficient, this solver may fail. In that case, use
    :func:`_llsq_solver_pinv`.

    :param A: (m, d) array like
    :param y: (m,) array_like
    :returns x: (d,) ndarray, least square solution

    """
    Aadj = A.T.conj()
    return np.linalg.solve(Aadj.dot(A), Aadj.dot(y))


def _llsq_solve_pinv(A, y):
    """Same as :func:`llsq_solve_fast` but more robust, albeit slower.

    :param A: (m, d) array like
    :param y: (m,) array_like
    :returns x: (d,) ndarray, least square solution

    """
    B = np.linalg.pinv(A.T @  A)
    return B @ A.T @ y
