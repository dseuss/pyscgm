import numpy as np
from scipy import linalg
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
