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
    # Enlarge test matrices due to oversampling
    rows, cols = 10 * rows, 10 * cols
    rf_size = rank + 10
    assert min(rows, cols) > rf_size

    A = random_lowrank(rows, cols, rank, rgen=rgen, dtype=dtype)
    A /= np.linalg.norm(A, ord='fro')
    Q = m.approx_range_finder(A, rf_size, 7, rgen=rgen,
                              piter_normalizer=piter_normalizer)

    assert Q.shape == (rows, rf_size)
    normdist = np.linalg.norm(A - Q * (Q.H * A), ord='fro')
    assert normdist < 1e-7
