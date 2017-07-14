import numpy as np
import pytest as pt
from numpy.testing import assert_allclose
from pyscgm.measurements import Rank1MeasurmentMap
from pyscgm.extmath import standard_normal

from . import _utils as u


@pt.mark.parametrize('rows, cols', pt.TESTARGS_MATRIXDIMS)
@pt.mark.parametrize('dtype', pt.DTYPES)
@pt.mark.parametrize('test_rounds', [100])
def test_sketch_lowrank(rows, cols, dtype, test_rounds, rgen):
    A = u.random_lowrank(rows, cols, rank=1, rgen=rgen, dtype=dtype)
    # Recall that random_lowrank returns normalized matrix
    U, _, V = u.svds(A, 1)
    measurement_map = Rank1MeasurmentMap(4 * min(rows, cols), (rows, cols))
    map_dense = measurement_map.left_measurements[:, :, None] \
        * measurement_map.right_measurements[:, None, :] \
        / np.sqrt(measurement_map.nr_measurements)
    outcomes = measurement_map(U, V)
    outcomes_ref = map_dense.reshape((-1, rows * cols)).dot(A.ravel())
    assert_allclose(outcomes, outcomes_ref)

    outcomes = measurement_map(A)
    assert_allclose(outcomes, outcomes_ref)

    A_recons = measurement_map.H(outcomes)
    A_recons_ref = np.tensordot(outcomes, map_dense, axes=(0, 0))
    for _ in range(test_rounds):
        x = standard_normal((cols,), dtype=dtype, rgen=rgen)
        assert_allclose(A_recons * x, A_recons_ref.dot(x))
    for _ in range(test_rounds):
        x = standard_normal((rows,), dtype=dtype, rgen=rgen)
        assert_allclose(A_recons.H * x, A_recons_ref.conj().T.dot(x))
