import numpy as np
import pyscgm.cgm as m
import pytest as pt
from numpy.testing import assert_allclose, assert_array_equal

from . import _utils as u


def generate_setting(dim, rank, rgen):
    X = u.random_lowrankh(dim, rank, rgen=rgen, psd=True)
    A = rgen.randn(4 * dim * rank, dim, dim)
    y = A.reshape((len(A), -1)) @ X.ravel()
    return X, A, y


@pt.mark.parametrize('estimator', [m.scgm_estimator])
def test_scgm_estimator(estimator, rgen):
    X, A, y = generate_setting(30, 3, rgen)
    X_sharp = estimator(A, y)
    assert_allclose(np.linalg.norm(X - X_sharp), 0, atol=1e-2)
