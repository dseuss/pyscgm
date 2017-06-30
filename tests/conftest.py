import numpy as np
import pytest as pt


def pytest_namespace():
    return dict(
        DTYPES=[np.float_, np.complex_],
        TESTARGS_MATRIXDIMS=[(20, 20), (50, 20)],
        TESTARGS_RANKS=[1, 5],
        PITER_NORMALIZERS=[None, 'qr', 'lu', 'auto'],
    )


@pt.fixture(scope="module")
def rgen():
    return np.random.RandomState(seed=3476583865)
