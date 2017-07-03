import numpy as np
import pytest as pt


def pytest_namespace():
    return dict(
        DTYPES=[np.float_, np.complex_],
        TESTARGS_MATRIXDIMS=[(50, 50), (100, 50)],
        TESTARGS_RANKS=[1, 10, 'fullrank'],
        PITER_NORMALIZERS=[None, 'qr', 'lu', 'auto'],
    )


@pt.fixture(scope="module")
def rgen():
    return np.random.RandomState(seed=3476583865)
