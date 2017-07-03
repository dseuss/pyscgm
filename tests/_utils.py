import numpy as np
import scipy.sparse.linalg as la


def svds(*args, **kwargs):
    """Thin wrapper around `scipy.sparse.linalg.svds` guaranteeting ascending
    order of singular values
    """
    u, s, v = la.svds(*args, **kwargs)
    i = np.argsort(s)
    return u[:, i], s[i], v[i, :]


def eigsh(*args, **kwargs):
    """Thin wrapper around `scipy.sparse.linalg.eigsh` guaranteeting ascending
    order of eigenvalues
    """
    vals, vecs = la.eigsh(*args, **kwargs)
    i = np.argsort(vals)
    return vals[i], vecs[:, i]


def random_lowrank(rows, cols, rank, rgen, dtype):
    """Returns a random lowrank matrix of given shape and dtype"""
    if dtype == np.float_:
        A = rgen.randn(rows, rank)
        B = rgen.randn(cols, rank)
    elif dtype == np.complex_:
        A = rgen.randn(rows, rank) + 1.j * rgen.randn(rows, rank)
        B = rgen.randn(cols, rank) + 1.j * rgen.randn(cols, rank)
    else:
        raise ValueError("{} is not a valid dtype".format(dtype))

    return A.dot(B.conj().T)


def random_lowrankh(rows, rank, rgen=np.random, dtype=np.float_, psd=False):
    """Returns a random hermitian lowrank matrix of given shape and dtype"""
    if dtype == np.float_:
        A = rgen.randn(rows, rank)
    elif dtype == np.complex_:
        A = rgen.randn(rows, rank) + 1.j * rgen.randn(rows, rank)
    else:
        raise ValueError("{} is not a valid dtype".format(dtype))

    if psd:
        return A.dot(A.conj().T)
    else:
        signs = rgen.choice([-1, 1], size=rank)
        return (A * signs).dot(A.conj().T)


def normalize_svec(U):
    """Normalizes the singular vectors U to normal form, such that the first
    component of each singular vector has argument 0."""
    if np.isrealobj(U):
        return U / np.sign(U[0])
    else:
        phases = np.exp(1.j * np.angle(U[-1]))
        return U / phases


