import numpy as np
import scipy.sparse.linalg as la
from scipy.sparse import random


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


def random_lowrank(rows, cols, rank, rgen=np.random, dtype=np.float_):
    """Returns a random lowrank matrix of given shape and dtype"""
    if dtype == np.float_:
        A = rgen.randn(rows, rank)
        B = rgen.randn(cols, rank)
    elif dtype == np.complex_:
        A = rgen.randn(rows, rank) + 1.j * rgen.randn(rows, rank)
        B = rgen.randn(cols, rank) + 1.j * rgen.randn(cols, rank)
    else:
        raise ValueError("{} is not a valid dtype".format(dtype))

    C = A.dot(B.conj().T)
    return C / np.linalg.norm(C)


def random_fullrank(rows, cols, **kwargs):
    """Returns a random matrix of given shape and dtype. Should provide
    same interface as random_lowrank"""
    kwargs.pop('rank', None)
    return random_lowrank(rows, cols, min(rows, cols), **kwargs)


def random_lowrankh(rows, rank, rgen=np.random, dtype=np.float_, psd=False):
    """Returns a random hermitian lowrank matrix of given shape and dtype"""
    if dtype == np.float_:
        A = rgen.randn(rows, rank)
    elif dtype == np.complex_:
        A = rgen.randn(rows, rank) + 1.j * rgen.randn(rows, rank)
    else:
        raise ValueError("{} is not a valid dtype".format(dtype))

    if psd:
        C = A.dot(A.conj().T)
    else:
        signs = rgen.choice([-1, 1], size=rank)
        C = (A * signs).dot(A.conj().T)
    return C / np.linalg.norm(C)


def random_fullrankh(rows, **kwargs):
    kwargs.pop('rank', None)
    return random_lowrankh(rows, rows, **kwargs)


def random_sparse(rows, cols, rank, rgen=np.random, dtype=np.float_,
                  format='csr'):
    if dtype == np.float_:
        return random(rows, cols, rank / min(rows, cols), format=format,
                    dtype=dtype, data_rvs=rgen.randn)
    elif dtype == np.complex_:
        return random_sparse(rows, cols, rank, rgen=rgen) + \
            1.j * random_sparse(rows, cols, rank, rgen=rgen)
    else:
        raise ValueError("{} is not a valid dtype".format(dtype))


def random_sparseh(rows, rank, rgen=np.random, dtype=np.float_, format='csr'):
    A = random_sparse(rows, rows, rank, rgen=rgen, dtype=dtype, format=format)
    return A + A.H


def normalize_svec(U):
    """Normalizes the singular vectors U to normal form, such that the first
    component of each singular vector has argument 0."""
    if np.isrealobj(U):
        return U / np.sign(U[0])
    else:
        phases = np.exp(1.j * np.angle(U[-1]))
        return U / phases


def lowrank_approx(A, rank):
    # compute the best rank r approximation to A
    U, sigma, V = svds(A, k=rank)
    return (U * sigma).dot(V)
