import numpy as np


def truncated_svd(A, k):
    """Compute the truncated SVD of the matrix `A` i.e. the `k` largest
    singular values as well as the corresponding singular vectors. It might
    return less singular values/vectors, if one dimension of `A` is smaller
    than `k`.

    In the background it performs a full SVD. Therefore, it might be
    inefficient when `k` is much smaller than the dimensions of `A`.

    :param A: A real or complex matrix
    :param k: Number of singular values/vectors to compute
    :returns: u, s, v, where
        u: left-singular vectors
        s: singular values
        v: right-singular vectors

    """
    u, s, v = np.linalg.svd(A)
    k_prime = min(k, len(s))
    return u[:, :k_prime], s[:k_prime], v[:k_prime].conj().T


def random_lowrank(rows, cols, rank, rgen, dtype):
    """Returns a random lowrank matrix of given shape and dtype"""
    if dtype == np.float_:
        A = np.asmatrix(rgen.randn(rows, rank))
        B = np.asmatrix(rgen.randn(cols, rank))
    elif dtype == np.complex_:
        A = np.asmatrix(rgen.randn(rows, rank) + 1.j * rgen.randn(rows, rank))
        B = np.asmatrix(rgen.randn(cols, rank) + 1.j * rgen.randn(cols, rank))
    else:
        raise ValueError("{} is not a valid dtype".format(dtype))

    return A * B.H


def normalize_svec(U):
    """Normalizes the singular vectors U to normal form, such that the first
    component of each singular vector has argument 0."""
    if np.isreal(U.dtype):
        return U / np.sign(U[0])[None, :]
    else:
        phases = np.exp(1.j * np.argument(U[0]))
        return U / phases[None, :]


