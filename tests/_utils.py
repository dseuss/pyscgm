import numpy as np


def truncated_svd(A, k):
    """Compute the truncated SVD of the matrix `A` i.e. the `k` largest
    singular values as well as the corresponding singular vectors. It might
    return less singular values/vectors, if one dimension of `A` is smaller
    than `k`.

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


def truncated_eigh(A, k):
    """Compute the truncated eigenvalue decomposition  of the Hermitian matrix
    `A` i.e. the `k` eigenvalues, which have the largest magnitude, as well
    the corresponding eigenvectors.

    :param A: A real or complex Hermitian matrix
    :param k: Number of eigenvalues/vectors to compute
    :returns: vals, vecs

    """
    vals, vecs = np.linalg.eigh(A)
    # get the k largest elements by magintude, but keep the original order,
    # which respects the sign
    sel = np.sort(np.argsort(np.abs(vals))[-k:])
    return vals[sel], vecs[:, sel]


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


