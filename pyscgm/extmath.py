import numpy as np
from scipy import linalg


def approx_range_finder(A, size, n_iter, piter_normalizer='auto',
                        rgen=np.random):
    """Computes an orthonormal matrix whose range approximates the range of A.

    Parameters
    ----------
    A: 2D array
        The input data matrix

    size: integer
        Size of the return array

    n_iter: integer
        Number of power iterations used to stabilize the result

    piter_normalizer: 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter`<=2 and switches to LU otherwise.

    rgen: RandomState
        A random number generator instance

    Returns
    -------
    Q: numpy matrix
        A (A.shape[0] x size) projection matrix, the range of which
        approximates well the range of the input matrix A.

    Notes
    -----

    Follows Algorithm 4.3/4.4 of
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) http://arxiv.org/pdf/0909.4061

    An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014

    Original implementation from scikit-learn.

    """
    A = np.asmatrix(A)

    # Generating normal random vectors with shape: (A.shape[1], size)
    # note that real normal vectors are sufficient, no complex values necessary
    Q = rgen.randn(A.shape[1], size)

    # Deal with "auto" mode
    if piter_normalizer == 'auto':
        if n_iter <= 2:
            piter_normalizer = 'none'
        else:
            piter_normalizer = 'LU'

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for i in range(n_iter):
        if piter_normalizer == 'none':
            Q = A * Q
            Q = A.H * Q
        elif piter_normalizer == 'LU':
            Q, _ = linalg.lu(A * Q, permute_l=True)
            Q, _ = linalg.lu(A.H * Q, permute_l=True)
        elif piter_normalizer == 'QR':
            Q, _ = linalg.qr(A * Q, mode='economic')
            Q, _ = linalg.qr(A.H * Q, mode='economic')

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = linalg.qr(A * Q, mode='economic')
    return np.asmatrix(Q)
