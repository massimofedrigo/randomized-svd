import numpy as np


def rsvd(X: np.ndarray, t: int, p: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Randomized Singular Value Decomposition of a general matrix.

    This function acts as a smart wrapper that automatically selects the optimal
    computational strategy based on the matrix shape (tall-and-skinny vs
    short-and-fat) to minimize memory usage and floating-point operations.

    Parameters
    ----------
    X : np.ndarray
        The input matrix of shape (m, n).
    t : int
        The target rank (projection dimension).
        Must be an integer satisfying 1 <= t <= min(m, n).
    p : int, optional
        Number of power iterations (default is 0).
        Increasing this value improves the accuracy of the approximation
        when the singular values decay slowly, at the cost of extra computation.
        Common values are 1 or 2.

    Returns
    -------
    U : np.ndarray
        Unitary left singular vectors of shape (m, t).
    S : np.ndarray
        Diagonal matrix of singular values of shape (t, t).
    Vt : np.ndarray
        Unitary right singular vectors (transposed) of shape (t, n).

    Raises
    ------
    TypeError
        If parameter t or p is not an integer.
    ValueError
        If t is out of the valid bounds [1, min(m, n)] or p < 0.

    References
    ----------
    .. [1] Brunton, S. L., & Kutz, J. N. (2019). Data-Driven Science and
           Engineering: Machine Learning, Dynamical Systems, and Control.
           Cambridge University Press, USA, 1st Edition.
    .. [2] Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure
           with randomness: Probabilistic algorithms for constructing approximate
           matrix decompositions. SIAM review.
    """
    m, n = X.shape

    # 1. Type Validation
    if not isinstance(t, int):
        raise TypeError(f"Parameter t must be an integer, got {type(t).__name__}.")
    if not isinstance(p, int):
        raise TypeError(f"Parameter p must be an integer, got {type(p).__name__}.")

    # 2. Value Validation
    if t < 1 or t > min(m, n):
        raise ValueError(
            f"Parameter t={t} must be between 1 and min(m, n)={min(m, n)}."
        )
    if p < 0:
        raise ValueError(f"Parameter p must be non-negative, got {p}.")

    # 3. Dispatching Strategy
    if m >= n:
        # Optimization for Tall & Skinny matrices
        return _rsvd_tall(X, t, p)
    else:
        # Optimization for Short & Fat matrices
        return _rsvd_wide(X, t, p)


def _rsvd_tall(X: np.ndarray, t: int, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implementation of Randomized SVD for square or tall-and-skinny matrices (m >= n).
    Includes Power Iterations for improved accuracy.
    """
    m, n = X.shape

    # 1. Random Projection
    # Generate random test matrix P (n x t)
    P = np.random.randn(n, t)

    # Sketch the column space of X
    Z = X @ P  # (m x t)

    # 2. Power Iterations (Randomized Subspace Iteration)
    # This step enhances the approximation accuracy for slowly decaying spectra.
    # We apply the power scheme: Z = (X X.T)^p Z
    # We include QR decomposition at each step to maintain numerical stability
    # (orthogonality), as per Halko et al. (2011), Algo 4.4.
    for _ in range(p):
        # Move to the row space and orthogonalize
        Z, _ = np.linalg.qr(X.T @ Z, mode='reduced')
        # Move back to column space and orthogonalize
        Z, _ = np.linalg.qr(X @ Z, mode='reduced')

    # 3. QR Decomposition (Final orthonormal basis)
    # Form an orthonormal basis Q for the range of Z
    Q, _ = np.linalg.qr(Z, mode='reduced')  # Q is (m x t)

    # 4. Orthogonal Projection
    # Project X into the low-rank subspace defined by Q
    Y = Q.T @ X  # (t x n)

    # 5. Deterministic SVD on small matrix
    # Uy is (t x t), s is (t,), Vt is (t x n)
    Uy, s, Vt = np.linalg.svd(Y, full_matrices=False)
    S = np.diag(s)

    # 6. Reconstruction
    # Lift the left singular vectors back to the original space
    U = Q @ Uy  # (m x t)

    return U, S, Vt


def _rsvd_wide(X: np.ndarray, t: int, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implementation of Randomized SVD for short-and-fat matrices (m < n).
    """
    # Compute Randomized SVD on the transpose (which is tall-and-skinny)
    # Pass 'p' recursively to the underlying implementation
    U_trans, S, Vt_trans = _rsvd_tall(X.T, t, p)

    # Map results back to original dimensions
    U = Vt_trans.T
    Vt = U_trans.T

    return U, S, Vt
