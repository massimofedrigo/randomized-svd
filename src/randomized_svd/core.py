import numpy as np


def rsvd(
    X: np.ndarray,
    t: int,
    p: int = 0,
    oversampling: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Randomized Singular Value Decomposition (rSVD) of a general matrix.

    This function uses a randomized algorithm to compute an approximate low-rank
    SVD of the input matrix X. It includes optional parameters for power iterations
    (to handle slowly decaying singular values) and oversampling (to improve
    approximation accuracy).

    It automatically dispatches to the optimal strategy for "Tall-and-Skinny"
    vs "Short-and-Fat" matrices.

    Parameters
    ----------
    X : np.ndarray
        The input matrix of shape (m, n).
    t : int
        The target rank (number of singular components to return).
        Must satisfy 1 <= t <= min(m, n).
    p : int, optional
        Number of power iterations (default is 0).
        Increasing this value improves accuracy for matrices with slowly decaying
        spectra, at the cost of additional computational time.
        Recommended values: 0, 1, or 2.
    oversampling : int, optional
        Additional sampling factors to use during projection (default is 10).
        The algorithm effectively computes a decomposition of rank (t + oversampling)
        internally to capture more variance, then truncates the result to rank t.
        Higher values increase accuracy but use slightly more memory.

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
        If t, p, or oversampling are not integers.
    ValueError
        If parameters are out of valid bounds (e.g. t <= 0, p < 0).

    References
    ----------
    .. [1] Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure
           with randomness: Probabilistic algorithms for constructing approximate
           matrix decompositions. SIAM review, 53(2), 217-288.
    .. [2] Brunton, S. L., & Kutz, J. N. (2019). Data-Driven Science and
           Engineering. Cambridge University Press.
    """
    m, n = X.shape

    # 1. Type Validation
    if not isinstance(t, int):
        raise TypeError(f"Parameter t must be an integer, got {type(t).__name__}.")
    if not isinstance(p, int):
        raise TypeError(f"Parameter p must be an integer, got {type(p).__name__}.")
    if not isinstance(oversampling, int):
        raise TypeError(f"Parameter oversampling must be an integer, got {type(oversampling).__name__}.")

    # 2. Value Validation
    if t < 1 or t > min(m, n):
        raise ValueError(
            f"Parameter t={t} must be between 1 and min(m, n)={min(m, n)}."
        )
    if p < 0:
        raise ValueError(f"Parameter p must be non-negative, got {p}.")
    if oversampling < 0:
        raise ValueError(f"Parameter oversampling must be non-negative, got {oversampling}.")

    # 3. Dispatching Strategy
    if m >= n:
        # Optimization for Tall & Skinny matrices
        return _rsvd_tall(X, t, p, oversampling)
    else:
        # Optimization for Short & Fat matrices
        return _rsvd_wide(X, t, p, oversampling)


def _rsvd_tall(
    X: np.ndarray,
    t: int,
    p: int,
    oversampling: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implementation of Randomized SVD for square or tall-and-skinny matrices (m >= n).

    Includes Power Iterations (p) and Oversampling logic.
    """
    m, n = X.shape

    # 1. Define Internal Rank (k)
    # We project onto a subspace of size k = t + oversampling.
    # We must ensure k does not exceed the matrix dimensions.
    k = min(t + oversampling, n, m)

    # 2. Random Projection
    # Generate random test matrix P (n x k)
    P = np.random.randn(n, k)

    # Sketch the column space of X
    Z = X @ P  # (m x k)

    # 3. Power Iterations (Randomized Subspace Iteration)
    # Z = (X X.T)^p Z
    # Stabilized with QR at each step (Halko et al., Algo 4.4)
    for _ in range(p):
        # Move to row space and orthogonalize
        Z, _ = np.linalg.qr(X.T @ Z, mode='reduced')
        # Move back to column space and orthogonalize
        Z, _ = np.linalg.qr(X @ Z, mode='reduced')

    # 4. QR Decomposition (Final orthonormal basis)
    # Form an orthonormal basis Q for the range of Z
    Q, _ = np.linalg.qr(Z, mode='reduced')  # Q is (m x k)

    # 5. Orthogonal Projection
    # Project X into the low-rank subspace defined by Q
    Y = Q.T @ X  # (k x n)

    # 6. Deterministic SVD on small matrix
    # Uy is (k x k), s is (k,), Vt is (k x n)
    Uy, s, Vt = np.linalg.svd(Y, full_matrices=False)

    # 7. Reconstruction
    # Lift the left singular vectors back to the original space
    U = Q @ Uy  # (m x k)

    # 8. Truncation
    # We computed 'k' components for accuracy, but the user requested 't'.
    # We discard the extra 'oversampling' components here.
    return U[:, :t], np.diag(s[:t]), Vt[:t, :]


def _rsvd_wide(
    X: np.ndarray,
    t: int,
    p: int,
    oversampling: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implementation of Randomized SVD for short-and-fat matrices (m < n).
    """
    # Compute Randomized SVD on the transpose (which is tall-and-skinny)
    # Pass 'p' and 'oversampling' recursively
    U_trans, S, Vt_trans = _rsvd_tall(X.T, t, p, oversampling)

    # Map results back to original dimensions
    # U of transpose -> Vt of original (transposed)
    # Vt of transpose -> U of original (transposed)
    U = Vt_trans.T
    Vt = U_trans.T

    return U, S, Vt
