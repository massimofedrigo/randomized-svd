import numpy as np

def rsvd(X: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        If parameter t is not an integer.
    ValueError
        If t is out of the valid bounds [1, min(m, n)].

    References
    ----------
    .. [1] Brunton, S. L., & Kutz, J. N. (2019). Data-Driven Science and
           Engineering: Machine Learning, Dynamical Systems, and Control.
           Cambridge University Press, USA, 1st Edition.
    """
    m, n = X.shape

    # 1. Type Validation
    if not isinstance(t, int):
        raise TypeError(f"Parameter t must be an integer, got {type(t).__name__}.")

    # 2. Value Validation
    if t < 1 or t > min(m, n):
        raise ValueError(
            f"Parameter t={t} must be between 1 and min(m, n)={min(m, n)}."
        )

    # 3. Dispatching Strategy
    if m >= n:
        # Optimization for Tall & Skinny matrices
        return _rsvd_tall(X, t)
    else:
        # Optimization for Short & Fat matrices
        return _rsvd_wide(X, t)


def _rsvd_tall(X: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implementation of Randomized SVD for square or tall-and-skinny matrices (m >= n).

    Algorithm Steps:
    1. Generate a Gaussian random projection matrix P.
    2. Project X into a lower-dimensional subspace Z = X @ P.
    3. Orthogonalize Z using QR decomposition to obtain Q.
    4. Project X onto the orthogonal basis Q to obtain Y = Q.T @ X.
    5. Compute deterministic SVD on the small matrix Y.
    6. Reconstruct high-dimensional singular vectors U.

    Parameters
    ----------
    X : np.ndarray
        Input matrix of shape (m, n) where m >= n.
    t : int
        Target rank.

    Returns
    -------
    U, S, Vt : tuple[np.ndarray, np.ndarray, np.ndarray]
        The SVD components.
    """
    m, n = X.shape

    # 1. Random Projection
    # Generate random test matrix P (n x t)
    P = np.random.randn(n, t)
    # Sketch the column space of X
    Z = X @ P  # (m x t)

    # 2. QR Decomposition
    # Form an orthonormal basis Q for the range of Z
    Q, _ = np.linalg.qr(Z, mode='reduced')  # Q is (m x t)

    # 3. Orthogonal Projection
    # Project X into the low-rank subspace defined by Q
    Y = Q.T @ X  # (t x n)

    # 4. Deterministic SVD on small matrix
    # Uy is (t x t), s is (t,), Vt is (t x n)
    Uy, s, Vt = np.linalg.svd(Y, full_matrices=False)
    S = np.diag(s)

    # 5. Reconstruction
    # Lift the left singular vectors back to the original space
    U = Q @ Uy  # (m x t)

    return U, S, Vt


def _rsvd_wide(X: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implementation of Randomized SVD for short-and-fat matrices (m < n).

    This method avoids creating a large projection matrix by exploiting the
    transpose property of SVD. It computes the SVD of X.T (which is tall)
    and maps the results back to X.

    Mathematical derivation:
        If X = U * S * Vt
        Then X.T = V * S * Ut
        Let SVD(X.T) = U_hat * S_hat * Vt_hat
        Mapping: U = Vt_hat.T, S = S_hat, Vt = U_hat.T

    Parameters
    ----------
    X : np.ndarray
        Input matrix of shape (m, n) where m < n.
    t : int
        Target rank.

    Returns
    -------
    U, S, Vt : tuple[np.ndarray, np.ndarray, np.ndarray]
        The SVD components mapped back to the original orientation.
    """
    # Compute Randomized SVD on the transpose (which is tall-and-skinny)
    # U_trans corresponds to V of original X
    # Vt_trans corresponds to U.T of original X
    U_trans, S, Vt_trans = _rsvd_tall(X.T, t)

    # Map results back to original dimensions
    U = Vt_trans.T
    Vt = U_trans.T

    return U, S, Vt
