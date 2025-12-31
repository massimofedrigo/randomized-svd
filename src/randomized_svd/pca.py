import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from typing import Union, Tuple, cast
from .core import rsvd

ArrayOrSparse = Union[np.ndarray, sp.spmatrix]


class CenteredMatrix(LinearOperator):
    """
    A LinearOperator wrapper that performs 'Virtual Centering'.

    It represents the matrix (X - mu) without ever materializing it in memory.
    This is crucial for sparse matrices where X is sparse but (X - mu) would be dense.

    Math:
        Forward:  (X - mu) @ V  = (X @ V) - (mu @ V_col_sums)
        Backward: (X - mu).T @ U = X.T @ U - mu.T @ (U_row_sums)
    """
    def __init__(self, X: ArrayOrSparse, mean: np.ndarray):
        self.X = X
        self.mean_ = mean  # Shape (n,)
        self.dtype = X.dtype
        self.shape = X.shape

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        """Forward multiplication: y = (X - mu) @ v"""
        # 1. Standard multiplication X @ v
        # Works efficiently for both dense and sparse X
        y = self.X @ v

        # 2. Centering correction
        # scalar = dot(mean, v)
        # correction = scalar * ones_vector (broadcasted subtraction)
        correction = np.dot(self.mean_, v)

        return cast(np.ndarray, y - correction)

    def _matmat(self, V: np.ndarray) -> np.ndarray:
        """
        Forward matrix multiplication: Y = (X - mu) @ V.
        Optimized for block operations (used in rSVD).
        """
        # 1. Batch multiplication X @ V
        Y = self.X @ V

        # 2. Batch centering correction
        # We calculate the dot product of the mean vector with every column of V.
        # Result shape: (1, k)
        correction = self.mean_ @ V

        # Broadcast subtraction: subtract correction from every row of Y
        return cast(np.ndarray, Y - correction)

    def _rmatvec(self, u: np.ndarray) -> np.ndarray:
        """Transpose multiplication: y = (X - mu).T @ u"""
        # 1. Standard transpose multiplication X.T @ u
        y = self.X.T @ u

        # 2. Centering correction
        # sum_u = sum of elements in u
        # correction = sum_u * mean_vector
        correction = u.sum() * self.mean_

        return cast(np.ndarray, y - correction)

    def _rmatmat(self, U: np.ndarray) -> np.ndarray:
        """
        Transpose matrix multiplication: Y = (X - mu).T @ U.
        Math: X.T @ U - outer(mean, sum_cols(U))
        """
        # 1. Batch transpose multiplication
        Y = self.X.T @ U

        # 2. Batch centering correction
        # Sum columns of U -> shape (k,)
        u_sums = U.sum(axis=0)

        # Outer product: mean (n,1) * u_sums (1,k) -> (n, k)
        correction = np.outer(self.mean_, u_sums)

        return cast(np.ndarray, Y - correction)


def rpca(
    X: ArrayOrSparse,
    t: int,
    p: int = 0,
    oversampling: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Randomized Principal Component Analysis (rPCA).

    This function performs PCA using a randomized SVD solver.
    It automatically handles centering of the data.

    CRITICAL: Uses 'Virtual Centering' to handle sparse matrices efficiently.
    It never creates the dense (X - mu) matrix, avoiding MemoryErrors.

    Parameters
    ----------
    X : {np.ndarray, scipy.sparse.spmatrix}
        Input matrix of shape (m, n).
    t : int
        Target rank (number of components).
    p : int, optional
        Power iterations. Default 0.
    oversampling : int, optional
        Oversampling parameter. Default 10.

    Returns
    -------
    U : np.ndarray
        Principal components in feature space (only if X was transposed, otherwise projection).
        Actually, standard PCA returns (U, S, Vt).
        - Vt contains the Principal Axes (Eigenvectors of Covariance).
        - U * S contains the projected data (Scores).
    S : np.ndarray
        Singular values.
    Vt : np.ndarray
        Principal axes (components).
    """
    # 1. Compute Mean
    # Note: For sparse matrices, .mean() returns a np.matrix which can cause bugs.
    # We cast to np.asarray and squeeze to 1D array (n,).
    if sp.issparse(X):
        mean_vector = np.asarray(X.mean(axis=0)).squeeze()
    else:
        mean_vector = X.mean(axis=0)

    # 2. Virtual Centering
    # Wrap X in a LinearOperator that subtracts mean on the fly during multiplication.
    X_centered = CenteredMatrix(X, mean_vector)

    # 3. Randomized SVD on the virtual matrix
    # rsvd is compatible with LinearOperator thanks to duck-typing (@ operator)
    U, S, Vt = rsvd(X_centered, t, p=p, oversampling=oversampling)
    # S is a diagonal matrix, must be converted in 1D array
    return U, np.diag(S), Vt
