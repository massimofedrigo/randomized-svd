import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore[import-untyped]
from sklearn.utils.validation import check_is_fitted, check_array, check_random_state  # type: ignore[import-untyped]
from typing import Optional, Union, Any, cast
from .core import rsvd


class RandomizedSVD(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible wrapper for Randomized SVD.

    This class serves as a drop-in replacement for `sklearn.decomposition.TruncatedSVD`,
    but uses the faster randomized algorithm implemented in `randomized-svd`.

    Parameters
    ----------
    n_components : int, default=50
        The target rank (k).
    n_oversamples : int, default=10
        Additional sampling factors (oversampling) to improve accuracy.
    n_iter : int, default=0
        Number of power iterations (p). Recommended p=2 for slow spectral decay.
    random_state : int, RandomState instance or None, default=None
        Seed for reproducibility.

    Attributes
    ----------
    components_ : np.ndarray of shape (n_components, n_features)
        The right singular vectors (Vt).
    singular_values_ : np.ndarray of shape (n_components,)
        The singular values (S).
    """

    def __init__(
        self,
        n_components: int = 50,
        n_oversamples: int = 10,
        n_iter: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ):
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "RandomizedSVD":
        """
        Fit the model on X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 1. Input Validation (Checks sparsity, NaNs, infinite values)
        X = check_array(X, accept_sparse=['csr', 'csc'])

        # 2. Handle Random State
        # Since core.py uses np.random directly, we seed the global state momentarily.
        # This ensures reproducibility if the user passes an integer seed.
        if self.random_state is not None:
            rng = check_random_state(self.random_state)
            # Extract a seed from the sklearn RandomState to feed numpy
            seed = rng.randint(0, 2**32 - 1)
            np.random.seed(seed)

        # 3. Core Algorithm Execution
        # Note: core.rsvd returns (U, S, Vt).
        # sklearn stores components_ as Vt and singular_values_ as S (1D array).
        U, S, Vt = rsvd(
            X,
            t=self.n_components,
            p=self.n_iter,
            oversampling=self.n_oversamples
        )

        # 4. Store Attributes
        self.components_ = Vt
        # Extract diagonal from S matrix to match sklearn format (1D array)
        self.singular_values_ = np.diag(S)

        return self

    def transform(self, X: Any) -> np.ndarray:
        """
        Perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_components)
            Reduced version of X.
        """
        check_is_fitted(self, ['components_', 'singular_values_'])
        X = check_array(X, accept_sparse=['csr', 'csc'])

        # Project data: X @ V (where V is components_.T)
        result = X @ self.components_.T
        return cast(np.ndarray, result)

    def inverse_transform(self, X: Any) -> np.ndarray:
        """
        Transform X back to its original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)

        Returns
        -------
        X_original : np.ndarray of shape (n_samples, n_features)
            Reconstructed data.
        """
        check_is_fitted(self, ['components_'])
        X = check_array(X)

        # Reconstruct: X_reduced @ Vt
        result = X @ self.components_
        return cast(np.ndarray, result)
