import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore[import-untyped]
from sklearn.utils.validation import check_is_fitted, check_array, check_random_state  # type: ignore[import-untyped]
from typing import Optional, Union, Any, cast, Literal
from .core import rsvd
from .threshold import optimal_rank


class RandomizedSVD(BaseEstimator, TransformerMixin):
    """
    Randomized SVD with Automatic Denoising (Gavish-Donoho).

    Unlike sklearn.decomposition.TruncatedSVD, this class can automatically
    determine the optimal rank to separate signal from noise.

    Parameters
    ----------
    n_components : int, default=50
        If rank_selection='manual', this is the exact rank k.
        If rank_selection='auto', this acts as an UPPER BOUND (buffer).
        The algorithm will compute this many components, then truncate
        based on the noise estimate.

    rank_selection : {'manual', 'auto'}, default='manual'
        - 'manual': Works exactly like standard TruncatedSVD.
        - 'auto': Uses Gavish-Donoho method to truncate singular values
          below the optimal noise threshold.

    sigma : float, optional, default=None
        Only used if rank_selection='auto'.
        The known noise level. If None, it is estimated from the median
        singular value (Marchenko-Pastur distribution).

    n_oversamples : int, default=10
        Additional sampling factors for accuracy.

    n_iter : int, default=0
        Power iterations (p). Recommended p=2 for higher accuracy.

    random_state : int, RandomState instance or None, default=None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 50,
        rank_selection: Literal['manual', 'auto'] = 'manual',
        sigma: Optional[float] = None,
        n_oversamples: int = 10,
        n_iter: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ):
        self.n_components = n_components
        self.rank_selection = rank_selection
        self.sigma = sigma
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "RandomizedSVD":
        # 1. Validation
        X = check_array(X, accept_sparse=['csr', 'csc'])
        m, n = X.shape

        # 2. Random State Management
        if self.random_state is not None:
            rng = check_random_state(self.random_state)
            seed = rng.randint(0, 2**32 - 1)
            original_state = np.random.get_state()
            np.random.seed(seed)
            try:
                self._run_fit(X, m, n)
            finally:
                np.random.set_state(original_state)
        else:
            self._run_fit(X, m, n)

        return self

    def _run_fit(self, X: Any, m: int, n: int) -> None:
        # Run rSVD with the requested components (acting as upper bound if auto)
        U, S_matrix, Vt = rsvd(
            X,
            t=self.n_components,
            p=self.n_iter,
            oversampling=self.n_oversamples
        )

        # Extract 1D singular values (core.py returns diagonal matrix)
        S = np.diag(S_matrix)

        # 3. Automatic Rank Selection (The Killer Feature)
        if self.rank_selection == 'auto':
            # Calcola il rank ottimale usando il modulo threshold.py
            k_opt = optimal_rank(m, n, S, sigma=self.sigma)

            # Safety: k_opt non può superare i componenti calcolati
            k_opt = min(k_opt, self.n_components)

            # Se k_opt è 0 (tutto rumore), teniamo almeno 1 componente per evitare crash
            if k_opt == 0:
                k_opt = 1

            # Troncamento dei risultati
            self.n_components_ = k_opt  # Salviamo il rank effettivo trovato
            self.components_ = Vt[:k_opt, :]
            self.singular_values_ = S[:k_opt]

            # Nota: U non viene salvato in sklearn standard (occupa troppa RAM),
            # serve solo per transform(), ma qui stiamo solo facendo fit.
        else:
            # Manual mode
            self.n_components_ = self.n_components
            self.components_ = Vt
            self.singular_values_ = S

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ['components_', 'singular_values_'])
        X = check_array(X, accept_sparse=['csr', 'csc'])
        return cast(np.ndarray, X @ self.components_.T)

    def inverse_transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ['components_'])
        X = check_array(X)
        return cast(np.ndarray, X @ self.components_)
