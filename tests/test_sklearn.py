import numpy as np
import pytest
from typing import cast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from randomized_svd import RandomizedSVD


# --- Fixtures ---

@pytest.fixture
def random_data() -> np.ndarray:
    """Generates a consistent random matrix for testing."""
    np.random.seed(42)
    return cast(np.ndarray, np.random.randn(100, 20))


# --- Test Suite ---

class TestSklearnAPI:
    """
    Tests for the Scikit-Learn wrapper (RandomizedSVD class).
    """

    def test_sklearn_standard_attributes(self) -> None:
        """Verifies that the class follows sklearn naming conventions."""
        model = RandomizedSVD()
        assert hasattr(model, "fit")
        assert hasattr(model, "transform")
        assert hasattr(model, "get_params")
        assert hasattr(model, "set_params")

    def test_input_validation(self) -> None:
        """Ensures bad inputs raise appropriate errors (Sad Paths)."""
        model = RandomizedSVD(n_components=5)

        # Test 1: Transform before fit
        X_test = np.random.randn(10, 20)
        with pytest.raises(NotFittedError):
            model.transform(X_test)

        # Test 2: Invalid dimensions in fit
        # n_components (5) > n_features (3)
        X_tiny = np.random.randn(10, 3)
        with pytest.raises(ValueError):
            model.fit(X_tiny)

    def test_integration_pipeline(self, random_data: np.ndarray) -> None:
        """Test that RandomizedSVD works inside a Scikit-Learn Pipeline."""
        X = random_data
        y = np.random.randint(0, 2, 100)  # Fake labels

        # Create a pipeline: Scale -> rSVD
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('rsvd', RandomizedSVD(n_components=5, random_state=42))
        ])

        # Fit and Transform should work in one go
        X_transformed = pipe.fit_transform(X, y)

        assert X_transformed.shape == (100, 5)

    def test_reproducibility(self, random_data: np.ndarray) -> None:
        """Test that setting random_state produces identical results."""
        X = random_data

        model1 = RandomizedSVD(n_components=5, random_state=42)
        X1 = model1.fit_transform(X)

        model2 = RandomizedSVD(n_components=5, random_state=42)
        X2 = model2.fit_transform(X)

        np.testing.assert_allclose(X1, X2, err_msg="Results are not reproducible despite fixed seed")

    def test_inverse_transform(self) -> None:
        """Test accurate reconstruction (inverse_transform)."""
        # Create simple low rank data
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(100, 5))
        V, _ = np.linalg.qr(np.random.randn(20, 5))
        S = np.diag([10, 8, 6, 4, 2])
        X = U @ S @ V.T  # Rank 5 exact

        model = RandomizedSVD(n_components=5, random_state=42)
        X_reduced = model.fit_transform(X)
        X_reconstructed = model.inverse_transform(X_reduced)

        # Error should be very low for low-rank matrix
        err = np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X)
        assert err < 1e-5
