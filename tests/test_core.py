import numpy as np
import pytest
from randomized_svd import rsvd


# --- Fixtures & Helpers ---

@pytest.fixture
def random_matrix_generator():
    """
    Returns a factory function to generate matrices with consistent seeding.
    This ensures tests are reproducible despite the randomized algorithm.
    """
    def _generate(m, n, seed=42):
        np.random.seed(seed)
        return np.random.randn(m, n)
    return _generate


@pytest.fixture
def low_rank_matrix_generator():
    """
    Generates a matrix with a specific exact rank to test accuracy.
    X = U * S * Vt
    """
    def _generate(m, n, rank, seed=42):
        np.random.seed(seed)
        # Random orthogonal matrices
        U, _ = np.linalg.qr(np.random.randn(m, rank))
        V, _ = np.linalg.qr(np.random.randn(n, rank))
        # Rapidly decaying singular values
        S = np.diag(np.linspace(10, 1, rank))
        return U @ S @ V.T
    return _generate


# --- Test Cases ---

class TestRSVDInputValidation:
    """
    Cover edge cases for bad inputs.
    Crucial for a robust library.
    """

    def test_invalid_t_type(self):
        X = np.zeros((10, 10))
        with pytest.raises(TypeError, match="must be an integer"):
            rsvd(X, t=5.5)  # type: ignore

    def test_t_too_small(self):
        X = np.zeros((10, 10))
        with pytest.raises(ValueError, match="must be between 1"):
            rsvd(X, t=0)

    def test_t_too_large(self):
        X = np.zeros((10, 5))
        # t cannot be larger than min(10, 5) = 5
        with pytest.raises(ValueError, match="must be between 1"):
            rsvd(X, t=6)


class TestRSVDDispatchAndShapes:
    """
    This class verifies that the wrapper correctly handles
    both Tall (m >= n) and Wide (m < n) matrices.
    """

    @pytest.mark.parametrize("shape", [
        (100, 50),   # Tall (triggers _rsvd_tall)
        (50, 100),   # Wide (triggers _rsvd_wide)
        (50, 50)     # Square (triggers _rsvd_tall)
    ])
    def test_output_dimensions(self, shape, random_matrix_generator):
        m, n = shape
        t = 10
        X = random_matrix_generator(m, n)

        U, S, Vt = rsvd(X, t)

        # Assert shapes comply with SVD definition
        assert U.shape == (m, t), f"U shape mismatch for {shape}"
        assert S.shape == (t, t), f"S shape mismatch for {shape}"
        assert Vt.shape == (t, n), f"Vt shape mismatch for {shape}"

    def test_full_rank_edge_case(self, random_matrix_generator):
        """Test asking for the maximum possible rank t = min(m, n)"""
        m, n = 20, 10
        t = 10
        X = random_matrix_generator(m, n)

        U, S, Vt = rsvd(X, t)

        assert U.shape == (20, 10)
        assert S.shape == (10, 10)


class TestRSVDMathematics:
    """
    Verifies mathematical properties: accuracy and orthogonality.
    """

    def test_reconstruction_accuracy(self, low_rank_matrix_generator):
        """
        If X is truly low-rank, rSVD should reconstruct it with very low error.
        """
        m, n = 100, 80
        true_rank = 10
        X = low_rank_matrix_generator(m, n, rank=true_rank)

        # We ask for a slightly larger rank to capture everything
        t = 12
        U, S, Vt = rsvd(X, t)

        # Reconstruct: X_approx = U * S * Vt
        X_approx = U @ S @ Vt

        # Relative Frobenius Norm Error
        error = np.linalg.norm(X - X_approx) / np.linalg.norm(X)

        # We expect error close to machine epsilon or very small for exact low-rank
        assert error < 1e-10, f"Reconstruction error too high: {error}"

    @pytest.mark.parametrize("shape", [(50, 30), (30, 50)])
    def test_orthogonality(self, shape, random_matrix_generator):
        """
        U and Vt must be orthonormal matrices (or semi-unitary).
        U.T @ U = I
        Vt @ Vt.T = I
        """
        m, n = shape
        t = 5
        X = random_matrix_generator(m, n)

        U, _, Vt = rsvd(X, t)

        # Check U orthogonality: U.T @ U should be Identity(t)
        UtU = U.T @ U
        I_t = np.eye(t)

        # Check Vt orthogonality: Vt @ Vt.T should be Identity(t)
        # Note: usually Vt is (t x n), so Vt @ Vt.T is (t x t)
        VVt = Vt @ Vt.T

        np.testing.assert_allclose(UtU, I_t, atol=1e-10, err_msg="U is not orthogonal")
        np.testing.assert_allclose(VVt, I_t, atol=1e-10, err_msg="Vt is not orthogonal")
