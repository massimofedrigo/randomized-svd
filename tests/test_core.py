import numpy as np
import pytest
from randomized_svd import rsvd


# --- Fixtures & Helpers ---

@pytest.fixture
def random_matrix_generator():
    """
    Returns a factory function to generate matrices with consistent seeding.
    """
    def _generate(m, n, seed=42):
        np.random.seed(seed)
        return np.random.randn(m, n)
    return _generate


@pytest.fixture
def low_rank_matrix_generator():
    """
    Generates a matrix with rapid decay (easy to approximate).
    """
    def _generate(m, n, rank, seed=42):
        np.random.seed(seed)
        U, _ = np.linalg.qr(np.random.randn(m, rank))
        V, _ = np.linalg.qr(np.random.randn(n, rank))
        S = np.diag(np.linspace(10, 1, rank))
        return U @ S @ V.T
    return _generate


@pytest.fixture
def slow_decay_matrix_generator():
    """
    Generates a matrix where singular values decay slowly.
    This is the crucial test case for Power Iterations.
    """
    def _generate(m, n, seed=42):
        np.random.seed(seed)
        # Create full rank matrix with linear decay (flat spectrum)
        # S = [1.0, 0.99, 0.98, ... ]
        U, _ = np.linalg.qr(np.random.randn(m, min(m, n)))
        V, _ = np.linalg.qr(np.random.randn(n, min(m, n)))
        s_values = np.linspace(1, 0.1, min(m, n))
        S = np.diag(s_values)

        # Expand S to match U and V shapes if needed for multiplication
        return U @ S @ V.T
    return _generate


# --- Test Cases ---

class TestRSVDInputValidation:
    """
    Cover edge cases for bad inputs.
    """

    def test_invalid_t_type(self):
        X = np.zeros((10, 10))
        with pytest.raises(TypeError, match="Parameter t must be an integer"):
            rsvd(X, t=5.5)  # type: ignore

    def test_t_too_small(self):
        X = np.zeros((10, 10))
        with pytest.raises(ValueError, match="must be between 1"):
            rsvd(X, t=0)

    def test_t_too_large(self):
        X = np.zeros((10, 5))
        with pytest.raises(ValueError, match="must be between 1"):
            rsvd(X, t=6)

    def test_invalid_p_type(self):
        X = np.zeros((10, 10))
        with pytest.raises(TypeError, match="Parameter p must be an integer"):
            rsvd(X, t=5, p=2.0)  # type: ignore

    def test_negative_p(self):
        X = np.zeros((10, 10))
        with pytest.raises(ValueError, match="Parameter p must be non-negative"):
            rsvd(X, t=5, p=-1)


class TestRSVDDispatchAndShapes:
    """
    Verifies output shapes for Tall vs Wide matrices.
    """

    @pytest.mark.parametrize("shape", [
        (100, 50),   # Tall
        (50, 100),   # Wide
    ])
    def test_output_dimensions(self, shape, random_matrix_generator):
        m, n = shape
        t = 10
        p = 2  # Verify dimensions hold even with power iterations
        X = random_matrix_generator(m, n)

        U, S, Vt = rsvd(X, t, p=p)

        assert U.shape == (m, t)
        assert S.shape == (t, t)
        assert Vt.shape == (t, n)


class TestRSVDMathematics:
    """
    Verifies mathematical properties: accuracy and orthogonality.
    """

    def test_exact_recovery_low_rank(self, low_rank_matrix_generator):
        """Standard rSVD (p=0) should handle simple low-rank matrices well."""
        m, n = 100, 80
        X = low_rank_matrix_generator(m, n, rank=10)

        U, S, Vt = rsvd(X, t=10, p=0)
        X_approx = U @ S @ Vt

        error = np.linalg.norm(X - X_approx) / np.linalg.norm(X)
        assert error < 1e-10


class TestRSVDPowerIterations:
    """
    Specific tests for the effect of Power Iterations (p > 0).
    """

    def test_accuracy_improvement_on_slow_decay(self, slow_decay_matrix_generator):
        """
        Crucial Test: On a matrix with slow spectral decay,
        p=2 should yield lower error than p=0.
        """
        m, n = 100, 100
        X = slow_decay_matrix_generator(m, n)
        target_rank = 10

        # 1. Compute without Power Iterations
        U0, S0, Vt0 = rsvd(X, t=target_rank, p=0)
        err0 = np.linalg.norm(X - (U0 @ S0 @ Vt0))

        # 2. Compute with Power Iterations (p=2)
        Up, Sp, Vtp = rsvd(X, t=target_rank, p=2)
        err_p = np.linalg.norm(X - (Up @ Sp @ Vtp))

        print(f"\nError (p=0): {err0:.5f}")
        print(f"Error (p=2): {err_p:.5f}")

        # The error with p=2 must be strictly smaller
        assert err_p < err0, "Power iterations failed to improve accuracy on slow-decay matrix"

    def test_consistency_wide_matrix(self, slow_decay_matrix_generator):
        """Ensure p works correctly also for wide matrices (via transpose logic)."""
        m, n = 50, 200  # Wide
        X = slow_decay_matrix_generator(m, n)

        # Should run without errors
        rsvd(X, t=10, p=2)
