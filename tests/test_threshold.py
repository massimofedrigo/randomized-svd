import numpy as np
import pytest
from typing import cast
from scipy import integrate
from randomized_svd import optimal_threshold, optimal_rank
from randomized_svd.threshold import (
    _limit_aspect_ratio,
    _optimal_lambda,
    _marchenko_pastur_median,
    _marchenko_pastur_density
)


# --- Internal Logic Tests ---

class TestGavishDonohoMath:
    """
    Mathematical validation ensuring internal formulas align with
    theoretical values from Gavish & Donoho (2014).
    """

    @pytest.mark.parametrize("m, n, expected_beta", [
        (100, 50, 0.5),
        (50, 100, 0.5),
        (100, 100, 1.0)
    ])
    def test_aspect_ratio_calculation(self, m: int, n: int, expected_beta: float) -> None:
        """Verify beta is correctly computed as min(m,n) / max(m,n)."""
        assert _limit_aspect_ratio(m, n) == expected_beta

    @pytest.mark.parametrize("m, n", [
        (100, 0),   # Zero dimension
        (100, -5),  # Negative dimension
        (-10, 10)   # Negative dimension
    ])
    def test_aspect_ratio_validation(self, m: int, n: int) -> None:
        """Verify invalid dimensions raise ValueError."""
        with pytest.raises(ValueError):
            _limit_aspect_ratio(m, n)

    def test_optimal_lambda_square_matrix(self) -> None:
        """
        Known theoretical value: For a square matrix (beta=1), lambda must be 4/sqrt(3).
        Reference: Gavish & Donoho 2014, Eq (11).
        """
        beta = 1.0
        expected = 4 / np.sqrt(3)  # ≈ 2.309
        calculated = _optimal_lambda(beta)
        np.testing.assert_allclose(calculated, expected, rtol=1e-5)

    def test_marchenko_pastur_integral(self) -> None:
        """
        Validation: The Marchenko-Pastur density is a PDF, so its integral
        over the entire support [b_minus, b_plus] must be 1.0.
        """
        beta = 0.5
        b_minus = (1 - np.sqrt(beta))**2
        b_plus = (1 + np.sqrt(beta))**2

        # Integrate the PDF
        integral, _ = integrate.quad(_marchenko_pastur_density, b_minus, b_plus, args=(beta,))
        np.testing.assert_allclose(integral, 1.0, atol=1e-4)

    def test_marchenko_pastur_median_logic(self) -> None:
        """
        Validation: The computed median must split the integral exactly in half (CDF = 0.5).
        """
        beta = 0.5
        median = _marchenko_pastur_median(beta)
        b_minus = (1 - np.sqrt(beta))**2

        # Integrate from lower bound to median -> should be 0.5
        area, _ = integrate.quad(_marchenko_pastur_density, b_minus, median, args=(beta,))
        np.testing.assert_allclose(area, 0.5, atol=1e-4)


# --- Public API Tests ---

class TestOptimalThresholdAPI:
    """
    Functional tests for the public API: optimal_threshold and optimal_rank.
    """

    @pytest.fixture
    def pure_noise_sv(self) -> np.ndarray:
        """Fixture: Returns Singular Values of a 100x100 pure Gaussian noise matrix."""
        np.random.seed(42)
        noise = np.random.randn(100, 100)
        _, S, _ = np.linalg.svd(noise, full_matrices=False)
        return cast(np.ndarray, S)

    @pytest.fixture
    def signal_plus_noise_sv(self) -> np.ndarray:
        """Fixture: Returns Singular Values of a Rank-5 Signal + Noise matrix."""
        np.random.seed(42)
        m, n = 100, 100
        rank = 5
        signal_strength = 50.0

        # Construct Low-Rank Signal
        U, _ = np.linalg.qr(np.random.randn(m, rank))
        V, _ = np.linalg.qr(np.random.randn(n, rank))
        S_signal = np.diag(np.linspace(signal_strength, signal_strength/2, rank))
        X_signal = U @ S_signal @ V.T

        # Add White Noise (sigma = 1.0)
        X_noise = np.random.randn(m, n)

        # SVD of the mixture
        X = X_signal + X_noise
        _, S, _ = np.linalg.svd(X, full_matrices=False)
        return cast(np.ndarray, S)

    def test_rank_is_zero_for_pure_noise_known_sigma(self, pure_noise_sv: np.ndarray) -> None:
        """
        If sigma is known (1.0) and provided, a pure noise matrix N(0,1)
        should ideally yield rank 0 (or close to 0 due to finite size effects).
        """
        m, n = 100, 100
        # Gavish-Donoho is asymptotic; on finite 100x100 matrices,
        # allowing k <= 1 is a statistical safety margin.
        k = optimal_rank(m, n, pure_noise_sv, sigma=1.0)

        assert k <= 1, f"Failed to suppress noise. Found rank {k}, expected 0 or 1."

    def test_rank_is_minimal_for_pure_noise_unknown_sigma(self, pure_noise_sv: np.ndarray) -> None:
        """
        If sigma is unknown (None), the algorithm estimates it from the median.
        It should still identify the matrix as mostly noise.
        """
        m, n = 100, 100
        k = optimal_rank(m, n, pure_noise_sv, sigma=None)

        # Estimation via median is robust for MP distributions.
        assert k <= 2, f"Failed to suppress noise with auto-sigma. Found rank {k}."

    def test_signal_recovery_exact_rank(self, signal_plus_noise_sv: np.ndarray) -> None:
        """
        Verify that the algorithm recovers the exact rank (5) of the signal
        embedded in noise.
        """
        m, n = 100, 100
        expected_rank = 5

        # Case 1: Known Sigma
        k_known = optimal_rank(m, n, signal_plus_noise_sv, sigma=1.0)
        assert k_known == expected_rank, f"Known Sigma: Expected rank {expected_rank}, got {k_known}"

        # Case 2: Unknown Sigma (Auto-estimation)
        k_unknown = optimal_rank(m, n, signal_plus_noise_sv, sigma=None)
        assert k_unknown == expected_rank, f"Unknown Sigma: Expected rank {expected_rank}, got {k_unknown}"

    def test_shape_invariance(self) -> None:
        """
        The threshold depends on aspect ratio beta.
        Transposing the matrix (swapping m, n) should not change beta
        (since we use min/max) and thus should yield the same threshold.
        """
        S = np.array([10.0, 9.0, 8.0, 1.0, 0.5, 0.1])

        # Tall matrix (1000x10) -> beta = 0.01
        tau_tall = optimal_threshold(1000, 10, S, sigma=None)

        # Wide matrix (10x1000) -> beta = 0.01
        tau_wide = optimal_threshold(10, 1000, S, sigma=None)

        assert np.isclose(tau_tall, tau_wide), "Threshold should be invariant to transposition."

    def test_edge_case_empty_singular_values(self) -> None:
        """
        Robustness check: Ensure function handles empty input arrays gracefully
        (e.g., return rank 0 instead of crashing).
        """
        S_empty = np.array([])
        m, n = 100, 100

        # Should calculate a valid positive cutoff based on dims and sigma
        cutoff = optimal_threshold(m, n, S_empty, sigma=1.0)
        assert cutoff > 0.0

        # Should return rank 0 since no singular values exceed cutoff
        rank = optimal_rank(m, n, S_empty, sigma=1.0)
        assert rank == 0
