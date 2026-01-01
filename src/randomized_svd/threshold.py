import numpy as np
from scipy import integrate, optimize
from functools import lru_cache
from typing import Optional


def _limit_aspect_ratio(m: int, n: int) -> float:
    """
    Computes the aspect ratio beta = min(m, n) / max(m, n).

    This is the definition used in Gavish & Donoho (2014).
    """
    if m <= 0 or n <= 0:
        raise ValueError("Matrix dimensions must be strictly positive.")

    return min(m, n) / max(m, n)


@lru_cache(maxsize=None)
def _optimal_lambda(beta: float) -> float:
    """
    Computes the optimal lambda(beta) for the known-noise case.

    Reference:
    Gavish & Donoho (2014), Eq. (11)

    Parameters
    ----------
    beta : float
        Aspect ratio in (0, 1].

    Returns
    -------
    float
        Optimal hard-threshold scaling factor.
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError("beta must lie in the interval (0, 1].")

    w = (8.0 * beta) / (beta + 1.0 + np.sqrt(beta**2 + 14.0 * beta + 1.0))
    return float(np.sqrt(2.0 * (beta + 1.0) + w))


def _marchenko_pastur_density(x: float, beta: float) -> float:
    """
    Marchenko–Pastur probability density function.

    MP(x) = sqrt((b_plus - x)(x - b_minus)) / (2*pi*beta*x)

    The density is supported on [b_minus, b_plus].
    """
    sqrt_beta = np.sqrt(beta)
    b_plus = (1.0 + sqrt_beta) ** 2
    b_minus = (1.0 - sqrt_beta) ** 2

    if x <= b_minus or x >= b_plus:
        return 0.0

    return float(
        np.sqrt((b_plus - x) * (x - b_minus))
        / (2.0 * np.pi * beta * x)
    )


@lru_cache(maxsize=None)
def _marchenko_pastur_median(beta: float) -> float:
    """
    Computes the median of the Marchenko–Pastur distribution numerically.

    The median m satisfies:
        ∫_{b_minus}^{m} MP(x) dx = 0.5
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError("beta must lie in the interval (0, 1].")

    sqrt_beta = np.sqrt(beta)
    b_plus = (1.0 + sqrt_beta) ** 2
    b_minus = (1.0 - sqrt_beta) ** 2

    def cdf_minus_half(x: float) -> float:
        value, _ = integrate.quad(
            _marchenko_pastur_density,
            b_minus,
            x,
            args=(beta,),
            limit=200
        )
        return float(value) - 0.5

    return float(optimize.brentq(cdf_minus_half, b_minus, b_plus))


@lru_cache(maxsize=None)
def _optimal_omega(beta: float) -> float:
    """
    Computes omega(beta) for the unknown-noise case.

    omega(beta) = lambda(beta) / sqrt(median(MP(beta)))
    """
    lambda_val = _optimal_lambda(beta)
    mp_median = _marchenko_pastur_median(beta)
    return float(lambda_val / np.sqrt(mp_median))


def optimal_threshold(
    m: int,
    n: int,
    S: np.ndarray,
    sigma: Optional[float] = None
) -> float:
    """
    Computes the Gavish–Donoho optimal hard threshold.

    Parameters
    ----------
    m, n : int
        Dimensions of the data matrix.
    S : np.ndarray
        Singular values of the observed matrix.
    sigma : float, optional
        Noise standard deviation. If None, the unknown-noise
        formulation is used.

    Returns
    -------
    float
        Threshold tau: singular values below tau should be discarded.
    """
    if S.ndim != 1:
        raise ValueError("S must be a one-dimensional array.")

    beta = _limit_aspect_ratio(m, n)

    if sigma is not None:
        if sigma <= 0.0:
            raise ValueError("sigma must be strictly positive.")

        lambda_val = _optimal_lambda(beta)
        tau = lambda_val * np.sqrt(max(m, n)) * sigma
    else:
        omega_val = _optimal_omega(beta)
        tau = omega_val * float(np.median(S))

    return float(tau)


def optimal_rank(
    m: int,
    n: int,
    S: np.ndarray,
    sigma: Optional[float] = None
) -> int:
    """
    Computes the optimal rank induced by the Gavish–Donoho threshold.

    Returns
    -------
    int
        Number of singular values strictly above the optimal threshold.
    """
    tau = optimal_threshold(m, n, S, sigma)
    return int(np.sum(S > tau))
