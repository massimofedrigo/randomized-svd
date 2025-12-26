import numpy as np


def optimal_threshold(m: int, n: int, gamma: float) -> int:
    """
    Compute the optimal hard threshold for singular value truncation.

    This function calculates the optimal target rank 't' based on the asymptotic
    formula provided by Gavish and Donoho (2014) for removing white noise from
    a matrix.

    Parameters
    ----------
    m : int
        Number of rows of the input matrix.
    n : int
        Number of columns of the input matrix.
    gamma : float
        The estimated noise level (sigma) of the white noise.
        Must be strictly positive.

    Returns
    -------
    int
        The optimal target rank 't'. The value is ceil-approximated and
        clamped such that t <= min(m, n).

    Raises
    ------
    ValueError
        If dimensions are not positive or if gamma is <= 0.

    References
    ----------
    .. [1] Gavish, M., & Donoho, D. L. (2014). The optimal hard threshold for
           singular values is 4/sqrt(3). IEEE Transactions on Information Theory,
           60(8), 5040-5053.
    """

    # 1. Input Validation
    if m <= 0 or n <= 0:
        raise ValueError(f"Matrix dimensions must be positive. Got m={m}, n={n}.")
    if gamma <= 0:
        raise ValueError(f"Noise level gamma must be positive. Got {gamma}.")

    # 2. Aspect Ratio calculation (beta must be in (0, 1])
    # We ensure beta is <= 1 regardless of matrix orientation
    beta = min(m / n, n / m)

    # 3. Compute Lambda coefficient (Gavish-Donoho formula)
    # Breaking down the formula for readability and numerical stability
    # Formula: lambda(b) = sqrt(2*(b+1) + 8*b / (b + 1 + sqrt(b^2 + 14*b + 1)))

    term1 = 2 * (beta + 1)

    numerator = 8 * beta
    denominator = beta + 1 + np.sqrt(beta**2 + 14 * beta + 1)

    lambda_val = np.sqrt(term1 + (numerator / denominator))

    # 4. Compute raw threshold
    # The thesis specifies using sqrt(n) where n is the number of columns.
    # Note: If the matrix is transposed externally, 'n' refers to the original columns.
    raw_threshold = lambda_val * np.sqrt(n) * gamma

    # 5. Approximation and Clamping
    # Using ceil approximation as per thesis requirements
    t_val = np.ceil(raw_threshold)

    # Ensure rank does not exceed matrix dimensions
    t = int(min(t_val, min(m, n)))

    return t
