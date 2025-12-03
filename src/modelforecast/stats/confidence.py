"""Wilson score interval for confidence intervals on binomial proportions."""

from math import sqrt


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Calculate Wilson score confidence interval for binomial proportion.

    Unlike normal approximation, Wilson intervals:
    - Work for small sample sizes (n=10-20)
    - Work near extreme values (0% or 100%)
    - Never produce impossible intervals (<0 or >100%)

    Args:
        successes: Number of successful trials
        trials: Total number of trials
        confidence: Confidence level (0.95 for 95%, 0.99 for 99%)

    Returns:
        Tuple of (lower_bound, upper_bound) as proportions [0.0, 1.0]

    Examples:
        >>> wilson_interval(9, 10, 0.95)
        (0.597, 0.997)  # 90% success with n=10
        >>> wilson_interval(0, 10, 0.95)
        (0.0, 0.308)    # 0% success with n=10
    """
    if trials == 0:
        return (0.0, 1.0)

    # Z-score for confidence level
    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.99:
        z = 2.576
    else:
        # Approximation for other confidence levels
        from math import erf

        z = sqrt(2) * erf((2 * confidence - 1))

    p_hat = successes / trials

    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    margin = (z / denominator) * sqrt(
        p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)
    )

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)


def format_interval(lower: float, upper: float, as_percentage: bool = True) -> str:
    """
    Format confidence interval for display.

    Args:
        lower: Lower bound [0.0, 1.0]
        upper: Upper bound [0.0, 1.0]
        as_percentage: If True, format as percentage (default)

    Returns:
        Formatted string like "[76,97]" or "[0.76,0.97]"

    Examples:
        >>> format_interval(0.76, 0.97)
        '[76,97]'
        >>> format_interval(0.76, 0.97, as_percentage=False)
        '[0.76,0.97]'
    """
    if as_percentage:
        return f"[{int(lower * 100)},{int(upper * 100)}]"
    else:
        return f"[{lower:.2f},{upper:.2f}]"
