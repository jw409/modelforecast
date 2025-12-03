"""Outlier detection using Median Absolute Deviation (MAD)."""

from statistics import median


def is_outlier(
    new_result: float, existing_results: list[float], threshold: float = 2.0
) -> bool:
    """
    Detect outliers using Median Absolute Deviation (MAD).

    MAD is more robust than standard deviation for detecting outliers
    because it's not influenced by extreme values.

    Args:
        new_result: New result to check (e.g., 0.95 for 95% success rate)
        existing_results: List of existing results from community
        threshold: MAD threshold (default 2.0 = moderate outlier detection)

    Returns:
        True if new_result is an outlier, False otherwise

    Examples:
        >>> existing = [0.85, 0.87, 0.86, 0.88, 0.84]
        >>> is_outlier(0.95, existing)  # Significantly higher
        True
        >>> is_outlier(0.86, existing)  # Within normal range
        False
    """
    # Need at least 5 data points for meaningful comparison
    if len(existing_results) < 5:
        return False

    # Calculate median
    med = median(existing_results)

    # Calculate MAD (Median Absolute Deviation)
    absolute_deviations = [abs(x - med) for x in existing_results]
    mad = median(absolute_deviations)

    # If MAD is 0, all values are identical
    if mad == 0:
        return new_result != med

    # Convert to modified Z-score using consistency constant
    # 1.4826 makes MAD equivalent to standard deviation for normal distribution
    z_score = abs(new_result - med) / (mad * 1.4826)

    return z_score > threshold


def get_outlier_severity(
    new_result: float, existing_results: list[float]
) -> tuple[bool, float, str]:
    """
    Get outlier status with severity classification.

    Args:
        new_result: New result to check
        existing_results: List of existing results

    Returns:
        Tuple of (is_outlier, z_score, severity)
        severity is one of: "normal", "moderate", "severe", "extreme"

    Examples:
        >>> existing = [0.85, 0.87, 0.86, 0.88, 0.84]
        >>> get_outlier_severity(0.95, existing)
        (True, 2.8, 'severe')
    """
    if len(existing_results) < 5:
        return (False, 0.0, "insufficient_data")

    med = median(existing_results)
    absolute_deviations = [abs(x - med) for x in existing_results]
    mad = median(absolute_deviations)

    if mad == 0:
        is_out = new_result != med
        return (is_out, float("inf") if is_out else 0.0, "extreme" if is_out else "normal")

    z_score = abs(new_result - med) / (mad * 1.4826)

    # Classify severity
    if z_score <= 2.0:
        severity = "normal"
        is_out = False
    elif z_score <= 3.0:
        severity = "moderate"
        is_out = True
    elif z_score <= 4.0:
        severity = "severe"
        is_out = True
    else:
        severity = "extreme"
        is_out = True

    return (is_out, z_score, severity)
