"""Statistical functions for confidence intervals and outlier detection."""

from .confidence import format_interval, wilson_interval
from .outliers import get_outlier_severity, is_outlier

__all__ = [
    "wilson_interval",
    "format_interval",
    "is_outlier",
    "get_outlier_severity",
]
