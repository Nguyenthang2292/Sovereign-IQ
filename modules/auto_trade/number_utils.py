"""Number coercion utilities for auto trade."""

from typing import Any, Union


def coerce_float(value: Any, default: float = 0.0) -> float:
    """
    Coerce a value to float, returning default on failure.

    Args:
        value: Value to coerce (int, str, float, or other).
        default: Default value when coercion fails.

    Returns:
        float: The coerced value or default.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
