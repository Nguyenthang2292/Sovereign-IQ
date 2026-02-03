from __future__ import annotations

from typing import Any


def coerce_float(value: Any, default: float = 0.0) -> float:
    """
    Convert a value to float safely.

    Motivation: some external APIs (e.g. CCXT) may return numeric fields as None.
    This helper avoids `float(None)` and provides a stable default.
    """
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default

