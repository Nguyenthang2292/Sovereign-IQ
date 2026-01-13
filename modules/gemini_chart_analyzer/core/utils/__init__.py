from pathlib import Path
from typing import List, Optional, Tuple

from modules.common.utils import normalize_timeframe

"""
Utility modules for gemini chart analyzer.

This module provides utility functions for multi-timeframe analysis and common utilities.
"""


# Default timeframes for multi-timeframe analysis
DEFAULT_TIMEFRAMES = ["15m", "30m", "1h"]


def normalize_timeframes(timeframes: List[str]) -> List[str]:
    """
    Normalize and validate a list of timeframes.

    Args:
        timeframes: List of timeframe strings

    Returns:
        List of normalized timeframes (sorted by weight, descending)
    """
    if not timeframes:
        return []

    normalized = []
    seen = set()
    for tf in timeframes:
        if not tf or not tf.strip():
            continue
        normalized_tf = normalize_timeframe(tf)
        if normalized_tf not in seen:
            normalized.append(normalized_tf)
            seen.add(normalized_tf)

    # Sort by weight (larger timeframes first)
    return sort_timeframes_by_weight(normalized)


def get_timeframe_weight(timeframe: str, weights: Optional[dict] = None) -> float:
    """
    Get weight for a timeframe.

    Args:
        timeframe: Timeframe string
        weights: Optional custom weights dict (if None, uses default)

    Returns:
        Weight value (default: 0.1 if not found)
    """
    from config.gemini_chart_analyzer import TIMEFRAME_WEIGHTS

    if weights is None:
        weights = TIMEFRAME_WEIGHTS

    normalized_tf = normalize_timeframe(timeframe)
    return weights.get(normalized_tf, 0.1)


def sort_timeframes_by_weight(timeframes: List[str], weights: Optional[dict] = None) -> List[str]:
    """
    Sort timeframes by weight (descending - larger timeframes first).

    Args:
        timeframes: List of timeframe strings
        weights: Optional custom weights dict

    Returns:
        Sorted list of timeframes
    """
    from config.gemini_chart_analyzer import TIMEFRAME_WEIGHTS

    if weights is None:
        weights = TIMEFRAME_WEIGHTS

    # Sort by weight descending
    return sorted(timeframes, key=lambda tf: get_timeframe_weight(tf, weights), reverse=True)


def validate_timeframes(timeframes: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate a list of timeframes.

    Args:
        timeframes: List of timeframe strings

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not timeframes:
        return False, "Timeframes list cannot be empty"

    # Check for duplicates after normalization
    normalized = [normalize_timeframe(tf) for tf in timeframes]
    if len(normalized) != len(set(normalized)):
        return False, "Duplicate timeframes found"

    return True, None


def find_project_root(start_path: Path) -> Path:
    """
    Find project root by walking up from start_path, looking for marker files like pyproject.toml or .git.

    Args:
        start_path: Starting path for search

    Returns:
        Path to project root, or start_path if no marker found
    """
    current = Path(start_path).resolve()

    # Marker files/directories to identify project root
    markers = ["pyproject.toml", ".git"]

    # Walk up each level until reaching filesystem root
    while current != current.parent:
        # Check each marker
        for marker in markers:
            marker_path = current / marker
            if marker_path.exists():
                return current
        current = current.parent

    # If not found, return resolved start_path
    return Path(start_path).resolve()
