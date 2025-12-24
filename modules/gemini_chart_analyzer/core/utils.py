"""
Utility functions for multi-timeframe analysis.
"""

from typing import List, Optional, Tuple
from modules.common.utils import normalize_timeframe


# Default timeframes for multi-timeframe analysis
DEFAULT_TIMEFRAMES = ['15m', '1h', '4h', '1d']


def normalize_timeframes(timeframes: List[str]) -> List[str]:
    """
    Normalize và validate danh sách timeframes.
    
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
    Get weight cho timeframe.
    
    Args:
        timeframe: Timeframe string
        weights: Optional custom weights dict (nếu None, dùng default)
    
    Returns:
        Weight value (default: 0.1 nếu không tìm thấy)
    """
    from modules.gemini_chart_analyzer.core.signal_aggregator import DEFAULT_TIMEFRAME_WEIGHTS
    
    if weights is None:
        weights = DEFAULT_TIMEFRAME_WEIGHTS
    
    normalized_tf = normalize_timeframe(timeframe)
    return weights.get(normalized_tf, 0.1)


def sort_timeframes_by_weight(timeframes: List[str], weights: Optional[dict] = None) -> List[str]:
    """
    Sort timeframes theo weight (descending - lớn hơn trước).
    
    Args:
        timeframes: List of timeframe strings
        weights: Optional custom weights dict
    
    Returns:
        Sorted list of timeframes
    """
    from modules.gemini_chart_analyzer.core.signal_aggregator import DEFAULT_TIMEFRAME_WEIGHTS
    
    if weights is None:
        weights = DEFAULT_TIMEFRAME_WEIGHTS
    
    # Sort by weight descending
    return sorted(
        timeframes,
        key=lambda tf: get_timeframe_weight(tf, weights),
        reverse=True
    )


def validate_timeframes(timeframes: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate danh sách timeframes.
    
    Args:
        timeframes: List of timeframe strings
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not timeframes:
        return False, "Timeframes list cannot be empty"
    
    if len(timeframes) == 0:
        return False, "At least one timeframe is required"
    
    # Check for duplicates after normalization
    normalized = [normalize_timeframe(tf) for tf in timeframes]
    if len(normalized) != len(set(normalized)):
        return False, "Duplicate timeframes found"
    
    return True, None

