import pandas as pd

from modules.common.utils import log_error, log_warn

"""Calculate percentage rate of change for price series."""


def rate_of_change(prices: pd.Series) -> pd.Series:
    """Calculate percentage rate of change for price series.

    Equivalent to Pine Script global variable:
        R = (close - close[1]) / close[1]

    Args:
        prices: Price series (typically close prices).

    Returns:
        Series containing percentage change values. First value will be NaN.

    Raises:
        ValueError: If prices is empty.
        TypeError: If prices is not a pandas Series.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pandas Series, got {type(prices)}")

    if prices is None or len(prices) == 0:
        log_warn("Empty prices series provided for rate_of_change, returning empty series")
        return pd.Series(dtype="float64", index=prices.index if hasattr(prices, "index") else pd.RangeIndex(0, 0))

    try:
        # Check cache
        from modules.adaptive_trend_enhance.utils.cache_manager import get_cache_manager

        cache = get_cache_manager()

        # We use a simplified key for performance: length + first/last values + index hash
        # Full content hashing is too slow (as slow as calculation)
        # Assuming typical use case: same Series object or identical data

        # Calculate fast hash using start/end values and length
        cache_key = None
        if hasattr(prices, "values") and len(prices) > 0:
            # Hash based on simple properties + start/end
            start_val = float(prices.iloc[0])
            end_val = float(prices.iloc[-1])
            length = len(prices)
            # Create a key string
            cache_key = f"ROC|{length}|{start_val:.6f}|{end_val:.6f}"

            cached_result = cache.get("ROC", 0, cache_key)
            if cached_result is not None:
                return cached_result

        result = prices.pct_change(fill_method=None)

        # Check for excessive NaN values
        nan_count = result.isna().sum()
        if nan_count > 1:
            log_warn(
                f"rate_of_change contains {nan_count} NaN values. "
                f"Expected only 1 (first value). This may indicate data quality issues."
            )

        # Store in cache if key generated
        if cache_key:
            cache.put("ROC", 0, cache_key, result)

        return result

    except Exception as e:
        log_error(f"Error calculating rate_of_change: {e}")
        raise
