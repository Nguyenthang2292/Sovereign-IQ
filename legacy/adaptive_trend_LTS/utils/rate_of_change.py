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
        from modules.adaptive_trend_LTS_mini.utils.cache_manager import get_cache_manager

        cache = get_cache_manager()

        # We use a simplified key for performance: length + first/last values + index hash
        # Full content hashing is too slow (as slow as calculation)
        # Assuming typical use case: same Series object or identical data

        # Calculate robust cache key using pandas hash
        # This is faster (single pass) and more robust against collisions than statistical properties
        try:
            from pandas.util import hash_pandas_object

            series_hash = hash_pandas_object(prices, index=True).sum()
            cache_key = f"ROC|{series_hash}"

            cached_result = cache.get("ROC", 0, cache_key)
            if cached_result is not None:
                return cached_result
        except ImportError:
            # Fallback if hash_pandas_object is not available
            log_warn("hash_pandas_object not found, skipping cache check for rate_of_change")
            cache_key = None
        except Exception as e:
            log_warn(f"Error calculating cache key: {e}, skipping cache")
            cache_key = None

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
