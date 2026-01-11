
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pickle

import ccxt
import pandas as pd
import pandas as pd

"""
Example: Using CCXT with cached data to avoid rate limits.

This is an alternative approach to mocking - fetch real data once,
cache it, and reuse in tests. This is useful when you want to test
with real market data but avoid rate limits.

NOTE: This is just an example. The recommended approach is to use
mock data as implemented in conftest.py.
"""



# Cache directory
CACHE_DIR = Path("tests/cache/ohlcv")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache expiration (24 hours)
CACHE_EXPIRY_HOURS = 24


def get_cached_ohlcv(symbol: str, timeframe: str, limit: int, force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """
    Get OHLCV data from cache or fetch from exchange.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        timeframe: Timeframe (e.g., "1h", "4h")
        limit: Number of candles
        force_refresh: Force refresh even if cache exists

    Returns:
        DataFrame with OHLCV data or None if fetch fails
    """
    # Create cache file path
    cache_key = f"{symbol.replace('/', '_')}_{timeframe}_{limit}"
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    cache_meta_file = CACHE_DIR / f"{cache_key}.meta"

    # Check if cache exists and is valid
    if not force_refresh and cache_file.exists() and cache_meta_file.exists():
        try:
            # Check cache age
            with open(cache_meta_file, "r") as f:
                cache_time = datetime.fromisoformat(f.read().strip())

            if datetime.now() - cache_time < timedelta(hours=CACHE_EXPIRY_HOURS):
                # Load from cache
                with open(cache_file, "rb") as f:
                    df = pickle.load(f)
                print(f"Loaded {symbol} data from cache ({len(df)} candles)")
                return df
        except Exception as e:
            print(f"Error loading cache: {e}. Will fetch fresh data.")

    # Fetch from exchange (no credentials needed for public OHLCV data)
    try:
        print(f"Fetching {symbol} data from Binance...")
        exchange = ccxt.binance(
            {
                "enableRateLimit": True,  # Respect rate limits
                "options": {
                    "defaultType": "spot",  # Use spot market
                },
            }
        )

        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Cache the data
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)

        with open(cache_meta_file, "w") as f:
            f.write(datetime.now().isoformat())

        print(f"Cached {symbol} data ({len(df)} candles)")
        return df

    except Exception as e:
        print(f"Error fetching {symbol} data: {e}")
        return None


def create_cached_data_fetcher():
    """Create a DataFetcher-like object that uses cached data."""
    from types import SimpleNamespace

    def cached_fetch(symbol, **kwargs):
        limit = kwargs.get("limit", 200)
        timeframe = kwargs.get("timeframe", "1h")

        df = get_cached_ohlcv(symbol, timeframe, limit)
        if df is None:
            return pd.DataFrame(), None

        return df, "binance"

    return SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=cached_fetch,
    )


# Example usage in tests:
"""
def test_with_cached_data():
    # This will fetch data once, cache it, and reuse in subsequent runs
    data_fetcher = create_cached_data_fetcher()
    
    backtester = FullBacktester(data_fetcher)
    
    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=200,
        signal_type="LONG",
    )
    
    assert 'trades' in result
    assert 'metrics' in result
"""
