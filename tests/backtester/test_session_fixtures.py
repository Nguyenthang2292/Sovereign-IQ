"""
Tests demonstrating session fixtures for memory optimization.

These tests use session-scoped fixtures to reduce RAM usage by sharing data across tests.
Run with: pytest tests/backtester/test_session_fixtures.py -v
"""


def test_session_fixture_basic(session_mock_data_fetcher):
    """Test using session data fetcher for basic backtesting."""
    from modules.backtester import FullBacktester

    backtester = FullBacktester(session_mock_data_fetcher)
    result = backtester.backtest(symbol="BTC/USDT", timeframe="1h", signal_type="LONG", lookback=50)

    assert "trades" in result
    assert "metrics" in result
    assert isinstance(result["trades"], list)
    assert "win_rate" in result["metrics"]


def test_optimized_fixture_basic(optimized_mock_data_fetcher):
    """Test using optimized data fetcher (function scope with session data)."""
    from modules.backtester import FullBacktester

    backtester = FullBacktester(optimized_mock_data_fetcher)
    result = backtester.backtest(symbol="BTC/USDT", timeframe="1h", signal_type="LONG", lookback=50)

    assert "trades" in result
    assert "metrics" in result
    assert isinstance(result["trades"], list)
    assert "win_rate" in result["metrics"]


def test_session_data_direct(session_small_df):
    """Test using session DataFrame directly."""
    assert len(session_small_df) == 50
    assert "open" in session_small_df.columns
    assert "close" in session_small_df.columns
    assert "high" in session_small_df.columns
    assert "low" in session_small_df.columns
    assert "volume" in session_small_df.columns

    # Verify data is reasonable
    assert session_small_df["close"].min() > 0
    assert session_small_df["high"].max() > session_small_df["low"].min()


def test_session_medium_data(session_medium_df):
    """Test using session medium DataFrame."""
    assert len(session_medium_df) == 150
    assert "open" in session_medium_df.columns
    assert "close" in session_medium_df.columns
    assert "high" in session_medium_df.columns
    assert "low" in session_medium_df.columns
    assert "volume" in session_medium_df.columns

    # Verify data is reasonable
    assert session_medium_df["close"].min() > 0
    assert session_medium_df["high"].max() > session_medium_df["low"].min()


def test_session_data_consistency(session_small_df, session_mock_data_fetcher):
    """Test that session data fetcher returns the same data as session DataFrame."""
    # Fetch data using the session fetcher
    fetched_df, exchange = session_mock_data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=50)

    # Should return the same data
    assert exchange == "binance"
    assert len(fetched_df) == len(session_small_df)
    assert fetched_df["close"].equals(session_small_df["close"])


def test_session_fixture_performance(session_mock_data_fetcher):
    """Test that session fixtures provide consistent performance."""
    import time

    from modules.backtester import FullBacktester

    backtester = FullBacktester(session_mock_data_fetcher)

    start_time = time.time()
    result = backtester.backtest(symbol="BTC/USDT", timeframe="1h", signal_type="LONG", lookback=50)
    elapsed = time.time() - start_time

    # Should complete quickly (< 0.5 seconds typically)
    assert elapsed < 1.0, f"Test took too long: {elapsed:.3f}s"
    assert "trades" in result
    assert "metrics" in result
