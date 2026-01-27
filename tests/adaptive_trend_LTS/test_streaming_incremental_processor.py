"""Tests for StreamingIncrementalProcessor class."""

import pytest
import pandas as pd
import numpy as np

from modules.adaptive_trend_LTS.core.compute_atc_signals.streaming_incremental_processor import (
    StreamingIncrementalProcessor,
)


@pytest.fixture
def sample_config():
    """Sample ATC configuration."""
    return {
        "ema_len": 28,
        "hull_len": 28,
        "wma_len": 28,
        "dema_len": 28,
        "lsma_len": 28,
        "kama_len": 28,
        "De": 0.03,
        "La": 0.02,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
    }


@pytest.fixture
def sample_prices():
    """Sample price series for initialization."""
    np.random.seed(42)
    return pd.Series(np.random.randn(100).cumsum() + 1000)


@pytest.fixture
def streaming_processor(sample_config):
    """StreamingIncrementalProcessor instance."""
    return StreamingIncrementalProcessor(sample_config)


def test_streaming_processor_initialization(streaming_processor):
    """Test that StreamingIncrementalProcessor initializes correctly."""
    assert streaming_processor.batch_atc is not None
    assert streaming_processor.get_symbol_count() == 0
    assert streaming_processor.get_processed_count() == 0


def test_streaming_processor_initialize_symbol(streaming_processor, sample_prices):
    """Test initializing a symbol with historical data."""
    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)

    assert streaming_processor.get_symbol_count() == 1
    assert "BTCUSDT" in streaming_processor.get_symbols()

    # Check that BatchIncrementalATC is initialized
    state = streaming_processor.get_symbol_state("BTCUSDT")
    assert state is not None
    assert "ma_values" in state
    assert "equity" in state
    assert state["initialized"]


def test_streaming_processor_process_live_bar(streaming_processor, sample_prices):
    """Test processing a single live bar."""
    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)

    new_price = sample_prices.iloc[-1] * 1.01
    signal = streaming_processor.process_live_bar("BTCUSDT", new_price)

    assert signal is not None
    assert -1.0 <= signal <= 1.0
    assert streaming_processor.get_processed_count() == 1


def test_streaming_processor_process_live_bar_nonexistent(streaming_processor):
    """Test processing a bar for nonexistent symbol."""
    signal = streaming_processor.process_live_bar("NONEXISTENT", 50000.0)

    assert signal is None


def test_streaming_processor_process_live_bars(streaming_processor, sample_prices):
    """Test processing multiple live bars."""
    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)
    streaming_processor.initialize_symbol("ETHUSDT", sample_prices)

    price_updates = {
        "BTCUSDT": sample_prices.iloc[-1] * 1.01,
        "ETHUSDT": sample_prices.iloc[-1] * 0.99,
    }

    signals = streaming_processor.process_live_bars(price_updates)

    assert len(signals) == 2
    assert "BTCUSDT" in signals
    assert "ETHUSDT" in signals
    for signal in signals.values():
        assert -1.0 <= signal <= 1.0


def test_streaming_processor_process_live_bars_partial_nonexistent(streaming_processor, sample_prices):
    """Test processing with some nonexistent symbols."""
    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)

    price_updates = {
        "BTCUSDT": sample_prices.iloc[-1] * 1.01,
        "NONEXISTENT": 50000.0,
    }

    signals = streaming_processor.process_live_bars(price_updates)

    assert len(signals) == 1
    assert "BTCUSDT" in signals
    assert "NONEXISTENT" not in signals


def test_streaming_processor_get_signal(streaming_processor, sample_prices):
    """Test getting current signal for a symbol."""
    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)

    signal = streaming_processor.get_signal("BTCUSDT")

    # Signal should be None initially (no update yet)
    assert signal is None

    # Update and check again
    new_price = sample_prices.iloc[-1] * 1.01
    streaming_processor.process_live_bar("BTCUSDT", new_price)

    signal_after = streaming_processor.get_signal("BTCUSDT")
    assert signal_after is not None


def test_streaming_processor_get_all_signals(streaming_processor, sample_prices):
    """Test getting all current signals."""
    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)
    streaming_processor.initialize_symbol("ETHUSDT", sample_prices)
    streaming_processor.initialize_symbol("ADAUSDT", sample_prices)

    signals = streaming_processor.get_all_signals()

    assert len(signals) == 3
    for symbol in ["BTCUSDT", "ETHUSDT", "ADAUSDT"]:
        assert symbol in signals


def test_streaming_processor_get_signal_nonexistent(streaming_processor):
    """Test getting signal for nonexistent symbol."""
    signal = streaming_processor.get_signal("NONEXISTENT")
    assert signal is None


def test_streaming_processor_remove_symbol(streaming_processor, sample_prices):
    """Test removing a symbol."""
    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)
    streaming_processor.initialize_symbol("ETHUSDT", sample_prices)

    # Remove one symbol
    result = streaming_processor.remove_symbol("BTCUSDT")

    assert result is True
    assert streaming_processor.get_symbol_count() == 1
    assert "BTCUSDT" not in streaming_processor.get_symbols()
    assert "ETHUSDT" in streaming_processor.get_symbols()


def test_streaming_processor_remove_nonexistent(streaming_processor):
    """Test removing a nonexistent symbol."""
    result = streaming_processor.remove_symbol("NONEXISTENT")

    assert result is False


def test_streaming_processor_reset_symbol(streaming_processor, sample_prices):
    """Test resetting a specific symbol."""
    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)
    streaming_processor.initialize_symbol("ETHUSDT", sample_prices)

    # Update both symbols
    price_updates = {
        "BTCUSDT": sample_prices.iloc[-1] * 1.01,
        "ETHUSDT": sample_prices.iloc[-1] * 0.99,
    }
    streaming_processor.process_live_bars(price_updates)

    # Reset one symbol
    result = streaming_processor.reset_symbol("BTCUSDT")

    assert result is True
    # After reset, symbol should not be initialized
    state_btc = streaming_processor.get_symbol_state("BTCUSDT")
    assert state_btc["initialized"] is False

    # Other symbol should still be initialized
    state_eth = streaming_processor.get_symbol_state("ETHUSDT")
    assert state_eth["initialized"] is True


def test_streaming_processor_reset_all(streaming_processor, sample_prices):
    """Test resetting all symbols."""
    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)
    streaming_processor.initialize_symbol("ETHUSDT", sample_prices)

    # Update both symbols
    price_updates = {
        "BTCUSDT": sample_prices.iloc[-1] * 1.01,
        "ETHUSDT": sample_prices.iloc[-1] * 0.99,
    }
    streaming_processor.process_live_bars(price_updates)

    # Reset all
    streaming_processor.reset_all()

    # All symbols should be un-initialized
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        state = streaming_processor.get_symbol_state(symbol)
        assert state is not None
        assert state["initialized"] is False


def test_streaming_processor_get_symbol_state(streaming_processor, sample_prices):
    """Test getting full state for a symbol."""
    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)

    state = streaming_processor.get_symbol_state("BTCUSDT")

    assert state is not None
    assert "ma_values" in state
    assert "equity" in state
    assert "initialized" in state


def test_streaming_processor_get_symbol_state_nonexistent(streaming_processor):
    """Test getting state for nonexistent symbol."""
    state = streaming_processor.get_symbol_state("NONEXISTENT")
    assert state is None


def test_streaming_processor_get_all_states(streaming_processor, sample_prices):
    """Test getting all states."""
    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)
    streaming_processor.initialize_symbol("ETHUSDT", sample_prices)

    states = streaming_processor.get_all_states()

    assert len(states) == 2
    assert "BTCUSDT" in states
    assert "ETHUSDT" in states
    for symbol, state in states.items():
        assert "ma_values" in state
        assert "equity" in state


def test_streaming_processor_get_symbol_count(streaming_processor, sample_prices):
    """Test getting symbol count."""
    assert streaming_processor.get_symbol_count() == 0

    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)
    assert streaming_processor.get_symbol_count() == 1

    streaming_processor.initialize_symbol("ETHUSDT", sample_prices)
    assert streaming_processor.get_symbol_count() == 2


def test_streaming_processor_get_symbols(streaming_processor, sample_prices):
    """Test getting all symbols."""
    assert streaming_processor.get_symbols() == []

    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)
    streaming_processor.initialize_symbol("ETHUSDT", sample_prices)

    symbols = streaming_processor.get_symbols()

    assert len(symbols) == 2
    assert "BTCUSDT" in symbols
    assert "ETHUSDT" in symbols


def test_streaming_processor_get_processed_count(streaming_processor, sample_prices):
    """Test getting processed count."""
    assert streaming_processor.get_processed_count() == 0

    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)

    # Process multiple bars
    for i in range(5):
        price = 1000 + i * 10
        streaming_processor.process_live_bar("BTCUSDT", float(price))

    assert streaming_processor.get_processed_count() == 5


def test_streaming_processor_get_state(streaming_processor, sample_prices):
    """Test getting overall processor state."""
    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)
    streaming_processor.initialize_symbol("ETHUSDT", sample_prices)

    # Process some bars
    streaming_processor.process_live_bars(
        {
            "BTCUSDT": 1100.0,
            "ETHUSDT": 600.0,
        }
    )

    state = streaming_processor.get_state()

    assert "symbol_count" in state
    assert state["symbol_count"] == 2
    assert state["processed_count"] == 2
    assert "BTCUSDT" in state["symbols"]
    assert "ETHUSDT" in state["symbols"]


def test_streaming_processor_local_state_management(streaming_processor, sample_prices):
    """Test that state persists locally without external store."""
    # Initialize multiple symbols
    for i in range(10):
        symbol = f"SYMBOL_{i}"
        prices = sample_prices.iloc[:100]
        streaming_processor.initialize_symbol(symbol, prices)

    assert streaming_processor.get_symbol_count() == 10

    # Process updates
    for i in range(20):
        for symbol_id in range(10):
            symbol = f"SYMBOL_{symbol_id}"
            price = sample_prices.iloc[-1] * (1 + 0.01 * i)
            streaming_processor.process_live_bar(symbol, float(price))

    # Verify state is maintained locally
    for symbol_id in range(10):
        symbol = f"SYMBOL_{symbol_id}"
        state = streaming_processor.get_symbol_state(symbol)
        assert state is not None
        assert state["initialized"] is True
        assert "ma_values" in state

    assert streaming_processor.get_processed_count() == 200


def test_streaming_processor_multiple_updates(streaming_processor, sample_prices):
    """Test multiple sequential updates maintain state correctly."""
    streaming_processor.initialize_symbol("BTCUSDT", sample_prices)

    # Store initial signal
    initial_signal = streaming_processor.get_signal("BTCUSDT")
    assert initial_signal is None

    # Update multiple times
    signals = []
    for i in range(10):
        price = sample_prices.iloc[-1] * (1 + 0.01 * i)
        signal = streaming_processor.process_live_bar("BTCUSDT", float(price))
        signals.append(signal)

    # All signals should be valid
    for signal in signals:
        assert signal is not None
        assert -1.0 <= signal <= 1.0
