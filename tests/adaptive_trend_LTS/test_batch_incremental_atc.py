"""Tests for BatchIncrementalATC class."""

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_incremental_atc import BatchIncrementalATC


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
    return pd.Series(np.random.randn(100).cumsum() + 100)


@pytest.fixture
def batch_atc(sample_config):
    """BatchIncrementalATC instance with sample config."""
    return BatchIncrementalATC(sample_config)


def test_batch_initialization(batch_atc, sample_config):
    """Test that BatchIncrementalATC initializes correctly."""
    assert batch_atc.config == sample_config
    assert batch_atc.instances == {}
    assert not batch_atc.initialized


def test_batch_add_symbol(batch_atc, sample_prices):
    """Test adding a symbol to the batch."""
    batch_atc.add_symbol("BTCUSDT", sample_prices)

    assert "BTCUSDT" in batch_atc.instances
    assert batch_atc.get_symbol_count() == 1
    assert "BTCUSDT" in batch_atc.get_symbols()


def test_batch_add_multiple_symbols(batch_atc, sample_prices):
    """Test adding multiple symbols to the batch."""
    batch_atc.add_symbol("BTCUSDT", sample_prices)
    batch_atc.add_symbol("ETHUSDT", sample_prices)
    batch_atc.add_symbol("ADAUSDT", sample_prices)

    assert batch_atc.get_symbol_count() == 3
    assert len(batch_atc.get_symbols()) == 3


def test_batch_add_duplicate_symbol(batch_atc, sample_prices):
    """Test adding a duplicate symbol replaces the existing one."""
    batch_atc.add_symbol("BTCUSDT", sample_prices)
    first_instance = batch_atc.instances["BTCUSDT"]

    # Add same symbol again with different data
    new_prices = sample_prices * 1.1
    batch_atc.add_symbol("BTCUSDT", new_prices)
    second_instance = batch_atc.instances["BTCUSDT"]

    # Should be different instances
    assert first_instance is not second_instance
    assert batch_atc.get_symbol_count() == 1


def test_batch_update_symbol(batch_atc, sample_prices):
    """Test updating a single symbol."""
    batch_atc.add_symbol("BTCUSDT", sample_prices)

    new_price = sample_prices.iloc[-1] * 1.01
    signal = batch_atc.update_symbol("BTCUSDT", new_price)

    assert signal is not None
    assert -1.0 <= signal <= 1.0


def test_batch_update_nonexistent_symbol(batch_atc):
    """Test updating a symbol that doesn't exist."""
    signal = batch_atc.update_symbol("NONEXISTENT", 50000.0)
    assert signal is None


def test_batch_update_all(batch_atc, sample_prices):
    """Test updating all symbols at once."""
    batch_atc.add_symbol("BTCUSDT", sample_prices)
    batch_atc.add_symbol("ETHUSDT", sample_prices)
    batch_atc.add_symbol("ADAUSDT", sample_prices)

    price_updates = {
        "BTCUSDT": sample_prices.iloc[-1] * 1.01,
        "ETHUSDT": sample_prices.iloc[-1] * 0.99,
        "ADAUSDT": sample_prices.iloc[-1] * 1.02,
    }

    signals = batch_atc.update_all(price_updates)

    assert len(signals) == 3
    assert "BTCUSDT" in signals
    assert "ETHUSDT" in signals
    assert "ADAUSDT" in signals
    for signal in signals.values():
        assert -1.0 <= signal <= 1.0


def test_batch_update_all_partial_nonexistent(batch_atc, sample_prices):
    """Test updating with some nonexistent symbols."""
    batch_atc.add_symbol("BTCUSDT", sample_prices)

    price_updates = {
        "BTCUSDT": sample_prices.iloc[-1] * 1.01,
        "NONEXISTENT": 1000.0,
    }

    signals = batch_atc.update_all(price_updates)

    assert len(signals) == 1
    assert "BTCUSDT" in signals
    assert "NONEXISTENT" not in signals


def test_batch_get_all_signals(batch_atc, sample_prices):
    """Test getting all current signals."""
    batch_atc.add_symbol("BTCUSDT", sample_prices)
    batch_atc.add_symbol("ETHUSDT", sample_prices)

    # Update symbols to compute signals
    price_updates = {
        "BTCUSDT": sample_prices.iloc[-1] * 1.01,
        "ETHUSDT": sample_prices.iloc[-1] * 0.99,
    }
    batch_atc.update_all(price_updates)

    signals = batch_atc.get_all_signals()

    assert len(signals) == 2
    assert "BTCUSDT" in signals
    assert "ETHUSDT" in signals
    for signal in signals.values():
        assert -1.0 <= signal <= 1.0


def test_batch_get_symbol_signal(batch_atc, sample_prices):
    """Test getting signal for a specific symbol."""
    batch_atc.add_symbol("BTCUSDT", sample_prices)

    # Update to compute signal
    new_price = sample_prices.iloc[-1] * 1.01
    batch_atc.update_symbol("BTCUSDT", new_price)

    signal = batch_atc.get_symbol_signal("BTCUSDT")
    assert signal is not None
    assert -1.0 <= signal <= 1.0


def test_batch_get_symbol_signal_nonexistent(batch_atc):
    """Test getting signal for a nonexistent symbol."""
    signal = batch_atc.get_symbol_signal("NONEXISTENT")
    assert signal is None


def test_batch_remove_symbol(batch_atc, sample_prices):
    """Test removing a symbol from the batch."""
    batch_atc.add_symbol("BTCUSDT", sample_prices)
    batch_atc.add_symbol("ETHUSDT", sample_prices)

    result = batch_atc.remove_symbol("BTCUSDT")

    assert result is True
    assert "BTCUSDT" not in batch_atc.instances
    assert batch_atc.get_symbol_count() == 1


def test_batch_remove_nonexistent_symbol(batch_atc):
    """Test removing a nonexistent symbol."""
    result = batch_atc.remove_symbol("NONEXISTENT")
    assert result is False


def test_batch_reset_symbol(batch_atc, sample_prices):
    """Test resetting a specific symbol."""
    batch_atc.add_symbol("BTCUSDT", sample_prices)
    batch_atc.add_symbol("ETHUSDT", sample_prices)

    # Update one symbol
    new_price = sample_prices.iloc[-1] * 1.01
    batch_atc.update_symbol("BTCUSDT", new_price)

    # Reset the symbol
    result = batch_atc.reset_symbol("BTCUSDT")

    assert result is True
    # After reset, symbol should not be initialized
    assert not batch_atc.instances["BTCUSDT"].state["initialized"]


def test_batch_reset_nonexistent_symbol(batch_atc):
    """Test resetting a nonexistent symbol."""
    result = batch_atc.reset_symbol("NONEXISTENT")
    assert result is False


def test_batch_reset_all(batch_atc, sample_prices):
    """Test resetting all symbols."""
    batch_atc.add_symbol("BTCUSDT", sample_prices)
    batch_atc.add_symbol("ETHUSDT", sample_prices)

    # Update symbols
    price_updates = {
        "BTCUSDT": sample_prices.iloc[-1] * 1.01,
        "ETHUSDT": sample_prices.iloc[-1] * 0.99,
    }
    batch_atc.update_all(price_updates)

    # Reset all
    batch_atc.reset_all()

    # All symbols should be un-initialized
    for symbol in batch_atc.instances:
        assert not batch_atc.instances[symbol].state["initialized"]


def test_batch_get_symbol_state(batch_atc, sample_prices):
    """Test getting full state for a specific symbol."""
    batch_atc.add_symbol("BTCUSDT", sample_prices)

    state = batch_atc.get_symbol_state("BTCUSDT")

    assert state is not None
    assert "ma_values" in state
    assert "equity" in state
    assert "signal" in state
    assert "initialized" in state


def test_batch_get_symbol_state_nonexistent(batch_atc):
    """Test getting state for a nonexistent symbol."""
    state = batch_atc.get_symbol_state("NONEXISTENT")
    assert state is None


def test_batch_get_all_states(batch_atc, sample_prices):
    """Test getting all states."""
    batch_atc.add_symbol("BTCUSDT", sample_prices)
    batch_atc.add_symbol("ETHUSDT", sample_prices)

    states = batch_atc.get_all_states()

    assert len(states) == 2
    assert "BTCUSDT" in states
    assert "ETHUSDT" in states
    for state in states.values():
        assert "ma_values" in state
        assert "equity" in state


def test_batch_shared_state_management(batch_atc, sample_prices):
    """Test that state updates correctly for all symbols in batch."""
    batch_atc.add_symbol("BTCUSDT", sample_prices)
    batch_atc.add_symbol("ETHUSDT", sample_prices)

    # Get initial signals
    _ = batch_atc.get_all_signals()

    # Update all symbols
    price_updates = {
        "BTCUSDT": sample_prices.iloc[-1] * 1.01,
        "ETHUSDT": sample_prices.iloc[-1] * 0.99,
    }
    batch_atc.update_all(price_updates)

    # Get new signals
    _ = batch_atc.get_all_signals()

    # All signals should have values
    assert len(batch_atc.get_all_signals()) == 2
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        assert symbol in batch_atc.get_all_signals()
        assert -1.0 <= batch_atc.get_all_signals()[symbol] <= 1.0


def test_batch_symbol_count(batch_atc, sample_prices):
    """Test getting symbol count."""
    assert batch_atc.get_symbol_count() == 0

    batch_atc.add_symbol("BTCUSDT", sample_prices)
    assert batch_atc.get_symbol_count() == 1

    batch_atc.add_symbol("ETHUSDT", sample_prices)
    assert batch_atc.get_symbol_count() == 2

    batch_atc.remove_symbol("BTCUSDT")
    assert batch_atc.get_symbol_count() == 1


def test_batch_multiple_updates(batch_atc, sample_prices):
    """Test multiple sequential batch updates."""
    batch_atc.add_symbol("BTCUSDT", sample_prices)
    batch_atc.add_symbol("ETHUSDT", sample_prices)

    # Perform multiple updates
    for i in range(10):
        price_updates = {
            "BTCUSDT": sample_prices.iloc[-1] * (1 + i * 0.01),
            "ETHUSDT": sample_prices.iloc[-1] * (1 - i * 0.01),
        }
        signals = batch_atc.update_all(price_updates)

        # Verify all signals are valid
        assert len(signals) == 2
        for signal in signals.values():
            assert -1.0 <= signal <= 1.0
