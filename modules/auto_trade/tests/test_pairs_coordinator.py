"""
Tests for PairsCoordinator.

Unit tests for pairs trading orchestration including
regime detection, position sizing, and atomic execution.
"""

from unittest.mock import MagicMock, patch

import pytest

from modules.auto_trade.execution.correlation_scanner import HedgeCandidate
from modules.auto_trade.execution.order_builder import OrderTicket
from modules.auto_trade.execution.pairs_coordinator import (
    PairExecutionResult,
    PairsCoordinator,
    PairsSettings,
    PairsState,
)


def test_pairs_settings_defaults():
    """Test PairsSettings values."""
    settings = PairsSettings()

    assert settings.enabled is False
    assert settings.min_correlation == 0.65
    assert settings.lookback == 100
    assert settings.timeframe == "1h"
    assert settings.refresh_interval == 7200
    assert settings.adx_low == 20
    assert settings.adx_high == 30
    assert settings.stat_arb_direction == "opposite"
    assert settings.momentum_direction == "opposite"
    assert settings.blended_direction == "correlation_based"
    assert settings.drift_threshold == 0.15
    assert settings.hedge_leverage_min == 1
    assert settings.hedge_leverage_max == 5


def test_pairs_settings_custom():
    """Test PairsSettings with custom values."""
    settings = PairsSettings(
        enabled=True,
        min_correlation=0.75,
        lookback=200,
        timeframe="4h",
        refresh_interval=3600,
        adx_low=25,
        adx_high=35,
        stat_arb_direction="same",
        momentum_direction="same",
        blended_direction="opposite",
        drift_threshold=0.20,
        hedge_leverage_min=2,
        hedge_leverage_max=10,
    )

    assert settings.enabled is True
    assert settings.min_correlation == 0.75
    assert settings.lookback == 200
    assert settings.timeframe == "4h"
    assert settings.refresh_interval == 3600
    assert settings.adx_low == 25
    assert settings.adx_high == 35
    assert settings.stat_arb_direction == "same"
    assert settings.momentum_direction == "same"
    assert settings.blended_direction == "opposite"
    assert settings.drift_threshold == 0.20
    assert settings.hedge_leverage_min == 2
    assert settings.hedge_leverage_max == 10


def test_pairs_state_defaults():
    """Test PairsState default values."""
    state = PairsState()

    assert state.enabled is False
    assert state.active_pairs == {}
    assert state.last_scan_time is None


def test_pairs_coordinator_initialization():
    """Test PairsCoordinator initialization."""
    coordinator = PairsCoordinator()

    assert coordinator._correlation_scanner is None
    assert coordinator._order_executor is None
    assert coordinator._order_builder is None
    assert coordinator._risk_manager is None
    assert coordinator._settings is not None
    assert coordinator._state is not None


def test_pairs_coordinator_with_settings():
    """Test PairsCoordinator with custom settings."""
    settings = PairsSettings(enabled=True, min_correlation=0.80)
    coordinator = PairsCoordinator(settings=settings)

    assert coordinator._settings.enabled is True
    assert coordinator._settings.min_correlation == 0.80


def test_should_activate_pairs_disabled():
    """Test should_activate_pairs returns False when disabled."""
    coordinator = PairsCoordinator(settings=PairsSettings(enabled=False))

    result = coordinator.should_activate_pairs("BTC/USDT")

    assert result is False


def test_should_activate_pairs_enabled():
    """Test should_activate_pairs returns True when enabled."""
    coordinator = PairsCoordinator(settings=PairsSettings(enabled=True))

    result = coordinator.should_activate_pairs("BTC/USDT")

    assert result is True


def test_update_settings():
    """Test updating pairs trading settings."""
    coordinator = PairsCoordinator()
    new_settings = PairsSettings(enabled=True, min_correlation=0.90)

    coordinator.update_settings(new_settings)

    assert coordinator._settings.enabled is True
    assert coordinator._settings.min_correlation == 0.90


@patch("modules.auto_trade.execution.pairs_coordinator.CorrelationScanner")
def test_find_hedge_symbol_success(mock_scanner_cls):
    """Test finding hedge symbol successfully."""
    mock_scanner = MagicMock()
    mock_scanner_cls.return_value = mock_scanner

    candidate = HedgeCandidate(
        symbol="ETH/USDT",
        correlation=0.85,
        hedge_ratio=1.2,
        kalman_hedge_ratio=1.15,
        score=1.0,
    )
    mock_scanner.scan_hedge_candidates.return_value = [candidate]

    coordinator = PairsCoordinator()
    coordinator._correlation_scanner = mock_scanner

    result = coordinator.find_hedge_symbol("BTC/USDT", ["ETH/USDT", "BNB/USDT"])

    assert result is not None
    assert result.symbol == "ETH/USDT"
    assert result.correlation == 0.85
    mock_scanner.scan_hedge_candidates.assert_called_once()


@patch("modules.auto_trade.execution.pairs_coordinator.CorrelationScanner")
def test_find_hedge_symbol_no_candidates(mock_scanner_cls):
    """Test finding hedge symbol with no candidates."""
    mock_scanner = MagicMock()
    mock_scanner_cls.return_value = mock_scanner
    mock_scanner.scan_hedge_candidates.return_value = []

    coordinator = PairsCoordinator()
    coordinator._correlation_scanner = mock_scanner

    result = coordinator.find_hedge_symbol("BTC/USDT")

    assert result is None


@patch("modules.auto_trade.execution.pairs_coordinator.CorrelationScanner")
def test_determine_regime_stat_arb(mock_scanner_cls):
    """Test regime determination for STAT_ARB."""
    mock_scanner = MagicMock()
    mock_scanner_cls.return_value = mock_scanner
    mock_scanner.calculate_adx_for_regime.return_value = "STAT_ARB"

    coordinator = PairsCoordinator()
    coordinator._correlation_scanner = mock_scanner

    regime = coordinator.determine_regime("BTC/USDT", "ETH/USDT")

    assert regime == "STAT_ARB"


@patch("modules.auto_trade.execution.pairs_coordinator.CorrelationScanner")
def test_determine_regime_momentum(mock_scanner_cls):
    """Test regime determination for MOMENTUM."""
    mock_scanner = MagicMock()
    mock_scanner_cls.return_value = mock_scanner
    mock_scanner.calculate_adx_for_regime.return_value = "MOMENTUM"

    coordinator = PairsCoordinator()
    coordinator._correlation_scanner = mock_scanner

    regime = coordinator.determine_regime("BTC/USDT", "ETH/USDT")

    assert regime == "MOMENTUM"


@patch("modules.auto_trade.execution.pairs_coordinator.CorrelationScanner")
def test_determine_regime_fallback_to_blended(mock_scanner_cls):
    """Test regime determination falls back to BLENDED on error."""
    mock_scanner = MagicMock()
    mock_scanner_cls.return_value = mock_scanner
    mock_scanner.calculate_adx_for_regime.return_value = None

    coordinator = PairsCoordinator()
    coordinator._correlation_scanner = mock_scanner

    regime = coordinator.determine_regime("BTC/USDT", "ETH/USDT")

    assert regime == "BLENDED"


@patch("modules.auto_trade.execution.pairs_coordinator.CorrelationScanner")
def test_determine_regime_uses_settings_thresholds(mock_scanner_cls):
    """Test determine_regime passes configured ADX thresholds to scanner."""
    mock_scanner = MagicMock()
    mock_scanner_cls.return_value = mock_scanner
    mock_scanner.calculate_adx_for_regime.return_value = "STAT_ARB"

    settings = PairsSettings(adx_low=18, adx_high=28)
    coordinator = PairsCoordinator(settings=settings)
    coordinator._correlation_scanner = mock_scanner

    regime = coordinator.determine_regime("BTC/USDT", "ETH/USDT")

    assert regime == "STAT_ARB"
    mock_scanner.calculate_adx_for_regime.assert_called_once_with(
        "BTC/USDT",
        "ETH/USDT",
        adx_low=18,
        adx_high=28,
    )


def test_determine_hedge_direction_opposite():
    """Test hedge direction with opposite config."""
    coordinator = PairsCoordinator(
        settings=PairsSettings(
            stat_arb_direction="opposite",
            momentum_direction="opposite",
            blended_direction="opposite",
        )
    )

    result = coordinator.determine_hedge_direction("BUY", "STAT_ARB")
    assert result == "SELL"

    result = coordinator.determine_hedge_direction("SELL", "STAT_ARB")
    assert result == "BUY"


def test_determine_hedge_direction_same():
    """Test hedge direction with same config."""
    coordinator = PairsCoordinator(
        settings=PairsSettings(
            stat_arb_direction="same",
            momentum_direction="same",
            blended_direction="same",
        )
    )

    result = coordinator.determine_hedge_direction("BUY", "STAT_ARB")
    assert result == "BUY"

    result = coordinator.determine_hedge_direction("SELL", "STAT_ARB")
    assert result == "SELL"


def test_determine_hedge_direction_correlation_based():
    """Test hedge direction with correlation_based config (defaults to opposite)."""
    coordinator = PairsCoordinator(
        settings=PairsSettings(
            stat_arb_direction="correlation_based",
            momentum_direction="correlation_based",
            blended_direction="correlation_based",
        )
    )

    result = coordinator.determine_hedge_direction("BUY", "STAT_ARB")
    assert result == "SELL"


def test_pair_execution_result_dataclass():
    """Test PairExecutionResult dataclass."""
    result = PairExecutionResult(
        success=True,
        pair_id="test-pair-123",
    )

    assert result.success is True
    assert result.pair_id == "test-pair-123"
    assert result.signal_ticket is None
    assert result.hedge_ticket is None
    assert result.error is None
    assert result.rollback_performed is False


def test_pair_execution_result_with_error():
    """Test PairExecutionResult with error."""
    result = PairExecutionResult(
        success=False,
        pair_id="test-pair-123",
        error="Insufficient balance",
    )

    assert result.success is False
    assert result.error == "Insufficient balance"


def test_calculate_position_sizes_stat_arb_no_crash_and_clamps_leverage():
    """STAT_ARB sizing should not crash and should clamp leverage by settings."""
    coordinator = PairsCoordinator(
        settings=PairsSettings(
            hedge_leverage_min=1,
            hedge_leverage_max=5,
        )
    )

    signal_size, hedge_size, signal_lev, hedge_lev = coordinator.calculate_position_sizes(
        regime="STAT_ARB",
        signal_symbol="BTC/USDT",
        hedge_symbol="ETH/USDT",
        signal_side="BUY",
        total_position_size=100.0,
        hedge_ratio=10.0,
        hedge_correlation=0.8,
        signal_leverage=3,
    )

    assert signal_size > 0
    assert hedge_size > 0
    assert signal_lev == 3
    assert hedge_lev == 5


def test_calculate_position_sizes_blended_no_crash_and_clamps_leverage():
    """BLENDED sizing should not crash and should clamp leverage by settings."""
    coordinator = PairsCoordinator(
        settings=PairsSettings(
            hedge_leverage_min=2,
            hedge_leverage_max=4,
        )
    )

    signal_size, hedge_size, signal_lev, hedge_lev = coordinator.calculate_position_sizes(
        regime="BLENDED",
        signal_symbol="BTC/USDT",
        hedge_symbol="ETH/USDT",
        signal_side="SELL",
        total_position_size=200.0,
        hedge_ratio=5.0,
        hedge_correlation=0.6,
        signal_leverage=2,
    )

    assert signal_size > 0
    assert hedge_size > 0
    assert signal_lev == 2
    assert hedge_lev == 4


@pytest.mark.asyncio
async def test_execute_pair_atomically_success_tracks_active_pair():
    """Atomic execution succeeds when both legs succeed."""
    coordinator = PairsCoordinator()
    order_executor = MagicMock()
    order_executor.place_order.side_effect = [
        {"success": True},
        {"success": True},
    ]

    signal_ticket = OrderTicket(symbol="BTC/USDT", side="BUY", amount=100.0, leverage=2)
    hedge_ticket = OrderTicket(symbol="ETH/USDT", side="SELL", amount=80.0, leverage=2)

    result = await coordinator.execute_pair_atomically(signal_ticket, hedge_ticket, order_executor)

    assert result.success is True
    assert result.error is None
    assert result.pair_id in coordinator._state.active_pairs
    assert order_executor.place_order.call_count == 2


@pytest.mark.asyncio
async def test_execute_pair_atomically_rolls_back_on_hedge_failure():
    """Atomic execution rolls back signal leg when hedge leg fails."""
    coordinator = PairsCoordinator()
    order_executor = MagicMock()
    order_executor.place_order.side_effect = [
        {"success": True},
        {"success": False, "error": "hedge failed"},
        {"success": True},
    ]

    signal_ticket = OrderTicket(symbol="BTC/USDT", side="BUY", amount=100.0, leverage=2)
    hedge_ticket = OrderTicket(symbol="ETH/USDT", side="SELL", amount=80.0, leverage=2)

    result = await coordinator.execute_pair_atomically(signal_ticket, hedge_ticket, order_executor)

    assert result.success is False
    assert result.rollback_performed is True
    assert "rollback succeeded" in (result.error or "")
    assert order_executor.place_order.call_count == 3
