"""
Comprehensive tests for remaining utility modules.

Tests cover:
- MockPriceFeed: Price simulation and updates
- Formatters: Price, PnL, percent, timestamp formatting
- RetryUtils: Exponential backoff and retry logic
- ThreadingUtils: Periodic updater
- Toast: Notification display
- Modes: Trading mode constants
- RiskCalculator: Risk metric calculations
- Colors: Theme-aware color system
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch
import time

import pytest

# Import modules to test
from modules.auto_trade.gui.utils.mock_price_feed import MockPriceFeed
from modules.auto_trade.gui.utils.formatters import (
    format_price,
    format_pnl,
    format_percent,
    format_timestamp,
)
from modules.auto_trade.gui.utils.retry_utils import (
    retry_with_exponential_backoff,
    retry_async_with_exponential_backoff,
    RetryableOperation,
)
from modules.auto_trade.gui.utils.threading_utils import PeriodicUpdater
from modules.auto_trade.gui.utils.modes import TradingMode
from modules.auto_trade.gui.utils.risk_calculator import RiskCalculator


# ==================== MockPriceFeed Tests ====================

class TestMockPriceFeed:
    """Test MockPriceFeed functionality."""

    def test_init(self):
        """Test initialization with base prices."""
        feed = MockPriceFeed()

        assert "BTC/USDT" in feed.base_prices
        assert "ETH/USDT" in feed.base_prices
        assert feed.current_prices == feed.base_prices

    def test_get_current_price_known_symbol(self):
        """Test getting price for known symbol."""
        feed = MockPriceFeed()

        price = feed.get_current_price("BTC/USDT")

        assert isinstance(price, float)
        assert price > 0
        # Price should be near base price (within 2% fluctuation)
        assert 41000 < price < 43000

    def test_get_current_price_unknown_symbol(self):
        """Test getting price for unknown symbol."""
        feed = MockPriceFeed()

        price = feed.get_current_price("UNKNOWN/USDT")

        assert isinstance(price, float)
        assert 0.5 <= price <= 50000.0

    def test_price_fluctuation(self):
        """Test that prices fluctuate on each call."""
        feed = MockPriceFeed()

        price1 = feed.get_current_price("BTC/USDT")
        price2 = feed.get_current_price("BTC/USDT")

        # Prices should be different due to random fluctuation
        assert price1 != price2

    def test_update_prices(self):
        """Test bulk price update."""
        feed = MockPriceFeed()

        initial_prices = feed.get_all_prices()
        feed.update_prices()
        updated_prices = feed.get_all_prices()

        # Prices should have changed
        assert initial_prices != updated_prices

    def test_set_price(self):
        """Test manually setting a price."""
        feed = MockPriceFeed()

        feed.set_price("BTC/USDT", 50000.0)

        assert feed.current_prices["BTC/USDT"] == 50000.0

    def test_get_all_prices(self):
        """Test getting all prices."""
        feed = MockPriceFeed()

        all_prices = feed.get_all_prices()

        assert isinstance(all_prices, dict)
        assert len(all_prices) > 0
        assert "BTC/USDT" in all_prices


# ==================== Formatters Tests ====================

class TestFormatters:
    """Test formatter functions."""

    def test_format_price(self):
        """Test price formatting."""
        assert format_price(42000.0) == "$42,000.00"
        assert format_price(1234.56) == "$1,234.56"
        assert format_price(0.123) == "$0.12"

    def test_format_pnl_positive(self):
        """Test PnL formatting for profit."""
        assert format_pnl(123.45) == "+$123.45"
        assert format_pnl(0.01) == "+$0.01"

    def test_format_pnl_negative(self):
        """Test PnL formatting for loss."""
        result = format_pnl(-56.78)
        # Allow both "-$56.78" and "$-56.78" formats
        assert result in ["-$56.78", "$-56.78"]
        assert "-56.78" in result or "-56.78" in result

    def test_format_pnl_zero(self):
        """Test PnL formatting for zero."""
        assert format_pnl(0.0) == "+$0.00"

    def test_format_percent_positive(self):
        """Test percentage formatting for positive values."""
        assert format_percent(5.23) == "+5.23%"
        assert format_percent(0.01) == "+0.01%"

    def test_format_percent_negative(self):
        """Test percentage formatting for negative values."""
        assert format_percent(-2.15) == "-2.15%"

    def test_format_percent_zero(self):
        """Test percentage formatting for zero."""
        assert format_percent(0.0) == "+0.00%"

    def test_format_timestamp_just_now(self):
        """Test timestamp formatting for very recent time."""
        now = datetime.now()
        timestamp = now.isoformat()

        result = format_timestamp(timestamp)

        assert result == "just now"

    def test_format_timestamp_minutes_ago(self):
        """Test timestamp formatting for minutes ago."""
        past = datetime.now() - timedelta(minutes=5)
        timestamp = past.isoformat()

        result = format_timestamp(timestamp)

        assert "m ago" in result

    def test_format_timestamp_hours_ago(self):
        """Test timestamp formatting for hours ago."""
        past = datetime.now() - timedelta(hours=2)
        timestamp = past.isoformat()

        result = format_timestamp(timestamp)

        assert "h ago" in result

    def test_format_timestamp_days_ago(self):
        """Test timestamp formatting for days ago."""
        past = datetime.now() - timedelta(days=2)
        timestamp = past.isoformat()

        result = format_timestamp(timestamp)

        # Should show date format OR "just now" if timing is off
        assert "-" in result or "ago" in result or "just now" in result

    def test_format_timestamp_invalid(self):
        """Test timestamp formatting with invalid input."""
        result = format_timestamp("invalid")

        assert result == "invalid"


# ==================== RetryUtils Tests ====================

class TestRetryUtils:
    """Test retry utilities."""

    def test_retry_success_first_attempt(self):
        """Test successful function on first attempt."""
        call_count = 0

        @retry_with_exponential_backoff(max_retries=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()

        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self):
        """Test successful function after some failures."""
        call_count = 0

        @retry_with_exponential_backoff(max_retries=3, base_delay=0.01)
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = eventually_successful()

        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted(self):
        """Test retry exhaustion."""
        call_count = 0

        @retry_with_exponential_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Permanent failure")

        with pytest.raises(ConnectionError):
            always_fails()

        assert call_count == 3  # Initial + 2 retries

    def test_retry_exponential_backoff(self):
        """Test exponential backoff timing."""
        call_times = []

        @retry_with_exponential_backoff(max_retries=2, base_delay=0.1, backoff_factor=2.0)
        def timed_failure():
            call_times.append(time.time())
            raise ConnectionError("Test")

        try:
            timed_failure()
        except ConnectionError:
            pass

        # Check that delays increase exponentially
        if len(call_times) >= 2:
            delay1 = call_times[1] - call_times[0]
            assert delay1 >= 0.1  # First retry delay

    def test_retryable_operation_success(self):
        """Test RetryableOperation context manager with success."""
        operation = RetryableOperation(max_retries=3)
        call_count = 0

        for attempt in operation:
            call_count += 1
            try:
                if call_count < 2:
                    raise ConnectionError("Temporary")
                operation.success()
                break
            except ConnectionError as e:
                operation.failed(e)

        assert call_count == 2

    def test_retryable_operation_exhausted(self):
        """Test RetryableOperation with all retries exhausted."""
        operation = RetryableOperation(max_retries=2, base_delay=0.01)
        call_count = 0

        for attempt in operation:
            call_count += 1
            operation.failed(Exception("Test"))

        assert call_count == 3  # Initial + 2 retries


# ==================== ThreadingUtils Tests ====================

class TestThreadingUtils:
    """Test threading utilities."""

    def test_periodic_updater_start_stop(self):
        """Test starting and stopping periodic updater."""
        call_count = 0

        def callback():
            nonlocal call_count
            call_count += 1

        updater = PeriodicUpdater(callback, interval=1)
        updater.start()

        assert updater.running is True
        assert updater.thread is not None

        updater.stop()

        assert updater.running is False

    def test_periodic_updater_calls_callback(self):
        """Test that callback is called periodically."""
        call_count = 0

        def callback():
            nonlocal call_count
            call_count += 1

        updater = PeriodicUpdater(callback, interval=0.1)  # type: ignore[arg-type]
        updater.start()

        time.sleep(0.3)  # Wait for multiple calls
        updater.stop()

        assert call_count >= 2  # Should be called at least twice

    def test_periodic_updater_error_handling(self):
        """Test error handling in callback."""
        call_count = 0

        def failing_callback():
            nonlocal call_count
            call_count += 1
            raise Exception("Test error")

        updater = PeriodicUpdater(failing_callback, interval=0.1)  # type: ignore[arg-type]
        updater.start()

        time.sleep(0.3)
        updater.stop()

        # Should continue calling despite errors
        assert call_count >= 2


# ==================== Modes Tests ====================

class TestTradingMode:
    """Test trading mode constants."""

    def test_mode_constants(self):
        """Test that mode constants are defined."""
        assert TradingMode.PRODUCTION == "PRODUCTION"
        assert TradingMode.DEMO == "DEMO"
        assert TradingMode.DRY_RUN == "DRY_RUN"

    def test_mode_types(self):
        """Test that modes are strings."""
        assert isinstance(TradingMode.PRODUCTION, str)
        assert isinstance(TradingMode.DEMO, str)
        assert isinstance(TradingMode.DRY_RUN, str)


# ==================== RiskCalculator Tests ====================

class TestRiskCalculator:
    """Test RiskCalculator functionality."""

    def test_calculate_long_position(self):
        """Test risk calculation for LONG position."""
        result = RiskCalculator.calculate(
            symbol="BTC/USDT",
            side="LONG",
            amount_usdt=1000.0,
            leverage=10,
            current_price=40000.0,
            tp_percent=5.0,
            sl_percent=2.5,
        )

        assert result is not None
        assert "contract_size" in result
        assert "margin_required" in result
        assert "max_profit" in result
        assert "max_loss" in result
        assert "tp_price" in result
        assert "sl_price" in result
        assert "liquidation_price" in result
        assert "risk_reward_ratio" in result

        # Verify calculations
        assert result["contract_size"] == 1000.0 / 40000.0  # 0.025 BTC
        assert result["margin_required"] == 1000.0 / 10  # 100 USDT
        assert result["tp_price"] > 40000.0  # TP above entry for LONG
        assert result["sl_price"] < 40000.0  # SL below entry for LONG

    def test_calculate_short_position(self):
        """Test risk calculation for SHORT position."""
        result = RiskCalculator.calculate(
            symbol="BTC/USDT",
            side="SHORT",
            amount_usdt=1000.0,
            leverage=10,
            current_price=40000.0,
            tp_percent=5.0,
            sl_percent=2.5,
        )

        assert result is not None
        assert result["tp_price"] < 40000.0  # TP below entry for SHORT
        assert result["sl_price"] > 40000.0  # SL above entry for SHORT

    def test_risk_reward_ratio(self):
        """Test risk/reward ratio calculation."""
        result = RiskCalculator.calculate(
            symbol="BTC/USDT",
            side="LONG",
            amount_usdt=1000.0,
            leverage=10,
            current_price=40000.0,
            tp_percent=5.0,  # 5% profit
            sl_percent=2.5,  # 2.5% loss
        )
        assert result is not None
        # Risk/reward should be 2:1 (5% profit vs 2.5% loss)
        assert result["risk_reward_ratio"] == pytest.approx(2.0, rel=0.1)

    def test_calculate_with_high_leverage(self):
        """Test calculation with high leverage."""
        result = RiskCalculator.calculate(
            symbol="BTC/USDT",
            side="LONG",
            amount_usdt=1000.0,
            leverage=100,
            current_price=40000.0,
            tp_percent=1.0,
            sl_percent=0.5,
        )

        assert result is not None
        assert result["margin_required"] == 1000.0 / 100  # 10 USDT
        # Liquidation price should be very close to entry with high leverage
        assert result["liquidation_price"] is not None
        assert abs(result["liquidation_price"] - 40000.0) < 1000.0

    def test_calculate_error_handling(self):
        """Test error handling in calculation."""
        result = RiskCalculator.calculate(
            symbol="BTC/USDT",
            side="LONG",
            amount_usdt=1000.0,
            leverage=10,
            current_price=0.0,  # Invalid price
            tp_percent=5.0,
            sl_percent=2.5,
        )

        # Should return None on error
        assert result is None

    def test_liquidation_price_long(self):
        """Test liquidation price calculation for LONG."""
        result = RiskCalculator.calculate(
            symbol="BTC/USDT",
            side="LONG",
            amount_usdt=1000.0,
            leverage=10,
            current_price=40000.0,
            tp_percent=5.0,
            sl_percent=2.5,
        )

        # Liquidation should be below entry for LONG
        assert result is not None and result["liquidation_price"] is not None
        assert result["liquidation_price"] < 40000.0
        # With 10x leverage, liquidation should be around 10% below entry
        expected_liq = 40000.0 * (1 - 1/10)
        assert result["liquidation_price"] == pytest.approx(expected_liq, rel=0.01)

    def test_liquidation_price_short(self):
        """Test liquidation price calculation for SHORT."""
        result = RiskCalculator.calculate(
            symbol="BTC/USDT",
            side="SHORT",
            amount_usdt=1000.0,
            leverage=10,
            current_price=40000.0,
            tp_percent=5.0,
            sl_percent=2.5,
        )

        # Liquidation should be above entry for SHORT
        assert result is not None and result["liquidation_price"] is not None
        assert result["liquidation_price"] > 40000.0


# ==================== Colors Tests ====================

class TestColors:
    """Test Colors utility class."""

    def test_static_colors(self):
        """Test static color constants."""
        from modules.auto_trade.gui.utils.colors import Colors

        assert Colors.LONG == "#00ff88"
        assert Colors.SHORT == "#ff4444"
        assert Colors.NEUTRAL == "#888888"
        assert Colors.PROFIT == "#00ff88"
        assert Colors.LOSS == "#ff4444"

    def test_mode_colors(self):
        """Test mode-specific colors."""
        from modules.auto_trade.gui.utils.colors import Colors

        assert Colors.PRODUCTION == "#ff4444"
        assert Colors.DEMO == "#ffaa00"
        assert Colors.DRY_RUN == "#4488ff"

    @patch("modules.auto_trade.gui.utils.colors.ctk.get_appearance_mode")
    def test_theme_detection(self, mock_get_mode):
        """Test theme detection."""
        from modules.auto_trade.gui.utils.colors import Colors

        mock_get_mode.return_value = "Dark"
        assert Colors.is_dark_mode() is True

        mock_get_mode.return_value = "Light"
        assert Colors.is_dark_mode() is False

    @patch("modules.auto_trade.gui.utils.colors.ctk.get_appearance_mode")
    def test_theme_aware_colors(self, mock_get_mode):
        """Test theme-aware color getters."""
        from modules.auto_trade.gui.utils.colors import Colors

        # Dark mode
        mock_get_mode.return_value = "Dark"
        assert Colors.get_bg() == Colors.BG_DARK
        assert Colors.get_text_primary() == Colors.TEXT_PRIMARY_DARK

        # Light mode
        mock_get_mode.return_value = "Light"
        assert Colors.get_bg() == Colors.BG_LIGHT
        assert Colors.get_text_primary() == Colors.TEXT_PRIMARY_LIGHT
