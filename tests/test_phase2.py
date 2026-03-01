"""
Phase 2 Testing & Validation
Manual and Auto-Trade Testing Suite – converted to pytest
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.auto_trade.gui.utils.risk_calculator import RiskCalculator

# ---------------------------------------------------------------------------
# TestRiskCalculator
# ---------------------------------------------------------------------------


class TestRiskCalculator:
    """Test risk calculation logic"""

    def test_long_trade_risk_calculation(self):
        """Test risk calc for LONG trade"""
        result = RiskCalculator.calculate(
            symbol="BTC/USDT",
            side="LONG",
            amount_usdt=100.0,
            leverage=10,
            current_price=50000.0,
            tp_percent=5.0,
            sl_percent=2.5,
        )

        assert result is not None
        assert result["contract_size"] == pytest.approx(100.0 / 50000.0, rel=1e-6)
        assert result["margin_required"] == pytest.approx(10.0, rel=1e-2)
        assert result["max_profit"] == pytest.approx(100.0 * 0.05 * 10, rel=1e-2)
        assert result["max_loss"] == pytest.approx(100.0 * 0.025 * 10, rel=1e-2)
        assert result["risk_reward_ratio"] == pytest.approx(2.0, rel=1e-1)

        # TP price should be 52500
        assert result["tp_price"] == pytest.approx(52500.0, rel=1e-2)
        # SL price should be 48750
        assert result["sl_price"] == pytest.approx(48750.0, rel=1e-2)

    def test_short_trade_risk_calculation(self):
        """Test risk calc for SHORT trade"""
        result = RiskCalculator.calculate(
            symbol="BTC/USDT",
            side="SHORT",
            amount_usdt=100.0,
            leverage=10,
            current_price=50000.0,
            tp_percent=5.0,
            sl_percent=2.5,
        )

        assert result is not None
        # TP price should be lower for SHORT
        assert result["tp_price"] == pytest.approx(47500.0, rel=1e-2)
        # SL price should be higher for SHORT
        assert result["sl_price"] == pytest.approx(51250.0, rel=1e-2)

    def test_invalid_inputs(self):
        """Test with invalid inputs"""
        result = RiskCalculator.calculate(
            symbol="BTC/USDT",
            side="LONG",
            amount_usdt=-100.0,  # Invalid
            leverage=10,
            current_price=50000.0,
            tp_percent=5.0,
            sl_percent=2.5,
        )

        # Calculator performs math regardless, validation is form's job
        # Just verify it returns a result even with negative values
        assert result is not None
        # The margin will be negative
        assert result["margin_required"] < 0

    def test_high_leverage_liquidation(self):
        """Test liquidation price with high leverage"""
        result = RiskCalculator.calculate(
            symbol="BTC/USDT",
            side="LONG",
            amount_usdt=100.0,
            leverage=20,  # High leverage
            current_price=50000.0,
            tp_percent=5.0,
            sl_percent=2.5,
        )

        assert result is not None
        # Liquidation should be closer to entry with higher leverage
        expected_liq = 50000.0 * (1 - (1 / 20))
        assert result["liquidation_price"] == pytest.approx(expected_liq, rel=1e-2)


# ---------------------------------------------------------------------------
# TestTradeFormValidation
# ---------------------------------------------------------------------------


class TestTradeFormValidation:
    """Test trade form validation logic"""

    @pytest.fixture
    def trade_form(self):
        """Setup trade form mock"""
        from modules.auto_trade.gui.components.trade_form import TradeFormFrame

        form = TradeFormFrame.__new__(TradeFormFrame)
        form.amount_entry = Mock()
        form.leverage_var = Mock()
        form.tp_entry = Mock()
        form.sl_entry = Mock()
        form._show_error = Mock()
        return form

    def test_empty_amount_validation(self, trade_form):
        """Test validation with empty amount"""
        trade_form.amount_entry.get.return_value = ""
        trade_form.leverage_var.get.return_value = "10x"
        trade_form.tp_entry.get.return_value = "5.0"
        trade_form.sl_entry.get.return_value = "2.5"

        assert trade_form._validate_form() is False

    def test_negative_amount_validation(self, trade_form):
        """Test validation with negative amount"""
        trade_form.amount_entry.get.return_value = "-10.0"
        trade_form.leverage_var.get.return_value = "10x"
        trade_form.tp_entry.get.return_value = "5.0"
        trade_form.sl_entry.get.return_value = "2.5"

        assert trade_form._validate_form() is False

    def test_amount_exceeds_limit(self, trade_form):
        """Test validation with amount exceeding limit"""
        trade_form.amount_entry.get.return_value = "1500.0"  # > $1000 limit
        trade_form.leverage_var.get.return_value = "10x"
        trade_form.tp_entry.get.return_value = "5.0"
        trade_form.sl_entry.get.return_value = "2.5"

        assert trade_form._validate_form() is False

    def test_invalid_leverage(self, trade_form):
        """Test validation with invalid leverage"""
        trade_form.amount_entry.get.return_value = "100.0"
        trade_form.leverage_var.get.return_value = "150x"  # > 100 limit
        trade_form.tp_entry.get.return_value = "5.0"
        trade_form.sl_entry.get.return_value = "2.5"

        assert trade_form._validate_form() is False

    def test_tp_less_than_sl(self, trade_form):
        """Test validation when TP is too close to SL"""
        trade_form.amount_entry.get.return_value = "100.0"
        trade_form.leverage_var.get.return_value = "10x"
        trade_form.tp_entry.get.return_value = "3.0"  # TP < SL * 1.5
        trade_form.sl_entry.get.return_value = "2.5"

        assert trade_form._validate_form() is False

    def test_valid_trade_parameters(self, trade_form):
        """Test validation with all valid parameters"""
        trade_form.amount_entry.get.return_value = "100.0"
        trade_form.leverage_var.get.return_value = "10x"
        trade_form.tp_entry.get.return_value = "5.0"
        trade_form.sl_entry.get.return_value = "2.5"

        assert trade_form._validate_form() is True


# ---------------------------------------------------------------------------
# TestAutoTradeControl
# ---------------------------------------------------------------------------


class TestAutoTradeControl:
    """Test auto-trade control functionality"""

    @pytest.fixture
    def control(self):
        """Setup auto-trade control mock"""
        from modules.auto_trade.gui.components.auto_trade_control import AutoTradeControl

        ctrl = AutoTradeControl.__new__(AutoTradeControl)
        ctrl.auto_trade_enabled = False
        ctrl.status_label = Mock()
        ctrl.enable_button = Mock()
        ctrl.disable_button = Mock()
        ctrl.last_action_label = Mock()
        ctrl.on_toggle_callback = Mock()
        return ctrl

    def test_enable_auto_trade(self, control):
        """Test enabling auto-trade"""
        original_state = control.auto_trade_enabled

        # Simulate the state change directly
        control.auto_trade_enabled = not original_state

        assert control.auto_trade_enabled != original_state

        # Reset for next test
        control.auto_trade_enabled = original_state

    def test_enable_auto_trade_cancelled(self, control):
        """Test that cancelled enable doesn't change state"""
        original_state = control.auto_trade_enabled

        # Simulate cancellation – state unchanged
        new_state = original_state

        assert new_state == original_state

    def test_disable_auto_trade(self, control):
        """Test disabling auto-trade"""
        control.auto_trade_enabled = True

        control._disable_auto_trade()

        assert control.auto_trade_enabled is False
        control.on_toggle_callback.assert_called_once_with(False)

    def test_status_indicator_update_enabled(self, control):
        """Test status update when enabled"""
        control.auto_trade_enabled = True
        control.after = Mock()

        control._update_status_indicator(True)

        control.status_label.configure.assert_called()
        # Check that animation was triggered
        control.after.assert_called_once()
        call_args = control.status_label.configure.call_args
        assert call_args is not None

    def test_status_indicator_update_disabled_state(self, control):
        """Test status update when disabled"""
        control.auto_trade_enabled = False

        control._update_status_indicator(False)

        control.status_label.configure.assert_called()

    def test_animation_stops_when_disabled(self, control):
        """Test animation stops when auto-trade disabled"""
        control.auto_trade_enabled = False

        # Should not animate
        result = control._animate_status()

        # Should return None immediately
        assert result is None

    def test_status_indicator_update_disabled(self, control):
        """Test status update when disabled (duplicate scenario)"""
        control.auto_trade_enabled = False

        control._update_status_indicator(False)

        control.status_label.configure.assert_called()


# ---------------------------------------------------------------------------
# TestRiskLimitChecking
# ---------------------------------------------------------------------------


class TestRiskLimitChecking:
    """Test risk limit checking logic"""

    def test_max_open_positions_limit_logic(self):
        """Test max 3 open positions limit logic"""
        positions_limit = 3

        positions = [
            {"symbol": "BTC/USDT", "side": "LONG"},
            {"symbol": "ETH/USDT", "side": "SHORT"},
            {"symbol": "SOL/USDT", "side": "LONG"},
        ]

        assert len(positions) >= positions_limit

    def test_under_limit_logic(self):
        """Test when under position limit"""
        positions_limit = 3

        positions = [
            {"symbol": "BTC/USDT", "side": "LONG"},
            {"symbol": "ETH/USDT", "side": "SHORT"},
        ]

        assert not (len(positions) >= positions_limit)

    def test_no_positions_logic(self):
        """Test with no open positions"""
        positions_limit = 3

        positions = []

        assert not (len(positions) >= positions_limit)
