"""
Tests for Negative Breakeven Logic
==================================

Unit tests for negative breakeven calculations and trigger conditions.

Created: 2026-02-06
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from modules.auto_trade.execution.negative_breakeven import (
    calculate_profit_pct,
    has_hit_stop_loss,
    should_trigger_negative_be,
    calculate_take_profit_for_be,
    NegativeBreakevenLogic,
)


class TestCalculateProfitPct:
    """Tests for calculate_profit_pct function."""

    def test_long_position_profit(self):
        """LONG position with profit."""
        assert calculate_profit_pct(100.0, 103.0, "LONG") == 3.0

    def test_long_position_loss(self):
        """LONG position with loss."""
        assert calculate_profit_pct(100.0, 97.0, "LONG") == -3.0

    def test_short_position_profit(self):
        """SHORT position with profit."""
        assert calculate_profit_pct(100.0, 97.0, "SHORT") == 3.0

    def test_short_position_loss(self):
        """SHORT position with loss."""
        assert calculate_profit_pct(100.0, 103.0, "SHORT") == -3.0

    def test_breakeven(self):
        """Position at breakeven."""
        assert calculate_profit_pct(100.0, 100.0, "LONG") == 0.0
        assert calculate_profit_pct(100.0, 100.0, "SHORT") == 0.0

    def test_invalid_entry_price(self):
        """Invalid entry price should return 0."""
        assert calculate_profit_pct(0.0, 100.0, "LONG") == 0.0
        assert calculate_profit_pct(-1.0, 100.0, "LONG") == 0.0

    def test_case_insensitive_side(self):
        """Side parameter should be case insensitive."""
        assert calculate_profit_pct(100.0, 103.0, "long") == 3.0
        assert calculate_profit_pct(100.0, 103.0, "LONG") == 3.0
        assert calculate_profit_pct(100.0, 97.0, "short") == 3.0
        assert calculate_profit_pct(100.0, 97.0, "SHORT") == 3.0


class TestHasHitStopLoss:
    """Tests for has_hit_stop_loss function."""

    def test_long_hit_sl(self):
        """LONG position hits stop loss."""
        assert has_hit_stop_loss(95.0, 98.0, "LONG") is True  # mark < stop_loss
        assert has_hit_stop_loss(98.0, 98.0, "LONG") is True  # mark == stop_loss

    def test_long_not_hit_sl(self):
        """LONG position hasn't hit stop loss."""
        assert has_hit_stop_loss(100.0, 98.0, "LONG") is False
        assert has_hit_stop_loss(99.0, 98.0, "LONG") is False

    def test_short_hit_sl(self):
        """SHORT position hits stop loss."""
        assert has_hit_stop_loss(105.0, 102.0, "SHORT") is True  # mark > stop_loss
        assert has_hit_stop_loss(102.0, 102.0, "SHORT") is True  # mark == stop_loss

    def test_short_not_hit_sl(self):
        """SHORT position hasn't hit stop loss."""
        assert has_hit_stop_loss(100.0, 102.0, "SHORT") is False
        assert has_hit_stop_loss(101.0, 102.0, "SHORT") is False


class TestShouldTriggerNegativeBE:
    """Tests for should_trigger_negative_be function."""

    def test_trigger_when_loss_exceeds_threshold(self):
        """Should trigger when loss >= threshold."""
        assert (
            should_trigger_negative_be(
                profit_pct=-3.0,
                threshold_pct=2.0,
                mark_price=97.0,
                stop_loss=95.0,
                side="LONG",
                be_moved=False,
            )
            is True
        )

    def test_no_trigger_when_loss_below_threshold(self):
        """Should NOT trigger when loss < threshold."""
        assert (
            should_trigger_negative_be(
                profit_pct=-1.0,
                threshold_pct=2.0,
                mark_price=99.0,
                stop_loss=95.0,
                side="LONG",
                be_moved=False,
            )
            is False
        )

    def test_no_trigger_when_already_moved(self):
        """Should NOT trigger when be_moved is True."""
        assert (
            should_trigger_negative_be(
                profit_pct=-3.0,
                threshold_pct=2.0,
                mark_price=97.0,
                stop_loss=95.0,
                side="LONG",
                be_moved=True,
            )
            is False
        )

    def test_no_trigger_when_sl_hit(self):
        """Should NOT trigger when stop loss is hit."""
        assert (
            should_trigger_negative_be(
                profit_pct=-3.0,
                threshold_pct=2.0,
                mark_price=94.0,
                stop_loss=95.0,
                side="LONG",
                be_moved=False,
            )
            is False
        )

    def test_no_trigger_when_threshold_zero(self):
        """Should NOT trigger when threshold is 0 or negative."""
        assert (
            should_trigger_negative_be(
                profit_pct=-3.0,
                threshold_pct=0.0,
                mark_price=97.0,
                stop_loss=95.0,
                side="LONG",
                be_moved=False,
            )
            is False
        )

        assert (
            should_trigger_negative_be(
                profit_pct=-3.0,
                threshold_pct=-1.0,
                mark_price=97.0,
                stop_loss=95.0,
                side="LONG",
                be_moved=False,
            )
            is False
        )

    def test_trigger_at_exact_threshold(self):
        """Should trigger when loss == threshold."""
        assert (
            should_trigger_negative_be(
                profit_pct=-2.0,
                threshold_pct=2.0,
                mark_price=98.0,
                stop_loss=95.0,
                side="LONG",
                be_moved=False,
            )
            is True
        )

    def test_no_trigger_when_in_profit(self):
        """Should NOT trigger when position is in profit."""
        assert (
            should_trigger_negative_be(
                profit_pct=3.0,
                threshold_pct=2.0,
                mark_price=103.0,
                stop_loss=95.0,
                side="LONG",
                be_moved=False,
            )
            is False
        )


class TestCalculateTakeProfitForBE:
    """Tests for calculate_take_profit_for_be function."""

    def test_returns_entry_price(self):
        """Should always return entry price."""
        assert calculate_take_profit_for_be(100.0, "LONG") == 100.0
        assert calculate_take_profit_for_be(100.0, "SHORT") == 100.0
        assert calculate_take_profit_for_be(50000.0, "LONG") == 50000.0


class TestNegativeBreakevenLogic:
    """Tests for NegativeBreakevenLogic class."""

    def test_static_calculate_profit_pct(self):
        """Test static method for calculate_profit_pct."""
        assert NegativeBreakevenLogic.calculate_profit_pct(100.0, 103.0, "LONG") == 3.0
        assert NegativeBreakevenLogic.calculate_profit_pct(100.0, 97.0, "SHORT") == 3.0

    def test_should_trigger_true(self):
        """Test should_trigger returns True when conditions met."""
        assert (
            NegativeBreakevenLogic.should_trigger(
                entry_price=100.0,
                mark_price=97.0,
                stop_loss=95.0,
                side="LONG",
                threshold_pct=2.0,
                be_moved=False,
            )
            is True
        )

    def test_should_trigger_false(self):
        """Test should_trigger returns False when conditions not met."""
        assert (
            NegativeBreakevenLogic.should_trigger(
                entry_price=100.0,
                mark_price=99.0,
                stop_loss=95.0,
                side="LONG",
                threshold_pct=2.0,
                be_moved=False,
            )
            is False
        )

    def test_get_new_take_profit(self):
        """Test get_new_take_profit returns entry price."""
        assert NegativeBreakevenLogic.get_new_take_profit(100.0) == 100.0


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_flow_long_position(self):
        """Test complete flow for LONG position."""
        entry_price = 100.0
        mark_price = 97.0  # 3% loss
        stop_loss = 95.0
        threshold_pct = 2.0

        # Calculate profit
        profit_pct = calculate_profit_pct(entry_price, mark_price, "LONG")
        assert profit_pct == -3.0

        # Check SL not hit
        assert has_hit_stop_loss(mark_price, stop_loss, "LONG") is False

        # Should trigger
        should_trigger = should_trigger_negative_be(
            profit_pct, threshold_pct, mark_price, stop_loss, "LONG", be_moved=False
        )
        assert should_trigger is True

        # New TP should be entry price
        new_tp = calculate_take_profit_for_be(entry_price, "LONG")
        assert new_tp == entry_price

    def test_full_flow_short_position(self):
        """Test complete flow for SHORT position."""
        entry_price = 100.0
        mark_price = 103.0  # 3% loss for SHORT
        stop_loss = 105.0
        threshold_pct = 2.0

        # Calculate profit
        profit_pct = calculate_profit_pct(entry_price, mark_price, "SHORT")
        assert profit_pct == -3.0

        # Check SL not hit
        assert has_hit_stop_loss(mark_price, stop_loss, "SHORT") is False

        # Should trigger
        should_trigger = should_trigger_negative_be(
            profit_pct, threshold_pct, mark_price, stop_loss, "SHORT", be_moved=False
        )
        assert should_trigger is True

        # New TP should be entry price
        new_tp = calculate_take_profit_for_be(entry_price, "SHORT")
        assert new_tp == entry_price

    def test_no_trigger_after_sl_hit(self):
        """Should not trigger if SL is already hit."""
        entry_price = 100.0
        mark_price = 94.0  # Below SL for LONG
        stop_loss = 95.0
        threshold_pct = 2.0

        # Calculate profit
        profit_pct = calculate_profit_pct(entry_price, mark_price, "LONG")
        assert profit_pct == -6.0  # Exceeds threshold

        # Check SL hit
        assert has_hit_stop_loss(mark_price, stop_loss, "LONG") is True

        # Should NOT trigger because SL is hit
        should_trigger = should_trigger_negative_be(
            profit_pct, threshold_pct, mark_price, stop_loss, "LONG", be_moved=False
        )
        assert should_trigger is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
