"""
Unit Tests for Trailing Stop Module
=====================================

Tests the core trailing stop calculation logic.

Created: 2026-02-06
"""

import pytest

from execution.trailing_stop import (
    calculate_next_threshold,
    calculate_trailing_stop,
    get_trailing_stop_info,
)


class TestCalculateTrailingStop:
    """Test calculate_trailing_stop function."""

    def test_long_step_0_be_triggered(self):
        """Test step 0 (BE) triggered when profit >= 0%."""
        result = calculate_trailing_stop(
            entry_price=100.0,
            current_price=101.0,  # 1% profit
            side="LONG",
            step_index=0,
            step_pct=2.0,
        )

        assert result.should_step is True
        assert result.new_sl_price == 100.0  # Entry price (BE)
        assert result.next_step_index == 1
        assert "step 0 triggered" in result.message.lower()

    def test_long_step_0_not_triggered(self):
        """Test step 0 not triggered when price below entry."""
        result = calculate_trailing_stop(
            entry_price=100.0,
            current_price=99.0,  # -1% profit
            side="LONG",
            step_index=0,
            step_pct=2.0,
        )

        assert result.should_step is False
        assert result.new_sl_price is None
        assert result.next_step_index == 0

    def test_long_step_1_triggered(self):
        """Test step 1 triggered when profit >= 2%."""
        result = calculate_trailing_stop(
            entry_price=100.0,
            current_price=102.5,  # 2.5% profit
            side="LONG",
            step_index=1,
            step_pct=2.0,
            current_sl=100.0,  # SL already at BE
        )

        assert result.should_step is True
        assert result.new_sl_price == 102.0  # entry + 2%
        assert result.next_step_index == 2

    def test_long_step_1_not_triggered(self):
        """Test step 1 not triggered when profit < 2%."""
        result = calculate_trailing_stop(
            entry_price=100.0,
            current_price=101.0,  # 1% profit
            side="LONG",
            step_index=1,
            step_pct=2.0,
            current_sl=100.0,
        )

        assert result.should_step is False
        assert result.new_sl_price is None

    def test_short_step_0_be_triggered(self):
        """Test step 0 (BE) triggered for SHORT when profit >= 0%."""
        result = calculate_trailing_stop(
            entry_price=100.0,
            current_price=99.0,  # 1% profit for SHORT
            side="SHORT",
            step_index=0,
            step_pct=2.0,
        )

        assert result.should_step is True
        assert result.new_sl_price == 100.0  # Entry price (BE)
        assert result.next_step_index == 1

    def test_short_step_1_triggered(self):
        """Test step 1 triggered for SHORT when profit >= 2%."""
        result = calculate_trailing_stop(
            entry_price=100.0,
            current_price=97.5,  # 2.5% profit
            side="SHORT",
            step_index=1,
            step_pct=2.0,
            current_sl=100.0,
        )

        assert result.should_step is True
        assert result.new_sl_price == 98.0  # entry - 2%
        assert result.next_step_index == 2

    def test_max_steps_limit(self):
        """Test that max_steps is respected."""
        result = calculate_trailing_stop(
            entry_price=100.0,
            current_price=110.0,  # 10% profit
            side="LONG",
            step_index=5,
            step_pct=2.0,
            limit_steps=True,
            max_steps=5,
        )

        assert result.should_step is False
        assert "maximum steps" in result.message.lower()

    def test_invalid_side(self):
        """Test error handling for invalid side."""
        result = calculate_trailing_stop(
            entry_price=100.0,
            current_price=105.0,
            side="INVALID",
            step_index=0,
            step_pct=2.0,
        )

        assert result.should_step is False
        assert "invalid side" in result.message.lower()

    def test_invalid_step_pct(self):
        """Test error handling for invalid step percentage."""
        result = calculate_trailing_stop(
            entry_price=100.0,
            current_price=105.0,
            side="LONG",
            step_index=0,
            step_pct=-1.0,
        )

        assert result.should_step is False
        assert "must be positive" in result.message.lower()

    def test_new_sl_not_better_than_current(self):
        """Test that we don't step if new SL is not better."""
        result = calculate_trailing_stop(
            entry_price=100.0,
            current_price=102.5,
            side="LONG",
            step_index=1,
            step_pct=2.0,
            current_sl=103.0,  # Current SL is already better
        )

        assert result.should_step is False
        assert "not better than current" in result.message.lower()


class TestCalculateNextThreshold:
    """Test calculate_next_threshold function."""

    def test_long_step_0_threshold(self):
        """Test threshold calculation for LONG step 0."""
        price, pct = calculate_next_threshold(
            entry_price=100.0,
            side="LONG",
            step_index=0,
            step_pct=2.0,
        )

        assert price == 100.0  # Entry price
        assert pct == 0.0

    def test_long_step_1_threshold(self):
        """Test threshold calculation for LONG step 1."""
        price, pct = calculate_next_threshold(
            entry_price=100.0,
            side="LONG",
            step_index=1,
            step_pct=2.0,
        )

        assert price == 102.0  # entry + 2%
        assert pct == 2.0

    def test_long_step_3_threshold(self):
        """Test threshold calculation for LONG step 3."""
        price, pct = calculate_next_threshold(
            entry_price=100.0,
            side="LONG",
            step_index=3,
            step_pct=2.0,
        )

        assert price == 106.0  # entry + 6%
        assert pct == 6.0

    def test_short_step_0_threshold(self):
        """Test threshold calculation for SHORT step 0."""
        price, pct = calculate_next_threshold(
            entry_price=100.0,
            side="SHORT",
            step_index=0,
            step_pct=2.0,
        )

        assert price == 100.0  # Entry price
        assert pct == 0.0

    def test_short_step_1_threshold(self):
        """Test threshold calculation for SHORT step 1."""
        price, pct = calculate_next_threshold(
            entry_price=100.0,
            side="SHORT",
            step_index=1,
            step_pct=2.0,
        )

        assert price == 98.0  # entry - 2%
        assert pct == 2.0


class TestGetTrailingStopInfo:
    """Test get_trailing_stop_info function."""

    def test_info_with_profit_below_threshold(self):
        """Test info when profit is below next threshold."""
        info = get_trailing_stop_info(
            entry_price=100.0,
            current_price=101.0,
            side="LONG",
            step_index=1,
            step_pct=2.0,
        )

        assert info["current_step"] == 1
        assert info["current_profit_pct"] == 1.0
        assert info["next_threshold_price"] == 102.0
        assert info["next_threshold_pct"] == 2.0
        assert info["distance_to_next_step"] == 1.0
        assert info["sl_at_next_step"] == 102.0

    def test_info_with_profit_above_threshold(self):
        """Test info when profit is above next threshold."""
        info = get_trailing_stop_info(
            entry_price=100.0,
            current_price=103.0,
            side="LONG",
            step_index=1,
            step_pct=2.0,
        )

        assert info["current_profit_pct"] == 3.0
        assert info["distance_to_next_step"] == -1.0  # Already passed

    def test_short_info(self):
        """Test info for SHORT position."""
        info = get_trailing_stop_info(
            entry_price=100.0,
            current_price=98.0,
            side="SHORT",
            step_index=0,
            step_pct=2.0,
        )

        assert info["current_step"] == 0
        assert info["current_profit_pct"] == 2.0
        assert info["next_threshold_price"] == 100.0
        assert info["distance_to_next_step"] == -2.0  # Already at/above threshold

    def test_steps_remaining_with_limit(self):
        """Test steps_remaining when limit_steps is True."""
        info = get_trailing_stop_info(
            entry_price=100.0,
            current_price=101.0,
            side="LONG",
            step_index=2,
            step_pct=2.0,
            limit_steps=True,
            max_steps=5,
        )

        assert info["steps_remaining"] == 3
        assert info["is_max_steps_reached"] is False

    def test_max_steps_reached(self):
        """Test when max_steps is reached."""
        info = get_trailing_stop_info(
            entry_price=100.0,
            current_price=101.0,
            side="LONG",
            step_index=5,
            step_pct=2.0,
            limit_steps=True,
            max_steps=5,
        )

        assert info["steps_remaining"] == 0
        assert info["is_max_steps_reached"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
