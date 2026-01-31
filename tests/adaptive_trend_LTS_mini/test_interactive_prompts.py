"""Tests for modules/adaptive_trend_LTS_mini/cli/interactive_prompts.py.

Tests cover:
1. Timeframe selection (valid/custom/invalid/default/empty)
2. Interactive mode selection (auto/manual/timeframe-only/exit/invalid)
3. Input validation (non-numeric/out-of-range/edge cases)
4. Helper functions (_find_timeframe_index, _prompt_custom_timeframe, _validate_input_length)
5. Integration tests for full user flow
"""

from unittest.mock import MagicMock, patch

import pytest

from modules.adaptive_trend_LTS_mini.cli.interactive_prompts import (
    MAX_INPUT_LENGTH,
    PROMPT_DISPLAY_WIDTH,
    InteractiveModeResult,
    UserExitRequested,
    _display_timeframe_menu,
    _find_timeframe_index,
    _prompt_custom_timeframe,
    _validate_input_length,
    prompt_interactive_mode,
    prompt_timeframe,
)


class TestHelperFunctions:
    """Test helper functions."""

    def test_validate_input_length_valid(self):
        """Test valid input length."""
        assert _validate_input_length("short input") is True
        assert _validate_input_length("a" * MAX_INPUT_LENGTH) is True

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.log_error")
    def test_validate_input_length_too_long(self, mock_log_error):
        """Test input that's too long."""
        long_input = "a" * (MAX_INPUT_LENGTH + 1)
        assert _validate_input_length(long_input) is False
        mock_log_error.assert_called_once()
        assert "Input too long" in mock_log_error.call_args[0][0]

    def test_find_timeframe_index_found(self):
        """Test finding existing timeframe."""
        timeframes = [("15m", "15 minutes"), ("1h", "1 hour"), ("4h", "4 hours")]
        assert _find_timeframe_index(timeframes, "1h") == 1
        assert _find_timeframe_index(timeframes, "15m") == 0
        assert _find_timeframe_index(timeframes, "4h") == 2

    def test_find_timeframe_index_not_found(self):
        """Test timeframe not in list returns 0."""
        timeframes = [("15m", "15 minutes"), ("1h", "1 hour")]
        assert _find_timeframe_index(timeframes, "5m") == 0
        assert _find_timeframe_index(timeframes, "invalid") == 0

    def test_find_timeframe_index_empty_list(self):
        """Test empty timeframe list."""
        assert _find_timeframe_index([], "1h") == 0

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    def test_prompt_custom_timeframe_valid(self, mock_prompt):
        """Test custom timeframe with valid input."""
        mock_prompt.return_value = "4h"
        result = _prompt_custom_timeframe("1h")
        assert result == "4h"

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    def test_prompt_custom_timeframe_empty_returns_default(self, mock_prompt):
        """Test empty input returns default."""
        mock_prompt.return_value = ""
        result = _prompt_custom_timeframe("1h")
        assert result == "1h"

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.log_error")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    def test_prompt_custom_timeframe_invalid_then_valid(self, mock_prompt, mock_log_error):
        """Test invalid input then valid input."""
        mock_prompt.side_effect = ["invalid", "1h"]
        result = _prompt_custom_timeframe("15m")
        assert result == "1h"
        mock_log_error.assert_called()

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.log_error")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    def test_prompt_custom_timeframe_too_long(self, mock_prompt, mock_log_error):
        """Test input that's too long."""
        long_input = "x" * (MAX_INPUT_LENGTH + 1)
        mock_prompt.side_effect = [long_input, "1h"]
        result = _prompt_custom_timeframe("15m")
        assert result == "1h"
        assert any("Input too long" in str(call) for call in mock_log_error.call_args_list)

    @patch("builtins.print")
    def test_display_timeframe_menu(self, mock_print):
        """Test timeframe menu display."""
        timeframes = [("15m", "15 minutes"), ("1h", "1 hour")]
        _display_timeframe_menu(timeframes, "1h", 1)

        # Should print header and options
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("SELECT TIMEFRAME" in str(call) for call in print_calls)

    def test_constants_defined(self):
        """Test that constants are properly defined."""
        assert PROMPT_DISPLAY_WIDTH == 60
        assert MAX_INPUT_LENGTH == 100


class TestPromptTimeframe:
    """Test prompt_timeframe function."""

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts._display_timeframe_menu")
    def test_select_valid_option(self, mock_display, mock_prompt):
        """Test selecting a valid timeframe option."""
        mock_prompt.return_value = "1"
        result = prompt_timeframe("1h")
        assert result == "15m"

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts._display_timeframe_menu")
    def test_select_default_option(self, mock_display, mock_prompt):
        """Test selecting default option (7)."""
        mock_prompt.return_value = "7"
        result = prompt_timeframe("1h")
        assert result == "1h"

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts._display_timeframe_menu")
    def test_select_custom_option(self, mock_display, mock_prompt):
        """Test selecting custom timeframe option (6)."""
        mock_prompt.side_effect = ["6", "8h"]
        result = prompt_timeframe("1h")
        assert result == "8h"

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts._display_timeframe_menu")
    def test_empty_input_returns_default(self, mock_display, mock_prompt):
        """Test empty input returns default timeframe."""
        mock_prompt.return_value = ""
        result = prompt_timeframe("4h")
        assert result == "4h"

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.log_error")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts._display_timeframe_menu")
    def test_non_numeric_input(self, mock_display, mock_prompt, mock_log_error):
        """Test non-numeric input handling."""
        mock_prompt.side_effect = ["abc", "1"]
        result = prompt_timeframe("1h")
        assert result == "15m"
        mock_log_error.assert_called()
        assert "Invalid input" in mock_log_error.call_args[0][0]

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.log_error")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts._display_timeframe_menu")
    def test_out_of_range_input(self, mock_display, mock_prompt, mock_log_error):
        """Test out-of-range number input."""
        mock_prompt.side_effect = ["99", "1"]
        result = prompt_timeframe("1h")
        assert result == "15m"
        mock_log_error.assert_called()
        assert "Invalid choice" in mock_log_error.call_args[0][0]

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.log_error")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts._display_timeframe_menu")
    def test_negative_number_input(self, mock_display, mock_prompt, mock_log_error):
        """Test negative number input."""
        mock_prompt.side_effect = ["-1", "1"]
        result = prompt_timeframe("1h")
        assert result == "15m"
        # Negative numbers fail isdigit() check
        mock_log_error.assert_called()

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.log_error")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts._display_timeframe_menu")
    def test_zero_input(self, mock_display, mock_prompt, mock_log_error):
        """Test zero input."""
        mock_prompt.side_effect = ["0", "1"]
        result = prompt_timeframe("1h")
        assert result == "15m"
        mock_log_error.assert_called()
        assert "Invalid choice" in mock_log_error.call_args[0][0]

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts._display_timeframe_menu")
    def test_whitespace_handling(self, mock_display, mock_prompt):
        """Test whitespace is stripped from input."""
        mock_prompt.return_value = "  2  "
        result = prompt_timeframe("1h")
        assert result == "30m"

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.log_error")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts._display_timeframe_menu")
    def test_input_too_long(self, mock_display, mock_prompt, mock_log_error):
        """Test input that exceeds maximum length."""
        long_input = "1" * (MAX_INPUT_LENGTH + 1)
        mock_prompt.side_effect = [long_input, "1"]
        result = prompt_timeframe("1h")
        assert result == "15m"
        assert any("Input too long" in str(call) for call in mock_log_error.call_args_list)


class TestPromptInteractiveMode:
    """Test prompt_interactive_mode function."""

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_timeframe")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("builtins.print")
    def test_select_auto_mode(self, mock_print, mock_prompt, mock_timeframe):
        """Test selecting auto mode."""
        mock_prompt.return_value = "1"
        mock_timeframe.return_value = "1h"

        result = prompt_interactive_mode("1h")

        assert result["mode"] == "auto"
        assert result["timeframe"] == "1h"

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_timeframe")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("builtins.print")
    def test_select_manual_mode(self, mock_print, mock_prompt, mock_timeframe):
        """Test selecting manual mode."""
        mock_prompt.return_value = "2"
        mock_timeframe.return_value = "4h"

        result = prompt_interactive_mode("1h")

        assert result["mode"] == "manual"
        assert result["timeframe"] == "4h"

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_timeframe")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("builtins.print")
    def test_select_timeframe_only(self, mock_print, mock_prompt, mock_timeframe):
        """Test selecting timeframe-only option (3)."""
        mock_prompt.return_value = "3"
        mock_timeframe.return_value = "2h"

        result = prompt_interactive_mode("1h")

        assert result["mode"] is None
        assert result["timeframe"] == "2h"

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.log_warn")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("builtins.print")
    def test_select_exit(self, mock_print, mock_prompt, mock_log_warn):
        """Test selecting exit option (4) raises UserExitRequested."""
        mock_prompt.return_value = "4"

        with pytest.raises(UserExitRequested):
            prompt_interactive_mode("1h")

        mock_log_warn.assert_called_once()

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.log_error")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_timeframe")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("builtins.print")
    def test_invalid_choice_then_valid(self, mock_print, mock_prompt, mock_timeframe, mock_log_error):
        """Test invalid choice then valid choice."""
        mock_prompt.side_effect = ["99", "1"]
        mock_timeframe.return_value = "1h"

        result = prompt_interactive_mode("1h")

        assert result["mode"] == "auto"
        mock_log_error.assert_called()

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_timeframe")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("builtins.print")
    def test_whitespace_in_choice(self, mock_print, mock_prompt, mock_timeframe):
        """Test whitespace is stripped from choice."""
        mock_prompt.return_value = "  1  "
        mock_timeframe.return_value = "1h"

        result = prompt_interactive_mode("1h")

        assert result["mode"] == "auto"

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.log_error")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_timeframe")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("builtins.print")
    def test_input_too_long(self, mock_print, mock_prompt, mock_timeframe, mock_log_error):
        """Test input that exceeds maximum length."""
        long_input = "1" * (MAX_INPUT_LENGTH + 1)
        mock_prompt.side_effect = [long_input, "1"]
        mock_timeframe.return_value = "1h"

        result = prompt_interactive_mode("1h")

        assert result["mode"] == "auto"
        assert any("Input too long" in str(call) for call in mock_log_error.call_args_list)

    def test_return_type_is_typeddict(self):
        """Test that return type is InteractiveModeResult TypedDict."""
        # This test verifies the type annotation exists
        from typing import get_type_hints

        hints = get_type_hints(prompt_interactive_mode)
        assert hints["return"] == InteractiveModeResult


class TestIntegration:
    """Integration tests for full user flow."""

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("builtins.print")
    def test_full_flow_auto_mode_standard_timeframe(self, mock_print, mock_prompt):
        """Test complete flow: auto mode → standard timeframe."""
        # User selects: auto mode (1) → 1h timeframe (3)
        mock_prompt.side_effect = ["1", "3"]

        result = prompt_interactive_mode("1h")

        assert result["mode"] == "auto"
        assert result["timeframe"] == "1h"

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("builtins.print")
    def test_full_flow_manual_mode_custom_timeframe(self, mock_print, mock_prompt):
        """Test complete flow: manual mode → custom timeframe."""
        # User selects: manual mode (2) → custom (6) → 8h
        mock_prompt.side_effect = ["2", "6", "8h"]

        result = prompt_interactive_mode("1h")

        assert result["mode"] == "manual"
        assert result["timeframe"] == "8h"

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("builtins.print")
    def test_full_flow_with_errors(self, mock_print, mock_prompt):
        """Test complete flow with user errors."""
        # User makes mistakes: invalid mode → invalid timeframe → success
        mock_prompt.side_effect = ["invalid", "1", "abc", "2"]

        result = prompt_interactive_mode("1h")

        assert result["mode"] == "auto"
        assert result["timeframe"] == "30m"

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("builtins.print")
    def test_full_flow_timeframe_only(self, mock_print, mock_prompt):
        """Test selecting timeframe only (no mode)."""
        # User selects: timeframe only (3) → custom (6) → 12h
        mock_prompt.side_effect = ["3", "6", "12h"]

        result = prompt_interactive_mode("1h")

        assert result["mode"] is None
        assert result["timeframe"] == "12h"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts._display_timeframe_menu")
    def test_very_long_valid_number(self, mock_display, mock_prompt):
        """Test very long but valid number (within MAX_INPUT_LENGTH)."""
        mock_prompt.side_effect = ["1" * 50, "1"]
        result = prompt_timeframe("1h")
        # Should parse the large number and fail validation, then accept "1"
        assert result == "15m"

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts._display_timeframe_menu")
    def test_special_characters_in_input(self, mock_display, mock_prompt):
        """Test special characters in input."""
        mock_prompt.side_effect = ["@#$", "1"]
        result = prompt_timeframe("1h")
        assert result == "15m"

    def test_user_exit_requested_exception_message(self):
        """Test UserExitRequested exception can be instantiated."""
        exc = UserExitRequested("User requested exit")
        assert str(exc) == "User requested exit"

        # Test without message
        exc2 = UserExitRequested()
        assert isinstance(exc2, Exception)
