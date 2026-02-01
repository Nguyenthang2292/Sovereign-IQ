import unittest
from unittest.mock import patch
from modules.adaptive_trend_LTS_mini.cli.interactive_prompts import (
    prompt_timeframe,
    prompt_interactive_mode,
    UserExitRequested,
)


class TestInteractivePrompts(unittest.TestCase):
    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    def test_prompt_timeframe_valid_selection(self, mock_input):
        """Test selecting a valid valid timeframe option from the menu."""
        # Options: 1) 15m, 2) 30m, 3) 1h, ...
        mock_input.return_value = "1"  # Select 15m
        result = prompt_timeframe()
        self.assertEqual(result, "15m")

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    def test_prompt_timeframe_custom(self, mock_input):
        """Test selecting custom timeframe option."""
        # Assuming 5 default options, 6 is custom (len + 1)
        # Mock input sequence: "6" (custom choice), then "2h" (custom value)
        mock_input.side_effect = ["6", "2h"]
        result = prompt_timeframe()
        self.assertEqual(result, "2h")

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    def test_prompt_timeframe_default(self, mock_input):
        """Test selecting default option."""
        # Assuming 5 default options, 7 is default (len + 2)
        mock_input.return_value = "7"
        result = prompt_timeframe(default_timeframe="4h")
        self.assertEqual(result, "4h")

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    def test_prompt_timeframe_retry_then_valid(self, mock_input):
        """Test invalid input followed by valid input."""
        # "99" (invalid), "abc" (invalid), "1" (valid 15m)
        mock_input.side_effect = ["99", "abc", "1"]
        result = prompt_timeframe()
        self.assertEqual(result, "15m")

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    def test_prompt_timeframe_custom_validation(self, mock_input):
        """Test custom timeframe validation loop."""
        # "6" (custom), "invalid" (bad format), "3h" (valid)
        mock_input.side_effect = ["6", "invalid", "3h"]
        result = prompt_timeframe()
        self.assertEqual(result, "3h")

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    def test_prompt_interactive_mode_auto(self, mock_input):
        """Test selecting Auto mode."""
        # "1" = Auto mode
        # prompt_interactive_mode calls prompt_timeframe for timeframe selection
        # Need to return "1" for mode, then "4h" result for prompt_timeframe
        # (however prompt_timeframe is called inside)
        # But wait, prompt_interactive_mode calls `prompt_timeframe`
        # We can mock prompt_timeframe separately
        with patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_timeframe") as mock_tf:
            mock_tf.return_value = "4h"
            mock_input.return_value = "1"

            result = prompt_interactive_mode()
            self.assertEqual(result, {"mode": "auto", "timeframe": "4h"})

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    def test_prompt_interactive_mode_manual(self, mock_input):
        """Test selecting Manual mode."""
        with patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_timeframe") as mock_tf:
            mock_tf.return_value = "1d"
            mock_input.return_value = "2"

            result = prompt_interactive_mode()
            self.assertEqual(result, {"mode": "manual", "timeframe": "1d"})

    @patch("modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input")
    def test_prompt_interactive_mode_exit(self, mock_input):
        """Test selecting Exit option."""
        mock_input.return_value = "4"
        with self.assertRaises(UserExitRequested):
            prompt_interactive_mode()


if __name__ == "__main__":
    unittest.main()
