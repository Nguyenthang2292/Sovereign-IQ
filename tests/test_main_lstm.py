"""
Test for main_lstm.py - LSTM Model Manager CLI.

Tests the main entry point with mocked CLI interactions.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_lstm import main


class TestMainLSTM:
    """Test main_lstm.py CLI functionality."""

    @patch("main_lstm.parse_args")
    @patch("main_lstm.generate_signal_workflow")
    def test_main_with_symbol_args(self, mock_generate_signal, mock_parse_args):
        """Test main function with symbol and timeframe arguments (non-interactive mode)."""
        # Mock args with symbol and timeframe
        mock_args = MagicMock()
        mock_args.symbol = "BTCUSDT"
        mock_args.timeframe = "1h"
        mock_args.model_path = None
        mock_args.limit = 500
        mock_parse_args.return_value = mock_args

        # Call main
        main()

        # Verify generate_signal_workflow was called with correct args
        mock_generate_signal.assert_called_once_with(model_path=None, symbol="BTCUSDT", timeframe="1h", limit=500)

    @patch("main_lstm.parse_args")
    @patch("main_lstm.display_main_menu")
    @patch("main_lstm.prompt_menu_choice")
    @patch("main_lstm.generate_signal_workflow")
    @patch("modules.common.utils.prompt_user_input")
    def test_main_interactive_mode_choice_2(
        self, mock_prompt_user_input, mock_generate_signal, mock_prompt_choice, mock_display_menu, mock_parse_args
    ):
        """Test interactive mode with choice 2 (generate signal)."""
        # Mock args without symbol/timeframe (interactive mode)
        mock_args = MagicMock()
        mock_args.symbol = None
        mock_args.timeframe = None
        mock_args.model_path = None
        mock_args.limit = 500
        mock_parse_args.return_value = mock_args

        # Mock menu choice 2, then quit
        mock_prompt_choice.side_effect = ["2", "3"]  # Choice 2 then quit
        mock_prompt_user_input.return_value = ""  # Just press Enter to continue

        # Call main
        main()

        # Verify functions were called
        mock_display_menu.assert_called()
        mock_generate_signal.assert_called_once_with(model_path=None, limit=500)

    @patch("main_lstm.parse_args")
    @patch("main_lstm.display_main_menu")
    @patch("main_lstm.prompt_menu_choice")
    @patch("main_lstm.train_model_menu")
    @patch("modules.common.utils.prompt_user_input")
    @patch("main_lstm.prompt_symbol")
    @patch("main_lstm.prompt_timeframe")
    @patch("main_lstm.generate_signal_workflow")
    def test_main_interactive_mode_choice_1_with_signal_generation(
        self,
        mock_generate_signal,
        mock_prompt_timeframe,
        mock_prompt_symbol,
        mock_prompt_user_input,
        mock_train_model,
        mock_prompt_choice,
        mock_display_menu,
        mock_parse_args,
    ):
        """Test interactive mode with choice 1 (train model) and then signal generation."""
        # Mock args
        mock_args = MagicMock()
        mock_args.symbol = None
        mock_args.timeframe = None
        mock_args.model_path = None
        mock_args.limit = 500
        mock_parse_args.return_value = mock_args

        # Mock training successful
        mock_train_model.return_value = (True, "/path/to/model.pth")

        # Mock user inputs
        mock_prompt_choice.side_effect = ["1", "3"]  # Train then quit
        mock_prompt_user_input.side_effect = ["y", ""]  # Yes to generate signal, empty to quit

        # Mock symbol and timeframe prompts
        mock_prompt_symbol.return_value = "ETHUSDT"
        mock_prompt_timeframe.return_value = "4h"

        # Call main
        main()

        # Verify training was called
        mock_train_model.assert_called_once()

        # Verify signal generation prompts were called
        mock_prompt_symbol.assert_called_once()
        mock_prompt_timeframe.assert_called_once()

        # Verify generate_signal_workflow was called with trained model
        mock_generate_signal.assert_called_once_with(
            model_path="/path/to/model.pth", symbol="ETHUSDT", timeframe="4h", limit=500
        )

    @patch("main_lstm.parse_args")
    @patch("main_lstm.display_main_menu")
    @patch("main_lstm.prompt_menu_choice")
    @patch("main_lstm.train_model_menu")
    @patch("modules.common.utils.prompt_user_input")
    def test_main_interactive_mode_choice_1_training_failed(
        self, mock_prompt_user_input, mock_train_model, mock_prompt_choice, mock_display_menu, mock_parse_args
    ):
        """Test interactive mode with choice 1 (train model) when training fails."""
        # Mock args
        mock_args = MagicMock()
        mock_args.symbol = None
        mock_args.timeframe = None
        mock_parse_args.return_value = mock_args

        # Mock training failed
        mock_train_model.return_value = (False, None)

        # Mock menu choices
        mock_prompt_choice.side_effect = ["1", "3"]  # Train then quit
        mock_prompt_user_input.return_value = ""  # Just quit after training

        # Call main
        main()

        # Verify training was called
        mock_train_model.assert_called_once()

        # Verify no signal generation since training failed
        # (prompt_user_input for signal generation not called)

    @patch("main_lstm.parse_args")
    @patch("main_lstm.display_main_menu")
    @patch("main_lstm.prompt_menu_choice")
    def test_main_interactive_mode_invalid_choice(self, mock_prompt_choice, mock_display_menu, mock_parse_args):
        """Test interactive mode with invalid menu choice."""
        # Mock args
        mock_args = MagicMock()
        mock_args.symbol = None
        mock_args.timeframe = None
        mock_parse_args.return_value = mock_args

        # Mock invalid choice then quit
        mock_prompt_choice.side_effect = ["invalid", "3"]

        # Call main
        main()

        # Verify menu was displayed twice (invalid choice + quit)
        assert mock_display_menu.call_count == 2

    @patch("main_lstm.parse_args")
    @patch("main_lstm.display_main_menu")
    @patch("main_lstm.prompt_menu_choice")
    def test_main_interactive_mode_choice_3_quit(self, mock_prompt_choice, mock_display_menu, mock_parse_args):
        """Test interactive mode with choice 3 (quit)."""
        # Mock args
        mock_args = MagicMock()
        mock_args.symbol = None
        mock_args.timeframe = None
        mock_parse_args.return_value = mock_args

        # Mock choice 3 (quit)
        mock_prompt_choice.return_value = "3"

        # Call main
        main()

        # Verify menu was displayed once
        mock_display_menu.assert_called_once()

    @patch("main_lstm.parse_args")
    @patch("main_lstm.display_main_menu")
    @patch("main_lstm.prompt_menu_choice")
    @patch("main_lstm.generate_signal_workflow")
    @patch("modules.common.utils.prompt_user_input")
    def test_main_interactive_mode_generate_then_quit(
        self, mock_prompt_user_input, mock_generate_signal, mock_prompt_choice, mock_display_menu, mock_parse_args
    ):
        """Test interactive mode generate signal then quit."""
        # Mock args
        mock_args = MagicMock()
        mock_args.symbol = None
        mock_args.timeframe = None
        mock_args.model_path = None
        mock_args.limit = 500
        mock_parse_args.return_value = mock_args

        # Mock choice 2 then quit
        mock_prompt_choice.side_effect = ["2", "q"]

        # Mock quit confirmation
        mock_prompt_user_input.return_value = "q"

        # Call main
        main()

        # Verify signal generation was called
        mock_generate_signal.assert_called_once_with(model_path=None, limit=500)

    @patch("main_lstm.parse_args")
    def test_main_keyboard_interrupt(self, mock_parse_args):
        """Test main function handles KeyboardInterrupt."""
        # Mock args to cause KeyboardInterrupt in interactive mode
        mock_args = MagicMock()
        mock_args.symbol = None
        mock_args.timeframe = None
        mock_parse_args.return_value = mock_args

        # Mock display_main_menu to raise KeyboardInterrupt
        with patch("main_lstm.display_main_menu", side_effect=KeyboardInterrupt()):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    @patch("main_lstm.parse_args")
    def test_main_unexpected_exception(self, mock_parse_args):
        """Test main function handles unexpected exceptions."""
        # Mock args
        mock_args = MagicMock()
        mock_args.symbol = None
        mock_args.timeframe = None
        mock_parse_args.return_value = mock_args

        # Mock display_main_menu to raise unexpected exception
        with patch("main_lstm.display_main_menu", side_effect=ValueError("Test error")):
            with patch("builtins.print"):
                with patch("traceback.format_exc", return_value="Traceback here"):
                    with pytest.raises(SystemExit) as exc_info:
                        main()

                    # Should exit with code 1
                    assert exc_info.value.code == 1

                    # Verify error messages were printed
                    # (print calls for error and traceback)


def test_main_lstm_imports():
    """Test that main_lstm.py can be imported without errors."""
    try:
        import main_lstm

        # Verify main function exists
        assert hasattr(main_lstm, "main")
        assert callable(main_lstm.main)
    except ImportError as e:
        pytest.fail(f"Failed to import main_lstm: {e}")


def test_main_lstm_module_structure():
    """Test that main_lstm.py has expected module structure."""
    import main_lstm

    # Check for expected imports/attributes
    expected_attrs = [
        "main",
        "parse_args",
        "display_main_menu",
        "generate_signal_workflow",
        "train_model_menu",
        "prompt_menu_choice",
        "prompt_symbol",
        "prompt_timeframe",
    ]

    for attr in expected_attrs:
        assert hasattr(main_lstm, attr), f"main_lstm missing expected attribute: {attr}"
