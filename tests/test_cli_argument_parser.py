
from unittest.mock import MagicMock, patch

from cli.argument_parser import (
from cli.argument_parser import (

"""
Tests for cli/argument_parser.py interactive_config_menu.

Tests cover:
- Main menu display
- Navigation with back option
- Configuration updates
- Review and confirm
"""


    _configure_decision_matrix,
    _configure_hmm,
    _configure_random_forest,
    _configure_spc,
    _configure_timeframe,
    _configure_xgboost,
    _display_main_menu,
    _format_current_value,
    _prompt_with_back,
    interactive_config_menu,
    parse_args,
)
from config import (
    DECISION_MATRIX_MIN_VOTES,
    DECISION_MATRIX_VOTING_THRESHOLD,
)


class TestFormatCurrentValue:
    """Test _format_current_value helper function."""

    def test_format_none(self):
        """Test formatting None value."""
        result = _format_current_value(None)
        assert result == "not set"

    def test_format_bool_true(self):
        """Test formatting True boolean."""
        result = _format_current_value(True)
        assert result == "enabled"

    def test_format_bool_false(self):
        """Test formatting False boolean."""
        result = _format_current_value(False)
        assert result == "disabled"

    def test_format_string(self):
        """Test formatting string value."""
        result = _format_current_value("1h")
        assert result == "1h"

    def test_format_number(self):
        """Test formatting number value."""
        result = _format_current_value(2)
        assert result == "2"


class TestPromptWithBack:
    """Test _prompt_with_back helper function."""

    @patch("cli.argument_parser.prompt_user_input_with_backspace")
    def test_prompt_with_back_returns_continue(self, mock_prompt):
        """Test that normal input returns continue action."""
        mock_prompt.return_value = ("y", False)  # (user_input, is_back)
        value, action = _prompt_with_back("Test prompt: ", default="n")
        assert value == "y"
        assert action == "continue"

    @patch("cli.argument_parser.prompt_user_input_with_backspace")
    def test_prompt_with_back_returns_main(self, mock_prompt):
        """Test that backspace (is_back=True) returns main action."""
        mock_prompt.return_value = (None, True)  # (user_input, is_back)
        value, action = _prompt_with_back("Test prompt: ", default="n")
        assert value is None
        assert action == "main"

    @patch("cli.argument_parser.prompt_user_input")
    def test_prompt_with_back_no_back_option(self, mock_prompt):
        """Test prompt without back option."""
        mock_prompt.return_value = "y"
        value, action = _prompt_with_back("Test prompt: ", default="n", allow_back=False)
        assert value == "y"
        assert action == "continue"


class TestDisplayMainMenu:
    """Test _display_main_menu function."""

    def test_display_main_menu_hybrid(self):
        """Test main menu display for hybrid mode."""

        class Config:
            timeframe = "1h"
            enable_spc = True
            spc_k = 2
            enable_xgboost = False
            enable_hmm = False
            enable_random_forest = False
            use_decision_matrix = True

        config = Config()

        with patch("builtins.print") as mock_print:
            _display_main_menu(config, mode="hybrid")
            assert mock_print.called

        # Check that config values are accessible
        assert config.timeframe == "1h"
        assert config.enable_spc is True

    def test_display_main_menu_voting(self):
        """Test main menu display for voting mode."""

        class Config:
            timeframe = "30m"
            enable_spc = True
            spc_k = 3
            enable_xgboost = True
            enable_hmm = True
            enable_random_forest = True
            use_decision_matrix = False

        config = Config()

        with patch("builtins.print") as mock_print:
            _display_main_menu(config, mode="voting")
            assert mock_print.called


class TestConfigureFunctions:
    """Test individual configure functions."""

    @patch("cli.argument_parser.prompt_timeframe")
    def test_configure_timeframe(self, mock_prompt_timeframe, config_factory):
        """Test timeframe configuration."""
        config = config_factory(timeframe="1h", no_menu=False)
        mock_prompt_timeframe.return_value = "30m"

        action, changed = _configure_timeframe(config, mode="hybrid")

        assert action == "main"
        assert changed is True  # Changed from "1h" to "30m"
        assert config.timeframe == "30m"
        assert config.no_menu is True

    @patch("cli.argument_parser._prompt_with_back")
    def test_configure_spc(self, mock_prompt, config_factory):
        """Test SPC configuration."""
        config = config_factory(enable_spc=False, spc_k=2)
        mock_prompt.return_value = ("3", "continue")

        action, changed = _configure_spc(config, mode="hybrid")

        assert action == "main"
        assert changed is True  # Changed from k=2 to k=3
        assert config.enable_spc is True
        assert config.spc_k == 3

    @patch("cli.argument_parser._prompt_with_back")
    def test_configure_spc_back(self, mock_prompt, config_factory):
        """Test SPC configuration with back option."""
        config = config_factory(enable_spc=False, spc_k=2)
        orig_enable_spc = config.enable_spc
        orig_spc_k = config.spc_k
        mock_prompt.return_value = (None, "main")

        action, changed = _configure_spc(config, mode="hybrid")

        assert action == "main"
        assert changed is False  # No changes made when going back
        assert config.enable_spc == orig_enable_spc
        assert config.spc_k == orig_spc_k

    @patch("cli.argument_parser._prompt_with_back")
    def test_configure_xgboost(self, mock_prompt, config_factory):
        """Test XGBoost configuration."""
        config = config_factory(enable_xgboost=False)
        mock_prompt.return_value = ("y", "continue")

        action, changed = _configure_xgboost(config, mode="hybrid")

        assert action == "main"
        assert changed is True  # Changed from False to True
        assert config.enable_xgboost is True

    @patch("cli.argument_parser._prompt_with_back")
    def test_configure_hmm(self, mock_prompt, config_factory):
        """Test HMM configuration."""
        config = config_factory(enable_hmm=False)
        mock_prompt.return_value = ("y", "continue")

        action, changed = _configure_hmm(config, mode="hybrid")

        assert action == "main"
        assert changed is True  # Changed from False to True
        assert config.enable_hmm is True

    @patch("cli.argument_parser._prompt_with_back")
    def test_configure_random_forest(self, mock_prompt, config_factory):
        """Test Random Forest configuration."""
        config = config_factory(enable_random_forest=False)
        mock_prompt.return_value = ("y", "continue")

        action, changed = _configure_random_forest(config, mode="hybrid")

        assert action == "main"
        assert changed is True  # Changed from False to True
        assert config.enable_random_forest is True

    @patch("cli.argument_parser._prompt_with_back")
    def test_configure_decision_matrix_back_at_threshold(self, mock_prompt, config_factory):
        """Test Decision Matrix configuration with back option at threshold prompt."""
        config = config_factory()
        config.voting_threshold = 0.5
        config.min_votes = 2
        orig_threshold = config.voting_threshold
        orig_min_votes = config.min_votes
        mock_prompt.return_value = (None, "main")  # User presses 'b' at threshold prompt

        action, changed = _configure_decision_matrix(config, mode="hybrid")

        assert action == "main"
        assert changed is False  # No changes made when going back
        assert config.voting_threshold == orig_threshold
        assert config.min_votes == orig_min_votes
        assert config.use_decision_matrix is True

    @patch("cli.argument_parser._prompt_with_back")
    def test_configure_decision_matrix_back_at_min_votes(self, mock_prompt, config_factory):
        """Test Decision Matrix configuration with back option at min_votes prompt - threshold should be rolled back."""
        config = config_factory()
        config.voting_threshold = 0.5
        config.min_votes = 2
        orig_threshold = config.voting_threshold
        orig_min_votes = config.min_votes
        # First call returns new threshold value, second call returns 'back'
        mock_prompt.side_effect = [
            ("0.7", "continue"),  # User changes threshold to 0.7
            (None, "main"),  # User presses 'b' at min_votes prompt
        ]

        action, changed = _configure_decision_matrix(config, mode="hybrid")

        assert action == "main"
        assert changed is False  # No changes made because user went back
        assert config.voting_threshold == orig_threshold  # Should be rolled back to original
        assert config.min_votes == orig_min_votes  # Should remain unchanged
        assert config.use_decision_matrix is True

    @patch("builtins.input")
    @patch("cli.argument_parser._prompt_with_back")
    def test_configure_decision_matrix_success(self, mock_prompt, mock_input, config_factory):
        """Test Decision Matrix configuration with successful changes to both values."""
        config = config_factory()
        config.voting_threshold = 0.5
        config.min_votes = 2
        # Both values changed successfully
        mock_prompt.side_effect = [
            ("0.7", "continue"),  # User changes threshold to 0.7
            ("3", "continue"),  # User changes min_votes to 3
        ]
        mock_input.return_value = ""  # User presses Enter at final prompt

        action, changed = _configure_decision_matrix(config, mode="hybrid")

        assert action == "main"
        assert changed is True  # Changes were made
        assert config.voting_threshold == 0.7
        assert config.min_votes == 3
        assert config.use_decision_matrix is True

    @patch("builtins.input")
    @patch("cli.argument_parser._prompt_with_back")
    def test_configure_decision_matrix_change_min_votes_only(self, mock_prompt, mock_input, config_factory):
        """Test Decision Matrix configuration with change only to min_votes."""
        config = config_factory()
        config.voting_threshold = 0.5
        config.min_votes = 2
        orig_threshold = config.voting_threshold
        # Only min_votes changed (user presses enter for threshold, changes min_votes)
        mock_prompt.side_effect = [
            ("", "continue"),  # User presses enter, keeping threshold at 0.5
            ("4", "continue"),  # User changes min_votes to 4
        ]
        mock_input.return_value = ""  # User presses Enter at final prompt

        action, changed = _configure_decision_matrix(config, mode="hybrid")

        assert action == "main"
        assert changed is True  # min_votes changed
        assert config.voting_threshold == orig_threshold  # Threshold unchanged
        assert config.min_votes == 4
        assert config.use_decision_matrix is True

    @patch("builtins.input")
    @patch("cli.argument_parser._prompt_with_back")
    def test_configure_decision_matrix_no_changes(self, mock_prompt, mock_input, config_factory):
        """Test Decision Matrix configuration with no changes (user presses enter for both)."""
        config = config_factory()
        config.voting_threshold = 0.5
        config.min_votes = 2
        orig_threshold = config.voting_threshold
        orig_min_votes = config.min_votes
        # User presses enter for both (keeping current values)
        mock_prompt.side_effect = [
            ("", "continue"),  # User presses enter, keeping threshold
            ("", "continue"),  # User presses enter, keeping min_votes
        ]
        mock_input.return_value = ""  # User presses Enter at final prompt

        action, changed = _configure_decision_matrix(config, mode="hybrid")

        assert action == "main"
        assert changed is False  # No changes made
        assert config.voting_threshold == orig_threshold
        assert config.min_votes == orig_min_votes
        assert config.use_decision_matrix is True

    @patch("builtins.input")
    @patch("cli.argument_parser._prompt_with_back")
    def test_configure_decision_matrix_uses_defaults(self, mock_prompt, mock_input, config_factory):
        """Test Decision Matrix configuration uses default values when attributes don't exist."""
        config = config_factory()  # No voting_threshold or min_votes set
        # Capture default constants to verify they are used
        expected_threshold = DECISION_MATRIX_VOTING_THRESHOLD
        expected_min_votes = DECISION_MATRIX_MIN_VOTES
        # User accepts defaults by pressing Enter (empty string)
        mock_prompt.side_effect = [
            ("", "continue"),  # Accept default threshold
            ("", "continue"),  # Accept default min_votes
        ]
        mock_input.return_value = ""  # User presses Enter at final prompt

        action, changed = _configure_decision_matrix(config, mode="hybrid")

        assert action == "main"
        assert changed is False  # No changes made when accepting defaults
        assert config.voting_threshold == expected_threshold
        assert config.min_votes == expected_min_votes
        assert config.use_decision_matrix is True


class TestInteractiveConfigMenu:
    """Test interactive_config_menu main function."""

    @patch("cli.argument_parser._display_main_menu")
    @patch("cli.argument_parser.prompt_user_input")
    @patch("cli.argument_parser._review_and_confirm")
    def test_interactive_config_menu_review_and_confirm(self, mock_review, mock_prompt, mock_display):
        """Test menu flow with review and confirm."""
        mock_prompt.return_value = "7"  # Review and Confirm
        mock_review.return_value = "done"

        config = interactive_config_menu(mode="hybrid")

        assert config is not None
        assert hasattr(config, "timeframe")
        assert hasattr(config, "enable_spc")

    @patch("cli.argument_parser._display_main_menu")
    @patch("cli.argument_parser.prompt_user_input")
    @patch("cli.argument_parser._configure_timeframe")
    def test_interactive_config_menu_configure_timeframe(self, mock_configure, mock_prompt, mock_display):
        """Test menu flow with timeframe configuration."""
        mock_prompt.side_effect = ["1", "7"]  # Configure timeframe, then review
        mock_configure.return_value = ("main", True)

        with patch("cli.argument_parser._review_and_confirm", return_value="done"):
            config = interactive_config_menu(mode="hybrid")

        assert config is not None
        mock_configure.assert_called_once()

    @patch("cli.argument_parser._display_main_menu")
    @patch("cli.argument_parser.prompt_user_input")
    @patch("cli.argument_parser._configure_spc")
    def test_interactive_config_menu_configure_spc(self, mock_configure, mock_prompt, mock_display):
        """Test menu flow with SPC configuration."""
        mock_prompt.side_effect = ["2", "7"]  # Configure SPC, then review
        mock_configure.return_value = ("main", True)

        with patch("cli.argument_parser._review_and_confirm", return_value="done"):
            config = interactive_config_menu(mode="hybrid")

        assert config is not None
        mock_configure.assert_called_once()

    @patch("cli.argument_parser._display_main_menu")
    @patch("cli.argument_parser.prompt_user_input")
    def test_interactive_config_menu_exit(self, mock_prompt, mock_display):
        """Test menu exit option."""
        mock_prompt.return_value = "8"  # Exit

        with patch("sys.exit") as mock_exit:
            try:
                interactive_config_menu(mode="hybrid")
            except SystemExit:
                pass

        mock_exit.assert_called_once_with(0)

    @patch("cli.argument_parser._display_main_menu")
    @patch("cli.argument_parser.prompt_user_input")
    def test_interactive_config_menu_invalid_choice(self, mock_prompt, mock_display):
        """Test menu with invalid choice."""
        mock_prompt.side_effect = ["99", "8"]  # Invalid choice, then exit

        with patch("sys.exit") as mock_exit:
            with patch("builtins.print") as mock_print:
                interactive_config_menu(mode="hybrid")

                # Should print error message for invalid choice
                # Check call arguments directly instead of converting to strings
                invalid_message_found = any(
                    call.args and "Invalid" in call.args[0] for call in mock_print.call_args_list
                )

                assert invalid_message_found, "Expected 'Invalid' error message was not printed"

                # Verify exit was called with code 0
                mock_exit.assert_called_once_with(0)

    def test_interactive_config_menu_defaults(self):
        """Test that menu initializes with default values."""
        with (
            patch("cli.argument_parser._display_main_menu"),
            patch("cli.argument_parser.prompt_user_input", return_value="7"),
            patch("cli.argument_parser._review_and_confirm", return_value="done"),
        ):
            config = interactive_config_menu(mode="hybrid")

            # Check default values
            assert config.timeframe is not None
            assert config.enable_spc is True
            assert config.no_menu is True
            assert config.limit == 500
            assert config.max_workers == 10

    def test_interactive_config_menu_voting_mode(self):
        """Test menu in voting mode."""
        with (
            patch("cli.argument_parser._display_main_menu"),
            patch("cli.argument_parser.prompt_user_input", return_value="7"),
            patch("cli.argument_parser._review_and_confirm", return_value="done"),
        ):
            config = interactive_config_menu(mode="voting")

            # Check voting mode specific defaults
            assert config.use_decision_matrix is False
            assert config.spc_strategy == "all"


class TestParseArgs:
    """Test parse_args function."""

    @patch("cli.argument_parser.sys.argv", ["script.py"])
    @patch("cli.argument_parser.interactive_config_menu")
    def test_parse_args_no_arguments_calls_interactive_menu(self, mock_interactive_menu):
        """Test that parse_args calls interactive menu when no arguments provided."""
        mock_interactive_menu.return_value = MagicMock()

        result = parse_args(mode="hybrid")

        mock_interactive_menu.assert_called_once_with(mode="hybrid")
        assert result is not None

    @patch("cli.argument_parser.sys.argv", ["script.py", "--timeframe", "1h"])
    def test_parse_args_with_timeframe(self):
        """Test parse_args with timeframe argument."""
        args = parse_args(mode="hybrid")

        assert args.timeframe == "1h"

    @patch("cli.argument_parser.sys.argv", ["script.py", "--timeframe", "30m", "--limit", "1000"])
    def test_parse_args_multiple_args(self):
        """Test parse_args with multiple arguments."""
        args = parse_args(mode="hybrid")

        assert args.timeframe == "30m"
        assert args.limit == 1000

    @patch("cli.argument_parser.sys.argv", ["script.py", "--enable-spc", "--spc-k", "3"])
    def test_parse_args_spc_config(self):
        """Test parse_args with SPC configuration."""
        args = parse_args(mode="hybrid")

        assert args.enable_spc is True  # Should be forced enabled
        assert args.spc_k == 3

    @patch("cli.argument_parser.sys.argv", ["script.py", "--enable-xgboost"])
    def test_parse_args_xgboost(self):
        """Test parse_args with XGBoost enabled."""
        args = parse_args(mode="hybrid")

        assert args.enable_xgboost is True

    @patch("cli.argument_parser.sys.argv", ["script.py", "--enable-hmm"])
    def test_parse_args_hmm(self):
        """Test parse_args with HMM enabled."""
        args = parse_args(mode="hybrid")

        assert args.enable_hmm is True

    @patch("cli.argument_parser.sys.argv", ["script.py", "--enable-random-forest"])
    def test_parse_args_random_forest(self):
        """Test parse_args with Random Forest enabled."""
        args = parse_args(mode="hybrid")

        assert args.enable_random_forest is True

    @patch("cli.argument_parser.sys.argv", ["script.py", "--voting-threshold", "0.7", "--min-votes", "3"])
    def test_parse_args_decision_matrix(self):
        """Test parse_args with Decision Matrix parameters."""
        args = parse_args(mode="hybrid")

        assert args.voting_threshold == 0.7
        assert args.min_votes == 3
        assert args.use_decision_matrix is True  # Should be forced enabled

    @patch("cli.argument_parser.sys.argv", ["script.py", "--timeframe", "1h"])
    def test_parse_args_force_enable_decision_matrix(self):
        """Test that force_enable_decision_matrix parameter works."""
        args = parse_args(mode="hybrid", force_enable_decision_matrix=True)

        assert args.use_decision_matrix is True

    @patch("cli.argument_parser.sys.argv", ["script.py", "--timeframe", "1h"])
    def test_parse_args_hybrid_mode_defaults(self):
        """Test parse_args defaults in hybrid mode."""
        args = parse_args(mode="hybrid")

        assert args.enable_spc is True  # Forced enabled
        assert args.use_decision_matrix is True  # Forced enabled

    @patch("cli.argument_parser.sys.argv", ["script.py", "--timeframe", "1h"])
    def test_parse_args_voting_mode(self):
        """Test parse_args in voting mode."""
        args = parse_args(mode="voting")

        assert args.enable_spc is True  # Forced enabled
        # In voting mode, verify decision matrix default behavior
        assert args.use_decision_matrix is False  # Default for voting mode

    @patch("cli.argument_parser.sys.argv", ["script.py", "--timeframe", "1h", "--osc-strategies", "5", "6", "7"])
    def test_parse_args_osc_strategies(self):
        """Test parse_args with oscillator strategies."""
        args = parse_args(mode="hybrid")

        assert args.osc_strategies == [5, 6, 7]

    @patch("cli.argument_parser.sys.argv", ["script.py", "--timeframe", "1h", "--hmm-strict-mode"])
    def test_parse_args_hmm_strict_mode(self):
        """Test parse_args with HMM strict mode."""
        args = parse_args(mode="hybrid")

        assert args.hmm_strict_mode is True

    @patch("cli.argument_parser.sys.argv", ["script.py", "--timeframe", "1h", "--robustness", "Wide"])
    def test_parse_args_robustness(self):
        """Test parse_args with robustness setting."""
        args = parse_args(mode="hybrid")

        assert args.robustness == "Wide"

    @patch("cli.argument_parser.sys.argv", ["script.py", "--timeframe", "1h", "--lambda", "0.8"])
    def test_parse_args_lambda_param(self):
        """Test parse_args with lambda parameter."""
        args = parse_args(mode="hybrid")

        assert args.lambda_param == 0.8
