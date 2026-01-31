"""
Tests for argument_parser.py

Comprehensive test coverage for CLI argument parsing.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.argument_parser import parse_args, ATCArguments


class TestDefaultValues:
    """Test default values for all arguments."""

    def test_default_symbol(self):
        """Test default symbol value."""
        args = parse_args([])
        assert args.symbol is None

    def test_default_quote(self):
        """Test default quote value."""
        args = parse_args([])
        assert args.quote == "USDT"

    def test_default_timeframe(self):
        """Test default timeframe value."""
        # Use patch to ensure we test against expected default regardless of config.py
        with patch("cli.argument_parser.DEFAULT_TIMEFRAME", "1h"):
            args = parse_args([])
            assert args.timeframe == "1h"

    def test_default_limit(self):
        """Test default limit value."""
        args = parse_args([])
        assert args.limit == 1500

    def test_default_ema_len(self):
        """Test default EMA length."""
        args = parse_args([])
        assert args.ema_len == 28

    def test_default_hma_len(self):
        """Test default HMA length."""
        args = parse_args([])
        assert args.hma_len == 28

    def test_default_wma_len(self):
        """Test default WMA length."""
        args = parse_args([])
        assert args.wma_len == 28

    def test_default_dema_len(self):
        """Test default DEMA length."""
        args = parse_args([])
        assert args.dema_len == 28

    def test_default_lsma_len(self):
        """Test default LSMA length."""
        args = parse_args([])
        assert args.lsma_len == 28

    def test_default_kama_len(self):
        """Test default KAMA length."""
        args = parse_args([])
        assert args.kama_len == 28

    def test_default_robustness(self):
        """Test default robustness value."""
        args = parse_args([])
        assert args.robustness == "Medium"

    def test_default_lambda_param(self):
        """Test default lambda_param value."""
        args = parse_args([])
        assert args.lambda_param == 0.02

    def test_default_decay(self):
        """Test default decay value."""
        args = parse_args([])
        assert args.decay == 0.03

    def test_default_cutout(self):
        """Test default cutout value."""
        args = parse_args([])
        assert args.cutout == 0

    def test_default_min_signal(self):
        """Test default min_signal value."""
        args = parse_args([])
        assert args.min_signal == 0.01

    def test_default_batch_size(self):
        """Test default batch_size value."""
        args = parse_args([])
        assert args.batch_size == 100

    def test_default_boolean_flags(self):
        """Test default boolean flag values."""
        args = parse_args([])
        assert args.no_prompt is False
        assert args.no_menu is False
        assert args.list_symbols is False
        assert args.auto is False


class TestArgumentParsing:
    """Test that arguments parse correctly."""

    def test_parse_symbol(self):
        """Test parsing symbol argument."""
        args = parse_args(["--symbol", "ETH/USDT"])
        assert args.symbol == "ETH/USDT"

    def test_parse_quote(self):
        """Test parsing quote argument."""
        args = parse_args(["--quote", "BTC"])
        assert args.quote == "BTC"

    def test_parse_timeframe(self):
        """Test parsing timeframe argument."""
        args = parse_args(["--timeframe", "15m"])
        assert args.timeframe == "15m"

    def test_parse_limit(self):
        """Test parsing limit argument."""
        args = parse_args(["--limit", "1000"])
        assert args.limit == 1000

    def test_parse_ema_len(self):
        """Test parsing EMA length."""
        args = parse_args(["--ema-len", "50"])
        assert args.ema_len == 50

    def test_parse_lambda_param(self):
        """Test parsing lambda_param argument."""
        args = parse_args(["--lambda-param", "0.05"])
        assert args.lambda_param == 0.05

    def test_parse_decay(self):
        """Test parsing decay argument."""
        args = parse_args(["--decay", "0.1"])
        assert args.decay == 0.1

    def test_parse_cutout(self):
        """Test parsing cutout argument."""
        args = parse_args(["--cutout", "10"])
        assert args.cutout == 10

    def test_parse_min_signal(self):
        """Test parsing min_signal argument."""
        args = parse_args(["--min-signal", "0.05"])
        assert args.min_signal == 0.05

    def test_parse_batch_size(self):
        """Test parsing batch_size argument."""
        args = parse_args(["--batch-size", "50"])
        assert args.batch_size == 50

    def test_parse_boolean_flags(self):
        """Test parsing boolean flags."""
        args = parse_args(["--no-prompt", "--no-menu", "--auto"])
        assert args.no_prompt is True
        assert args.no_menu is True
        assert args.auto is True

    def test_parse_list_symbols(self):
        """Test parsing list-symbols flag."""
        args = parse_args(["--list-symbols"])
        assert args.list_symbols is True

    def test_parse_max_symbols(self):
        """Test parsing max-symbols argument."""
        args = parse_args(["--max-symbols", "50"])
        assert args.max_symbols == 50


class TestTypeConversions:
    """Test type conversions work correctly."""

    def test_string_type(self):
        """Test string type conversion."""
        args = parse_args(["--symbol", "BTC/USDT"])
        assert isinstance(args.symbol, str)

    def test_int_type(self):
        """Test integer type conversion."""
        args = parse_args(["--limit", "1000"])
        assert isinstance(args.limit, int)

    def test_float_type(self):
        """Test float type conversion."""
        args = parse_args(["--lambda-param", "0.05"])
        assert isinstance(args.lambda_param, float)

    def test_bool_type(self):
        """Test boolean type conversion."""
        args = parse_args(["--auto"])
        assert isinstance(args.auto, bool)


class TestRobustnessChoices:
    """Test robustness choices work correctly."""

    def test_robustness_narrow(self):
        """Test narrow robustness choice."""
        args = parse_args(["--robustness", "Narrow"])
        assert args.robustness == "Narrow"

    def test_robustness_medium(self):
        """Test medium robustness choice."""
        args = parse_args(["--robustness", "Medium"])
        assert args.robustness == "Medium"

    def test_robustness_wide(self):
        """Test wide robustness choice."""
        args = parse_args(["--robustness", "Wide"])
        assert args.robustness == "Wide"

    def test_invalid_robustness(self, capsys):
        """Test invalid robustness value."""
        with pytest.raises(SystemExit):
            parse_args(["--robustness", "Invalid"])


class TestValidation:
    """Test input validation."""

    def test_negative_limit_fails(self, capsys):
        """Test that negative limit fails validation."""
        with pytest.raises(SystemExit):
            parse_args(["--limit", "-1"])

    def test_zero_limit_fails(self, capsys):
        """Test that zero limit fails validation."""
        with pytest.raises(SystemExit):
            parse_args(["--limit", "0"])

    def test_limit_too_large_fails(self, capsys):
        """Test that limit > 10000 fails validation."""
        with pytest.raises(SystemExit):
            parse_args(["--limit", "10001"])

    def test_negative_batch_size_fails(self, capsys):
        """Test that negative batch_size fails validation."""
        with pytest.raises(SystemExit):
            parse_args(["--batch-size", "-1"])

    def test_zero_batch_size_fails(self, capsys):
        """Test that zero batch_size fails validation."""
        with pytest.raises(SystemExit):
            parse_args(["--batch-size", "0"])

    def test_batch_size_too_large_fails(self, capsys):
        """Test that batch_size > 1000 fails validation."""
        with pytest.raises(SystemExit):
            parse_args(["--batch-size", "1001"])

    def test_min_signal_negative_fails(self, capsys):
        """Test that negative min_signal fails validation."""
        with pytest.raises(SystemExit):
            parse_args(["--min-signal", "-0.01"])

    def test_min_signal_zero_fails(self, capsys):
        """Test that zero min_signal fails validation."""
        with pytest.raises(SystemExit):
            parse_args(["--min-signal", "0"])

    def test_min_signal_too_large_fails(self, capsys):
        """Test that min_signal > 1.0 fails validation."""
        with pytest.raises(SystemExit):
            parse_args(["--min-signal", "1.5"])

    def test_negative_ema_len_fails(self, capsys):
        """Test that negative EMA length fails validation."""
        with pytest.raises(SystemExit):
            parse_args(["--ema-len", "-1"])

    def test_negative_hma_len_fails(self, capsys):
        """Test that negative HMA length fails validation."""
        with pytest.raises(SystemExit):
            parse_args(["--hma-len", "-1"])

    def test_negative_cutout_fails(self, capsys):
        """Test that negative cutout fails validation."""
        with pytest.raises(SystemExit):
            parse_args(["--cutout", "-1"])

    def test_negative_lambda_param_fails(self, capsys):
        """Test that negative lambda_param fails validation."""
        with pytest.raises(SystemExit):
            parse_args(["--lambda-param", "-0.01"])

    def test_negative_decay_fails(self, capsys):
        """Test that negative decay fails validation."""
        with pytest.raises(SystemExit):
            parse_args(["--decay", "-0.01"])


class TestVersionArgument:
    """Test --version argument."""

    def test_version_argument(self, capsys):
        """Test that --version displays version."""
        with pytest.raises(SystemExit):
            parse_args(["--version"])

        captured = capsys.readouterr()
        assert "1.0.0" in captured.out or "1.0.0" in captured.err


class TestHelpText:
    """Test help text contains all arguments."""

    def test_help_contains_all_arguments(self, capsys):
        """Test that help contains all major arguments."""
        with pytest.raises(SystemExit):
            parse_args(["--help"])

        captured = capsys.readouterr()
        assert "--symbol" in captured.out
        assert "--quote" in captured.out
        assert "--timeframe" in captured.out
        assert "--limit" in captured.out
        assert "--ema-len" in captured.out
        assert "--hma-len" in captured.out
        assert "--wma-len" in captured.out
        assert "--dema-len" in captured.out
        assert "--lsma-len" in captured.out
        assert "--kama-len" in captured.out
        assert "--robustness" in captured.out
        assert "--lambda-param" in captured.out
        assert "--decay" in captured.out
        assert "--cutout" in captured.out
        assert "--no-prompt" in captured.out
        assert "--no-menu" in captured.out
        assert "--list-symbols" in captured.out
        assert "--auto" in captured.out
        assert "--max-symbols" in captured.out
        assert "--min-signal" in captured.out
        assert "--batch-size" in captured.out
        assert "--version" in captured.out

    def test_help_shows_defaults(self, capsys):
        """Test that help shows correct default values."""
        with pytest.raises(SystemExit):
            parse_args(["--help"])

        captured = capsys.readouterr()
        assert "28" in captured.out
        assert "0.02" in captured.out
        assert "0.03" in captured.out


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_command_line(self):
        """Test empty command line (all defaults)."""
        args = parse_args([])
        assert args is not None

    def test_mix_of_flags_and_values(self):
        """Test mix of flags and values."""
        args = parse_args(["--symbol", "BTC/USDT", "--timeframe", "15m", "--auto", "--no-prompt"])
        assert args.symbol == "BTC/USDT"
        assert args.timeframe == "15m"
        assert args.auto is True
        assert args.no_prompt is True

    def test_unknown_argument_fails(self, capsys):
        """Test that unknown argument is rejected."""
        with pytest.raises(SystemExit):
            parse_args(["--unknown-argument"])


class TestMutuallyExclusiveGroups:
    """Test mutually exclusive argument groups."""

    def test_auto_and_list_symbols_conflict(self, capsys):
        """Test that --auto and --list-symbols cannot be used together."""
        with pytest.raises(SystemExit):
            parse_args(["--auto", "--list-symbols"])


class TestConfigImportFailure:
    """Test behavior when config import fails."""

    def test_fallback_defaults(self):
        """Test that fallback defaults are used when config is missing."""
        # Mock sys.modules to simulate config module missing
        with patch.dict(sys.modules, {"config": None}):
            # We need to reload the module to trigger the ImportError block
            # However, since parse_args is already imported, we'll just verify
            # the logic inside parse_args or check if we can patch the imported values.
            # A better way for this specific test might be to verify the constants
            # if they were exposed, but since they are module-level variables
            # set at import time, this is tricky to test after import.
            # Instead, we can verify that the default values in the parser
            # match what we expect from the fallback block if config was missing.
            # But since config IS present in this environment, it uses config values.
            # To properly test this, we would need to reload the module.
            import importlib
            import cli.argument_parser

            # Temporarily remove config from sys.modules to force ImportError
            with patch.dict(sys.modules):
                sys.modules.pop("config", None)
                # Mock __import__ to raise ImportError for config
                original_import = __import__

                def mock_import(name, *args, **kwargs):
                    if name == "config":
                        raise ImportError("No module named 'config'")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    importlib.reload(cli.argument_parser)
                    args = cli.argument_parser.parse_args([])
                    # Verify fallback defaults
                    # These should match the hardcoded values in the ImportError block
                    # DEFAULT_SYMBOL = "BTC/USDT"
                    # DEFAULT_QUOTE = "USDT"
                    # DEFAULT_TIMEFRAME = "1h"
                    # DEFAULT_LIMIT = 1500
                    assert args.symbol is None  # default is None in parser, but help text changes
                    assert args.quote == "USDT"
                    assert args.timeframe == "1h"
                    assert args.limit == 1500

            # Reload again to restore normal state for other tests
            importlib.reload(cli.argument_parser)


class TestATCArgumentsDataclass:
    """Test ATCArguments dataclass."""

    def test_from_namespace(self):
        """Test creating ATCArguments from argparse.Namespace."""
        args_ns = parse_args(["--symbol", "ETH/USDT", "--limit", "500"])
        args_obj = ATCArguments.from_namespace(args_ns)

        assert isinstance(args_obj, ATCArguments)
        assert args_obj.symbol == "ETH/USDT"
        assert args_obj.limit == 500
        assert args_obj.quote == "USDT"  # Default
        assert args_obj.ema_len == 28  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
