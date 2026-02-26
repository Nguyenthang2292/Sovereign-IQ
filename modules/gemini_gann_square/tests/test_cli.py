"""
CLI smoke tests for gemini_gann_square.
"""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import patch

from modules.gemini_gann_square.cli import argument_parser
from modules.gemini_gann_square.cli import gann_main
from modules.gemini_gann_square.cli import interactive_menu
from modules.gemini_gann_square.cli import runner


class TestArgumentParser:
    def test_parse_args_with_explicit_values(self):
        with patch(
            "sys.argv",
            [
                "gemini_gann_square",
                "--symbol",
                "BTC/USDT",
                "--timeframe",
                "4h",
                "--limit",
                "250",
                "--lookback",
                "7",
                "--output-dir",
                "tmp_charts",
            ],
        ):
            args = argument_parser.parse_args()

        assert args.symbol == "BTC/USDT"
        assert args.timeframe == "4h"
        assert args.limit == 250
        assert args.lookback == 7
        assert args.output_dir == "tmp_charts"


class TestMainRouting:
    def test_main_runs_analysis_when_symbol_and_timeframe_present(self):
        args = Namespace(symbol="BTC/USDT", timeframe="4h", limit=200, lookback=5, output_dir="charts")
        with patch("modules.gemini_gann_square.cli.gann_main.parse_args", return_value=args), patch(
            "modules.gemini_gann_square.cli.gann_main.run_analysis"
        ) as run_analysis_mock, patch("modules.gemini_gann_square.cli.gann_main.run_interactive_menu") as menu_mock:
            gann_main.main()

        run_analysis_mock.assert_called_once()
        menu_mock.assert_not_called()

    def test_main_runs_interactive_menu_when_missing_required_args(self):
        args = Namespace(symbol=None, timeframe=None, limit=200, lookback=5, output_dir="charts")
        with patch("modules.gemini_gann_square.cli.gann_main.parse_args", return_value=args), patch(
            "modules.gemini_gann_square.cli.gann_main.run_analysis"
        ) as run_analysis_mock, patch("modules.gemini_gann_square.cli.gann_main.run_interactive_menu") as menu_mock:
            gann_main.main()

        menu_mock.assert_called_once()
        run_analysis_mock.assert_not_called()


class TestInteractiveMenu:
    def test_run_interactive_menu_exit_path(self):
        with patch("builtins.input", side_effect=["3"]):
            interactive_menu.run_interactive_menu()


class TestRunner:
    def test_run_analysis_calls_engine_and_prints_result(self):
        fake_result = type("Result", (), {"display": lambda self: "ok"})()

        with patch("modules.gemini_gann_square.cli.runner.GannSignalEngine") as engine_cls, patch(
            "builtins.print"
        ) as print_mock:
            engine = engine_cls.return_value
            engine.analyze.return_value = fake_result

            runner.run_analysis(symbol="BTC/USDT", timeframe="4h")

        engine_cls.assert_called_once()
        engine.analyze.assert_called_once()
        print_mock.assert_called_with("ok")
