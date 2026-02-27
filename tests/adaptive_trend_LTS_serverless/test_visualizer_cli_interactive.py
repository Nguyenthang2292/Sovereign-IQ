from __future__ import annotations

import builtins

from modules.adaptive_trend_LTS_serverless.visualizer.cli import _collect_interactive_args


def _mock_input_factory(values: list[str]):
    data = iter(values)

    def _mock_input(_prompt: str = "") -> str:
        return next(data)

    return _mock_input


def test_collect_interactive_args_trailing_yes_uses_default(monkeypatch) -> None:
    """Regression: entering 'yes' for trailing stop must not crash."""
    inputs = [
        "BTCUSDT",    # symbol
        "1h",         # tf
        "300",        # limit
        "150",        # display
        "12",         # ma length
        "",           # output
        "y",          # mock
        "n",          # verbose
        "y",          # backtest
        "pct",        # bt_mode
        "2.0",        # bt_tp
        "1.0",        # bt_sl
        "yes",        # enable trailing stop
        "",           # trailing level -> default 1.0
        "10000",      # bt_capital
        "100",        # bt_max_hold
    ]

    monkeypatch.setattr(builtins, "input", _mock_input_factory(inputs))
    args = _collect_interactive_args()

    assert args.backtest is True
    assert args.bt_trail == 1.0


def test_collect_interactive_args_trailing_none(monkeypatch) -> None:
    inputs = [
        "BTCUSDT",    # symbol
        "1h",         # tf
        "300",        # limit
        "150",        # display
        "12",         # ma length
        "",           # output
        "y",          # mock
        "n",          # verbose
        "y",          # backtest
        "pct",        # bt_mode
        "2.0",        # bt_tp
        "1.0",        # bt_sl
        "n",          # enable trailing stop
        "",           # trailing stop optional -> None
        "10000",      # bt_capital
        "100",        # bt_max_hold
    ]

    monkeypatch.setattr(builtins, "input", _mock_input_factory(inputs))
    args = _collect_interactive_args()

    assert args.backtest is True
    assert args.bt_trail is None


def test_collect_interactive_args_non_mock_asks_function_and_region(monkeypatch) -> None:
    inputs = [
        "BTCUSDT",       # symbol
        "1h",            # tf
        "300",           # limit
        "150",           # display
        "12",            # ma length
        "",              # output
        "n",             # mock -> false, so prompt function/region
        "my-lambda-fn",  # function
        "ap-southeast-1",# region
        "n",             # verbose
        "n",             # backtest
    ]

    monkeypatch.setattr(builtins, "input", _mock_input_factory(inputs))
    args = _collect_interactive_args()

    assert args.mock is False
    assert args.function == "my-lambda-fn"
    assert args.region == "ap-southeast-1"
