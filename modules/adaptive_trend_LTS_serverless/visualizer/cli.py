"""
CLI entry point for the ATC Chart Visualizer.

Usage:
    python -m modules.adaptive_trend_LTS_serverless.visualizer --symbol BTCUSDT --tf 1h
    python -m modules.adaptive_trend_LTS_serverless.visualizer --symbol ETHUSDT --tf 4h --mock
    python -m modules.adaptive_trend_LTS_serverless.visualizer --symbol BTCUSDT --tf 1h --output out.png --display 200
"""

from __future__ import annotations

import argparse
import logging
import sys

from modules.adaptive_trend_LTS_serverless.lambda_client import (
    ATCLambdaClient,
    DEFAULT_FUNCTION_NAME,
    DEFAULT_REGION,
)
from modules.backtester_static import BacktestConfig

from .chart import ATCChartRenderer


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="atc-visualizer",
        description=(
            "Render a candlestick chart for a symbol/timeframe with "
            "ATC signal markers and all 6 MA lines (EMA, WMA, DEMA, LSMA, HMA, KAMA)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--symbol", required=True, metavar="SYM",
                   help="Trading pair, e.g. BTCUSDT")
    p.add_argument("--tf", required=True, metavar="TF",
                   help="Timeframe, e.g. 1h, 4h, 15m, 1d")

    # Data
    p.add_argument("--limit", type=int, default=300, metavar="N",
                   help="Total bars to fetch (used for MA warmup)")
    p.add_argument("--display", type=int, default=150, metavar="N",
                   help="Bars to display in chart")

    # MA
    p.add_argument("--ma-length", type=int, default=12, metavar="L",
                   help="Period for all MA lines")

    # Output
    p.add_argument("--output", metavar="FILE",
                   help="Output PNG path (auto-generated under ./charts/ if omitted)")

    # Lambda options
    p.add_argument("--mock", action="store_true",
                   help="Use mock Lambda (no AWS credentials needed)")
    p.add_argument("--function", default=DEFAULT_FUNCTION_NAME, metavar="NAME",
                   help="Lambda function name / ARN")
    p.add_argument("--region", default=DEFAULT_REGION, metavar="REGION",
                   help="AWS region")

    # Logging
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable debug logging")

    # Backtest
    bt = p.add_argument_group("backtest (optional)")
    bt.add_argument("--backtest", action="store_true",
                    help="Run static backtest and overlay TP/SL on chart")
    bt.add_argument("--bt-mode", choices=["pct", "atr"], default="pct",
                    help="TP/SL mode: pct = percent of entry, atr = ATR multiple")
    bt.add_argument("--bt-tp", type=float, default=2.0, metavar="N",
                    help="Take-profit level (%%  or ATR multiple)")
    bt.add_argument("--bt-sl", type=float, default=1.0, metavar="N",
                    help="Stop-loss level (%% or ATR multiple)")
    bt.add_argument("--bt-trail", type=float, default=None, metavar="N",
                    help="Trailing-stop level (same unit as SL); omit to disable")
    bt.add_argument("--bt-capital", type=float, default=10_000.0, metavar="N",
                    help="Initial capital for equity curve")
    bt.add_argument("--bt-max-hold", type=int, default=100, metavar="N",
                    help="Force-close position after N bars")

    return p


def _prompt_text(label: str, default: str) -> str:
    raw = input(f"{label} [{default}]: ").strip()
    return raw or default


def _prompt_int(label: str, default: int, min_value: int = 1) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if value < min_value:
                print(f"Please enter a value >= {min_value}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid integer.")


def _prompt_bool(label: str, default: bool) -> bool:
    default_hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{label} [{default_hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes", "1", "true", "t"}:
            return True
        if raw in {"n", "no", "0", "false", "f"}:
            return False
        print("Please answer yes or no (y/n).")


def _prompt_optional_float(
    label: str,
    default: float | None = None,
    allow_none: bool = True,
) -> float | None:
    """
    Prompt optional float value.

    Accepts:
      - empty input      -> default
      - "none"/"no"/"n" -> None (if allow_none=True)
    - "yes"/"y"      -> default (or 1.0 when default is None)
      - numeric input    -> parsed float
    """
    default_text = "none" if default is None else str(default)
    while True:
        raw = input(f"{label} [{default_text}]: ").strip().lower()

        if not raw:
            return default

        if allow_none and raw in {"none", "no", "n"}:
            return None

        if raw in {"yes", "y"}:
            return default if default is not None else 1.0

        try:
            value = float(raw)
            if value <= 0:
                print("Please enter a value > 0.")
                continue
            return value
        except ValueError:
            accepted = "number / yes / no / none"
            print(f"Please enter a valid {accepted}.")


def _collect_interactive_args() -> argparse.Namespace:
    print("\n=== ATC Visualizer Interactive Menu ===")
    symbol = _prompt_text("Symbol", "BTCUSDT")
    tf = _prompt_text("Timeframe", "1h")
    limit = _prompt_int("Total bars (limit)", 300, min_value=30)
    display = _prompt_int("Bars to display", 150, min_value=10)
    if display > limit:
        print(f"Bars to display ({display}) > limit ({limit}), auto-set display = {limit}.")
        display = limit

    ma_length = _prompt_int("MA length", 12, min_value=1)
    output = input("Output PNG path [auto]: ").strip() or None
    mock = _prompt_bool("Use mock Lambda", True)
    if mock:
        print("Mock mode enabled: skipping Lambda function/AWS region (not used).")
        function = DEFAULT_FUNCTION_NAME
        region = DEFAULT_REGION
    else:
        function = _prompt_text("Lambda function name / ARN", DEFAULT_FUNCTION_NAME)
        region = _prompt_text("AWS region", DEFAULT_REGION)
    verbose = _prompt_bool("Verbose logging", False)

    # Backtest
    print("\n--- Backtest (optional) ---")
    backtest = _prompt_bool("Enable backtest overlay", False)
    bt_mode = "pct"
    bt_tp = 2.0
    bt_sl = 1.0
    bt_trail: float | None = None
    bt_capital = 10_000.0
    bt_max_hold = 100
    if backtest:
        bt_mode = _prompt_text("TP/SL mode (pct/atr)", "pct")
        bt_tp    = float(_prompt_text("TP level", "2.0"))
        bt_sl    = float(_prompt_text("SL level", "1.0"))
        use_trailing = _prompt_bool("Enable trailing stop", False)
        if use_trailing:
            bt_trail = _prompt_optional_float("Trailing stop level", 1.0, allow_none=False)
        else:
            bt_trail = _prompt_optional_float("Trailing stop", None, allow_none=True)
        bt_capital   = float(_prompt_text("Initial capital", "10000"))
        bt_max_hold  = _prompt_int("Max hold bars", 100, min_value=1)

    return argparse.Namespace(
        symbol=symbol,
        tf=tf,
        limit=limit,
        display=display,
        ma_length=ma_length,
        output=output,
        mock=mock,
        function=function,
        region=region,
        verbose=verbose,
        backtest=backtest,
        bt_mode=bt_mode,
        bt_tp=bt_tp,
        bt_sl=bt_sl,
        bt_trail=bt_trail,
        bt_capital=bt_capital,
        bt_max_hold=bt_max_hold,
    )


def main(argv: list[str] | None = None) -> None:
    if argv is None and len(sys.argv) == 1:
        args = _collect_interactive_args()
    else:
        parser = _build_parser()
        args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    client = ATCLambdaClient(
        function_name=args.function,
        region=args.region,
        mock_mode=args.mock,
    )
    renderer = ATCChartRenderer(client, ma_length=args.ma_length)

    # Build optional BacktestConfig
    backtest_cfg: BacktestConfig | None = None
    if getattr(args, "backtest", False):
        backtest_cfg = BacktestConfig(
            mode=getattr(args, "bt_mode", "pct"),
            tp=getattr(args, "bt_tp", 2.0),
            sl=getattr(args, "bt_sl", 1.0),
            trailing_stop=getattr(args, "bt_trail", None),
            initial_capital=getattr(args, "bt_capital", 10_000.0),
            max_hold_bars=getattr(args, "bt_max_hold", 100),
        )
        logging.getLogger(__name__).info("Backtest config: %s", backtest_cfg.summary())

    try:
        out = renderer.render(
            symbol=args.symbol,
            timeframe=args.tf,
            limit=args.limit,
            display_last=args.display,
            output_path=args.output,
            backtest_config=backtest_cfg,
        )
        print(f"Chart saved → {out}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
