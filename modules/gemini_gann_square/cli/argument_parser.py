"""
Argument parser for Gemini Gann Square CLI.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Gann Square analyzer."""
    parser = argparse.ArgumentParser(
        prog="gemini_gann_square",
        description=(
            "Analyze a crypto symbol using Gann Square methodology + Google Gemini AI.\n"
            "Produces LONG / SHORT / SKIP signals with Entry, Stop Loss, and Take Profit levels."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.gemini_gann_square --symbol BTCUSDT --timeframe 4h
  python -m modules.gemini_gann_square --symbol ETH/USDT --timeframe 1h --limit 300 --lookback 7
  python -m modules.gemini_gann_square   (interactive menu)
""",
    )

    parser.add_argument(
        "--symbol",
        "-s",
        type=str,
        default=None,
        help="Trading symbol, e.g. BTCUSDT or BTC/USDT (default: interactive prompt)",
    )
    parser.add_argument(
        "--timeframe",
        "-t",
        type=str,
        default=None,
        help="Candle timeframe, e.g. 1h, 4h, 1d (default: interactive prompt)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=200,
        help="Number of candles to fetch (default: 200)",
    )
    parser.add_argument(
        "--lookback",
        "-n",
        type=int,
        default=5,
        help="Zigzag pivot lookback window — candles on each side (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="charts",
        dest="output_dir",
        help="Directory to save chart PNG files (default: charts/)",
    )

    return parser.parse_args()
