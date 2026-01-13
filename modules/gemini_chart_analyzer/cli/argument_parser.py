"""
Argument parser for Gemini Chart Analyzer CLI.

This module provides functions for parsing command-line arguments
and interactive configuration menu for chart analysis with Gemini AI.
"""

import argparse
import sys

from colorama import Fore

from modules.common.utils import color_text, normalize_timeframe

# ============================================================================
# Utility Functions
# ============================================================================


def _format_current_value(value) -> str:
    """Format current value for display in menu."""
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        if "periods" in value:
            return f"periods={value['periods']}"
        if "period" in value:
            return f"period={value['period']}"
        return str(value)
    return str(value)


def parse_args():
    """
    Parse command-line arguments for Gemini Chart Analyzer.

    If no arguments provided, returns None to trigger interactive menu.
    Otherwise, parses command-line arguments.

    Returns:
        argparse.Namespace object with parsed arguments, or None if no args provided
    """
    # Check if any arguments were provided
    if len(sys.argv) == 1:
        # No arguments, use interactive menu
        return None

    parser = argparse.ArgumentParser(
        description="Gemini Chart Analyzer - AI-powered technical analysis using Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_gemini_chart_analyzer.py --symbol BTC/USDT --timeframe 1h
  python main_gemini_chart_analyzer.py --symbol ETH/USDT --timeframe 4h --prompt-type simple
  python main_gemini_chart_analyzer.py --symbol BTC/USDT --timeframe 1h --no-ma --no-rsi
        """,
    )

    # Symbol and timeframe (required if using CLI)
    parser.add_argument(
        "--symbol",
        type=str,
        help="Trading symbol (e.g., BTC/USDT, ETH/USDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help=(
            "Timeframe for analysis (default: 1h). "
            "Options: 15m/m15, 30m/m30, 1h/h1, 4h/h4, 1d/d1, 1w/w1. "
            "Ignored if --timeframes is provided."
        ),
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        help=(
            "Multiple timeframes for multi-timeframe analysis "
            '(comma-separated, e.g., "15m,1h,4h,1d"). '
            "If provided, enables multi-timeframe mode."
        ),
    )

    # Indicators configuration
    parser.add_argument(
        "--ma-periods",
        type=str,
        help='Moving Average periods (comma-separated, e.g., "20,50,200")',
    )
    parser.add_argument(
        "--no-ma",
        action="store_true",
        help="Disable Moving Averages",
    )
    parser.add_argument(
        "--rsi-period",
        type=int,
        default=14,
        help="RSI period (default: 14)",
    )
    parser.add_argument(
        "--no-rsi",
        action="store_true",
        help="Disable RSI",
    )
    parser.add_argument(
        "--no-macd",
        action="store_true",
        help="Disable MACD",
    )
    parser.add_argument(
        "--bb-period",
        type=int,
        default=20,
        help="Bollinger Bands period (default: 20)",
    )
    parser.add_argument(
        "--enable-bb",
        action="store_true",
        help="Enable Bollinger Bands (disabled by default)",
    )

    # Gemini prompt configuration
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["detailed", "simple", "custom"],
        default="detailed",
        help="Prompt type for Gemini analysis (default: detailed)",
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        help="Custom prompt text (required if --prompt-type is custom)",
    )

    # Chart configuration
    parser.add_argument(
        "--chart-figsize",
        type=str,
        default="16,10",
        help="Chart figure size as width,height (default: 16,10)",
    )
    parser.add_argument(
        "--chart-dpi",
        type=int,
        default=150,
        help="Chart DPI (default: 150)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable automatic cleanup of old charts and results",
    )

    # Data fetching
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of candles to fetch (default: 500)",
    )

    args = parser.parse_args()

    # Validate that custom-prompt is provided when prompt-type is custom
    if args.prompt_type == "custom" and (not args.custom_prompt or args.custom_prompt.strip() == ""):
        parser.error("--custom-prompt is required when --prompt-type is 'custom'")

    # If symbol not provided, return None to trigger interactive menu
    # This allows running without args to get interactive menu
    if args.symbol is None:
        return None

    # Parse MA periods if provided
    if args.ma_periods:
        try:
            args.ma_periods_list = [int(p.strip()) for p in args.ma_periods.split(",")]
        except ValueError:
            print(color_text("Warning: Invalid MA periods format. Using default: 20,50,200", Fore.YELLOW))
            args.ma_periods_list = [20, 50, 200]
    else:
        args.ma_periods_list = None

    # Parse chart figsize
    try:
        width, height = map(int, args.chart_figsize.split(","))
        args.chart_figsize_tuple = (width, height)
    except (ValueError, AttributeError):
        print(color_text("Warning: Invalid figsize format. Using default: (16, 10)", Fore.YELLOW))
        args.chart_figsize_tuple = (16, 10)

    # Normalize timeframe (accept both '15m' and 'm15', '1h' and 'h1', etc.)
    args.timeframe = normalize_timeframe(args.timeframe)

    # Parse timeframes if provided (multi-timeframe mode)
    if args.timeframes:
        try:
            from modules.gemini_chart_analyzer.core.utils import normalize_timeframes

            timeframes_list = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
            args.timeframes_list = normalize_timeframes(timeframes_list)
            if not args.timeframes_list:
                print(color_text("Warning: No valid timeframes found. Using single timeframe mode.", Fore.YELLOW))
                args.timeframes_list = None
        except Exception as e:
            print(color_text(f"Warning: Error parsing timeframes: {e}. Using single timeframe mode.", Fore.YELLOW))
            args.timeframes_list = None
    else:
        args.timeframes_list = None

    return args
