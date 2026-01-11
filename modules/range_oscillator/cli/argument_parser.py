
import argparse

"""
Command-line argument parser for ATC + Range Oscillator combined signal filter.

This module provides the main argument parser for the ATC + Range Oscillator CLI,
defining all command-line options and their default values.
"""


try:
    from config import DEFAULT_TIMEFRAME
except ImportError:
    DEFAULT_TIMEFRAME = "15m"


def parse_args():
    """Parse command-line arguments for ATC + Range Oscillator combined signal filter."""
    # DEBUG POINT: Argument parser entry - Check default_timeframe

    parser = argparse.ArgumentParser(
        description="ATC + Range Oscillator Combined Signal Filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    def validate_timeframe(value):
        """Validate timeframe format."""
        if not isinstance(value, str) or len(value) < 2:
            raise argparse.ArgumentTypeError(f"Invalid timeframe format: {value}")
        return value

    parser.add_argument(
        "--timeframe",
        type=validate_timeframe,
        default=DEFAULT_TIMEFRAME,
        help=f"Timeframe for analysis (default: {DEFAULT_TIMEFRAME})",
    )
    parser.add_argument(
        "--no-menu",
        action="store_true",
        help="Disable interactive timeframe menu",
    )

    def validate_positive_int(value):
        """Validate positive integer."""
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"Value must be positive, got {ivalue}")
        return ivalue

    parser.add_argument(
        "--limit",
        type=validate_positive_int,
        default=500,
        help="Number of candles to fetch (default: 500, must be > 0)",
    )
    parser.add_argument(
        "--ema-len",
        type=int,
        default=28,
        help="EMA length (default: 28)",
    )
    parser.add_argument(
        "--hma-len",
        type=int,
        default=28,
        help="HMA length (default: 28)",
    )
    parser.add_argument(
        "--wma-len",
        type=int,
        default=28,
        help="WMA length (default: 28)",
    )
    parser.add_argument(
        "--dema-len",
        type=int,
        default=28,
        help="DEMA length (default: 28)",
    )
    parser.add_argument(
        "--lsma-len",
        type=int,
        default=28,
        help="LSMA length (default: 28)",
    )
    parser.add_argument(
        "--kama-len",
        type=int,
        default=28,
        help="KAMA length (default: 28)",
    )
    parser.add_argument(
        "--robustness",
        type=str,
        choices=["Narrow", "Medium", "Wide"],
        default="Medium",
        help="Robustness setting (default: Medium)",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.5,
        dest="lambda_param",
        help="Lambda parameter (default: 0.5)",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.1,
        help="Decay rate (default: 0.1)",
    )
    parser.add_argument(
        "--cutout",
        type=int,
        default=5,
        help="Number of bars to skip at start (default: 5)",
    )
    parser.add_argument(
        "--min-signal",
        type=float,
        default=0.01,
        help="Minimum signal strength to display (default: 0.01)",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Maximum number of symbols to scan (default: None = all)",
    )
    parser.add_argument(
        "--osc-length",
        type=int,
        default=50,
        help="Range Oscillator length parameter (default: 50)",
    )
    parser.add_argument(
        "--osc-mult",
        type=float,
        default=2.0,
        help="Range Oscillator multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--max-workers",
        type=validate_positive_int,
        default=10,
        help="Maximum number of parallel workers for Range Oscillator filtering (default: 10, must be > 0)",
    )

    def validate_strategy_id(value):
        """Validate strategy ID."""
        ivalue = int(value)
        valid_ids = {2, 3, 4, 6, 7, 8, 9}
        if ivalue not in valid_ids:
            raise argparse.ArgumentTypeError(f"Invalid strategy ID: {ivalue}. Valid IDs: {sorted(valid_ids)}")
        return ivalue

    parser.add_argument(
        "--osc-strategies",
        type=validate_strategy_id,
        nargs="+",
        default=None,
        help="Range Oscillator strategies to use (e.g., --osc-strategies 5 6 7 8 9). Valid IDs: 2, 3, 4, 6, 7, 8, 9",
    )

    # DEBUG POINT: Parse arguments - Check parsed values and handle exceptions
    # Check: timeframe, limit, osc_strategies, max_workers
    # On exception: Check exception_type, exception_msg
    try:
        args = parser.parse_args()
    except Exception:
        raise

    # Additional validation after parsing
    if args.max_workers is not None and args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    if args.min_signal < 0 or args.min_signal > 1:
        parser.error("--min-signal must be between 0 and 1")
    if args.lambda_param < 0 or args.lambda_param > 1:
        parser.error("--lambda must be between 0 and 1")
    if args.decay < 0 or args.decay > 1:
        parser.error("--decay must be between 0 and 1")

    return args
