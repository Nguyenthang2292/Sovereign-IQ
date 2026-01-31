"""
Command-line argument parser for ATC analysis.

This module provides main argument parser for ATC CLI,
defining all command-line options and their default values.
"""

import argparse
from dataclasses import dataclass
from typing import Optional, List

# Default values from config
try:
    from config import (
        DEFAULT_LIMIT,
        DEFAULT_QUOTE,
        DEFAULT_SYMBOL,
        DEFAULT_TIMEFRAME,
    )
except ImportError:
    DEFAULT_SYMBOL = "BTC/USDT"
    DEFAULT_QUOTE = "USDT"
    DEFAULT_TIMEFRAME = "1h"
    DEFAULT_LIMIT = 1500

# Default parameter values
DEFAULT_MA_LENGTH = 28
DEFAULT_LAMBDA_PARAM = 0.02
DEFAULT_DECAY = 0.03
DEFAULT_CUTOUT = 0
DEFAULT_MIN_SIGNAL = 0.01
DEFAULT_BATCH_SIZE = 100

# Security and validation limits
MAX_LIMIT = 10000
MAX_BATCH_SIZE = 1000

# Version information
VERSION = "1.0.0"


@dataclass
class ATCArguments:
    """Typed arguments for ATC analysis.

    This provides type-safe access to parsed arguments.
    """

    symbol: Optional[str]
    quote: str
    timeframe: str
    limit: int
    ema_len: int
    hma_len: int
    wma_len: int
    dema_len: int
    lsma_len: int
    kama_len: int
    robustness: str
    lambda_param: float
    decay: float
    cutout: int
    no_prompt: bool
    no_menu: bool
    list_symbols: bool
    max_symbols: Optional[int]
    min_signal: float
    auto: bool
    batch_size: int

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> "ATCArguments":
        """Convert argparse.Namespace to typed ATCArguments.

        Args:
            ns: argparse.Namespace to convert

        Returns:
            ATCArguments: Typed version of the namespace
        """
        return cls(**vars(ns))


def _add_ma_arguments(parser: argparse.ArgumentParser, ma_length: int) -> None:
    """Add Moving Average length arguments to parser.

    Args:
        parser: Argument parser to add arguments to
        ma_length: Default MA length for all indicators
    """
    ma_group = parser.add_argument_group("Moving Average Parameters")
    ma_types = ["ema", "hma", "wma", "dema", "lsma", "kama"]
    for ma_type in ma_types:
        ma_group.add_argument(
            f"--{ma_type}-len",
            type=int,
            default=ma_length,
            help=f"{ma_type.upper()} length (default: {ma_length})",
        )


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for ATC analysis.

    Args:
        args: Optional list of command-line arguments to parse.
              If None, sys.argv is used.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Adaptive Trend Classification (ATC) Analysis for Binance Futures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    # Basic options
    basic_group = parser.add_argument_group("Basic Options")
    basic_group.add_argument(
        "--symbol",
        type=str,
        default=None,
        help=f"Symbol pair to analyze (default: {DEFAULT_SYMBOL})",
    )
    basic_group.add_argument(
        "--quote",
        type=str,
        default=DEFAULT_QUOTE,
        help=f"Quote currency (default: {DEFAULT_QUOTE})",
    )
    basic_group.add_argument(
        "--timeframe",
        type=str,
        default=DEFAULT_TIMEFRAME,
        help=f"Timeframe for analysis (default: {DEFAULT_TIMEFRAME})",
    )
    basic_group.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of candles to fetch (default: {DEFAULT_LIMIT})",
    )

    _add_ma_arguments(parser, DEFAULT_MA_LENGTH)

    # Advanced parameters
    advanced_group = parser.add_argument_group("Advanced Parameters")
    advanced_group.add_argument(
        "--robustness",
        type=str,
        choices=["Narrow", "Medium", "Wide"],
        default="Medium",
        help="Robustness setting (default: Medium)",
    )
    advanced_group.add_argument(
        "--lambda-param",
        type=float,
        default=DEFAULT_LAMBDA_PARAM,
        dest="lambda_param",
        help=f"Lambda parameter for exponential growth (default: {DEFAULT_LAMBDA_PARAM})",
    )
    advanced_group.add_argument(
        "--decay",
        type=float,
        default=DEFAULT_DECAY,
        help=f"Decay rate (default: {DEFAULT_DECAY})",
    )
    advanced_group.add_argument(
        "--cutout",
        type=int,
        default=DEFAULT_CUTOUT,
        help=f"Number of bars to skip at start (default: {DEFAULT_CUTOUT})",
    )

    # Mode options
    mode_group = parser.add_argument_group("Mode Options")

    # Mutually exclusive options
    exclusive_mode_group = mode_group.add_mutually_exclusive_group()
    exclusive_mode_group.add_argument(
        "--list-symbols",
        action="store_true",
        help="List available futures symbols and exit",
    )
    exclusive_mode_group.add_argument(
        "--auto",
        action="store_true",
        help="Force auto mode (scan all symbols)",
    )

    mode_group.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive prompts",
    )
    mode_group.add_argument(
        "--no-menu",
        action="store_true",
        help="Disable interactive menu",
    )

    # Performance options
    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Maximum number of symbols to scan in auto mode",
    )
    perf_group.add_argument(
        "--min-signal",
        type=float,
        default=DEFAULT_MIN_SIGNAL,
        help=f"Minimum signal strength to display (default: {DEFAULT_MIN_SIGNAL})",
    )
    perf_group.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        dest="batch_size",
        help=f"Number of symbols to process in each batch before forcing GC (default: {DEFAULT_BATCH_SIZE}). "
        "Larger batches use more memory but may be faster. Smaller batches use less memory.",
    )

    args = parser.parse_args(args)

    # Validate numerical arguments
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.limit > MAX_LIMIT:
        parser.error(f"--limit too large (max: {MAX_LIMIT})")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.batch_size > MAX_BATCH_SIZE:
        parser.error(f"--batch-size too large (max: {MAX_BATCH_SIZE})")

    if not (0 < args.min_signal <= 1.0):
        parser.error("--min-signal must be between 0 and 1.0")

    # Validate MA lengths
    ma_types = ["ema", "hma", "wma", "dema", "lsma", "kama"]
    for ma in ma_types:
        ma_len = getattr(args, f"{ma}_len", 0)
        if ma_len <= 0:
            parser.error(f"--{ma}-len must be positive")

    # Validate cutout
    if args.cutout < 0:
        parser.error("--cutout must be non-negative")

    # Validate lambda_param
    if args.lambda_param < 0:
        parser.error("--lambda-param must be non-negative")

    # Validate decay
    if args.decay < 0:
        parser.error("--decay must be non-negative")

    return args
