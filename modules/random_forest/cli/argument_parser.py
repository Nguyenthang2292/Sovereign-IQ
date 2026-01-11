
import argparse

"""
Command-line argument parser for Random Forest model training and signal generation.

This module provides the main argument parser for the Random Forest CLI,
defining all command-line options and their default values.
"""


# Constants
DATA_PROCESSING_WAIT_TIME_IN_SECONDS = 2
DEFAULT_CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
DEFAULT_TEST_SYMBOL = "BTC/USDT"
DEFAULT_TEST_TIMEFRAME = "1h"
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
DEFAULT_TOP_SYMBOLS = 10


def parse_args():
    """Parse command-line arguments for Random Forest model training and signal generation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Crypto pair signal analysis using Random Forest")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Set logging level"
    )
    parser.add_argument(
        "--multi-timeframe", action="store_true", default=True, help="Use multiple timeframes for analysis"
    )
    parser.add_argument("--pairs", type=str, help="Comma-separated list of crypto pairs to analyze")
    parser.add_argument(
        "--top-symbols", type=int, default=0, help="Number of top symbols by volume to analyze (0 for all symbols)"
    )
    return parser.parse_args()
