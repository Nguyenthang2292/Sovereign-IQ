"""
Main entry point for Forex Market Batch Scanner.

Run batch forex market scanning with Google Gemini AI.
Scans 7 major pairs and common minor pairs (cross pairs without USD).

Note: OANDA exchange is not supported by ccxt library, so we use a hardcoded
list of major and minor forex pairs. The exchange (binance, kraken, etc.) is
only used for fetching price data, not for getting the symbol list.
"""

import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
if "__file__" in globals():
    project_root = Path(__file__).parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


import time
import traceback

from colorama import Fore
from colorama import init as colorama_init

from config.forex_pairs import FOREX_MAJOR_PAIRS, FOREX_MINOR_PAIRS
from modules.common.core.exchange_manager import PublicExchangeManager
from modules.common.ui.logging import log_error, log_info, log_success, log_warn
from modules.common.utils import (
    color_text,
    configure_windows_stdio,
    get_error_code,
    is_retryable_error,
    normalize_timeframe,
    safe_input,
    setup_windows_stdin,
)
from modules.gemini_chart_analyzer.core.scanners.forex_market_batch_scanner import ForexMarketBatchScanner
from modules.gemini_chart_analyzer.core.utils import DEFAULT_TIMEFRAMES, normalize_timeframes

setup_windows_stdin()
configure_windows_stdio()


def get_forex_symbols(
    exchange_name: Optional[str] = None, max_retries: int = 3, retry_delay: float = 1.0, use_exchange: bool = False
) -> List[str]:
    """
    Get forex symbols: 7 major pairs + minor pairs.

    Args:
        exchange_name: Exchange name to fetch minor pairs from (optional)
                      If None or exchange not supported, uses hardcoded list
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay in seconds for exponential backoff (default: 1.0)
        use_exchange: If True, attempt to fetch from exchange; if False, use hardcoded list

    Returns:
        List of forex symbol strings (e.g., ['EUR/USD', 'GBP/USD', 'EUR/GBP', ...])
        Starts with 7 major pairs, followed by minor pairs (cross pairs without USD)
    """
    # Start with major pairs
    symbols = FOREX_MAJOR_PAIRS.copy()

    # If not using exchange, return hardcoded list
    if not use_exchange or exchange_name is None:
        minor_pairs = FOREX_MINOR_PAIRS.copy()
        all_symbols = symbols + minor_pairs
        log_info(
            f"Using hardcoded forex pairs: {len(FOREX_MAJOR_PAIRS)} major + "
            f"{len(minor_pairs)} minor = {len(all_symbols)} total"
        )
        return all_symbols

    # Try to fetch from exchange

    for attempt in range(max_retries):
        try:
            # Connect to exchange
            public_exchange_manager = PublicExchangeManager()
            exchange = public_exchange_manager.connect_to_exchange_with_no_credentials(exchange_name)

            log_info(f"Loading markets from {exchange_name}...")
            markets = exchange.load_markets()

            # Filter for minor pairs (cross pairs without USD)
            minor_pairs = []
            major_pairs_set = set(FOREX_MAJOR_PAIRS)

            for symbol, market in markets.items():
                # Only active spot markets
                if not market.get("active", True):
                    continue
                if market.get("type") != "spot":
                    continue

                # Skip if already in major pairs
                if symbol in major_pairs_set:
                    continue

                # Check if it's a minor pair (cross pair without USD)
                # Minor pairs are pairs that don't contain USD
                base = market.get("base", "")
                quote = market.get("quote", "")

                if "USD" not in base and "USD" not in quote:
                    minor_pairs.append(symbol)

            # Sort minor pairs alphabetically
            minor_pairs.sort()

            # Combine: major pairs first, then minor pairs
            all_symbols = symbols + minor_pairs

            log_success(
                f"Found {len(FOREX_MAJOR_PAIRS)} major pairs and {len(minor_pairs)} minor pairs from {exchange_name}"
            )
            log_info(f"Total forex symbols: {len(all_symbols)}")

            return all_symbols
        except Exception as e:
            error_message = str(e)

            # Determine if error is retryable
            is_retryable = is_retryable_error(e)
            error_code = get_error_code(e)

            # Network errors, rate limits, timeouts are retryable
            if error_code in [429, 500, 502, 503, 504]:
                is_retryable = True
            elif "timeout" in error_message.lower() or "network" in error_message.lower():
                is_retryable = True
            elif "rate limit" in error_message.lower():
                is_retryable = True

            # Log the error
            if attempt < max_retries - 1 and is_retryable:
                wait_time = retry_delay * (2**attempt)
                log_warn(
                    f"Retryable error getting forex symbols (attempt {attempt + 1}/{max_retries}): "
                    f"{error_message}. Waiting {wait_time}s before retrying..."
                )
                time.sleep(wait_time)
                continue
            else:
                # Non-retryable error or final attempt failed
                log_warn(f"Error getting forex symbols from exchange: {error_message}")
                break

    # Fallback to hardcoded list
    log_warn("Falling back to hardcoded forex pairs list...")
    minor_pairs = FOREX_MINOR_PAIRS.copy()
    all_symbols = symbols + minor_pairs
    log_info(
        f"Using hardcoded forex pairs: {len(FOREX_MAJOR_PAIRS)} major + {len(minor_pairs)} "
        f"minor = {len(all_symbols)} total"
    )
    return all_symbols


def main_forex():
    """
    Main entry point for Forex Batch Scanner.

    Interactive menu for batch scanning forex market with Gemini.
    """
    colorama_init(autoreset=True)

    try:
        print()
        print(color_text("=" * 60, Fore.CYAN))
        print(color_text("FOREX MARKET BATCH SCANNER", Fore.CYAN))
        print(color_text("=" * 60, Fore.CYAN))
        print()
        log_info("Forex pairs: 7 Major Pairs + Common Minor Pairs")
        log_info("Data source: TradingView scraper (OANDA:<SYMBOL_NAME>)")
        print()

        # Get timeframe(s) - single or multi
        print("\nAnalysis mode:")
        print("  1. Single timeframe")
        print("  2. Multi-timeframe (recommended)")
        mode = safe_input(color_text("Select mode (1/2) [2]: ", Fore.YELLOW), default="2")
        if not mode:
            mode = "2"

        timeframe = None
        timeframes = None

        if mode == "2":
            # Multi-timeframe mode
            print(f"\nDefault timeframes: {', '.join(DEFAULT_TIMEFRAMES)}")
            print("Timeframes: 15m, 30m, 1h, 4h, 1d, 1w (comma-separated)")
            timeframes_input = safe_input(
                color_text(f"Enter timeframes (comma-separated) [{', '.join(DEFAULT_TIMEFRAMES)}]: ", Fore.YELLOW),
                default="",
            )
            if not timeframes_input:
                timeframes = DEFAULT_TIMEFRAMES
            else:
                try:
                    timeframes_list = [tf.strip() for tf in timeframes_input.split(",") if tf.strip()]
                    timeframes = normalize_timeframes(timeframes_list)
                    if not timeframes:
                        log_warn("No valid timeframes, using default")
                        timeframes = DEFAULT_TIMEFRAMES
                except Exception as e:
                    log_warn(f"Error parsing timeframes: {e}, using default")
                    timeframes = DEFAULT_TIMEFRAMES
        else:
            # Single timeframe mode
            print("\nTimeframes: 15m, 30m, 1h, 4h, 1d, 1w")
            timeframe = safe_input(color_text("Enter timeframe [1h]: ", Fore.YELLOW), default="1h")
            if not timeframe:
                timeframe = "1h"

            # Normalize timeframe
            try:
                timeframe = normalize_timeframe(timeframe)
            except Exception as e:
                log_warn(f"Error parsing timeframe: {e}, using default '1h'")
                timeframe = "1h"

        # Get max symbols (optional)
        max_symbols_input = safe_input(
            color_text("Max symbols to scan (press Enter for all): ", Fore.YELLOW), default=""
        )
        max_symbols = None
        if max_symbols_input:
            try:
                max_symbols = int(max_symbols_input)
                if max_symbols < 1:
                    log_warn(f"max_symbols ({max_symbols}) must be >= 1, resetting to default (all symbols)")
                    max_symbols = None
            except ValueError:
                log_warn("Invalid input, scanning all symbols")

        # Get cooldown
        cooldown_input = safe_input(
            color_text("Cooldown between batches in seconds [2.5]: ", Fore.YELLOW), default="2.5"
        )
        cooldown = 2.5
        if cooldown_input:
            try:
                cooldown = float(cooldown_input)
                if cooldown < 0.0:
                    log_warn(f"cooldown ({cooldown}) must be >= 0.0, clamping to 0.0")
                    cooldown = 0.0
            except ValueError:
                log_warn("Invalid input, using default 2.5s")

        # Get candles limit
        limit_input = safe_input(color_text("Number of candles per symbol [500]: ", Fore.YELLOW), default="500")
        limit = 500
        if limit_input:
            try:
                limit = int(limit_input)
                if limit < 1:
                    log_warn(f"limit ({limit}) must be >= 1, clamping to 1")
                    limit = 1
            except ValueError:
                log_warn("Invalid input, using default 500")

        # Get forex symbols
        log_info("=" * 60)
        log_info("GETTING FOREX SYMBOLS")
        log_info("=" * 60)

        try:
            # Try to get from exchange, fallback to hardcoded list
            # Note: OANDA is not supported by ccxt, so we use hardcoded list by default
            forex_symbols = get_forex_symbols(exchange_name=None, use_exchange=False)

            if not forex_symbols:
                log_error("No forex symbols found")
                sys.exit(1)

            log_success(f"Found {len(forex_symbols)} forex symbols")
            log_info(f"Major pairs: {len(FOREX_MAJOR_PAIRS)}")
            log_info(f"Minor pairs: {len(forex_symbols) - len(FOREX_MAJOR_PAIRS)}")

        except Exception as e:
            log_error(f"Failed to get forex symbols: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Initialize forex scanner using TradingView scraper
        # Note: Uses TradingView scraper with format OANDA:<SYMBOL_NAME>
        # No exchange selection needed as we use TradingView for forex data
        try:
            scanner = ForexMarketBatchScanner(cooldown_seconds=cooldown)

            # Run scan with forex symbols
            results = scanner.scan_market(
                timeframe=timeframe,
                timeframes=timeframes,
                max_symbols=max_symbols,
                limit=limit,
                initial_symbols=forex_symbols,
                skip_cleanup=False,
            )

            # Display results
            print()
            print(color_text("=" * 60, Fore.GREEN))
            print(color_text("SCAN RESULTS", Fore.GREEN))
            print(color_text("=" * 60, Fore.GREEN))
            print()

            # Display LONG signals with confidence
            long_symbols = results.get("long_symbols", [])
            print(color_text(f"LONG Signals ({len(long_symbols)}):", Fore.GREEN))
            if results.get("long_symbols_with_confidence"):
                print("  (Sorted by confidence: High → Low)")
                for symbol, confidence in results["long_symbols_with_confidence"]:
                    clamped_confidence = min(max(confidence, 0.0), 1.0)
                    length = int(clamped_confidence * 10)
                    confidence_bar = "█" * length
                    print(f"  {symbol:15s} | Confidence: {confidence:.2f} {confidence_bar}")
            elif long_symbols:
                for i in range(0, len(long_symbols), 5):
                    row = long_symbols[i : i + 5]
                    print("  " + "  ".join(f"{s:12s}" for s in row))
            else:
                print("  None")

            print()
            # Display SHORT signals with confidence
            short_symbols = results.get("short_symbols", [])
            print(color_text(f"SHORT Signals ({len(short_symbols)}):", Fore.RED))
            if results.get("short_symbols_with_confidence"):
                print("  (Sorted by confidence: High → Low)")
                for symbol, confidence in results["short_symbols_with_confidence"]:
                    clamped_confidence = min(max(confidence, 0.0), 1.0)
                    length = int(clamped_confidence * 10)
                    confidence_bar = "█" * length
                    print(f"  {symbol:15s} | Confidence: {confidence:.2f} {confidence_bar}")
            elif short_symbols:
                for i in range(0, len(short_symbols), 5):
                    row = short_symbols[i : i + 5]
                    print("  " + "  ".join(f"{s:12s}" for s in row))
            else:
                print("  None")

            # Display summary
            if results.get("summary"):
                summary = results["summary"]
                if summary.get("avg_long_confidence", 0) > 0:
                    print()
                    print(color_text("Summary:", Fore.CYAN))
                    print(f"  Average LONG confidence: {summary['avg_long_confidence']:.2f}")
                if summary.get("avg_short_confidence", 0) > 0:
                    print(f"  Average SHORT confidence: {summary['avg_short_confidence']:.2f}")

            print()
            print(color_text("=" * 60, Fore.GREEN))
            results_file = results.get("results_file", "N/A")
            print(color_text(f"Results saved to: {results_file}", Fore.GREEN))
            print(color_text("=" * 60, Fore.GREEN))

        except Exception as e:
            log_error(f"Error during forex batch scan: {e}")
            traceback.print_exc()
            sys.exit(1)

    except KeyboardInterrupt:
        log_warn("\nExiting...")
        sys.exit(0)
    except Exception as e:
        log_error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main_forex()
