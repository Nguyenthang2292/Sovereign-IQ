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
from datetime import datetime

from colorama import Fore
from colorama import init as colorama_init

from config.forex_pairs import FOREX_MAJOR_PAIRS, FOREX_MINOR_PAIRS
from modules.adaptive_trend_LTS.utils.rust_build_checker import check_rust_backend
from modules.common.core.exchange_manager import PublicExchangeManager
from modules.common.ui.logging import log_error, log_info, log_success, log_warn
from modules.common.utils import (
    NavigationBack,
    color_text,
    configure_windows_stdio,
    get_error_code,
    is_retryable_error,
    normalize_timeframe,
    safe_input,
    setup_windows_stdin,
)
from modules.gemini_chart_analyzer.cli.config.display import display_loaded_configuration
from modules.gemini_chart_analyzer.cli.config.exporter import export_configuration
from modules.gemini_chart_analyzer.cli.config.loader import (
    list_configuration_files,
    load_configuration_from_file,
)
from modules.gemini_chart_analyzer.cli.prompts.timeframe import (
    prompt_market_coverage,
)
from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner_forex import ForexMarketBatchScanner
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
    Main entry point for Forex Batch Scanner with Backspace support.
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

        # Check Rust status
        rust_status = check_rust_backend()
        if not rust_status["available"]:
            print(f"\n{'=' * 60}")
            print("⚠️  PERFORMANCE WARNING: Rust backend not found.")
            print(f"To build: {rust_status['build_command']}")
            print(f"{'=' * 60}\n")
        else:
            print(f"\n{'=' * 60}")
            print("✅ Rust backend is ACTIVE (Optimal performance)")
            print(f"{'=' * 60}\n")

        # Configuration state
        config = {
            "mode": "2",
            "timeframe": "1h",
            "timeframes": DEFAULT_TIMEFRAMES,
            "max_symbols": None,
            "stage0_sample_percentage": None,
            "cooldown": 2.5,
            "limit": 500,
        }

        steps = ["load_config_prompt", "mode", "timeframes", "market_coverage", "cooldown", "limit", "export_config"]
        current_step = 0
        config_files = list_configuration_files()

        print(color_text("\n(Tip: Press Backspace or type 'b' at any prompt to go back)\n", Fore.CYAN))

        while current_step < len(steps):
            step_name = steps[current_step]
            try:
                if step_name == "load_config_prompt":
                    print("\nLoad Configuration:")
                    if config_files:
                        print(f"  {len(config_files)} configuration file(s) found in project root")
                    else:
                        print("  No configuration files found in project root")

                    load_config_input = safe_input(
                        color_text("Load configuration from file? (y/n) [n]: ", Fore.YELLOW),
                        default="n",
                        allow_back=False,
                    ).lower()

                    if load_config_input in ["y", "yes"]:
                        if config_files:
                            print("\nAvailable configuration files:")
                            for idx, config_file in enumerate(config_files[:10], 1):
                                try:
                                    mtime = datetime.fromtimestamp(config_file.stat().st_mtime)
                                    mtime_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
                                    print(f"  {idx}. {config_file.name} ({mtime_str})")
                                except (OSError, ValueError):
                                    print(f"  {idx}. {config_file.name}")

                        file_choice = safe_input(
                            color_text("\nSelect file number or enter full path (press Enter to skip): ", Fore.YELLOW),
                            default="",
                            allow_back=True,
                        ).strip()

                        if file_choice:
                            config_path = None
                            try:
                                file_idx = int(file_choice)
                                if config_files and 1 <= file_idx <= len(config_files):
                                    config_path = config_files[file_idx - 1]
                            except ValueError:
                                config_path = Path(file_choice)

                            if config_path:
                                loaded_data = load_configuration_from_file(config_path)
                                if loaded_data:
                                    display_loaded_configuration(loaded_data)
                                    print("\nConfiguration Options:")
                                    print("  1. Use loaded configuration as-is")
                                    print("  2. Use as defaults and adjust")
                                    print("  3. Start fresh (ignore loaded config)")
                                    use_choice = safe_input(
                                        color_text("Select option (1/2/3) [2]: ", Fore.YELLOW),
                                        default="2",
                                        allow_back=True,
                                    )
                                    if use_choice == "1":
                                        # Map loaded data to config
                                        config["mode"] = loaded_data.get("analysis_mode_id", "2")
                                        config["timeframe"] = loaded_data.get("timeframe", "1h")
                                        config["timeframes"] = loaded_data.get("timeframes", DEFAULT_TIMEFRAMES)
                                        config["max_symbols"] = loaded_data.get("max_symbols")
                                        config["stage0_sample_percentage"] = loaded_data.get("stage0_sample_percentage")
                                        config["cooldown"] = loaded_data.get("cooldown", 2.5)
                                        config["limit"] = loaded_data.get("limit", 500)
                                        current_step = steps.index("export_config")
                                        continue
                                    elif use_choice == "2":
                                        # Use as defaults
                                        config["mode"] = loaded_data.get("analysis_mode_id", "2")
                                        config["timeframe"] = loaded_data.get("timeframe", "1h")
                                        config["timeframes"] = loaded_data.get("timeframes", DEFAULT_TIMEFRAMES)
                                        config["max_symbols"] = loaded_data.get("max_symbols")
                                        config["stage0_sample_percentage"] = loaded_data.get("stage0_sample_percentage")
                                        config["cooldown"] = loaded_data.get("cooldown", 2.5)
                                        config["limit"] = loaded_data.get("limit", 500)
                    current_step += 1

                elif step_name == "mode":
                    print("\nAnalysis mode:")
                    print("  1. Single timeframe")
                    print("  2. Multi-timeframe (recommended)")
                    config["mode"] = safe_input(
                        color_text("Select mode (1/2) [2]: ", Fore.YELLOW), default="2", allow_back=False
                    )
                    current_step += 1

                elif step_name == "timeframes":
                    if config["mode"] == "2":
                        print(f"\nDefault timeframes: {', '.join(DEFAULT_TIMEFRAMES)}")
                        print("Timeframes: 15m, 30m, 1h, 4h, 1d, 1w (comma-separated)")
                        tfs_input = safe_input(
                            color_text(
                                f"Enter timeframes (comma-separated) [{', '.join(DEFAULT_TIMEFRAMES)}]: ", Fore.YELLOW
                            ),
                            default="",
                            allow_back=True,
                        )
                        if tfs_input:
                            try:
                                tfs_list = [tf.strip() for tf in tfs_input.split(",") if tf.strip()]
                                config["timeframes"] = normalize_timeframes(tfs_list)
                            except Exception:
                                log_warn("Invalid timeframes format, using default")
                        current_step += 1
                    else:
                        print("\nTimeframes: 15m, 30m, 1h, 4h, 1d, 1w")
                        tf = safe_input(
                            color_text("Enter timeframe [1h]: ", Fore.YELLOW), default="1h", allow_back=True
                        )
                        try:
                            config["timeframe"] = normalize_timeframe(tf)
                        except Exception:
                            config["timeframe"] = "1h"
                        current_step += 1

                elif step_name == "market_coverage":
                    config["max_symbols"], config["stage0_sample_percentage"] = prompt_market_coverage(
                        loaded_config={
                            "max_symbols": config["max_symbols"],
                            "stage0_sample_percentage": config["stage0_sample_percentage"],
                        }
                    )
                    current_step += 1

                elif step_name == "cooldown":
                    cd = safe_input(
                        color_text("Cooldown between batches in seconds [2.5]: ", Fore.YELLOW),
                        default="2.5",
                        allow_back=True,
                    )
                    try:
                        config["cooldown"] = float(cd)
                    except ValueError:
                        config["cooldown"] = 2.5
                    current_step += 1

                elif step_name == "limit":
                    lim = safe_input(
                        color_text("Number of candles per symbol [500]: ", Fore.YELLOW),
                        default=str(config["limit"]),
                        allow_back=True,
                    )
                    try:
                        config["limit"] = int(lim)
                    except ValueError:
                        config["limit"] = 500
                    current_step += 1

                elif step_name == "export_config":
                    print("\nExport Configuration:")
                    export_config_input = safe_input(
                        color_text("Export configuration to file? (y/n) [n]: ", Fore.YELLOW),
                        default="n",
                        allow_back=True,
                    ).lower()
                    if export_config_input in ["y", "yes"]:
                        print("\nExport Format:")
                        print("  1. YAML (Recommended)")
                        print("  2. JSON")
                        format_choice = safe_input(
                            color_text("Select format (1/2) [1]: ", Fore.YELLOW), default="1", allow_back=True
                        )
                        export_format = "json" if format_choice == "2" else "yaml"

                        export_data = {
                            "analysis_mode_id": config["mode"],
                            "timeframe": config["timeframe"],
                            "timeframes": config["timeframes"],
                            "max_symbols": config["max_symbols"],
                            "stage0_sample_percentage": config["stage0_sample_percentage"],
                            "cooldown": config["cooldown"],
                            "limit": config["limit"],
                            "export_timestamp": datetime.now().isoformat(),
                        }
                        export_configuration(export_data, format=export_format)
                    current_step += 1

            except NavigationBack:
                if current_step > 0:
                    current_step -= 1
                    print(color_text(f"\n[Go back to: {steps[current_step]}]", Fore.CYAN))
                else:
                    print(color_text("\nAlready at first step.", Fore.YELLOW))

        # Common code to run scan follows
        timeframe = config["timeframe"] if config["mode"] == "1" else None
        timeframes = config["timeframes"] if config["mode"] == "2" else None
        max_symbols = config["max_symbols"]
        stage0_sample_percentage = config["stage0_sample_percentage"]
        cooldown = config["cooldown"]
        limit = config["limit"]

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
                stage0_sample_percentage=stage0_sample_percentage,
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
