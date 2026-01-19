"""
CLI Main Program for Market Batch Scanner.

Interactive menu for batch scanning entire market with Gemini.
"""

import sys
import traceback
from pathlib import Path

# Add project root to sys.path
if "__file__" in globals():
    project_root = Path(__file__).parent.parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

# Ensure stdin is available on Windows before configuring stdio
# This is critical when running the file directly (not via wrapper)
if sys.platform == "win32":
    try:
        if sys.stdin is None or (hasattr(sys.stdin, "closed") and sys.stdin.closed):
            sys.stdin = open("CON", "r", encoding="utf-8", errors="replace")
    except (OSError, IOError, AttributeError):
        pass  # Continue if we can't fix stdin

# Fix encoding issues on Windows
from modules.common.utils import configure_windows_stdio

configure_windows_stdio()

import time
import warnings
from typing import List, Optional

from colorama import Fore
from colorama import init as colorama_init

from modules.common.core.exchange_manager import PublicExchangeManager
from modules.common.ui.logging import log_info
from modules.common.utils import (
    color_text,
    log_error,
    log_warn,
    normalize_timeframe,
    safe_input,
)
from modules.gemini_chart_analyzer.cli.pre_filter import pre_filter_symbols_with_hybrid, pre_filter_symbols_with_voting
from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import MarketBatchScanner

# Suppress specific noisy warnings if needed
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PIL.Image")
colorama_init(autoreset=True)


class SymbolFetchError(Exception):
    """Custom exception for symbol fetching errors."""

    def __init__(self, message: str, original_exception: Optional[Exception] = None, is_retryable: bool = False):
        super().__init__(message)
        self.original_exception = original_exception
        self.is_retryable = is_retryable


def get_all_symbols_from_exchange(
    exchange_name: str = "binance", quote_currency: str = "USDT", max_retries: int = 3, retry_delay: float = 1.0
) -> List[str]:
    """
    Get all trading symbols from exchange with retry logic for transient errors.

    Args:
        exchange_name: Exchange name to connect to (default: 'binance')
        quote_currency: Quote currency to filter symbols (default: 'USDT')
        max_retries: Maximum number of retry attempts for transient errors (default: 3)
        retry_delay: Initial delay in seconds for exponential backoff (default: 1.0)

    Returns:
        List of symbol strings (e.g., ['BTC/USDT', 'ETH/USDT', ...])
        Empty list if no symbols found (but no error occurred)

    Raises:
        SymbolFetchError: If symbol fetching fails after all retries or encounters
                       a non-retryable error. The exception includes information
                       about whether the error is retryable and the original exception.
    """
    # Protect stdin before creating PublicExchangeManager (may close stdin on Windows)
    saved_stdin = None
    if sys.platform == "win32" and sys.stdin is not None:
        try:
            saved_stdin = sys.stdin
            if hasattr(sys.stdin, "closed") and sys.stdin.closed:
                try:
                    sys.stdin = open("CON", "r", encoding="utf-8", errors="replace")
                    saved_stdin = sys.stdin
                except (OSError, IOError):
                    pass
        except (AttributeError, ValueError, OSError, IOError):
            pass

    try:
        last_exception = None
        public_exchange_manager = PublicExchangeManager()

        for attempt in range(max_retries):
            try:
                # Use public exchange manager (no credentials needed for load_markets)
                exchange = public_exchange_manager.connect_to_exchange_with_no_credentials(exchange_name)

                # Load markets
                markets = exchange.load_markets()

                # Filter by quote currency and active status
                symbols = []
                for symbol, market in markets.items():
                    if (
                        market.get("quote") == quote_currency
                        and market.get("active", True)
                        and market.get("type") == "spot"
                    ):  # Only spot markets
                        symbols.append(symbol)

                # Sort alphabetically
                symbols.sort()

                # Success - return symbols (empty list is valid if no symbols match criteria)
                return symbols

            except Exception as e:
                last_exception = e
                error_message = str(e)

                # Determine if error is retryable (network errors, rate limits, temporary unavailability)
                error_code = None
                if hasattr(e, "status_code"):
                    error_code = e.status_code
                elif hasattr(e, "code"):
                    error_code = e.code
                elif "503" in error_message or "UNAVAILABLE" in error_message.upper():
                    error_code = 503
                elif "429" in error_message or "RATE_LIMIT" in error_message.upper():
                    error_code = 429

                is_retryable = (
                    error_code in [503, 429]
                    or "overloaded" in error_message.lower()
                    or "rate limit" in error_message.lower()
                    or "unavailable" in error_message.lower()
                    or "timeout" in error_message.lower()
                    or "connection" in error_message.lower()
                    or "network" in error_message.lower()
                )

                # Log the error
                if attempt < max_retries - 1 and is_retryable:
                    wait_time = retry_delay * (2**attempt)
                    log_warn(
                        f"Retryable error getting symbols (attempt {attempt + 1}/{max_retries}): {error_message}. "
                        f"Waiting {wait_time}s before retrying..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error or final attempt failed
                    log_error(f"Error getting symbols: {error_message}")
                    if is_retryable:
                        # Final retry failed
                        raise SymbolFetchError(
                            f"Failed to fetch symbols after {max_retries} attempts: {error_message}",
                            original_exception=e,
                            is_retryable=True,
                        ) from e
                    else:
                        # Non-retryable error
                        raise SymbolFetchError(
                            f"Failed to fetch symbols (non-retryable error): {error_message}",
                            original_exception=e,
                            is_retryable=False,
                        ) from e

        # This should never be reached, but just in case
        if last_exception:
            raise SymbolFetchError(
                f"Failed to fetch symbols after {max_retries} attempts",
                original_exception=last_exception,
                is_retryable=True,
            ) from last_exception
    finally:
        # Always restore stdin after exchange operations
        if sys.platform == "win32" and saved_stdin is not None:
            try:
                if sys.stdin is None or (hasattr(sys.stdin, "closed") and sys.stdin.closed):
                    if saved_stdin is not None and not (hasattr(saved_stdin, "closed") and saved_stdin.closed):
                        sys.stdin = saved_stdin
                    else:
                        try:
                            sys.stdin = open("CON", "r", encoding="utf-8", errors="replace")
                        except (OSError, IOError):
                            pass
            except (AttributeError, ValueError, OSError, IOError):
                pass


def interactive_batch_scan():
    """Interactive menu for batch scanning."""
    print()
    print(color_text("=" * 60, Fore.CYAN))
    print(color_text("MARKET BATCH SCANNER", Fore.CYAN))
    print(color_text("=" * 60, Fore.CYAN))
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
        from modules.gemini_chart_analyzer.core.utils import DEFAULT_TIMEFRAMES, normalize_timeframes

        print(f"\nDefault timeframes: {', '.join(DEFAULT_TIMEFRAMES)}")
        print("Timeframes: 15m, 30m,1h, 4h, 1d, 1w (comma-separated)")
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
        print("\nTimeframes: 15m, 30m,1h, 4h, 1d, 1w")
        timeframe = safe_input(color_text("Enter timeframe [1h]: ", Fore.YELLOW), default="1h")
        if not timeframe:
            timeframe = "1h"

        # Normalize timeframe with exception handling
        try:
            timeframe = normalize_timeframe(timeframe)
        except Exception as e:
            log_warn(f"Error parsing timeframe: {e}, defaulting to '1h'")
            timeframe = "1h"
            try:
                timeframe = normalize_timeframe(timeframe)
            except Exception as e2:
                log_error(f"Critical error normalizing default timeframe: {e2}")
                raise

    # Get max symbols (optional)
    max_symbols_input = safe_input(color_text("Max symbols to scan (press Enter for all): ", Fore.YELLOW), default="")
    max_symbols = None
    if max_symbols_input:
        try:
            max_symbols = int(max_symbols_input)
            # Validate: must be positive integer (>=1)
            if max_symbols < 1:
                log_warn(f"max_symbols ({max_symbols}) must be >= 1, resetting to default (all symbols)")
                max_symbols = None
        except ValueError:
            log_warn("Invalid input, scanning all symbols")

    # Get cooldown
    cooldown_input = safe_input(color_text("Cooldown between batches in seconds [2.5]: ", Fore.YELLOW), default="2.5")
    cooldown = 2.5
    if cooldown_input:
        try:
            cooldown = float(cooldown_input)
            # Validate: must be non-negative (>=0.0)
            if cooldown < 0.0:
                log_warn(f"cooldown ({cooldown}) must be >= 0.0, clamping to 0.0")
                cooldown = 0.0
        except ValueError:
            log_warn("Invalid input, using default 2.5s")

    # Get candles limit
    limit_input = safe_input(color_text("Number of candles per symbol [700]: ", Fore.YELLOW), default="700")
    limit = 700
    if limit_input:
        try:
            limit = int(limit_input)
            # Validate: must be positive integer (>=1)
            if limit < 1:
                log_warn(f"limit ({limit}) must be >= 1, clamping to 1")
                limit = 1
        except ValueError:
            log_warn("Invalid input, using default 700")

    # Get pre-filter option (yes/no)
    print("\nPre-filter option:")
    print("  Filter symbols using VotingAnalyzer or HybridAnalyzer before Gemini scan")
    print("  (Selects all symbols with signals)")
    pre_filter_input = safe_input(color_text("Enable pre-filter? (y/n) [y]: ", Fore.YELLOW), default="y").lower()
    if not pre_filter_input:
        pre_filter_input = "y"
    enable_pre_filter = pre_filter_input in ["y", "yes"]

    # Initialize SPC configuration variables (used in pre-filter)
    spc_preset = None
    spc_volatility_adjustment = False
    spc_use_correlation_weights = False
    spc_time_decay_factor = None
    spc_interpolation_mode = None
    spc_min_flip_duration = None
    spc_flip_confidence_threshold = None
    spc_enable_mtf = False
    spc_mtf_timeframes = None
    spc_mtf_require_alignment = None
    spc_config_mode = "3"  # Default: skip

    # Get pre-filter mode (voting/hybrid) if pre-filter is enabled
    pre_filter_mode = "voting"  # Default
    fast_mode = True  # Default to fast mode (recommended)
    if enable_pre_filter:
        print("\nPre-filter mode:")
        print("  1. Voting (Pure voting system - all indicators vote simultaneously)")
        print("  2. Hybrid (Sequential filtering: ATC → Range Oscillator → SPC → Decision Matrix)")
        mode_input = safe_input(color_text("Select pre-filter mode (1/2) [1]: ", Fore.YELLOW), default="1")
        if not mode_input:
            mode_input = "1"
        if mode_input == "2":
            pre_filter_mode = "hybrid"
        else:
            pre_filter_mode = "voting"

        # Get fast mode option
        print("\nPre-filter speed mode:")
        print("  1. Fast (ATC + Range Osc + SPC only - Recommended)")
        print("     Pre-filter with fast indicators, then run XGBoost/HMM/RF in main Gemini scan")
        print("  2. Full (All indicators including ML models - SLOW!)")
        fast_mode_input = safe_input(color_text("Select mode (1/2) [1]: ", Fore.YELLOW), default="1")
        if not fast_mode_input:
            fast_mode_input = "1"
        fast_mode = fast_mode_input != "2"  # True if not "2", False if "2"

        # SPC Enhancements Configuration
        print("\nSPC Enhancements Configuration:")
        print("  Configure Simplified Percentile Clustering enhancements")
        print("  1. Use preset (Conservative/Balanced/Aggressive)")
        print("  2. Custom configuration")
        print("  3. Skip (use defaults from config file)")
        spc_config_mode = safe_input(color_text("Select SPC config mode (1/2/3) [3]: ", Fore.YELLOW), default="3")
        if not spc_config_mode:
            spc_config_mode = "3"

        if spc_config_mode == "1":
            # Preset mode
            print("\nSPC Preset:")
            print("  1. Conservative (Most stable - choppy markets)")
            print("  2. Balanced (Recommended - most crypto markets) ⭐")
            print("  3. Aggressive (Most responsive - trending markets)")
            preset_input = safe_input(color_text("Select preset (1/2/3) [2]: ", Fore.YELLOW), default="2")
            if not preset_input:
                preset_input = "2"
            if preset_input == "1":
                spc_preset = "conservative"
            elif preset_input == "2":
                spc_preset = "balanced"
            elif preset_input == "3":
                spc_preset = "aggressive"
        elif spc_config_mode == "2":
            # Custom configuration
            print("\nSPC Enhancement Options:")

            # Volatility adjustment
            vol_adj_input = safe_input(
                color_text("Enable volatility-adaptive percentiles? (y/n) [n]: ", Fore.YELLOW), default="n"
            ).lower()
            spc_volatility_adjustment = vol_adj_input in ["y", "yes"]

            # Correlation weights
            corr_weights_input = safe_input(
                color_text("Enable correlation-based feature weighting? (y/n) [n]: ", Fore.YELLOW), default="n"
            ).lower()
            spc_use_correlation_weights = corr_weights_input in ["y", "yes"]

            # Time decay factor
            time_decay_input = safe_input(
                color_text("Time decay factor (1.0=no decay, 0.99=light, 0.95=moderate) [1.0]: ", Fore.YELLOW),
                default="1.0",
            )
            if time_decay_input:
                try:
                    spc_time_decay_factor = float(time_decay_input)
                    if not (0.5 <= spc_time_decay_factor <= 1.0):
                        log_warn("Time decay factor must be in [0.5, 1.0], using default 1.0")
                        spc_time_decay_factor = None
                except ValueError:
                    log_warn("Invalid time decay factor, using default")
                    spc_time_decay_factor = None

            # Interpolation mode
            print("\nInterpolation mode:")
            print("  1. Linear (default)")
            print("  2. Sigmoid (smooth transitions)")
            print("  3. Exponential (sticky to current cluster)")
            interp_input = safe_input(color_text("Select interpolation mode (1/2/3) [1]: ", Fore.YELLOW), default="1")
            if not interp_input:
                interp_input = "1"
            if interp_input == "2":
                spc_interpolation_mode = "sigmoid"
            elif interp_input == "3":
                spc_interpolation_mode = "exponential"
            else:
                spc_interpolation_mode = "linear"

            # Min flip duration
            flip_dur_input = safe_input(
                color_text("Minimum bars in cluster before flip (1-10) [3]: ", Fore.YELLOW), default="3"
            )
            if flip_dur_input:
                try:
                    spc_min_flip_duration = int(flip_dur_input)
                    if not (1 <= spc_min_flip_duration <= 10):
                        log_warn("Min flip duration must be in [1, 10], using default 3")
                        spc_min_flip_duration = None
                except ValueError:
                    log_warn("Invalid min flip duration, using default")
                    spc_min_flip_duration = None

            # Flip confidence threshold
            conf_thresh_input = safe_input(
                color_text("Flip confidence threshold (0.0-1.0) [0.6]: ", Fore.YELLOW), default="0.6"
            )
            if conf_thresh_input:
                try:
                    spc_flip_confidence_threshold = float(conf_thresh_input)
                    if not (0.0 <= spc_flip_confidence_threshold <= 1.0):
                        log_warn("Confidence threshold must be in [0.0, 1.0], using default 0.6")
                        spc_flip_confidence_threshold = None
                except ValueError:
                    log_warn("Invalid confidence threshold, using default")
                    spc_flip_confidence_threshold = None

            # Multi-timeframe (optional)
            mtf_input = safe_input(
                color_text("Enable multi-timeframe analysis? (y/n) [n]: ", Fore.YELLOW), default="n"
            ).lower()
            spc_enable_mtf = mtf_input in ["y", "yes"]
            if spc_enable_mtf:
                mtf_tf_input = safe_input(
                    color_text("MTF timeframes (comma-separated, e.g., 1h,4h,1d) [1h,4h]: ", Fore.YELLOW),
                    default="1h,4h",
                )
                if mtf_tf_input:
                    spc_mtf_timeframes = [tf.strip() for tf in mtf_tf_input.split(",") if tf.strip()]
                else:
                    spc_mtf_timeframes = ["1h", "4h"]

                align_input = safe_input(
                    color_text("Require all timeframes to align? (y/n) [y]: ", Fore.YELLOW), default="y"
                ).lower()
                spc_mtf_require_alignment = align_input in ["y", "yes"]

    # Confirm BEFORE running pre-filter (to avoid stdin issues)
    # This ensures all user input is collected before any operations that might affect stdin
    print()
    print(color_text("=" * 60, Fore.CYAN))
    print(color_text("CONFIGURATION", Fore.CYAN))
    print(color_text("=" * 60, Fore.CYAN))
    if timeframes:
        print(f"Timeframes: {', '.join(timeframes)} (Multi-timeframe mode)")
    else:
        print(f"Timeframe: {timeframe} (Single timeframe mode)")
    print(f"Max symbols: {max_symbols or 'All'}")
    print(f"Cooldown: {cooldown}s")
    print(f"Candles per symbol: {limit}")
    if enable_pre_filter:
        mode_display = "Voting mode" if pre_filter_mode == "voting" else "Hybrid mode"
        speed_display = "Fast (ATC/RangeOsc/SPC only)" if fast_mode else "Full (All indicators)"
        print(f"Pre-filter: Enabled ({mode_display}, {speed_display})")

        # Display SPC configuration
        if spc_config_mode != "3":
            print("\nSPC Enhancements:")
            if spc_preset:
                print(f"  Preset: {spc_preset.capitalize()}")
            else:
                enhancements_list = []
                if spc_volatility_adjustment:
                    enhancements_list.append("Volatility Adjustment")
                if spc_use_correlation_weights:
                    enhancements_list.append("Correlation Weighting")
                if spc_time_decay_factor and spc_time_decay_factor != 1.0:
                    enhancements_list.append(f"Time Decay ({spc_time_decay_factor})")
                if spc_interpolation_mode and spc_interpolation_mode != "linear":
                    enhancements_list.append(f"Interpolation ({spc_interpolation_mode})")
                if spc_min_flip_duration:
                    enhancements_list.append(f"Min Flip Duration ({spc_min_flip_duration})")
                if spc_flip_confidence_threshold:
                    enhancements_list.append(f"Confidence Threshold ({spc_flip_confidence_threshold})")
                if spc_enable_mtf:
                    mtf_display = f"MTF ({', '.join(spc_mtf_timeframes or [])})"
                    if spc_mtf_require_alignment:
                        mtf_display += " [Require Alignment]"
                    enhancements_list.append(mtf_display)

                if enhancements_list:
                    print(f"  Custom: {', '.join(enhancements_list)}")
                else:
                    print("  Using defaults from config file")
            print()
    else:
        print("Pre-filter: Disabled")
    print()

    confirm = safe_input(color_text("Start batch scan? (y/n) [y]: ", Fore.YELLOW), default="y").lower()
    if not confirm:
        confirm = "y"

    if confirm not in ["y", "yes"]:
        log_warn("Cancelled by user")
        return

    # Pre-filter symbols BEFORE initializing scanner (if enabled)
    pre_filtered_symbols = None
    if enable_pre_filter:
        try:
            log_info("=" * 60)
            log_info("PRE-FILTERING SYMBOLS")
            log_info("=" * 60)

            log_info("Step 1: Getting all symbols from exchange...")
            all_symbols = get_all_symbols_from_exchange(exchange_name="binance", quote_currency="USDT")

            if not all_symbols:
                log_warn("No symbols found from exchange, skipping pre-filter")
                pre_filtered_symbols = None
            else:
                # Determine primary timeframe for pre-filtering
                if timeframes:
                    primary_timeframe = timeframes[0]
                elif timeframe:
                    primary_timeframe = timeframe
                else:
                    primary_timeframe = "1h"

                # Run pre-filter according to the selected mode
                if pre_filter_mode == "hybrid":
                    try:
                        pre_filtered_symbols = pre_filter_symbols_with_hybrid(
                            all_symbols=all_symbols,
                            timeframe=primary_timeframe,
                            limit=limit,
                            fast_mode=fast_mode,
                            spc_preset=spc_preset,
                            spc_volatility_adjustment=spc_volatility_adjustment,
                            spc_use_correlation_weights=spc_use_correlation_weights,
                            spc_time_decay_factor=spc_time_decay_factor,
                            spc_interpolation_mode=spc_interpolation_mode,
                            spc_min_flip_duration=spc_min_flip_duration,
                            spc_flip_confidence_threshold=spc_flip_confidence_threshold,
                            spc_enable_mtf=spc_enable_mtf,
                            spc_mtf_timeframes=spc_mtf_timeframes,
                            spc_mtf_require_alignment=spc_mtf_require_alignment,
                        )
                    except Exception as e:
                        log_error(f"Exception during hybrid pre-filtering: {e}")
                        pre_filtered_symbols = None
                else:
                    try:
                        pre_filtered_symbols = pre_filter_symbols_with_voting(
                            all_symbols=all_symbols,
                            timeframe=primary_timeframe,
                            limit=limit,
                            fast_mode=fast_mode,
                            spc_preset=spc_preset,
                            spc_volatility_adjustment=spc_volatility_adjustment,
                            spc_use_correlation_weights=spc_use_correlation_weights,
                            spc_time_decay_factor=spc_time_decay_factor,
                            spc_interpolation_mode=spc_interpolation_mode,
                            spc_min_flip_duration=spc_min_flip_duration,
                            spc_flip_confidence_threshold=spc_flip_confidence_threshold,
                            spc_enable_mtf=spc_enable_mtf,
                            spc_mtf_timeframes=spc_mtf_timeframes,
                            spc_mtf_require_alignment=spc_mtf_require_alignment,
                        )
                    except Exception as e:
                        log_error(f"Exception during voting pre-filtering: {e}")
                        pre_filtered_symbols = None

                if pre_filtered_symbols is not None:
                    if len(pre_filtered_symbols) < len(all_symbols):
                        msg = (
                            f"Pre-filtered: {len(all_symbols)} → {len(pre_filtered_symbols)} "
                            f"symbols (all symbols with signals)"
                        )
                        log_info(msg)
                    elif len(pre_filtered_symbols) == len(all_symbols):
                        log_info(f"Pre-filtered: All {len(all_symbols)} symbols have signals (no filtering applied)")
                    else:
                        log_warn(
                            "Pre-filtered symbols count is greater than all symbols. "
                            "There may be an unexpected behavior."
                        )
                else:
                    log_info(f"Pre-filtered: No symbols with signals found, using all {len(all_symbols)} symbols")
                    pre_filtered_symbols = None

                print(f"Pre-filtered symbols: {pre_filtered_symbols}")

        except SymbolFetchError as e:
            log_error(f"Failed to fetch symbols from exchange: {e}")
            if e.is_retryable:
                log_error(
                    "This was a retryable error (network/rate limit). Please check your connection and try again."
                )
            else:
                log_error("This was a non-retryable error. Please check exchange configuration and API access.")
            log_warn("Continuing without pre-filter...")
            pre_filtered_symbols = None
        except Exception as e:
            log_error(f"Error during pre-filtering: {e}")
            log_warn("Continuing without pre-filter...")
            pre_filtered_symbols = None

    # Initialize scanner AFTER pre-filter (if enabled)
    try:
        scanner = MarketBatchScanner(cooldown_seconds=cooldown)

        # Run scan with pre-filtered symbols (if available)
        results = scanner.scan_market(
            timeframe=timeframe,
            timeframes=timeframes,
            max_symbols=max_symbols,
            limit=limit,
            initial_symbols=pre_filtered_symbols,
        )

        # Display results
        print()
        print(color_text("=" * 60, Fore.GREEN))
        print(color_text("SCAN RESULTS", Fore.GREEN))
        print(color_text("=" * 60, Fore.GREEN))
        print()

        # Display LONG signals with confidence (sorted by confidence)
        long_symbols = results.get("long_symbols", [])
        print(color_text(f"LONG Signals ({len(long_symbols)}):", Fore.GREEN))
        if results.get("long_symbols_with_confidence"):
            print("  (Sorted by confidence: High → Low)")
            for symbol, confidence in results["long_symbols_with_confidence"]:
                clamped_confidence = min(max(confidence, 0.0), 1.0)
                length = int(clamped_confidence * 10)
                confidence_bar = "█" * length  # Visual bar
                print(f"  {symbol:15s} | Confidence: {confidence:.2f} {confidence_bar}")
        elif long_symbols:
            # Fallback if confidence data not available
            for i in range(0, len(long_symbols), 5):
                row = long_symbols[i : i + 5]
                print("  " + "  ".join(f"{s:12s}" for s in row))
        else:
            print("  None")

        print()
        # Display SHORT signals with confidence (sorted by confidence)
        short_symbols = results.get("short_symbols", [])
        print(color_text(f"SHORT Signals ({len(short_symbols)}):", Fore.RED))
        if results.get("short_symbols_with_confidence"):
            print("  (Sorted by confidence: High → Low)")
            for symbol, confidence in results["short_symbols_with_confidence"]:
                clamped_confidence = min(max(confidence, 0.0), 1.0)
                length = int(clamped_confidence * 10)
                confidence_bar = "█" * length  # Visual bar
                print(f"  {symbol:15s} | Confidence: {confidence:.2f} {confidence_bar}")
        elif short_symbols:
            # Fallback if confidence data not available
            for i in range(0, len(short_symbols), 5):
                row = short_symbols[i : i + 5]
                print("  " + "  ".join(f"{s:12s}" for s in row))
        else:
            print("  None")

        # Display summary with average confidence
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
        log_error(f"Error during batch scan: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    # Ensure stdin is available on Windows (critical when running file directly)
    if sys.platform == "win32":
        try:
            if sys.stdin is None or (hasattr(sys.stdin, "closed") and sys.stdin.closed):
                sys.stdin = open("CON", "r", encoding="utf-8", errors="replace")
        except (OSError, IOError, AttributeError):
            pass  # Continue if we can't fix stdin

    try:
        interactive_batch_scan()
    except KeyboardInterrupt:
        log_warn("\nExiting...")
        sys.exit(0)
    except Exception as e:
        log_error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
