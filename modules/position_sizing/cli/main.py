"""
Position Sizing Calculator - CLI Entry Point.

Calculate optimal position sizes using Bayesian Kelly Criterion and Regime Switching
based on results from main_hybrid.py or main_voting.py.

Example:
    Run from command line:
        $ python main_position_sizing.py --source hybrid --account-balance 10000

Note:
    TODO: Consider implementing localization/i18n support if multi-language
    user interface is desired. Currently all code comments and docstrings
    are in English for consistency across the codebase.
"""

import logging
import sys
import threading
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd

from modules.common.utils import configure_windows_stdio

# Fix encoding issues on Windows for interactive CLI runs only
configure_windows_stdio()

from colorama import Fore, Style
from colorama import init as colorama_init

from config.position_sizing import (
    ENABLE_MULTITHREADING,
    ENABLE_PARALLEL_PROCESSING,
    USE_GPU,
)
from modules.common.core.data_fetcher import DataFetcher
from modules.common.utils import (
    color_text,
    days_to_candles,
    initialize_components,
    log_error,
    log_progress,
    log_success,
    log_warn,
    safe_input,
)
from modules.position_sizing.cli.argument_parser import (
    interactive_config_menu,
    parse_args,
    parse_symbols_string,
)
from modules.position_sizing.cli.display import (
    display_position_sizing_results,
)
from modules.position_sizing.core.position_sizer import PositionSizer
from modules.position_sizing.utils.data_loader import (
    load_symbols_from_file,
    load_symbols_from_results,
    validate_symbols,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)

# Constants
PARALLEL_PROCESSING_THRESHOLD = 100  # Threshold for deciding between parallel and sequential processing


# Custom exception for data-related errors
class InsufficientDataError(Exception):
    """Raised when there is insufficient data to perform calculations."""

    pass


class DataError(Exception):
    """Raised when there is an error with data processing or validation."""

    pass


def _try_timeframes_auto(
    symbols: List[Dict],
    account_balance: float,
    lookback_days: int,
    data_fetcher: DataFetcher,
    max_position_size: float,
    signal_mode: str,
    signal_calculation_mode: str,
    sequential: bool = False,
) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """
    Automatically test timeframes in parallel to find one with valid position sizing results.

    Runs timeframes in parallel: 15m, 30m, 1h.
    Stops immediately when the first timeframe with valid results is found (early stopping).
    Significantly faster than sequential execution.

    Note: Cancellation is cooperative. When a valid result is found, cancellation prevents
    new timeframes from starting, but does not interrupt in-progress calculations.
    Once position_sizer.calculate_portfolio_allocation() begins, it will run to completion.

    Args:
        symbols: List of symbol dictionaries
        account_balance: Account balance in USDT
        lookback_days: Number of days to look back
        data_fetcher: DataFetcher instance (assumed thread-safe or each thread uses separate calls)
        max_position_size: Maximum position size as fraction
        signal_mode: Signal calculation mode ('majority_vote' or 'single_signal')
        signal_calculation_mode: Signal calculation approach ('precomputed' or 'incremental')

    Returns:
        Tuple of (timeframe, results_df) if found, (None, None) if no valid results found
    """
    timeframes = ["15m", "30m", "1h"]

    # Create cancellation event so tasks can check and stop early
    cancel_event = threading.Event()

    def try_timeframe(timeframe: str) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
        """Try a single timeframe and return result."""
        try:
            # Check cancellation before starting
            if cancel_event.is_set():
                log_progress(f"Skipping timeframe {timeframe} (cancelled)")
                return (None, None)

            # Check cancellation immediately before creating PositionSizer
            # This must be as close as possible to the creation to minimize race conditions
            if cancel_event.is_set():
                log_progress(f"Cancelling timeframe {timeframe} before PositionSizer creation")
                return (None, None)

            log_progress(f"Trying timeframe: {timeframe}...")

            # Create PositionSizer with this timeframe
            position_sizer = PositionSizer(
                data_fetcher=data_fetcher,
                timeframe=timeframe,
                lookback_days=lookback_days,
                max_position_size=max_position_size,
                signal_mode=signal_mode,
                signal_calculation_mode=signal_calculation_mode,
            )

            # Check cancellation before calculation
            if cancel_event.is_set():
                log_progress(f"Cancelling timeframe {timeframe} before calculation")
                return (None, None)

            # Calculate position sizing
            results_df = position_sizer.calculate_portfolio_allocation(
                symbols=symbols,
                account_balance=account_balance,
                timeframe=timeframe,
                lookback=lookback_days,
            )

            # Check cancellation after calculation
            if cancel_event.is_set():
                log_progress(f"Cancelling timeframe {timeframe} after calculation")
                return (None, None)

            # Check results
            if not results_df.empty and "position_size_usdt" in results_df.columns:
                total_position_size = results_df["position_size_usdt"].sum()
                total_exposure_pct = (
                    results_df["position_size_pct"].sum() if "position_size_pct" in results_df.columns else 0.0
                )

                if total_position_size > 0 or total_exposure_pct > 0:
                    log_success(f"✓ Found valid position sizing at timeframe: {timeframe}")
                    log_progress(f"  Total Position Size: {total_position_size:.2f} USDT")
                    log_progress(f"  Total Exposure: {total_exposure_pct:.2f}%")
                    return (timeframe, results_df)

            log_warn(f"No position sizing result at {timeframe}")
            return (None, None)

        except (ValueError, KeyError, IndexError, pd.errors.EmptyDataError, InsufficientDataError, DataError) as e:
            # Recoverable data errors - log with context at WARNING level
            log_warn(f"Data error trying timeframe {timeframe}: {type(e).__name__}: {e}")
            logging.warning(f"Full traceback for data error at timeframe {timeframe}", exc_info=True)
            return (None, None)
        except (ConnectionError, TimeoutError, OSError) as e:
            # Unrecoverable network/system errors - log and return None (let other threads continue)
            log_error(f"Network/system error trying timeframe {timeframe}: {type(e).__name__}: {e}")
            logging.exception(f"Full traceback for network/system error at timeframe {timeframe}")
            # Don't re-raise in parallel execution - let other threads continue
            return (None, None)
        except Exception as e:
            # Other unexpected errors - log with full traceback
            log_error(f"Unexpected error trying timeframe {timeframe}: {type(e).__name__}: {e}")
            logging.exception(f"Full traceback for unexpected error at timeframe {timeframe}")
            return (None, None)

    # Run in parallel with ThreadPoolExecutor
    if sequential:
        # Sequential execution for testing - try timeframes one at a time
        log_progress("\nRunning timeframes sequentially for testing...")
        for timeframe in timeframes:
            result_timeframe, result_df = try_timeframe(timeframe)
            if result_timeframe is not None and result_df is not None:
                log_progress(f"Early stopping: Found result at {result_timeframe}")
                return (result_timeframe, result_df)

        log_warn("✗ No timeframe found with valid position sizing results")
        return (None, None)

    log_progress("\nRunning timeframes in parallel for faster execution...")
    executor = ThreadPoolExecutor(max_workers=len(timeframes))
    early_exit = False  # Flag to mark early stopping
    future_to_timeframe = None  # Initialize so finally block can check
    try:
        # Submit all tasks
        future_to_timeframe = {executor.submit(try_timeframe, tf): tf for tf in timeframes}

        # Wait for results, return immediately when first valid result is found
        for future in as_completed(future_to_timeframe):
            timeframe = future_to_timeframe[future]
            try:
                result_timeframe, result_df = future.result()

                # If valid result found, cancel remaining futures and return
                if result_timeframe is not None and result_df is not None:
                    log_progress(f"Early stopping: Found result at {result_timeframe}, cancelling other timeframes...")

                    # Set cancellation event so new tasks can check and skip starting.
                    # Note: This is cooperative cancellation - it only prevents new timeframes
                    # from starting, not interrupting in-progress calculations.
                    cancel_event.set()

                    # Cancel remaining futures
                    remaining_futures = [f for f in future_to_timeframe.keys() if not f.done()]
                    cancelled_count = 0
                    for remaining_future in remaining_futures:
                        if remaining_future.cancel():
                            cancelled_count += 1

                    if cancelled_count > 0:
                        log_progress(f"Cancelled {cancelled_count} pending timeframe(s)")

                    # Mark early exit so finally block knows not to wait
                    early_exit = True

                    return (result_timeframe, result_df)
            except Exception as e:
                log_warn(f"Exception getting result from timeframe {timeframe}: {e}")
                continue

        log_warn("✗ No timeframe found with valid position sizing results")
        return (None, None)
    finally:
        # Ensure executor is shutdown even on exception for cleanup
        # Use early_exit flag to decide whether to wait for tasks to complete
        # If exception occurs before finding result, early_exit remains False so
        # shutdown(wait=True) will block until tasks complete (slows exception
        # propagation) but ensures clean shutdown
        try:
            if sys.version_info >= (3, 9):
                executor.shutdown(wait=not early_exit, cancel_futures=True)
            else:
                # For Python < 3.9, cancel remaining futures manually
                # future_to_timeframe may not be defined if exception occurs early
                if future_to_timeframe is not None:
                    for future in future_to_timeframe.keys():
                        if not future.done():
                            future.cancel()
                executor.shutdown(wait=not early_exit)
        except Exception:
            # Ignore exceptions in finally block to avoid masking original exception
            pass


def _prompt_for_balance() -> float:
    """
    Prompt user for account balance input and validate it.

    Returns:
        float: The validated account balance in USDT

    Exits:
        sys.exit(1) if input cannot be converted to float or if value is not positive
    """
    account_balance_str = safe_input("\nEnter account balance (USDT): ", default="").strip()
    try:
        balance = float(account_balance_str)
        # Validate balance is positive (inside try block to ensure balance is defined)
        if balance <= 0:
            log_error("Account balance must be a positive number. Exiting.")
            sys.exit(1)
        return balance
    except ValueError:
        log_error("Invalid account balance. Exiting.")
        sys.exit(1)
        # This return should never execute in normal operation (sys.exit should exit),
        # but prevents UnboundLocalError if sys.exit is mocked in tests
        return 0.0  # Unreachable in production, but safe fallback for testing


def load_and_validate_symbols(args) -> List[Dict]:
    """
    Load symbols from the specified source and validate them.

    Args:
        args: Parsed arguments containing symbols_file, symbols, or source

    Returns:
        List of validated symbol dictionaries

    Raises:
        SystemExit: If no symbol source is specified or no valid symbols found
    """
    log_progress("\nLoading symbols...")
    symbols = []

    if args.symbols_file:
        symbols = load_symbols_from_file(args.symbols_file)
    elif args.symbols:
        # Parse comma-separated symbols and normalize (auto-add /USDT if needed)
        symbol_list = parse_symbols_string(args.symbols)
        symbols = [{"symbol": s, "signal": 1} for s in symbol_list]
    elif args.source:
        symbols = load_symbols_from_results(args.source)
    else:
        log_error("No symbol source specified. Please provide --symbols-file, --symbols, or --source")
        sys.exit(1)

    # Validate symbols
    symbols = validate_symbols(symbols)

    if not symbols:
        log_error("No valid symbols found. Exiting.")
        sys.exit(1)

    log_success(f"Loaded {len(symbols)} symbols")
    return symbols


def render_configuration_summary(
    args,
    account_balance: float,
    signal_mode: str,
    signal_calculation_mode: str,
    auto_timeframe: bool,
    shared_memory_available: bool,
) -> None:
    """
    Display the configuration summary including all optimization features status.

    Args:
        args: Parsed arguments
        account_balance: Account balance in USDT
        signal_mode: Signal mode ('single_signal' or 'majority_vote')
        signal_calculation_mode: Signal calculation mode ('precomputed' or 'incremental')
        auto_timeframe: Whether auto timeframe mode is enabled
        shared_memory_available: Whether shared memory is available for parallel processing
    """
    # Display configuration
    print(color_text("\n" + "=" * 100, Fore.CYAN, Style.BRIGHT))
    print(color_text("POSITION SIZING CONFIGURATION", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))

    print(f"\n{color_text('Configuration:', Fore.WHITE)}")
    print(f"  Account Balance: {color_text(f'{account_balance:.2f} USDT', Fore.YELLOW)}")
    if auto_timeframe:
        print(f"  Timeframe: {color_text('Auto (15m -> 30m -> 1h)', Fore.YELLOW)}")
    else:
        print(f"  Timeframe: {color_text(args.timeframe, Fore.YELLOW)}")
    print(f"  Lookback: {color_text(f'{args.lookback_days} days', Fore.YELLOW)}")
    print(f"  Max Position Size: {color_text(f'{args.max_position_size * 100:.1f}%', Fore.YELLOW)}")
    print(f"  Signal Mode: {color_text(signal_mode, Fore.YELLOW)}")
    if signal_mode == "single_signal":
        print("    └─ Using single signal (highest confidence) approach - Use any signal with highest confidence")
    else:
        print(
            f"    └─ Using majority vote approach - Requires at least {color_text('3 indicators', Fore.CYAN)} to agree"
        )
    print(f"  Signal Calculation Mode: {color_text(signal_calculation_mode, Fore.YELLOW)}")
    if signal_calculation_mode == "incremental":
        print(
            "    └─ Incremental mode (Position-Aware Skipping): Pre-compute indicators once, extract signals per period"
        )
        print("       • Pre-computes all indicators once for entire DataFrame (10-15x faster)")
        print("       • Skips signal calculation when position open (saves 30-50% time)")
        print("       • Combines signal calculation and trade simulation in single loop")
    else:
        print("    └─ Precomputed mode: Calculate all signals first, then simulate trades (default)")
        print("       • Better for analysis and debugging")

    if args.source:
        print(f"  Source: {color_text(args.source, Fore.YELLOW)}")
    elif args.symbols_file:
        print(f"  Symbols File: {color_text(args.symbols_file, Fore.YELLOW)}")
    elif args.symbols:
        print(f"  Symbols: {color_text(args.symbols, Fore.YELLOW)}")

    # Display optimization features status
    print(f"\n{color_text('Optimization Features (Architecture 5 - Hybrid Approach):', Fore.WHITE, Style.BRIGHT)}")
    print(color_text("-" * 100, Fore.CYAN))

    # 1. Vectorized Indicator Pre-computation (Always enabled in new implementation)
    vectorized_status = color_text("✅ ENABLED", Fore.GREEN)
    print(f"  1. {color_text('Vectorized Indicator Pre-computation:', Fore.WHITE)} {vectorized_status}")
    print("     └─ Pre-compute all indicators once for entire DataFrame (10-15x faster)")

    # 2. Incremental Signal Calculation (Always enabled in new implementation)
    incremental_status = color_text("✅ ENABLED", Fore.GREEN)
    print(f"  2. {color_text('Incremental Signal Calculation:', Fore.WHITE)} {incremental_status}")
    if signal_calculation_mode == "incremental":
        print("     └─ Position-Aware Skipping: Skip signal calculation when position open")
        print("        • Pre-computes all indicators once, then extracts signals incrementally")
        print("        • Combines signal calculation and trade simulation in single loop")
        print("        • Saves 30-50% computation time for long-held positions")
        print("        • 10-15x faster when no position (uses precomputed indicators)")
    else:
        print("     └─ Calculate signals from pre-computed data using DataFrame views")

    # 3. Shared Memory for Parallel Processing
    if ENABLE_PARALLEL_PROCESSING:
        parallel_status = color_text("✅ ENABLED", Fore.GREEN)
        shared_mem_status = (
            color_text("✅ AVAILABLE", Fore.GREEN)
            if shared_memory_available
            else color_text("❌ NOT AVAILABLE", Fore.YELLOW)
        )
        print(f"  3. {color_text('Parallel Processing:', Fore.WHITE)} {parallel_status}")
        print(f"     └─ Shared Memory: {shared_mem_status}")
        if shared_memory_available:
            print("        • Using shared memory for efficient inter-process data sharing")
            print("        • Reduces memory overhead by 50-70% compared to pickle")
        else:
            print("        • Falling back to pickle serialization")
        if auto_timeframe:
            # When auto_timeframe is enabled, the timeframe is not yet determined
            # so we skip the lookback_candles display here (will be shown after timeframe is resolved)
            print(
                "        •",
                color_text("Parallel processing decision", Fore.CYAN),
                "will be made after timeframe selection",
            )
        else:
            lookback_candles = days_to_candles(args.lookback_days, args.timeframe)
            if lookback_candles > PARALLEL_PROCESSING_THRESHOLD:
                msg = color_text("Will use parallel processing", Fore.CYAN)
                print("        •", msg, f"for {lookback_candles} periods")
            else:
                msg = color_text("Using sequential processing", Fore.CYAN)
                print("        •", msg, f"(dataset size <= {PARALLEL_PROCESSING_THRESHOLD})")
    else:
        parallel_status = color_text("❌ DISABLED", Fore.YELLOW)
        print(f"  3. {color_text('Parallel Processing:', Fore.WHITE)} {parallel_status}")
        print("     └─ Using optimized sequential vectorized processing")

    # 4. Multithreading
    multithreading_status = (
        color_text("✅ ENABLED", Fore.GREEN) if ENABLE_MULTITHREADING else color_text("❌ DISABLED", Fore.YELLOW)
    )
    print(f"  4. {color_text('Multithreading:', Fore.WHITE)} {multithreading_status}")
    if ENABLE_MULTITHREADING:
        print("     └─ Parallel indicator calculation using ThreadPoolExecutor")

    # 5. GPU Acceleration
    gpu_status = color_text("✅ ENABLED", Fore.GREEN) if USE_GPU else color_text("❌ DISABLED", Fore.YELLOW)
    print(f"  5. {color_text('GPU Acceleration:', Fore.WHITE)} {gpu_status}")
    if USE_GPU:
        print("     └─ GPU acceleration for ML models (XGBoost) if available")

    print(color_text("-" * 100, Fore.CYAN))
    print(f"  {color_text('Expected Performance:', Fore.CYAN)} 10-15x faster for large datasets (>1000 periods)")
    print(f"  {color_text('Memory Usage:', Fore.CYAN)} Reduced by 50-70% with shared memory")
    print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))


def execute_position_sizing(
    args,
    symbols: List[Dict],
    account_balance: float,
    data_fetcher: DataFetcher,
    signal_mode: str,
    signal_calculation_mode: str,
    auto_timeframe: bool,
    position_sizer: Optional[PositionSizer] = None,
) -> Tuple[Optional[str], pd.DataFrame, Optional[int]]:
    """
    Execute position sizing calculation with optional auto timeframe testing.

    Args:
        args: Parsed arguments (not modified by this function)
        symbols: List of symbol dictionaries
        account_balance: Account balance in USDT
        data_fetcher: DataFetcher instance
        signal_mode: Signal mode ('single_signal' or 'majority_vote')
        signal_calculation_mode: Signal calculation mode ('precomputed' or 'incremental')
        auto_timeframe: Whether auto timeframe mode is enabled
        position_sizer: PositionSizer instance (only required if auto_timeframe is False)

    Returns:
        Tuple of (found_timeframe, results_df, final_lookback_candles)
        - found_timeframe: The timeframe used (None if auto_timeframe failed)
        - results_df: DataFrame with position sizing results
        - final_lookback_candles: Number of candles for lookback (None if not auto_timeframe)

    Raises:
        SystemExit: If auto_timeframe fails to find valid timeframe
    """
    # Calculate position sizes
    log_progress(f"\nCalculating position sizes for {len(symbols)} symbols...")
    log_progress("This may take a few minutes...\n")

    # Check if auto timeframe mode is enabled
    found_timeframe = None
    final_lookback_candles = None

    if auto_timeframe:
        log_progress("\nAuto timeframe testing enabled. Trying timeframes: 15m -> 30m -> 1h")
        log_progress("Will stop at first timeframe with valid position sizing results.\n")

        found_timeframe, results_df = _try_timeframes_auto(
            symbols=symbols,
            account_balance=account_balance,
            lookback_days=args.lookback_days,
            data_fetcher=data_fetcher,
            max_position_size=args.max_position_size,
            signal_mode=signal_mode,
            signal_calculation_mode=signal_calculation_mode,
        )

        if found_timeframe:
            log_success(f"\n✓ Found valid position sizing at timeframe: {found_timeframe}")
            # Calculate exact lookback_candles with the resolved timeframe
            final_lookback_candles = days_to_candles(args.lookback_days, found_timeframe)
        else:
            log_warn("\n✗ No timeframe found with valid position sizing results")
            log_warn("All timeframes (15m, 30m, 1h) were tried but none produced valid results")
            log_warn("This may indicate:")
            log_warn("  - Insufficient trades for Kelly calculation")
            log_warn("  - Win rate too low (< 40%)")
            log_warn("  - Invalid avg_win or avg_loss values")
            sys.exit(1)
    else:
        # Original logic: use timeframe from args
        if position_sizer is None:
            raise ValueError("position_sizer must be provided when auto_timeframe is False")
        results_df = position_sizer.calculate_portfolio_allocation(
            symbols=symbols,
            account_balance=account_balance,
            timeframe=args.timeframe,
            lookback=args.lookback_days,  # This will be converted to candles internally
        )
        found_timeframe = args.timeframe

    return found_timeframe, results_df, final_lookback_candles


def main() -> None:
    """
    Main entry point for position sizing CLI.

    Workflow:
    1. Parse arguments or show interactive menu
    2. Load symbols from source
    3. Initialize components (ExchangeManager, DataFetcher, PositionSizer)
    4. Calculate position sizes for all symbols
    5. Display results
    """
    try:
        # Parse arguments
        args = parse_args()

        # Interactive menu if not skipped
        config = {}
        if not args.no_menu:
            config = interactive_config_menu()

            # Handle case when user cancels (chooses Exit) - config will be None
            if config is None:
                log_progress("\nConfiguration cancelled by user. Exiting.")
                sys.exit(0)

            # Merge config with args - always override with menu values if menu was shown
            for key, value in config.items():
                # Always override with menu values when menu is shown (user explicitly chose these values)
                setattr(args, key, value)

        # Initialize components first (needed for fetching balance)
        log_progress("\nInitializing components...")
        exchange_manager, data_fetcher = initialize_components()

        # Get account balance
        account_balance = args.account_balance

        # Try to fetch from Binance if requested
        if getattr(args, "fetch_balance", False):
            log_progress("Fetching account balance from Binance...")
            try:
                fetched_balance = data_fetcher.fetch_binance_account_balance()
                if fetched_balance is not None and fetched_balance > 0:
                    account_balance = fetched_balance
                    log_success(f"Fetched account balance from Binance: {account_balance:.2f} USDT")
                else:
                    log_warn("Could not fetch balance from Binance. Please enter manually.")
            except Exception as e:
                log_warn(f"Error fetching balance from Binance: {e}. Please enter manually.")

        # Prompt for balance if still not set
        if account_balance is None:
            log_progress("\nAccount balance not provided. Options:")
            print("  1. Enter balance manually")
            print(
                color_text("  2. Fetch from Binance (requires API credentials) [default]", Fore.MAGENTA, Style.BRIGHT)
            )
            default_choice_text = color_text("default=2", Fore.MAGENTA)
            choice = safe_input(f"\nSelect option (1/2, {default_choice_text}): ", default="2").strip()

            # Default to option 2 if empty input
            if not choice:
                choice = "2"

            if choice == "2":
                try:
                    fetched_balance = data_fetcher.fetch_binance_account_balance()
                    if fetched_balance is not None and fetched_balance > 0:
                        account_balance = fetched_balance
                        log_success(f"Fetched account balance from Binance: {account_balance:.2f} USDT")
                    else:
                        log_warn("Could not fetch balance from Binance. Please enter manually.")
                        account_balance = _prompt_for_balance()
                except Exception as e:
                    log_warn(f"Error fetching balance from Binance: {e}. Please enter manually.")
                    account_balance = _prompt_for_balance()
            else:
                account_balance = _prompt_for_balance()

        if account_balance <= 0:
            log_error("Account balance must be positive. Exiting.")
            sys.exit(1)

        # Load and validate symbols
        symbols = load_and_validate_symbols(args)

        # Get signal mode and calculation mode from args
        signal_mode = getattr(args, "signal_mode", "single_signal")
        signal_calculation_mode = getattr(args, "signal_calculation_mode", "precomputed")

        # Check auto timeframe mode (from config menu or command line)
        auto_timeframe = getattr(args, "auto_timeframe", False)

        # Initialize Position Sizer (only if not using auto timeframe)
        position_sizer = None
        if not auto_timeframe:
            position_sizer = PositionSizer(
                data_fetcher=data_fetcher,
                timeframe=args.timeframe,
                lookback_days=args.lookback_days,
                max_position_size=args.max_position_size,
                signal_mode=signal_mode,
                signal_calculation_mode=signal_calculation_mode,
            )

        # Check shared memory availability for configuration display
        try:
            from modules.backtester.core.shared_memory_utils import SHARED_MEMORY_AVAILABLE

            shared_memory_available = SHARED_MEMORY_AVAILABLE
        except ImportError:
            shared_memory_available = False

        # Display configuration summary
        render_configuration_summary(
            args=args,
            account_balance=account_balance,
            signal_mode=signal_mode,
            signal_calculation_mode=signal_calculation_mode,
            auto_timeframe=auto_timeframe,
            shared_memory_available=shared_memory_available,
        )

        # Execute position sizing
        found_timeframe, results_df, final_lookback_candles = execute_position_sizing(
            args=args,
            symbols=symbols,
            account_balance=account_balance,
            data_fetcher=data_fetcher,
            signal_mode=signal_mode,
            signal_calculation_mode=signal_calculation_mode,
            auto_timeframe=auto_timeframe,
            position_sizer=position_sizer,
        )

        # Display final configuration summary (after finding timeframe if auto mode)
        if auto_timeframe and found_timeframe and final_lookback_candles is not None:
            print(color_text("\n" + "=" * 100, Fore.CYAN, Style.BRIGHT))
            print(color_text("FINAL CONFIGURATION", Fore.CYAN, Style.BRIGHT))
            print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))
            print(f"  Selected Timeframe: {color_text(found_timeframe, Fore.GREEN, Style.BRIGHT)}")
            print(f"  Lookback Candles: {color_text(f'{final_lookback_candles}', Fore.GREEN, Style.BRIGHT)} periods")
            if ENABLE_PARALLEL_PROCESSING:
                if final_lookback_candles > PARALLEL_PROCESSING_THRESHOLD:
                    print(
                        f"  Parallel Processing: {color_text('Was used', Fore.GREEN)} "
                        f"({final_lookback_candles} > {PARALLEL_PROCESSING_THRESHOLD} — parallel)"
                    )
                else:
                    print(
                        f"  Parallel Processing: {color_text('Processed sequentially', Fore.YELLOW)} "
                        f"(<= {PARALLEL_PROCESSING_THRESHOLD} — sequential)"
                    )
            print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))

        # Check if results_df is empty before displaying or saving
        if results_df.empty:
            log_warn("\nNo position sizing results found")
            log_warn("The calculation completed but produced no results.")
            log_warn("This may indicate:")
            log_warn("  - No valid signals were generated for the symbols")
            log_warn("  - All position sizes were filtered out")
            log_warn("  - Insufficient data for calculations")
            return

        # Display results
        display_position_sizing_results(results_df)

        # Save results if output file specified
        if args.output:
            results_df.to_csv(args.output, index=False)
            log_success(f"\nResults saved to {args.output}")

        log_success("\nPosition sizing calculation completed!")

    except KeyboardInterrupt:
        print(color_text("\n\nExiting by user request.", Fore.YELLOW))
        sys.exit(0)
    except Exception as e:
        log_error(f"Error: {type(e).__name__}: {e}")
        log_error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
