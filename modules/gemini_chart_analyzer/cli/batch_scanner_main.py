"""
CLI Main Program for Market Batch Scanner.

Interactive menu for batch scanning entire market with Gemini.
"""

import sys
from pathlib import Path
import traceback

# Add project root to sys.path
if '__file__' in globals():
    project_root = Path(__file__).parent.parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

# Ensure stdin is available on Windows before configuring stdio
# This is critical when running the file directly (not via wrapper)
if sys.platform == 'win32':
    try:
        if sys.stdin is None or (hasattr(sys.stdin, 'closed') and sys.stdin.closed):
            sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
    except (OSError, IOError, AttributeError):
        pass  # Continue if we can't fix stdin

# Fix encoding issues on Windows
from modules.common.utils import configure_windows_stdio
configure_windows_stdio()

from colorama import Fore, init as colorama_init
from modules.common.utils import color_text, log_error, log_warn
from modules.common.utils import normalize_timeframe
from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import MarketBatchScanner
from modules.common.core.exchange_manager import PublicExchangeManager
from modules.common.ui.logging import log_info, log_success
import time
from typing import List, Optional
from modules.gemini_chart_analyzer.cli.pre_filter import pre_filter_symbols_with_voting, pre_filter_symbols_with_hybrid

import warnings
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
    exchange_name: str = 'binance',
    quote_currency: str = 'USDT',
    max_retries: int = 3,
    retry_delay: float = 1.0
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
    if sys.platform == 'win32' and sys.stdin is not None:
        try:
            saved_stdin = sys.stdin
            if hasattr(sys.stdin, 'closed') and sys.stdin.closed:
                try:
                    sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
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
                    if (market.get('quote') == quote_currency and 
                        market.get('active', True) and
                        market.get('type') == 'spot'):  # Only spot markets
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
                if hasattr(e, 'status_code'):
                    error_code = e.status_code
                elif hasattr(e, 'code'):
                    error_code = e.code
                elif '503' in error_message or 'UNAVAILABLE' in error_message.upper():
                    error_code = 503
                elif '429' in error_message or 'RATE_LIMIT' in error_message.upper():
                    error_code = 429
                
                is_retryable = (
                    error_code in [503, 429] or
                    'overloaded' in error_message.lower() or
                    'rate limit' in error_message.lower() or
                    'unavailable' in error_message.lower() or
                    'timeout' in error_message.lower() or
                    'connection' in error_message.lower() or
                    'network' in error_message.lower()
                )
                
                # Log the error
                if attempt < max_retries - 1 and is_retryable:
                    wait_time = retry_delay * (2 ** attempt)
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
                            is_retryable=True
                        ) from e
                    else:
                        # Non-retryable error
                        raise SymbolFetchError(
                            f"Failed to fetch symbols (non-retryable error): {error_message}",
                            original_exception=e,
                            is_retryable=False
                        ) from e
        
        # This should never be reached, but just in case
        if last_exception:
            raise SymbolFetchError(
                f"Failed to fetch symbols after {max_retries} attempts",
                original_exception=last_exception,
                is_retryable=True
            ) from last_exception
    finally:
        # Always restore stdin after exchange operations
        if sys.platform == 'win32' and saved_stdin is not None:
            try:
                if sys.stdin is None or (hasattr(sys.stdin, 'closed') and sys.stdin.closed):
                    if saved_stdin is not None and not (hasattr(saved_stdin, 'closed') and saved_stdin.closed):
                        sys.stdin = saved_stdin
                    else:
                        try:
                            sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
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
    mode = input(color_text("Select mode (1/2) [2]: ", Fore.YELLOW)).strip()
    if not mode:
        mode = '2'
    
    timeframe = None
    timeframes = None
    
    if mode == '2':
        # Multi-timeframe mode
        from modules.gemini_chart_analyzer.core.utils import DEFAULT_TIMEFRAMES, normalize_timeframes
        
        print(f"\nDefault timeframes: {', '.join(DEFAULT_TIMEFRAMES)}")
        print("Timeframes: 15m, 30m, 1h, 4h, 1d, 1w (comma-separated)")
        timeframes_input = input(color_text(f"Enter timeframes (comma-separated) [{', '.join(DEFAULT_TIMEFRAMES)}]: ", Fore.YELLOW)).strip()
        if not timeframes_input:
            timeframes = DEFAULT_TIMEFRAMES
        else:
            try:
                timeframes_list = [tf.strip() for tf in timeframes_input.split(',') if tf.strip()]
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
        timeframe = input(color_text("Enter timeframe [1h]: ", Fore.YELLOW)).strip()
        if not timeframe:
            timeframe = '1h'
        
        # Normalize timeframe with exception handling
        try:
            timeframe = normalize_timeframe(timeframe)
        except Exception as e:
            log_warn(f"Error parsing timeframe: {e}, defaulting to '1h'")
            timeframe = '1h'
            try:
                timeframe = normalize_timeframe(timeframe)
            except Exception as e2:
                log_error(f"Critical error normalizing default timeframe: {e2}")
                raise
    
    # Get max symbols (optional)
    max_symbols_input = input(color_text("Max symbols to scan (press Enter for all): ", Fore.YELLOW)).strip()
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
    cooldown_input = input(color_text("Cooldown between batches in seconds [2.5]: ", Fore.YELLOW)).strip()
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
    limit_input = input(color_text("Number of candles per symbol [500]: ", Fore.YELLOW)).strip()
    limit = 500
    if limit_input:
        try:
            limit = int(limit_input)
            # Validate: must be positive integer (>=1)
            if limit < 1:
                log_warn(f"limit ({limit}) must be >= 1, clamping to 1")
                limit = 1
        except ValueError:
            log_warn("Invalid input, using default 500")
    
    # Get pre-filter option (yes/no)
    print("\nPre-filter option:")
    print("  Filter symbols using VotingAnalyzer or HybridAnalyzer before Gemini scan")
    print("  (Selects all symbols with signals)")
    pre_filter_input = input(color_text("Enable pre-filter? (y/n) [n]: ", Fore.YELLOW)).strip().lower()
    if not pre_filter_input:
        pre_filter_input = 'n'
    enable_pre_filter = pre_filter_input in ['y', 'yes']
    
    # Get pre-filter mode (voting/hybrid) if pre-filter is enabled
    pre_filter_mode = 'voting'  # Default
    if enable_pre_filter:
        print("\nPre-filter mode:")
        print("  1. Voting (Pure voting system - all indicators vote simultaneously)")
        print("  2. Hybrid (Sequential filtering: ATC → Range Oscillator → SPC → Decision Matrix)")
        mode_input = input(color_text("Select pre-filter mode (1/2) [1]: ", Fore.YELLOW)).strip()
        if not mode_input:
            mode_input = '1'
        if mode_input == '2':
            pre_filter_mode = 'hybrid'
        else:
            pre_filter_mode = 'voting'
    
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
        mode_display = "Voting mode" if pre_filter_mode == 'voting' else "Hybrid mode"
        print(f"Pre-filter: Enabled ({mode_display})")
    else:
        print(f"Pre-filter: Disabled")
    print()
    
    confirm = input(color_text("Start batch scan? (y/n) [y]: ", Fore.YELLOW)).strip().lower()
    if not confirm:
        confirm = 'y'
    
    if confirm not in ['y', 'yes']:
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
            all_symbols = get_all_symbols_from_exchange(
                exchange_name='binance',
                quote_currency='USDT'
            )

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
                    primary_timeframe = '1h'

                # Run pre-filter according to the selected mode
                if pre_filter_mode == 'hybrid':
                    try:
                        pre_filtered_symbols = pre_filter_symbols_with_hybrid(
                            all_symbols=all_symbols,
                            timeframe=primary_timeframe,
                            limit=limit
                        )
                    except Exception as e:
                        log_error(f"Exception during hybrid pre-filtering: {e}")
                        pre_filtered_symbols = None
                else:
                    try:
                        pre_filtered_symbols = pre_filter_symbols_with_voting(
                            all_symbols=all_symbols,
                            timeframe=primary_timeframe,
                            limit=limit
                        )
                    except Exception as e:
                        log_error(f"Exception during voting pre-filtering: {e}")
                        pre_filtered_symbols = None

                if pre_filtered_symbols is not None:
                    if len(pre_filtered_symbols) < len(all_symbols):
                        log_info(f"Pre-filtered: {len(all_symbols)} → {len(pre_filtered_symbols)} symbols (all symbols with signals)")
                    elif len(pre_filtered_symbols) == len(all_symbols):
                        log_info(f"Pre-filtered: All {len(all_symbols)} symbols have signals (no filtering applied)")
                    else:
                        log_warn("Pre-filtered symbols count is greater than all symbols. There may be an unexpected behavior.")
                else:
                    log_info(f"Pre-filtered: No symbols with signals found, using all {len(all_symbols)} symbols")
                    pre_filtered_symbols = None

                print(f"Pre-filtered symbols: {pre_filtered_symbols}")
                
        except SymbolFetchError as e:
            log_error(f"Failed to fetch symbols from exchange: {e}")
            if e.is_retryable:
                log_error("This was a retryable error (network/rate limit). Please check your connection and try again.")
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
            initial_symbols=pre_filtered_symbols
        )
        
        # Display results
        print()
        print(color_text("=" * 60, Fore.GREEN))
        print(color_text("SCAN RESULTS", Fore.GREEN))
        print(color_text("=" * 60, Fore.GREEN))
        print()
        
        # Display LONG signals with confidence (sorted by confidence)
        long_symbols = results.get('long_symbols', [])
        print(color_text(f"LONG Signals ({len(long_symbols)}):", Fore.GREEN))
        if results.get('long_symbols_with_confidence'):
            print("  (Sorted by confidence: High → Low)")
            for symbol, confidence in results['long_symbols_with_confidence']:
                clamped_confidence = min(max(confidence, 0.0), 1.0)
                length = int(clamped_confidence * 10)
                confidence_bar = "█" * length  # Visual bar
                print(f"  {symbol:15s} | Confidence: {confidence:.2f} {confidence_bar}")
        elif long_symbols:
            # Fallback if confidence data not available
            for i in range(0, len(long_symbols), 5):
                row = long_symbols[i:i+5]
                print("  " + "  ".join(f"{s:12s}" for s in row))
        else:
            print("  None")
        
        print()
        # Display SHORT signals with confidence (sorted by confidence)
        short_symbols = results.get('short_symbols', [])
        print(color_text(f"SHORT Signals ({len(short_symbols)}):", Fore.RED))
        if results.get('short_symbols_with_confidence'):
            print("  (Sorted by confidence: High → Low)")
            for symbol, confidence in results['short_symbols_with_confidence']:
                clamped_confidence = min(max(confidence, 0.0), 1.0)
                length = int(clamped_confidence * 10)
                confidence_bar = "█" * length  # Visual bar
                print(f"  {symbol:15s} | Confidence: {confidence:.2f} {confidence_bar}")
        elif short_symbols:
            # Fallback if confidence data not available
            for i in range(0, len(short_symbols), 5):
                row = short_symbols[i:i+5]
                print("  " + "  ".join(f"{s:12s}" for s in row))
        else:
            print("  None")
        
        # Display summary with average confidence
        if results.get('summary'):
            summary = results['summary']
            if summary.get('avg_long_confidence', 0) > 0:
                print()
                print(color_text("Summary:", Fore.CYAN))
                print(f"  Average LONG confidence: {summary['avg_long_confidence']:.2f}")
            if summary.get('avg_short_confidence', 0) > 0:
                print(f"  Average SHORT confidence: {summary['avg_short_confidence']:.2f}")
        
        print()
        print(color_text("=" * 60, Fore.GREEN))
        results_file = results.get('results_file', 'N/A')
        print(color_text(f"Results saved to: {results_file}", Fore.GREEN))
        print(color_text("=" * 60, Fore.GREEN))        
    except Exception as e:
        log_error(f"Error during batch scan: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    # Ensure stdin is available on Windows (critical when running file directly)
    if sys.platform == 'win32':
        try:
            if sys.stdin is None or (hasattr(sys.stdin, 'closed') and sys.stdin.closed):
                sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
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


