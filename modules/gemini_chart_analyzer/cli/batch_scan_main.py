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

# Fix encoding issues on Windows
from modules.common.utils import configure_windows_stdio
configure_windows_stdio()

from colorama import Fore, init as colorama_init
from modules.common.utils import color_text, log_error, log_warn
from modules.common.utils import normalize_timeframe
from modules.gemini_chart_analyzer.core.market_batch_scanner import MarketBatchScanner

import warnings
# Suppress specific noisy warnings if needed
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PIL.Image")
colorama_init(autoreset=True)


def interactive_batch_scan():
    """Interactive menu for batch scanning."""
    print()
    print(color_text("=" * 60, Fore.CYAN))
    print(color_text("MARKET BATCH SCANNER", Fore.CYAN))
    print(color_text("=" * 60, Fore.CYAN))
    print()
    
    # Get timeframe
    print("Timeframes: 15m, 30m, 1h, 4h, 1d, 1w")
    timeframe = input(color_text("Enter timeframe [1h]: ", Fore.YELLOW)).strip()
    if not timeframe:
        timeframe = '1h'
    
    # Normalize timeframe
    timeframe = normalize_timeframe(timeframe)
    
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
    limit_input = input(color_text("Number of candles per symbol [200]: ", Fore.YELLOW)).strip()
    limit = 200
    if limit_input:
        try:
            limit = int(limit_input)
            # Validate: must be positive integer (>=1)
            if limit < 1:
                log_warn(f"limit ({limit}) must be >= 1, clamping to 1")
                limit = 1
        except ValueError:
            log_warn("Invalid input, using default 200")
    else:
        # Validate default value
        if limit < 1:
            log_warn(f"limit default ({limit}) must be >= 1, clamping to 1")
            limit = 1
    
    # Confirm
    print()
    print(color_text("=" * 60, Fore.CYAN))
    print(color_text("CONFIGURATION", Fore.CYAN))
    print(color_text("=" * 60, Fore.CYAN))
    print(f"Timeframe: {timeframe}")
    print(f"Max symbols: {max_symbols or 'All'}")
    print(f"Cooldown: {cooldown}s")
    print(f"Candles per symbol: {limit}")
    print()
    
    confirm = input(color_text("Start batch scan? (y/n) [y]: ", Fore.YELLOW)).strip().lower()
    if confirm and confirm not in ['y', 'yes']:
        log_warn("Cancelled by user")
        return
    
    # Initialize scanner
    try:
        scanner = MarketBatchScanner(cooldown_seconds=cooldown)
        
        # Run scan
        results = scanner.scan_market(
            timeframe=timeframe,
            max_symbols=max_symbols,
            limit=limit
        )
        
        # Display results
        print()
        print(color_text("=" * 60, Fore.GREEN))
        print(color_text("SCAN RESULTS", Fore.GREEN))
        print(color_text("=" * 60, Fore.GREEN))
        print()
        
        # Display LONG signals with confidence (sorted by confidence)
        print(color_text(f"LONG Signals ({len(results['long_symbols'])}):", Fore.GREEN))
        if results.get('long_symbols_with_confidence'):
            print("  (Sorted by confidence: High → Low)")
            for symbol, confidence in results['long_symbols_with_confidence']:
                confidence_bar = "█" * int(confidence * 10)  # Visual bar
                print(f"  {symbol:15s} | Confidence: {confidence:.2f} {confidence_bar}")
        elif results['long_symbols']:
            # Fallback if confidence data not available
            for i in range(0, len(results['long_symbols']), 5):
                row = results['long_symbols'][i:i+5]
                print("  " + "  ".join(f"{s:12s}" for s in row))
        else:
            print("  None")
        
        print()
        # Display SHORT signals with confidence (sorted by confidence)
        print(color_text(f"SHORT Signals ({len(results['short_symbols'])}):", Fore.RED))
        if results.get('short_symbols_with_confidence'):
            print("  (Sorted by confidence: High → Low)")
            for symbol, confidence in results['short_symbols_with_confidence']:
                confidence_bar = "█" * int(confidence * 10)  # Visual bar
                print(f"  {symbol:15s} | Confidence: {confidence:.2f} {confidence_bar}")
        elif results['short_symbols']:
            # Fallback if confidence data not available
            for i in range(0, len(results['short_symbols']), 5):
                row = results['short_symbols'][i:i+5]
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
    except KeyboardInterrupt:
        log_warn("\nCancelled by user")
        sys.exit(0)
    except Exception as e:
        log_error(f"Error during batch scan: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
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

