"""
Main CLI function for position sizing module.

This module provides the main entry point for calculating position sizes
using Bayesian Kelly Criterion and Regime Switching.
"""

import sys
import os
from pathlib import Path
import pandas as pd
from typing import Optional

# Add project root to Python path if running directly
if __name__ == "__main__":
    # Get the project root (3 levels up from this file)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from modules.common.utils import configure_windows_stdio
configure_windows_stdio()

from colorama import Fore, Style, init as colorama_init

from modules.common.utils import (
    color_text,
    log_error,
    log_progress,
    log_success,
    log_warn,
)
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.core.data_fetcher import DataFetcher
from modules.common.utils import initialize_components
from modules.position_sizing.core.position_sizer import PositionSizer
from modules.position_sizing.utils.data_loader import (
    load_symbols_from_results,
    load_symbols_from_file,
    validate_symbols,
)
from modules.position_sizing.cli.argument_parser import (
    parse_args,
    interactive_config_menu,
)
from modules.position_sizing.cli.display import (
    display_position_sizing_results,
    display_configuration,
)

colorama_init(autoreset=True)


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
        if not args.no_menu:
            config = interactive_config_menu()
            # Merge config with args
            for key, value in config.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)
        
        # Initialize components first (needed for fetching balance)
        log_progress("\nInitializing components...")
        exchange_manager, data_fetcher = initialize_components()
        
        # Get account balance
        account_balance = args.account_balance
        
        # Try to fetch from Binance if requested
        if (account_balance is None and (args.fetch_balance or getattr(args, 'fetch_balance', False))) or \
           (hasattr(args, 'fetch_balance') and args.fetch_balance):
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
            print(color_text("  2. Fetch from Binance (requires API credentials) [default]", Fore.MAGENTA, Style.BRIGHT))
            default_choice_text = color_text("default=2", Fore.MAGENTA)
            choice = input(f"\nSelect option (1/2, {default_choice_text}): ").strip()
            
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
                        account_balance_str = input("\nEnter account balance (USDT): ").strip()
                        try:
                            account_balance = float(account_balance_str)
                        except ValueError:
                            log_error("Invalid account balance. Exiting.")
                            sys.exit(1)
                except Exception as e:
                    log_warn(f"Error fetching balance from Binance: {e}. Please enter manually.")
                    account_balance_str = input("\nEnter account balance (USDT): ").strip()
                    try:
                        account_balance = float(account_balance_str)
                    except ValueError:
                        log_error("Invalid account balance. Exiting.")
                        sys.exit(1)
            else:
                account_balance_str = input("\nEnter account balance (USDT): ").strip()
                try:
                    account_balance = float(account_balance_str)
                except ValueError:
                    log_error("Invalid account balance. Exiting.")
                    sys.exit(1)
        
        if account_balance <= 0:
            log_error("Account balance must be positive. Exiting.")
            sys.exit(1)
        
        # Load symbols
        log_progress("\nLoading symbols...")
        symbols = []
        
        if args.symbols_file:
            symbols = load_symbols_from_file(args.symbols_file)
        elif args.symbols:
            # Parse comma-separated symbols and normalize (auto-add /USDT if needed)
            from modules.position_sizing.cli.argument_parser import parse_symbols_string
            symbol_list = parse_symbols_string(args.symbols)
            symbols = [{'symbol': s, 'signal': 1} for s in symbol_list]
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
        
        # Initialize Position Sizer
        position_sizer = PositionSizer(
            data_fetcher=data_fetcher,
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
            max_position_size=args.max_position_size,
        )
        
        # Display configuration
        config_dict = {
            'account_balance': account_balance,
            'timeframe': args.timeframe,
            'lookback_days': args.lookback_days,
            'max_position_size': args.max_position_size,
        }
        if args.source:
            config_dict['source'] = args.source
        elif args.symbols_file:
            config_dict['symbols_file'] = args.symbols_file
        elif args.symbols:
            config_dict['symbols'] = args.symbols
        
        display_configuration(config_dict)
        
        # Calculate position sizes
        log_progress(f"\nCalculating position sizes for {len(symbols)} symbols...")
        log_progress("This may take a few minutes...\n")
        
        results_df = position_sizer.calculate_portfolio_allocation(
            symbols=symbols,
            account_balance=account_balance,
            timeframe=args.timeframe,
            lookback=args.lookback_days,  # This will be converted to candles internally
        )
        
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
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()

