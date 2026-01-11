"""
Script đơn giản để tính tổng size positions đang mở (Long + Short) từ Binance Futures account.

Usage:
    python calculate_position_totals.py
"""

import sys
import threading

from colorama import Fore, Style
from colorama import init as colorama_init

from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.utils import (
    color_text,
    log_error,
    log_info,
    log_success,
)

# Initialize colorama
colorama_init(autoreset=True)


def calculate_position_totals():
    """
    Tính tổng size positions đang mở từ Binance Futures account.
    
    Returns:
        Tuple of (long_total, short_total, combined_total, positions_list)
        - long_total: Tổng size của LONG positions (USDT)
        - short_total: Tổng size của SHORT positions (USDT)
        - combined_total: Tổng tổng (Long + Short) (USDT)
        - positions_list: List of position dicts
    """
    # Initialize components
    shutdown_event = threading.Event()
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager, shutdown_event)
    
    # Fetch positions from Binance Futures
    try:
        positions = data_fetcher.fetch_binance_futures_positions()
    except ValueError as e:
        log_error(f"Error fetching positions: {e}")
        return None, None, None, []
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        return None, None, None, []
    
    if not positions:
        log_info("No open positions found.")
        return 0.0, 0.0, 0.0, []
    
    # Calculate totals by direction
    long_total = 0.0
    short_total = 0.0
    
    for pos in positions:
        size = pos.get("size_usdt", 0.0)
        direction = pos.get("direction", "").upper()
        
        if direction == "LONG":
            long_total += size
        elif direction == "SHORT":
            short_total += size
    
    combined_total = long_total + short_total
    
    return long_total, short_total, combined_total, positions


def display_positions(positions):
    """
    Hiển thị chi tiết từng position với màu sắc.
    
    Args:
        positions: List of position dictionaries
    """
    if not positions:
        return
    
    print("\n" + color_text("=== POSITION DETAILS ===", Fore.CYAN, Style.BRIGHT))
    print("-" * 80)
    
    # Sort positions: LONG first (green), then SHORT (red)
    sorted_positions = sorted(positions, key=lambda p: (p.get("direction", "").upper() != "LONG", p.get("symbol", "")))
    
    for pos in sorted_positions:
        symbol = pos.get("symbol", "N/A")
        direction = pos.get("direction", "N/A").upper()
        size = pos.get("size_usdt", 0.0)
        entry_price = pos.get("entry_price", 0.0)
        contracts = pos.get("contracts", 0.0)
        
        # Color code based on direction
        if direction == "LONG":
            direction_color = Fore.GREEN
            size_color = Fore.GREEN
        elif direction == "SHORT":
            direction_color = Fore.RED
            size_color = Fore.RED
        else:
            direction_color = Fore.WHITE
            size_color = Fore.WHITE
        
        # Format output
        direction_str = color_text(direction, direction_color, Style.BRIGHT)
        size_str = color_text(f"{size:.8f}", size_color)
        
        print(
            f"  {symbol:20s} | {direction_str:6s} | "
            f"Size: {size_str:15s} USDT | "
            f"Entry: {entry_price:.8f} | "
            f"Contracts: {contracts:.4f}"
        )


def display_totals(long_total, short_total, combined_total):
    """
    Hiển thị tổng size (Long, Short, Combined) với màu sắc.
    
    Args:
        long_total: Tổng size LONG positions (USDT)
        short_total: Tổng size SHORT positions (USDT)
        combined_total: Tổng tổng (USDT)
    """
    print("\n" + color_text("=== TOTALS ===", Fore.CYAN, Style.BRIGHT))
    print("-" * 80)
    
    # Display LONG total in green
    long_str = color_text(f"{long_total:.8f} USDT", Fore.GREEN, Style.BRIGHT)
    print(f"Total LONG Size:  {long_str}")
    
    # Display SHORT total in red
    short_str = color_text(f"{short_total:.8f} USDT", Fore.RED, Style.BRIGHT)
    print(f"Total SHORT Size: {short_str}")
    
    # Display combined total in yellow
    combined_str = color_text(f"{combined_total:.8f} USDT", Fore.YELLOW, Style.BRIGHT)
    print(f"Total Combined:   {combined_str}")
    print("-" * 80)


def main():
    """Main entry point."""
    print(color_text(
        "=== Calculate Position Totals (Binance Futures) ===",
        Fore.MAGENTA,
        Style.BRIGHT
    ))
    
    # Calculate totals
    long_total, short_total, combined_total, positions = calculate_position_totals()
    
    # Check for errors
    if long_total is None:
        print(color_text("\nFailed to calculate position totals.", Fore.RED))
        sys.exit(1)
    
    # Display results
    if not positions:
        print(color_text("\nNo open positions found.", Fore.YELLOW))
        sys.exit(0)
    
    # Display position details
    display_positions(positions)
    
    # Display totals
    display_totals(long_total, short_total, combined_total)
    
    print()  # Empty line at end


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color_text("\n\nInterrupted by user. Exiting...", Fore.YELLOW))
        sys.exit(0)
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

