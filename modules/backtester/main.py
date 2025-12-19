"""
Backtester Main Entry Point.

File main để test tính năng của backtester với dữ liệu BTC 1h mặc định.
"""

import warnings
import sys
from colorama import Fore, Style, init as colorama_init

from modules.common.utils import (
    configure_windows_stdio,
    initialize_components,
    color_text,
    log_error,
    log_progress,
    log_success,
    days_to_candles,
)
from modules.backtester import FullBacktester
from config.position_sizing import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_TIMEFRAME,
)

warnings.filterwarnings("ignore")
colorama_init(autoreset=True)
configure_windows_stdio()


def display_backtest_results(result: dict, symbol: str) -> None:
    """
    Hiển thị kết quả backtest một cách đẹp mắt.
    
    Args:
        result: Dictionary chứa kết quả backtest từ FullBacktester.backtest()
        symbol: Symbol được backtest
    """
    if not result or not result.get('trades'):
        print(color_text(f"\nKhông có kết quả backtest cho {symbol}", Fore.YELLOW))
        return
    
    trades = result['trades']
    metrics = result['metrics']
    
    # Header
    print("\n" + color_text("=" * 100, Fore.CYAN, Style.BRIGHT))
    print(color_text(f"BACKTEST RESULTS - {symbol}", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))
    
    # Performance Metrics
    print("\n" + color_text("PERFORMANCE METRICS", Fore.CYAN, Style.BRIGHT))
    print(color_text("-" * 100, Fore.CYAN))
    
    print(f"{color_text('Win Rate:', Fore.WHITE)} {color_text(f'{metrics.get(\"win_rate\", 0.0)*100:.2f}%', Fore.GREEN if metrics.get('win_rate', 0.0) > 0.5 else Fore.RED)}")
    print(f"{color_text('Number of Trades:', Fore.WHITE)} {color_text(str(metrics.get('num_trades', 0)), Fore.YELLOW)}")
    print(f"{color_text('Total Return:', Fore.WHITE)} {color_text(f'{metrics.get(\"total_return\", 0.0)*100:.2f}%', Fore.GREEN if metrics.get('total_return', 0.0) > 0 else Fore.RED)}")
    print(f"{color_text('Sharpe Ratio:', Fore.WHITE)} {color_text(f'{metrics.get(\"sharpe_ratio\", 0.0):.2f}', Fore.GREEN if metrics.get('sharpe_ratio', 0.0) > 1.0 else Fore.YELLOW)}")
    print(f"{color_text('Max Drawdown:', Fore.WHITE)} {color_text(f'{metrics.get(\"max_drawdown\", 0.0)*100:.2f}%', Fore.RED if metrics.get('max_drawdown', 0.0) < -0.2 else Fore.YELLOW)}")
    print(f"{color_text('Profit Factor:', Fore.WHITE)} {color_text(f'{metrics.get(\"profit_factor\", 0.0):.2f}', Fore.GREEN if metrics.get('profit_factor', 0.0) > 1.0 else Fore.RED)}")
    print(f"{color_text('Average Win:', Fore.WHITE)} {color_text(f'{metrics.get(\"avg_win\", 0.0)*100:.2f}%', Fore.GREEN)}")
    print(f"{color_text('Average Loss:', Fore.WHITE)} {color_text(f'{metrics.get(\"avg_loss\", 0.0)*100:.2f}%', Fore.RED)}")
    
    print(color_text("-" * 100, Fore.CYAN))
    
    # Trade Statistics
    if trades:
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        print("\n" + color_text("TRADE STATISTICS", Fore.CYAN, Style.BRIGHT))
        print(color_text("-" * 100, Fore.CYAN))
        print(f"{color_text('Winning Trades:', Fore.WHITE)} {color_text(str(len(winning_trades)), Fore.GREEN)}")
        print(f"{color_text('Losing Trades:', Fore.WHITE)} {color_text(str(len(losing_trades)), Fore.RED)}")
        
        if winning_trades:
            best_trade = max(winning_trades, key=lambda t: t.get('pnl', 0))
            print(f"{color_text('Best Trade:', Fore.WHITE)} {color_text(f'{best_trade.get(\"pnl_pct\", 0):.2f}%', Fore.GREEN)}")
        
        if losing_trades:
            worst_trade = min(losing_trades, key=lambda t: t.get('pnl', 0))
            print(f"{color_text('Worst Trade:', Fore.WHITE)} {color_text(f'{worst_trade.get(\"pnl_pct\", 0):.2f}%', Fore.RED)}")
        
        print(color_text("-" * 100, Fore.CYAN))
    
    # Recent Trades (last 10)
    if trades:
        print("\n" + color_text("RECENT TRADES (Last 10)", Fore.CYAN, Style.BRIGHT))
        print(color_text("-" * 100, Fore.CYAN))
        
        recent_trades = trades[-10:] if len(trades) > 10 else trades
        for i, trade in enumerate(recent_trades, 1):
            pnl_color = Fore.GREEN if trade.get('pnl', 0) > 0 else Fore.RED
            print(f"{i}. Entry: {trade.get('entry_time', 'N/A')} | "
                  f"Exit: {trade.get('exit_time', 'N/A')} | "
                  f"PnL: {color_text(f'{trade.get(\"pnl_pct\", 0):.2f}%', pnl_color)} | "
                  f"Reason: {trade.get('exit_reason', 'N/A')} | "
                  f"Hold: {trade.get('hold_periods', 0)} periods")
        
        print(color_text("-" * 100, Fore.CYAN))
    
    print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))


def main() -> None:
    """
    Main entry point để test backtester với BTC 1h.
    
    Khởi tạo components, tạo FullBacktester instance, và chạy backtest
    với BTC/USDT 1h timeframe mặc định.
    """
    # Configuration
    symbol = "BTC/USDT"
    timeframe = DEFAULT_TIMEFRAME  # "1h"
    lookback_days = DEFAULT_LOOKBACK_DAYS  # 90 days
    lookback = days_to_candles(lookback_days, timeframe)  # Convert days to candles
    signal_type = "LONG"  # Test với LONG signals
    initial_capital = 10000.0  # $10,000 initial capital
    
    print(color_text("\n" + "=" * 100, Fore.CYAN, Style.BRIGHT))
    print(color_text("BACKTESTER TEST - BTC 1H", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))
    
    print(f"\n{color_text('Configuration:', Fore.WHITE)}")
    print(f"  Symbol: {color_text(symbol, Fore.YELLOW)}")
    print(f"  Timeframe: {color_text(timeframe, Fore.YELLOW)}")
    print(f"  Lookback: {color_text(f'{lookback_days} days ({lookback} candles)', Fore.YELLOW)}")
    print(f"  Signal Type: {color_text(signal_type, Fore.YELLOW)}")
    print(f"  Initial Capital: {color_text(f'${initial_capital:,.2f}', Fore.YELLOW)}")
    
    try:
        # Initialize components
        log_progress("\nInitializing components...")
        exchange_manager, data_fetcher = initialize_components()
        log_success("Components initialized successfully")
        
        # Create backtester
        log_progress("\nCreating FullBacktester...")
        backtester = FullBacktester(data_fetcher=data_fetcher)
        log_success("FullBacktester created successfully")
        
        # Run backtest
        log_progress(f"\nRunning backtest for {symbol} ({timeframe})...")
        log_progress(f"This may take a while as it calculates signals for {lookback} periods...")
        
        result = backtester.backtest(
            symbol=symbol,
            timeframe=timeframe,
            lookback=lookback,
            signal_type=signal_type,
            initial_capital=initial_capital,
        )
        
        log_success("Backtest completed successfully")
        
        # Display results
        display_backtest_results(result, symbol)
        
    except KeyboardInterrupt:
        print(color_text("\n\nBacktest interrupted by user.", Fore.YELLOW))
        sys.exit(0)
    except Exception as e:
        log_error(f"Error running backtest: {type(e).__name__}: {e}")
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color_text("\nExiting program by user request.", Fore.YELLOW))
        sys.exit(0)
    except Exception as e:
        log_error(f"Error: {type(e).__name__}: {e}")
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

