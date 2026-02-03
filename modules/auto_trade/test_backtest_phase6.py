"""
Test script for Auto-Trade Backtesting (Phase 6.5).

Demonstrates the integration of the backtester module with auto-trade system.
"""

import sys
from pathlib import Path

# Add project root to path
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from colorama import Fore, Style
from colorama import init as colorama_init

from modules.auto_trade.backtest import AutoTradeBacktester
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.utils import color_text, days_to_candles, log_error, log_progress, log_success

colorama_init(autoreset=True)


def display_backtest_results(result: dict, symbol: str) -> None:
    """Display backtest results in a formatted way."""
    if not result or not result.get("trades"):
        print(color_text(f"\nNo backtest results for {symbol}", Fore.YELLOW))
        return

    trades = result["trades"]
    metrics = result["metrics"]

    # Header
    print("\n" + color_text("=" * 100, Fore.CYAN, Style.BRIGHT))
    print(color_text(f"AUTO-TRADE BACKTEST RESULTS - {symbol}", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))

    # Performance Metrics
    print("\n" + color_text("PERFORMANCE METRICS", Fore.CYAN, Style.BRIGHT))
    print(color_text("-" * 100, Fore.CYAN))

    win_rate = metrics.get("win_rate", 0.0)
    win_rate_pct = win_rate * 100
    win_rate_color = Fore.GREEN if win_rate > 0.5 else Fore.RED
    print(f"{color_text('Win Rate:', Fore.WHITE)} {color_text(f'{win_rate_pct:.2f}%', win_rate_color)}")

    print(f"{color_text('Number of Trades:', Fore.WHITE)} {color_text(str(metrics.get('num_trades', 0)), Fore.YELLOW)}")

    total_return = metrics.get("total_return", 0.0)
    total_return_pct = total_return * 100
    total_return_color = Fore.GREEN if total_return > 0 else Fore.RED
    print(f"{color_text('Total Return:', Fore.WHITE)} {color_text(f'{total_return_pct:.2f}%', total_return_color)}")

    # Auto-Trade Specific Metrics
    print("\n" + color_text("AUTO-TRADE SPECIFIC METRICS", Fore.CYAN, Style.BRIGHT))
    print(color_text("-" * 100, Fore.CYAN))

    leverage = metrics.get("leverage_used", 2)
    print(f"{color_text('Leverage Used:', Fore.WHITE)} {color_text(f'{leverage}x', Fore.YELLOW)}")

    be_moves = metrics.get("breakeven_moves", 0)
    print(f"{color_text('Break-Even Moves:', Fore.WHITE)} {color_text(str(be_moves), Fore.CYAN)}")

    martingale_trades = metrics.get("martingale_trades", 0)
    max_martingale = metrics.get("max_martingale_step", 0)
    print(f"{color_text('Martingale Trades:', Fore.WHITE)} {color_text(str(martingale_trades), Fore.YELLOW)}")
    print(f"{color_text('Max Martingale Step:', Fore.WHITE)} {color_text(str(max_martingale), Fore.YELLOW)}")

    # Trade Statistics
    if trades:
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

        print("\n" + color_text("TRADE STATISTICS", Fore.CYAN, Style.BRIGHT))
        print(color_text("-" * 100, Fore.CYAN))
        print(f"{color_text('Winning Trades:', Fore.WHITE)} {color_text(str(len(winning_trades)), Fore.GREEN)}")
        print(f"{color_text('Losing Trades:', Fore.WHITE)} {color_text(str(len(losing_trades)), Fore.RED)}")

    print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))


def test_basic_backtest():
    """Test basic backtesting functionality."""
    print(color_text("\n" + "=" * 100, Fore.CYAN, Style.BRIGHT))
    print(color_text("TEST 1: BASIC BACKTEST (Without Martingale)", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))

    try:
        # Initialize components
        log_progress("\nInitializing components...")
        exchange_manager = ExchangeManager()
        data_fetcher = DataFetcher(exchange_manager)
        log_success("Components initialized")

        # Create backtester with auto-trade parameters
        log_progress("\nCreating AutoTradeBacktester...")
        backtester = AutoTradeBacktester(
            data_fetcher=data_fetcher,
            stop_loss_pct=0.50,  # 50% stop loss
            take_profit_pct=0.05,  # 5% take profit
            risk_per_trade=0.95,  # 95% balance per trade
            leverage=2,  # 2x leverage
            enable_breakeven=True,  # Enable BE protection
            enable_martingale=False,  # Disable Martingale for first test
        )
        log_success("AutoTradeBacktester created")

        # Run backtest
        symbol = "BTC/USDT"
        timeframe = "1h"
        lookback_days = 12
        lookback = days_to_candles(lookback_days, timeframe)
        initial_capital = 10000.0

        log_progress(f"\nRunning backtest for {symbol} ({timeframe})...")
        log_progress(f"Parameters: {lookback_days} days ({lookback} candles), ${initial_capital:,.2f} capital")

        result = backtester.backtest_strategy(
            symbol=symbol,
            timeframe=timeframe,
            lookback=lookback,
            initial_capital=initial_capital,
        )

        log_success("Backtest completed")

        # Display results
        display_backtest_results(result, symbol)

        return result

    except Exception as e:
        log_error(f"Error in basic backtest: {e}")
        import traceback

        log_error(f"Traceback: {traceback.format_exc()}")
        return None


def test_martingale_backtest():
    """Test backtesting with Martingale strategy."""
    print(color_text("\n" + "=" * 100, Fore.CYAN, Style.BRIGHT))
    print(color_text("TEST 2: MARTINGALE BACKTEST", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))

    try:
        # Initialize components
        log_progress("\nInitializing components...")
        exchange_manager = ExchangeManager()
        data_fetcher = DataFetcher(exchange_manager)
        log_success("Components initialized")

        # Create backtester with Martingale enabled
        log_progress("\nCreating AutoTradeBacktester with Martingale...")
        backtester = AutoTradeBacktester(
            data_fetcher=data_fetcher,
            stop_loss_pct=0.50,  # 50% stop loss
            take_profit_pct=0.05,  # 5% take profit
            risk_per_trade=0.95,  # 95% balance per trade
            leverage=2,  # 2x initial leverage
            enable_breakeven=True,  # Enable BE protection
            enable_martingale=True,  # Enable Martingale
            martingale_max_steps=4,  # Max 4 steps
            martingale_max_leverage=16,  # Max 16x leverage
        )
        log_success("AutoTradeBacktester created with Martingale")

        # Run backtest
        symbol = "BTC/USDT"
        timeframe = "1h"
        lookback_days = 12
        lookback = days_to_candles(lookback_days, timeframe)
        initial_capital = 10000.0

        log_progress(f"\nRunning Martingale backtest for {symbol} ({timeframe})...")
        log_progress(f"Parameters: {lookback_days} days ({lookback} candles), ${initial_capital:,.2f} capital")
        log_progress("Martingale: Enabled (Max 4 steps, Max 16x leverage)")

        result = backtester.backtest_strategy(
            symbol=symbol,
            timeframe=timeframe,
            lookback=lookback,
            initial_capital=initial_capital,
        )

        log_success("Martingale backtest completed")

        # Display results
        display_backtest_results(result, symbol)

        # Validate Martingale safety
        if result and result.get("trades"):
            log_progress("\nValidating Martingale safety...")
            safety = backtester.validate_martingale_safety(result["trades"])

            print("\n" + color_text("MARTINGALE SAFETY ANALYSIS", Fore.CYAN, Style.BRIGHT))
            print(color_text("-" * 100, Fore.CYAN))

            safe_color = Fore.GREEN if safety["safe"] else Fore.RED
            safe_status = "SAFE" if safety["safe"] else "UNSAFE"
            print(f"{color_text('Overall Safety:', Fore.WHITE)} {color_text(safe_status, safe_color)}")
            max_losses_str = str(safety["max_consecutive_losses"])
            print(f"{color_text('Max Consecutive Losses:', Fore.WHITE)} {color_text(max_losses_str, Fore.YELLOW)}")
            max_lev = f"{safety['max_leverage_used']}x"
            print(f"{color_text('Max Leverage Used:', Fore.WHITE)} {color_text(max_lev, Fore.YELLOW)}")

            if safety["exceeded_max_steps"]:
                print(color_text("⚠️  WARNING: Exceeded maximum Martingale steps!", Fore.RED))
            if safety["exceeded_max_leverage"]:
                print(color_text("⚠️  WARNING: Exceeded maximum leverage!", Fore.RED))

        return result

    except Exception as e:
        log_error(f"Error in Martingale backtest: {e}")
        import traceback

        log_error(f"Traceback: {traceback.format_exc()}")
        return None


def main():
    """Main test function."""
    print(color_text("\n" + "=" * 100, Fore.CYAN, Style.BRIGHT))
    print(color_text("AUTO-TRADE BACKTESTING MODULE TEST (Phase 6.5)", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))

    print(color_text("\nThis test demonstrates the integration of the backtester module", Fore.WHITE))
    print(color_text("with the auto-trade system's specific requirements:", Fore.WHITE))
    print(color_text("  • 50% Stop Loss / 5% Take Profit", Fore.CYAN))
    print(color_text("  • 95% Balance Risk per Trade", Fore.CYAN))
    print(color_text("  • 2x Leverage (scales with Martingale)", Fore.CYAN))
    print(color_text("  • Break-Even Protection at 30% Drawdown", Fore.CYAN))
    print(color_text("  • Optional Martingale Loss Recovery", Fore.CYAN))

    # Run tests
    result1 = test_basic_backtest()

    print("\n")
    input(color_text("Press Enter to run Martingale backtest...", Fore.YELLOW))

    result2 = test_martingale_backtest()

    # Summary
    print("\n" + color_text("=" * 100, Fore.CYAN, Style.BRIGHT))
    print(color_text("TEST SUMMARY", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))

    if result1:
        print(
            color_text(
                f"✅ Test 1 (Basic): {result1['metrics']['num_trades']} trades, "
                f"{result1['metrics']['total_return'] * 100:.2f}% return",
                Fore.GREEN,
            )
        )
    else:
        print(color_text("❌ Test 1 (Basic): FAILED", Fore.RED))

    if result2:
        print(
            color_text(
                f"✅ Test 2 (Martingale): {result2['metrics']['num_trades']} trades, "
                f"{result2['metrics']['total_return'] * 100:.2f}% return, "
                f"{result2['metrics']['martingale_trades']} Martingale trades",
                Fore.GREEN,
            )
        )
    else:
        print(color_text("❌ Test 2 (Martingale): FAILED", Fore.RED))

    print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color_text("\n\nTest interrupted by user.", Fore.YELLOW))
        sys.exit(0)
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        import traceback

        log_error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
