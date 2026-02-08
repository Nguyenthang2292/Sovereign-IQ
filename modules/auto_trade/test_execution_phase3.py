"""
Test script for Phase 3: Module BINANCE SEND MARKET

This script demonstrates how to use the execution module components.

Usage:
    python test_execution_phase3.py --dry-run
    python test_execution_phase3.py --testnet
    python test_execution_phase3.py --help
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.auto_trade.core.signal_selector import FinalSignal
from modules.auto_trade.execution.order_manager import OrderManager
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager


def create_mock_signal() -> FinalSignal:
    """Create a mock signal for testing."""
    return FinalSignal(
        symbol="BTC/USDT",
        signal_type="LONG",
        confidence=0.85,
        score=85.0,
        entry_price=50000.0,
        stop_loss=25000.0,  # 50% down
        take_profit=52500.0,  # 5% up
        timestamp=0,
        sources={
            "xgboost_score": 0.9,
            "gemini_score": 0.8,
            "gemini_reasoning": "Strong bullish trend detected",
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Test Phase 3: Order Execution")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate orders without actual execution",
    )
    parser.add_argument("--testnet", action="store_true", help="Use Binance testnet")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force order execution even if positions exist",
    )
    parser.add_argument("--leverage", type=int, default=2, help="Leverage to use (default: 2x)")
    args = parser.parse_args()

    print("=" * 80)
    print("🧪 PHASE 3: ORDER EXECUTION TEST")
    print("=" * 80)
    print()

    # Load API credentials
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        print("❌ Error: BINANCE_API_KEY and BINANCE_API_SECRET must be set")
        print("   Set them in your environment or .env file")
        return 1

    if args.dry_run:
        print("🧪 Running in DRY RUN mode (no actual orders)")
    if args.testnet:
        print("🧪 Using Binance TESTNET")
    print()

    # Initialize DataFetcher
    print("1. Initializing DataFetcher...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager=exchange_manager)
    print("   ✅ DataFetcher initialized")
    print()

    # Initialize OrderManager
    print("2. Initializing OrderManager...")
    order_manager = OrderManager(
        data_fetcher=data_fetcher,
        api_key=api_key,
        api_secret=api_secret,
        testnet=args.testnet,
        dry_run=args.dry_run,
        balance_percentage=0.95,
        default_leverage=args.leverage,
        default_tp_pct=5.0,
        default_sl_pct=50.0,
    )
    print("   ✅ OrderManager initialized")
    print()

    # Check open positions
    print("3. Checking for open positions...")
    open_positions = order_manager.check_open_positions()
    if open_positions and not args.force:
        print("   ⚠️ Open positions detected. Use --force to override.")
        return 0
    print("   ✅ No blocking positions")
    print()

    # Create mock signal
    print("4. Creating mock signal...")
    signal = create_mock_signal()
    print(f"   Signal: {signal.symbol} {signal.signal_type}")
    print(f"   Confidence: {signal.confidence:.1%}")
    print(f"   Entry: ${signal.entry_price:,.2f}")
    print(f"   TP: ${signal.take_profit:,.2f} (+5%)")
    print(f"   SL: ${signal.stop_loss:,.2f} (-50%)")
    print("   ✅ Signal created")
    print()

    # Execute signal
    print("5. Executing signal...")
    print("-" * 80)
    order_result = order_manager.execute_signal(
        signal=signal, force_execution=args.force, leverage_override=args.leverage
    )
    print("-" * 80)
    print()

    # Display result
    if order_result:
        print("✅ ORDER EXECUTION SUCCESSFUL")
        print("=" * 80)
        print()

        if args.dry_run:
            print("🧪 DRY RUN - No actual order was placed")
            print(f"   Symbol: {order_result.get('symbol')}")
            print(f"   Side: {order_result.get('side', '').upper()}")
            print(f"   Amount: ${order_result.get('amount', 0):.2f} USDT")
            print(f"   Leverage: {order_result.get('leverage')}x")
        else:
            market_order = order_result.get("market_order", {})
            entry_price = order_result.get("entry_price")
            tp_order = order_result.get("take_profit_order")
            sl_order = order_result.get("stop_loss_order")

            print(f"📊 Market Order ID: {market_order.get('id')}")
            print(f"   Status: {market_order.get('status')}")
            print(f"   Filled at: ${entry_price:,.2f}")
            print()
            print(f"✅ Take Profit Order: {'Placed' if tp_order else 'Failed'}")
            if tp_order:
                print(f"   TP Order ID: {tp_order.get('id')}")
            print(f"✅ Stop Loss Order: {'Placed' if sl_order else 'Failed'}")
            if sl_order:
                print(f"   SL Order ID: {sl_order.get('id')}")

        print()
        return 0
    else:
        print("❌ ORDER EXECUTION FAILED")
        print("   Check logs above for details")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
