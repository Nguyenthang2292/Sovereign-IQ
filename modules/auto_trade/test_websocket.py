"""
WebSocket Integration Test for Auto Trade System

Tests all WebSocket components:
- Position monitoring
- Balance monitoring
- Order monitoring
- Break-even manager

Run this on testnet/demo environment first to verify WebSocket connections.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.auto_trade.websocket.client import BinanceWebSocketClient
from modules.auto_trade.monitoring.position_monitor import PositionMonitor
from modules.auto_trade.monitoring.breakeven_manager import BreakEvenMonitor
from modules.auto_trade.monitoring.account_monitor import BalanceMonitor, OrderMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def test_websocket_integration():
    """Test WebSocket integration."""

    # Load API credentials
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")
    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

    if not api_key or not api_secret:
        logger.error("API credentials not found. Set BINANCE_API_KEY and BINANCE_SECRET_KEY environment variables.")
        return

    logger.info("=" * 60)
    logger.info("WebSocket Integration Test")
    logger.info(f"Mode: {'TESTNET' if testnet else 'PRODUCTION'}")
    logger.info("=" * 60)

    # Initialize WebSocket client
    ws_client = BinanceWebSocketClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )

    try:
        # Connect to WebSocket
        logger.info("Connecting to WebSocket...")
        await ws_client.connect()

        # Initialize monitors
        position_monitor = PositionMonitor(ws_client, max_positions=5)
        balance_monitor = BalanceMonitor(ws_client)
        order_monitor = OrderMonitor(ws_client)

        # Add test callbacks
        position_monitor.add_callback(on_position_update)
        balance_monitor.add_callback(on_balance_update)
        order_monitor.add_callback(on_order_update)

        # Start monitors
        logger.info("Starting monitors...")
        await position_monitor.start()
        await balance_monitor.start()
        await order_monitor.start()

        # Start watching all streams
        logger.info("Starting WebSocket watchers...")
        await ws_client.start_watching_all()

        logger.info("✅ All monitors started. Watching for updates...")
        logger.info("Press Ctrl+C to stop\n")

        # Wait for updates (run for 60 seconds or until interrupted)
        try:
            await asyncio.sleep(60)
        except KeyboardInterrupt:
            logger.info("\nShutdown requested by user")

        # Stop monitors
        logger.info("Stopping monitors...")
        await position_monitor.stop()
        await balance_monitor.stop()
        await order_monitor.stop()

    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)

    finally:
        # Close WebSocket connection
        logger.info("Closing WebSocket connection...")
        await ws_client.close()

    logger.info("=" * 60)
    logger.info("Test completed")
    logger.info("=" * 60)


def on_position_update(position):
    """Handle position update."""
    logger.info(
        f"[POSITION] {position.symbol} {position.side.upper()}: "
        f"PnL=${position.unrealized_pnl:.2f} ({position.unrealized_pnl_percent:+.2f}%)"
    )


def on_balance_update(balance):
    """Handle balance update."""
    logger.info(f"[BALANCE] ${balance.total:.2f} USDT (free: ${balance.free:.2f})")


def on_order_update(order):
    """Handle order update."""
    logger.info(
        f"[ORDER] {order.status.upper()} - {order.symbol} {order.side.upper()} "
        f"{order.filled}/{order.amount} @ ${order.price}"
    )


async def test_breakeven_monitor():
    """Test break-even monitor with mock data."""

    logger.info("=" * 60)
    logger.info("Break-Even Monitor Test")
    logger.info("=" * 60)

    # Load API credentials
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")
    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

    if not api_key or not api_secret:
        logger.error("API credentials not found")
        return

    # Initialize WebSocket client
    ws_client = BinanceWebSocketClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )

    try:
        # Connect
        await ws_client.connect()

        # Initialize monitors
        position_monitor = PositionMonitor(ws_client)

        # Get initial balance
        initial_balance = await ws_client.get_initial_balance()
        account_balance = initial_balance.get("USDT", {}).get("total", 1000.0)

        logger.info(f"Account balance: ${account_balance:.2f} USDT")

        # Initialize break-even monitor
        be_monitor = BreakEvenMonitor(
            ws_client=ws_client,
            position_monitor=position_monitor,
            account_balance=account_balance,
            drawdown_threshold_percent=30.0,
            dry_run=True,  # Dry run mode for testing
        )

        # Start monitors
        await position_monitor.start()
        await be_monitor.start()

        # Start watching
        await ws_client.start_watching_all()

        logger.info("✅ Break-even monitor started (DRY RUN mode)")
        logger.info("Watching positions for 60 seconds...\n")

        # Wait
        try:
            await asyncio.sleep(60)
        except KeyboardInterrupt:
            logger.info("\nShutdown requested")

        # Stop
        await be_monitor.stop()
        await position_monitor.stop()

    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)

    finally:
        await ws_client.close()

    logger.info("=" * 60)
    logger.info("Test completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test WebSocket integration")
    parser.add_argument(
        "--test",
        choices=["all", "breakeven"],
        default="all",
        help="Which test to run (default: all)",
    )

    args = parser.parse_args()

    if args.test == "all":
        asyncio.run(test_websocket_integration())
    elif args.test == "breakeven":
        asyncio.run(test_breakeven_monitor())
