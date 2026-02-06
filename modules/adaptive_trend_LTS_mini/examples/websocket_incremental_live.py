"""Live WebSocket Integration: Incremental ATC with Real-Time Binance Data.

This example demonstrates real-time incremental ATC updates using live
Binance WebSocket price feeds (via ccxt.pro).

Features:
- Connects to Binance WebSocket for real-time kline (candlestick) data
- Updates incremental ATC on each completed bar
- Supports multiple symbols and timeframes
- Graceful shutdown and state persistence

Requirements:
    pip install ccxt

Usage:
    # Basic usage (BTC/USDT on 1m timeframe)
    python modules/adaptive_trend_LTS_mini/examples/websocket_incremental_live.py

    # Custom symbol and timeframe
    python modules/adaptive_trend_LTS_mini/examples/websocket_incremental_live.py --symbol ETH/USDT --timeframe 5m

    # Save state on exit
    python modules/adaptive_trend_LTS_mini/examples/websocket_incremental_live.py --save-state state.msgpack
"""

import argparse
import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

try:
    import ccxt.pro as ccxtpro
except ImportError:
    print("❌ ccxt.pro not installed. Install with: pip install ccxt")
    sys.exit(1)

from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import (
    AsyncIncrementalATC,
)
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig


class LiveIncrementalATCWebSocket:
    """Live incremental ATC with WebSocket integration."""

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        config: Optional[dict] = None,
        save_state_path: Optional[Path] = None,
    ):
        """Initialize live WebSocket ATC.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            timeframe: Timeframe (e.g., "1m", "5m", "15m")
            config: ATC configuration (uses defaults if None)
            save_state_path: Path to save state on exit
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.save_state_path = save_state_path

        # Create config
        if config is None:
            config = ATCConfig(
                ema_len=28,
                hma_len=28,
                wma_len=28,
                dema_len=28,
                lsma_len=28,
                kama_len=28,
                robustness="Medium",
                lambda_param=5.0,
                decay=0.005,
                cutout=100,
                use_rust_backend=False,
                use_o1_mas=True,
                use_rust_incremental=False,
            ).to_dict()

        self.config = config
        self.atc: Optional[AsyncIncrementalATC] = None

        # WebSocket exchange
        self.exchange = ccxtpro.binance({"enableRateLimit": True})

        # State
        self.is_initialized = False
        self.last_bar_time = None
        self.running = False
        self.update_count = 0

        # Signal tracking
        self.signal_history = []
        self.max_history = 100

    async def initialize_with_history(self, limit: int = 500):
        """Initialize ATC with historical data from exchange.

        Args:
            limit: Number of historical bars to fetch
        """
        print(f"\n[INIT] Fetching {limit} historical bars for {self.symbol} {self.timeframe}...")

        try:
            # Fetch historical OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(
                self.symbol, timeframe=self.timeframe, limit=limit
            )

            if not ohlcv or len(ohlcv) < 200:
                raise ValueError(f"Insufficient historical data: {len(ohlcv)} bars")

            # Extract close prices
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            prices = df["close"]

            print(f"[INIT] Initializing ATC with {len(prices)} bars...")
            self.atc = AsyncIncrementalATC(self.config)
            await self.atc.initialize(prices)

            self.is_initialized = True
            initial_signal = self.atc.state.get("signal", 0.0)
            print(f"[INIT] ✓ ATC initialized! Initial signal: {initial_signal:.4f}")
            print(f"[INIT] Last historical price: {prices.iloc[-1]:.2f}")

            # Track last bar time
            self.last_bar_time = ohlcv[-1][0]

        except Exception as e:
            print(f"[ERROR] Initialization failed: {e}")
            raise

    async def handle_price_update(self, kline: dict):
        """Handle incoming kline/candlestick data.

        Args:
            kline: OHLCV data from WebSocket
        """
        if not self.is_initialized:
            print("[WARN] Not initialized yet, skipping update")
            return

        # Extract data
        timestamp = kline[0]
        close_price = kline[4]

        # Only update on new/completed bars
        if self.last_bar_time is not None and timestamp <= self.last_bar_time:
            return  # Same bar, skip

        self.last_bar_time = timestamp

        # Update ATC
        try:
            signal = await self.atc.update(close_price)
            self.update_count += 1

            # Track signal
            self.signal_history.append(signal)
            if len(self.signal_history) > self.max_history:
                self.signal_history.pop(0)

            # Print update
            dt = datetime.fromtimestamp(timestamp / 1000)
            print(f"[UPDATE {self.update_count:04d}] {dt.strftime('%H:%M:%S')} | "
                  f"Price: {close_price:>8.2f} | Signal: {signal:>6.3f}")

        except Exception as e:
            print(f"[ERROR] Update failed: {e}")

    async def run(self):
        """Run WebSocket stream and process updates."""
        self.running = True

        try:
            # Initialize first
            await self.initialize_with_history()

            print(f"\n[STREAM] Starting WebSocket for {self.symbol} {self.timeframe}...")
            print("[STREAM] Press Ctrl+C to stop\n")

            # Watch kline stream
            while self.running:
                try:
                    kline = await self.exchange.watch_ohlcv(self.symbol, self.timeframe)

                    # Get latest bar
                    if kline and len(kline) > 0:
                        latest = kline[-1]
                        await self.handle_price_update(latest)

                except Exception as e:
                    print(f"[ERROR] WebSocket error: {e}")
                    await asyncio.sleep(1)

        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Fatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Clean shutdown."""
        self.running = False
        print("\n[SHUTDOWN] Closing WebSocket connection...")

        # Close exchange
        try:
            await self.exchange.close()
        except Exception as e:
            print(f"[WARN] Error closing exchange: {e}")

        # Save state if requested
        if self.save_state_path and self.atc:
            try:
                print(f"[SHUTDOWN] Saving state to {self.save_state_path}...")
                await self.atc.save_state(self.save_state_path)
                print("[SHUTDOWN] ✓ State saved")
            except Exception as e:
                print(f"[ERROR] Failed to save state: {e}")

        # Print summary
        print(f"\n[SUMMARY] Total updates: {self.update_count}")
        if self.signal_history:
            print(f"[SUMMARY] Signal range: [{min(self.signal_history):.4f}, "
                  f"{max(self.signal_history):.4f}]")
            print(f"[SUMMARY] Final signal: {self.signal_history[-1]:.4f}")

        print("\n[SHUTDOWN] Goodbye! 👋\n")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Live incremental ATC with Binance WebSocket"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Trading symbol (default: BTC/USDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1m",
        help="Timeframe (default: 1m)",
    )
    parser.add_argument(
        "--save-state",
        type=str,
        default=None,
        help="Path to save state on exit (optional)",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=500,
        help="Number of historical bars to fetch (default: 500)",
    )

    args = parser.parse_args()

    # Create client
    save_path = Path(args.save_state) if args.save_state else None
    client = LiveIncrementalATCWebSocket(
        symbol=args.symbol,
        timeframe=args.timeframe,
        save_state_path=save_path,
    )

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\n[SIGNAL] Received interrupt signal...")
        client.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    await client.run()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LIVE INCREMENTAL ATC WITH BINANCE WEBSOCKET")
    print("=" * 80)
    print("\nReal-time incremental ATC updates from live Binance data.")
    print("This connects to public Binance WebSocket (no API keys required).\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
