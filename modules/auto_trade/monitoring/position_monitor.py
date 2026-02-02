"""
Position Monitor Module

Monitors open positions in real-time, tracking P&L, drawdown, and position lifecycle.
Polls Binance Futures positions every 5 seconds (configurable).
"""

import time
from dataclasses import dataclass
from datetime import datetime
from threading import Event, Thread
from typing import Callable, List, Optional

from modules.common.core.data_fetcher import DataFetcher
from modules.common.ui.logging import log_error, log_info, log_warn


@dataclass
class PositionSnapshot:
    """Snapshot of a position at a point in time."""

    symbol: str
    side: str  # "LONG" or "SHORT"
    position_amt: float
    entry_price: float
    mark_price: float
    liquidation_price: Optional[float]
    unrealized_pnl: float
    unrealized_pnl_percent: float
    margin_type: str
    leverage: int
    timestamp: datetime

    @property
    def is_profitable(self) -> bool:
        """Check if position is in profit."""
        return self.unrealized_pnl > 0

    @property
    def drawdown_percent(self) -> float:
        """Calculate drawdown as negative percentage."""
        if self.unrealized_pnl_percent < 0:
            return abs(self.unrealized_pnl_percent)
        return 0.0


class PositionMonitor:
    """
    Monitors open positions in real-time.

    Example:
        >>> monitor = PositionMonitor(data_fetcher, api_key, api_secret)
        >>> monitor.add_callback(on_position_update)
        >>> monitor.start()
    """

    def __init__(
        self,
        data_fetcher: DataFetcher,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        poll_interval: float = 5.0,
        max_positions: int = 1,
    ):
        """
        Initialize PositionMonitor.

        Args:
            data_fetcher: DataFetcher instance
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet if True
            poll_interval: Polling interval in seconds (default: 5.0)
            max_positions: Maximum allowed open positions (default: 1)
        """
        self.data_fetcher = data_fetcher
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.poll_interval = poll_interval
        self.max_positions = max_positions

        self._running = False
        self._stop_event = Event()
        self._monitor_thread: Optional[Thread] = None
        self._callbacks: List[Callable[[PositionSnapshot], None]] = []
        self._last_position: Optional[PositionSnapshot] = None

        log_info(f"PositionMonitor initialized (poll_interval={poll_interval}s)")

    def add_callback(self, callback: Callable[[PositionSnapshot], None]):
        """
        Add a callback to be called when position updates.

        Args:
            callback: Function that takes PositionSnapshot as argument
        """
        self._callbacks.append(callback)
        log_info(f"Added position callback: {callback.__name__}")

    def start(self):
        """Start monitoring positions."""
        if self._running:
            log_warn("PositionMonitor is already running")
            return

        self._running = True
        self._stop_event.clear()
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        log_info("✅ PositionMonitor started")

    def stop(self):
        """Stop monitoring positions."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        log_info("⏹️ PositionMonitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        log_info("Position monitoring loop started")

        while self._running and not self._stop_event.is_set():
            try:
                positions = self.fetch_positions()

                if positions:
                    # Check max positions limit
                    if len(positions) > self.max_positions:
                        log_error(f"⚠️ Too many positions! Found {len(positions)}, max allowed: {self.max_positions}")

                    # Process each position
                    for position in positions:
                        self._process_position(position)
                else:
                    # No positions open
                    if self._last_position:
                        log_info("No open positions (position closed)")
                        self._last_position = None

            except Exception as e:
                log_error(f"Error in position monitoring loop: {e}", exc_info=True)

            # Wait for next poll interval
            self._stop_event.wait(timeout=self.poll_interval)

        log_info("Position monitoring loop stopped")

    def fetch_positions(self) -> List[dict]:
        """
        Fetch current open positions from Binance.

        Returns:
            List of position dicts, empty list if no positions
        """
        try:
            positions = self.data_fetcher.fetch_binance_futures_positions(
                api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet
            )

            if not positions:
                return []

            # Filter for non-zero positions
            open_positions = [p for p in positions if float(p.get("positionAmt", 0)) != 0]

            return open_positions

        except Exception as e:
            log_error(f"Failed to fetch positions: {e}", exc_info=True)
            return []

    def _process_position(self, position_data: dict):
        """
        Process a position update.

        Args:
            position_data: Position data from Binance API
        """
        try:
            # Parse position data
            snapshot = self._parse_position(position_data)

            # Calculate P&L percentage
            entry_price = snapshot.entry_price
            mark_price = snapshot.mark_price
            position_amt = snapshot.position_amt

            if entry_price > 0:
                if position_amt > 0:  # LONG
                    pnl_percent = ((mark_price - entry_price) / entry_price) * 100
                else:  # SHORT
                    pnl_percent = ((entry_price - mark_price) / entry_price) * 100

                snapshot.unrealized_pnl_percent = pnl_percent

            # Log if position changed significantly
            if self._has_significant_change(snapshot):
                log_info(
                    f"📊 {snapshot.symbol} {snapshot.side}: "
                    f"PnL=${snapshot.unrealized_pnl:.2f} ({snapshot.unrealized_pnl_percent:+.2f}%), "
                    f"Entry=${snapshot.entry_price:.2f}, Mark=${snapshot.mark_price:.2f}"
                )

            # Update last position
            self._last_position = snapshot

            # Trigger callbacks
            for callback in self._callbacks:
                try:
                    callback(snapshot)
                except Exception as e:
                    log_error(f"Error in position callback {callback.__name__}: {e}")

        except Exception as e:
            log_error(f"Error processing position: {e}", exc_info=True)

    def _parse_position(self, data: dict) -> PositionSnapshot:
        """Parse position data from Binance API."""
        position_amt = float(data.get("positionAmt", 0))
        side = "LONG" if position_amt > 0 else "SHORT"

        return PositionSnapshot(
            symbol=data.get("symbol", ""),
            side=side,
            position_amt=abs(position_amt),
            entry_price=float(data.get("entryPrice", 0)),
            mark_price=float(data.get("markPrice", 0)),
            liquidation_price=float(data.get("liquidationPrice", 0)) or None,
            unrealized_pnl=float(data.get("unRealizedProfit", 0)),
            unrealized_pnl_percent=0.0,  # Calculated in _process_position
            margin_type=data.get("marginType", "cross"),
            leverage=int(data.get("leverage", 1)),
            timestamp=datetime.now(),
        )

    def _has_significant_change(self, snapshot: PositionSnapshot) -> bool:
        """Check if position has changed significantly since last update."""
        if not self._last_position:
            return True

        # Check if PnL changed by more than 1%
        pnl_diff = abs(snapshot.unrealized_pnl_percent - self._last_position.unrealized_pnl_percent)
        return pnl_diff > 1.0

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    @property
    def current_position(self) -> Optional[PositionSnapshot]:
        """Get current position snapshot."""
        return self._last_position
