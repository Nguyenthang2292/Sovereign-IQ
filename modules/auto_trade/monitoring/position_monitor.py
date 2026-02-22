"""
Position Monitor Module (WebSocket Version)

Monitors open positions in real-time using WebSocket, tracking P&L, drawdown, and position lifecycle.
Replaces polling with real-time User Data Stream from Binance Futures.

Key improvements over REST polling:
- Real-time position updates (<100ms vs 5s polling)
- Instant P&L tracking
- Immediate position close detection
- Lower API rate limit usage
"""

import asyncio
from modules.common.ui.logging import log_info, log_error, log_warn, log_debug, log_success, log_system
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

from modules.auto_trade.websocket.client import BinanceWebSocketClient



@dataclass
class PositionSnapshot:
    """Snapshot of a position at a point in time."""

    symbol: str
    side: str  # "long" or "short"
    position_amt: float
    entry_price: float
    mark_price: float
    liquidation_price: Optional[float]
    unrealized_pnl: float
    unrealized_pnl_percent: float
    margin_type: str
    leverage: int
    timestamp: datetime
    # Total notional size in quote currency (e.g. USDT). Used for GUI "Size" in USD.
    notional: float = 0.0
    # Margin used for this position (collateral). From ccxt "collateral" or "initialMargin".
    margin_used: float = 0.0

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
    Monitors open positions in real-time using WebSocket.

    Replaces REST polling with WebSocket User Data Stream for:
    - Instant position updates
    - Real-time P&L tracking
    - Immediate position close detection

    Example:
        >>> monitor = PositionMonitor(ws_client, max_positions=1)
        >>> monitor.add_callback(on_position_update)
        >>> await monitor.start()
    """

    def __init__(
        self,
        ws_client: BinanceWebSocketClient,
        max_positions: int = 1,
        min_pnl_change_percent: float = 1.0,
    ):
        """
        Initialize PositionMonitor.

        Args:
            ws_client: WebSocket client instance
            max_positions: Maximum allowed open positions (default: 1)
            min_pnl_change_percent: Minimum P&L change to log (default: 1.0%)
        """
        self.ws_client = ws_client
        self.max_positions = max_positions
        self.min_pnl_change_percent = min_pnl_change_percent

        self._running = False
        self._callbacks: List[Callable[[PositionSnapshot], None]] = []
        self._last_positions: Dict[str, PositionSnapshot] = {}  # symbol -> last snapshot

        log_info(f"PositionMonitor initialized (max_positions={max_positions}, WebSocket mode)")

    def add_callback(self, callback: Callable[[PositionSnapshot], None]):
        """
        Add a callback to be called when position updates.

        Args:
            callback: Function that takes PositionSnapshot as argument
        """
        self._callbacks.append(callback)
        log_info(f"Added position callback: {callback.__name__}")

    async def start(self):
        """
        Start monitoring positions via WebSocket.
        """
        if self._running:
            log_warn("PositionMonitor is already running")
            return

        self._running = True

        # Fetch initial positions via REST
        initial_positions = await self.ws_client.get_initial_positions()

        if initial_positions:
            log_info(f"Loaded {len(initial_positions)} initial positions")
            self._process_positions_update(initial_positions)

        # Register WebSocket callback
        self.ws_client.on_position_update(self._handle_ws_position_update)

        log_info("✅ PositionMonitor started (WebSocket mode)")

    async def stop(self):
        """Stop monitoring positions."""
        if not self._running:
            return

        self._running = False
        log_info("⏹️  PositionMonitor stopped")

    def _handle_ws_position_update(self, positions: List[dict]):
        """
        Handle WebSocket position update.

        This is called by WebSocket client when positions change.

        Args:
            positions: List of position dicts from ccxt.pro
        """
        if not self._running:
            return

        try:
            self._process_positions_update(positions)
        except Exception as e:
            log_error(f"Error handling WebSocket position update: {e}", exc_info=True)

    def _process_positions_update(self, positions: List[dict]):
        """
        Process position updates.

        Args:
            positions: List of position dicts from ccxt.pro
        """
        # Check max positions limit
        if len(positions) > self.max_positions:
            log_error(f"⚠️  Too many positions! Found {len(positions)}, max allowed: {self.max_positions}")

        # Track current symbols
        current_symbols = set()

        # Process each position
        for position_data in positions:
            try:
                snapshot = self._parse_position(position_data)
                current_symbols.add(snapshot.symbol)

                # Calculate P&L percentage
                snapshot.unrealized_pnl_percent = self._calculate_pnl_percent(snapshot)

                # Log if position changed significantly
                if self._has_significant_change(snapshot):
                    log_info(
                        f"📊 {snapshot.symbol} {snapshot.side.upper()}: "
                        f"PnL=${snapshot.unrealized_pnl:.2f} ({snapshot.unrealized_pnl_percent:+.2f}%), "
                        f"Entry=${snapshot.entry_price:.2f}, Mark=${snapshot.mark_price:.2f}"
                    )

                # Update last position
                self._last_positions[snapshot.symbol] = snapshot

                # Trigger callbacks
                for callback in self._callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            asyncio.create_task(callback(snapshot))
                        else:
                            callback(snapshot)
                    except Exception as e:
                        log_error(f"Error in position callback {callback.__name__}: {e}")

            except Exception as e:
                log_error(f"Error processing position: {e}", exc_info=True)

        # Check for closed positions
        closed_symbols = set(self._last_positions.keys()) - current_symbols
        for symbol in closed_symbols:
            log_info(f"Position closed: {symbol}")
            del self._last_positions[symbol]

    def _parse_position(self, data: dict) -> PositionSnapshot:
        """
        Parse position data from ccxt.pro.

        ccxt.pro normalizes position data into unified format:
        {
            'symbol': 'BTC/USDT',
            'contracts': 0.001,
            'contractSize': 1,
            'side': 'long',  # or 'short'
            'notional': 34.5,
            'leverage': 10,
            'unrealizedPnl': -0.5,
            'collateral': 3.5,
            'marginType': 'isolated',  # or 'cross'
            'entryPrice': 35000,
            'markPrice': 34500,
            'liquidationPrice': 31500,
            'percentage': -1.43,  # P&L percentage
            'timestamp': 1623456789000,
            'info': {...}  # Raw exchange data
        }
        """
        symbol = data.get("symbol", "")
        contracts = float(data.get("contracts", 0))
        side = data.get("side", "long").lower()
        entry_price = float(data.get("entryPrice", 0))
        mark_price = float(data.get("markPrice", 0))
        liquidation_price = data.get("liquidationPrice")
        unrealized_pnl = float(data.get("unrealizedPnl", 0))
        margin_type = data.get("marginType", "cross").lower()
        # Notional value in quote currency (e.g. USDT)
        notional = float(data.get("notional", 0))
        # Margin used (collateral) from ccxt / Binance initialMargin (Binance positionRisk has initialMargin)
        raw_margin = (
            data.get("collateral")
            or data.get("initialMargin")
            or data.get("margin")
            or ((data.get("info") or {}).get("initialMargin"))
        )
        try:
            margin_used = abs(float(raw_margin)) if raw_margin is not None else 0.0
        except (TypeError, ValueError):
            margin_used = 0.0
        # Leverage: from API or derive from notional/initialMargin (Binance positionRisk doesn't return leverage)
        leverage_raw = data.get("leverage")
        if leverage_raw is not None:
            try:
                leverage = int(leverage_raw)
            except (TypeError, ValueError):
                leverage = 1
        else:
            leverage = 1
        if leverage <= 1 and margin_used > 0 and notional and abs(notional) > 0:
            leverage = max(1, int(round(abs(notional) / margin_used)))

        return PositionSnapshot(
            symbol=symbol,
            side=side,
            position_amt=abs(contracts),
            entry_price=entry_price,
            mark_price=mark_price,
            liquidation_price=float(liquidation_price) if liquidation_price else None,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_percent=0.0,  # Calculated below
            margin_type=margin_type,
            leverage=leverage,
            timestamp=datetime.now(),
            notional=abs(notional),
            margin_used=abs(margin_used),
        )

    def _calculate_pnl_percent(self, snapshot: PositionSnapshot) -> float:
        """
        Calculate P&L percentage.

        Args:
            snapshot: Position snapshot

        Returns:
            P&L percentage
        """
        if snapshot.entry_price <= 0:
            return 0.0

        if snapshot.side == "long":
            return ((snapshot.mark_price - snapshot.entry_price) / snapshot.entry_price) * 100
        else:  # short
            return ((snapshot.entry_price - snapshot.mark_price) / snapshot.entry_price) * 100

    def _has_significant_change(self, snapshot: PositionSnapshot) -> bool:
        """
        Check if position has changed significantly since last update.

        Args:
            snapshot: Current position snapshot

        Returns:
            True if significant change detected
        """
        last = self._last_positions.get(snapshot.symbol)

        if not last:
            return True  # New position

        # Check if P&L changed by threshold
        pnl_diff = abs(snapshot.unrealized_pnl_percent - last.unrealized_pnl_percent)
        return pnl_diff >= self.min_pnl_change_percent

    def get_open_positions(self) -> List[PositionSnapshot]:
        """
        Get current open positions.

        Returns:
            List of position snapshots
        """
        return list(self._last_positions.values())

    def get_position(self, symbol: str) -> Optional[PositionSnapshot]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position snapshot or None if not found
        """
        return self._last_positions.get(symbol)

    @property
    def position_count(self) -> int:
        """Get number of open positions."""
        return len(self._last_positions)

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running
