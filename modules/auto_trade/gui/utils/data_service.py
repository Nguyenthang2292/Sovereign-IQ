"""
Data Service Module

Unified data access layer that abstracts exchange data fetching,
database operations, and mock data for dry-run mode.
"""

import os
from typing import Any, Dict, List, Optional, Union, cast

# Local imports
from modules.auto_trade.gui.utils.mock_price_feed import MockPriceFeed

# Module imports (lazy loaded)
# from modules.common.core.data_fetcher import DataFetcher
# from modules.common.core.exchange_manager import ExchangeManager
# from modules.auto_trade.database import get_db_manager


class DataService:
    """
    Unified data service for managing exchange and database operations.

    Supports three modes:
    - DRY_RUN: Simulated trading with mock data
    - DEMO: Testnet trading with real API
    - PRODUCTION: Live trading with real API
    """

    def __init__(self, mode: str = "DRY_RUN"):
        """
        Initialize DataService.

        Args:
            mode: Operating mode ("DRY_RUN", "DEMO", or "PRODUCTION")
        """
        self.mode = mode
        self.data_fetcher: Optional[Any] = None
        self.database_manager: Optional[Any] = None
        self.exchange_manager: Optional[Any] = None

        # Initialize MockPriceFeed (always available as fallback)
        self.mock_price_feed = self._initialize_mock_price_feed()

        # Load API credentials from environment
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_API_SECRET", "")
        self.testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

        # Initialize exchange components only if not DRY_RUN
        if mode != "DRY_RUN":
            self._initialize_exchange_components()

        # Initialize database manager
        self._initialize_database_manager()

    def _initialize_mock_price_feed(self) -> Optional[MockPriceFeed]:
        """
        Initialize MockPriceFeed (always available as fallback).

        Returns:
            MockPriceFeed instance or None if initialization fails
        """
        try:
            return MockPriceFeed()
        except Exception as e:
            print(f"Warning: Could not initialize MockPriceFeed: {e}")
            return None

    def _initialize_exchange_components(self) -> None:
        """Initialize DataFetcher and ExchangeManager for non-DRY_RUN modes."""
        try:
            from modules.common.core.data_fetcher import DataFetcher
            from modules.common.core.exchange_manager import ExchangeManager

            exchange_manager = ExchangeManager(
                api_key=self.api_key or None,
                api_secret=self.api_secret or None,
                testnet=self.testnet,
            )
            self.exchange_manager = exchange_manager
            self.data_fetcher = DataFetcher(exchange_manager=exchange_manager)
        except Exception as e:
            print(f"Warning: Could not initialize DataFetcher: {e}")

    def _initialize_database_manager(self) -> None:
        """Initialize DatabaseManager for storing signals and trades."""
        try:
            from modules.auto_trade.database import get_db_manager

            # Use get_db_manager() which handles singleton and default path
            self.database_manager = get_db_manager()
        except Exception as e:
            print(f"Warning: Could not initialize DatabaseManager: {e}")

    def _get_mock_price_feed(self) -> MockPriceFeed:
        """
        Get MockPriceFeed instance (creates if not exists).

        Returns:
            MockPriceFeed instance
        """
        if self.mock_price_feed is None:
            self.mock_price_feed = MockPriceFeed()
        return self.mock_price_feed

    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.

        In DRY_RUN mode, uses MockPriceFeed for simulated prices.
        In other modes, fetches from exchange via DataFetcher.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT" or "BTCUSDT")

        Returns:
            Current price as float
        """
        try:
            if self.mode == "DRY_RUN":
                # Use centralized mock price feed
                return self._get_mock_price_feed().get_current_price(symbol)

            # For non-DRY_RUN modes, fetch from exchange
            if self.data_fetcher:
                # Normalize symbol format (BTC/USDT -> BTCUSDT for API calls)
                normalized_symbol = symbol.replace("/", "")
                ticker = self.data_fetcher.fetch_ticker(normalized_symbol)
                if ticker and "last" in ticker:
                    return float(ticker["last"])

            # Fallback to mock prices if exchange fetch fails
            return self._get_mock_price_feed().get_current_price(symbol)

        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
            # Return a safe default from centralized mock prices
            return self._get_mock_price_feed().get_current_price(symbol)

    def get_account_data(self) -> Optional[Dict]:
        try:
            if self.mode == "DRY_RUN":
                return self._get_dry_run_account_data()

            if self.data_fetcher and self.api_key and self.api_secret:
                # Fetch balance using DataFetcher
                balance = self.data_fetcher.fetch_binance_account_balance(
                    api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet, currency="USDT"
                )

                # Fetch positions using DataFetcher
                positions = self.data_fetcher.fetch_binance_futures_positions(
                    api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet
                )

                # Calculate margin used and unrealized PnL from positions
                margin_used = 0.0
                unrealized_pnl = 0.0

                if positions:
                    for pos in positions:
                        # size_usdt contains the position value in USDT
                        margin_used += float(pos.get("size_usdt", 0))
                        # Note: unrealized PnL calculation would require current prices
                        # For now, we'll leave it at 0.0

                return {
                    "balance": balance if balance else 0.0,
                    "available": balance if balance else 0.0,  # Simplified
                    "margin_used": margin_used,
                    "unrealized_pnl": unrealized_pnl,
                    "daily_pnl": 0.0,
                    "daily_pnl_percent": 0.0,
                }
            return self._get_demo_account_data()
        except Exception as e:
            print(f"Error fetching account data: {e}")
            return self._get_demo_account_data()

    def _get_demo_account_data(self) -> Dict:
        return {
            "balance": 1000.0,
            "available": 1000.0,
            "margin_used": 0.0,
            "unrealized_pnl": 0.0,
            "daily_pnl": 0.0,
            "daily_pnl_percent": 0.0,
        }

    def _get_dry_run_account_data(self) -> Dict:
        return {
            "balance": 10000.0,
            "available": 10000.0,
            "margin_used": 0.0,
            "unrealized_pnl": 0.0,
            "daily_pnl": 0.0,
            "daily_pnl_percent": 0.0,
        }

    def get_quick_stats(self) -> Optional[Dict]:
        try:
            open_positions = 0
            if self.mode != "DRY_RUN" and self.data_fetcher and self.api_key and self.api_secret:
                positions = self.data_fetcher.fetch_binance_futures_positions(
                    api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet
                )
                open_positions = len(positions) if positions else 0
            elif self.mode == "DRY_RUN":
                try:
                    from modules.auto_trade.gui.utils.dry_run_db import DryRunDB

                    db = DryRunDB()
                    positions = db.get_open_positions()
                    open_positions = len(positions)
                except ImportError:
                    pass

            today_trades = 0
            win_rate = 0.0

            if self.database_manager:
                try:
                    # Use database query functions
                    from modules.auto_trade.database import get_daily_stats

                    with self.database_manager.session_scope() as session:
                        daily_stats = get_daily_stats(session)
                        if daily_stats:
                            today_stats = daily_stats[0]
                            today_trades = today_stats.get("total_trades", 0)
                            win_rate = today_stats.get("win_rate", 0.0)
                except Exception as e:
                    print(f"Warning: Could not fetch database stats: {e}")

            return {
                "open_positions": open_positions,
                "today_trades": today_trades,
                "win_rate": win_rate,
                "mode": self.mode,
            }
        except Exception as e:
            print(f"Error fetching stats: {e}")
            return {"open_positions": 0, "today_trades": 0, "win_rate": 0.0, "mode": self.mode}

    def get_signals(self, min_score: float = 0.7, signal_types: Optional[List[str]] = None) -> List[Dict]:
        try:
            if self.database_manager:
                from modules.auto_trade.database import get_recent_signals

                with self.database_manager.session_scope() as session:
                    signals = get_recent_signals(session, limit=100)

                    filtered = []
                    for signal in signals:
                        # Use final_score (or confidence); cast for type checker (ORM instance yields float)
                        raw = signal.final_score if signal.final_score is not None else signal.confidence
                        score = float(cast(Union[float, int], raw))
                        if score >= min_score:
                            signal_type = signal.signal_type.upper()
                            if signal_types is None or signal_type in signal_types:
                                created_at = signal.created_at
                                filtered.append(
                                    {
                                        "symbol": signal.symbol,
                                        "signal": signal_type,
                                        "score": score,
                                        "time": signal.created_at.strftime("%Y-%m-%d %H:%M")
                                        if signal.created_at is not None
                                        else "",
                                        # Extra fields for freshness filtering (< 5 minutes)
                                        "created_at": created_at.isoformat() if created_at is not None else "",
                                        "created_at_ts": float(created_at.timestamp()) if created_at is not None else 0.0,
                                    }
                                )
                    if filtered:
                        return filtered
                    return self._get_demo_signals() if self.mode == "DRY_RUN" else []
            return self._get_demo_signals() if self.mode == "DRY_RUN" else []
        except Exception as e:
            print(f"Error fetching signals: {e}")
            return self._get_demo_signals()

    def _get_demo_signals(self) -> List[Dict]:
        return [
            {
                "symbol": "BTCUSDT",
                "signal": "LONG",
                "score": 0.85,
                "time": "2024-01-15 10:30",
                "created_at": "",
                "created_at_ts": 0.0,
            },
            {
                "symbol": "ETHUSDT",
                "signal": "SHORT",
                "score": 0.72,
                "time": "2024-01-15 10:25",
                "created_at": "",
                "created_at_ts": 0.0,
            },
            {
                "symbol": "SOLUSDT",
                "signal": "NEUTRAL",
                "score": 0.45,
                "time": "2024-01-15 10:20",
                "created_at": "",
                "created_at_ts": 0.0,
            },
        ]

    def get_positions(self) -> List[Dict]:
        try:
            if self.mode == "DRY_RUN":
                try:
                    from modules.auto_trade.gui.utils.dry_run_db import DryRunDB

                    db = DryRunDB()
                    price_feed = self._get_mock_price_feed()
                    positions = db.get_open_positions()

                    filtered_positions = []
                    for pos in positions:
                        symbol = pos.get("symbol", "")
                        side = pos.get("side", "LONG")
                        entry_price = float(pos.get("entry_price", 0))
                        size = float(pos.get("size", 0))
                        current_price = price_feed.get_current_price(symbol)

                        if side == "LONG":
                            pnl = (current_price - entry_price) * size
                        else:
                            pnl = (entry_price - current_price) * size

                        filtered_positions.append(
                            {
                                "symbol": symbol,
                                "side": side,
                                "size": size,
                                "entry_price": entry_price,
                                "current_price": current_price,
                                "pnl": pnl,
                            }
                        )

                    return filtered_positions
                except ImportError:
                    return []

            if self.data_fetcher and self.api_key and self.api_secret:
                positions = self.data_fetcher.fetch_binance_futures_positions(
                    api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet
                )

                filtered_positions = []
                for pos in positions:
                    # DataFetcher returns positions with: symbol, size_usdt, entry_price, direction, contracts
                    size_usdt = float(pos.get("size_usdt", 0))
                    contracts = float(pos.get("contracts", 0))

                    if contracts == 0:
                        continue

                    side = pos.get("direction", "LONG").upper()
                    entry_price = float(pos.get("entry_price", 0))

                    # For current price, we'd need to fetch from market data
                    # For now, use entry price as placeholder
                    current_price = entry_price

                    # Calculate PnL (simplified - would need current market price for accuracy)
                    pnl = 0.0
                    if side == "LONG":
                        pnl = (current_price - entry_price) * abs(contracts)
                    else:
                        pnl = (entry_price - current_price) * abs(contracts)

                    filtered_positions.append(
                        {
                            "symbol": pos.get("symbol", ""),
                            "side": side,
                            "size": abs(contracts),
                            "entry_price": entry_price,
                            "current_price": current_price,
                            "pnl": pnl,
                        }
                    )

                return filtered_positions
            return []
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []
