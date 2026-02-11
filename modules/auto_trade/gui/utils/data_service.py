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

    def __init__(self, mode: str = "DRY_RUN", settings_manager: Optional[Any] = None) -> None:
        """
        Initialize DataService.

        Args:
            mode: Operating mode ("DRY_RUN", "DEMO", or "PRODUCTION")
            settings_manager: Optional settings manager for TP/SL config (enables push missing TP/SL to Binance on refresh)
        """
        self.mode: str = mode
        self.settings_manager: Optional[Any] = settings_manager
        self.data_fetcher: Optional[Any] = None
        self.database_manager: Optional[Any] = None
        self.exchange_manager: Optional[Any] = None

        # Initialize MockPriceFeed (always available as fallback)
        self.mock_price_feed: Optional[MockPriceFeed] = self._initialize_mock_price_feed()

        # Load API credentials from environment
        self.api_key: str = os.getenv("BINANCE_API_KEY", "")
        self.api_secret: str = os.getenv("BINANCE_API_SECRET", "")
        self.testnet: bool = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

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

    def _reload_credentials(self) -> None:
        """Reload API credentials from .env (e.g. after user saves in Settings)."""
        try:
            from modules.auto_trade.gui.utils.credential_manager import CredentialManager

            cm = CredentialManager()
            exchange = "binance"
            creds = cm.load_credentials(exchange)
            self.api_key = (creds.get("api_key") or "").strip()
            self.api_secret = (creds.get("api_secret") or "").strip()
        except Exception as e:
            print(f"Warning: Could not reload credentials: {e}")

    def get_account_data(self) -> Optional[Dict]:
        try:
            if self.mode == "DRY_RUN":
                return self._get_dry_run_account_data()

            # Use latest credentials (e.g. after save in Settings)
            self._reload_credentials()

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
                    # Create BinanceClient once for all positions (efficient)
                    client = None
                    try:
                        from modules.auto_trade.execution.binance_client import BinanceClient
                        client = BinanceClient(
                            api_key=self.api_key,
                            api_secret=self.api_secret,
                            testnet=self.testnet,
                            dry_run=False,
                        )
                    except Exception as e:
                        print(f"[DataService] Could not create BinanceClient: {e}")

                    for pos in positions:
                        contracts = float(pos.get("contracts", 0))
                        if contracts == 0:
                            continue

                        # size_usdt contains the position value in USDT
                        margin_used += float(pos.get("size_usdt", 0))

                        # Calculate unrealized PnL with current mark price
                        symbol = pos.get("symbol", "")
                        entry_price = float(pos.get("entry_price", 0))
                        direction = pos.get("direction", "LONG").upper()

                        if client:
                            try:
                                ticker = client.exchange.fetch_ticker(symbol)
                                mark_price = float(ticker.get("info", {}).get("markPrice") or ticker.get("last") or entry_price)

                                if direction == "LONG":
                                    pos_pnl = (mark_price - entry_price) * abs(contracts)
                                else:
                                    pos_pnl = (entry_price - mark_price) * abs(contracts)

                                unrealized_pnl += pos_pnl
                            except Exception as e:
                                print(f"[DataService] Could not calc P&L for {symbol}: {e}")

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
                # Filter only positions with non-zero contracts (same as get_positions)
                if positions:
                    open_positions = len([p for p in positions if float(p.get("contracts", 0)) != 0])
                else:
                    open_positions = 0
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

                    from datetime import timezone

                    filtered = []
                    for signal in signals:
                        # Use final_score (or confidence); cast for type checker (ORM instance yields float)
                        raw = signal.final_score if signal.final_score is not None else signal.confidence
                        score = float(cast(Union[float, int], raw))
                        if score >= min_score:
                            signal_type = signal.signal_type.upper()
                            if signal_types is None or signal_type in signal_types:
                                created_at = signal.created_at
                                # SQLite strips timezone info; re-attach UTC so .timestamp() is correct
                                if created_at is not None and created_at.tzinfo is None:
                                    created_at = created_at.replace(tzinfo=timezone.utc)
                                filtered.append(
                                    {
                                        "symbol": signal.symbol,
                                        "signal": signal_type,
                                        "score": score,
                                        "time": created_at.strftime("%Y-%m-%d %H:%M")
                                        if created_at is not None
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
                        contracts = float(pos.get("size", 0))
                        current_price = price_feed.get_current_price(symbol)

                        # Notional size in USD for display (size * entry_price)
                        notional_usd = abs(contracts * entry_price)

                        if side == "LONG":
                            pnl = (current_price - entry_price) * contracts
                        else:
                            pnl = (entry_price - current_price) * contracts

                        # DRY_RUN mode also has TP/SL/BE in database
                        take_profit = pos.get("take_profit")
                        stop_loss = pos.get("stop_loss")
                        break_even = pos.get("break_even")

                        filtered_positions.append(
                            {
                                "symbol": symbol,
                                "side": side,
                                # For UI we treat size as notional (USD)
                                "size": notional_usd,
                                # Preserve contracts separately for advanced views if needed
                                "contracts": abs(contracts),
                                "entry_price": entry_price,
                                "current_price": current_price,
                                "pnl": pnl,
                                "take_profit": take_profit,
                                "stop_loss": stop_loss,
                                "break_even": break_even,
                            }
                        )

                    return filtered_positions
                except ImportError:
                    return []

            if self.data_fetcher and self.api_key and self.api_secret:
                positions = self.data_fetcher.fetch_binance_futures_positions(
                    api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet
                )
                print(f"[DataService] Fetched {len(positions) if positions else 0} positions from Binance")
                for p in (positions or []):
                    print(f"  - {p.get('symbol')}: {p.get('contracts')} contracts, direction={p.get('direction')}")

                # Create BinanceClient once for all positions (efficient)
                client = None
                try:
                    from modules.auto_trade.execution.binance_client import BinanceClient
                    client = BinanceClient(
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                        testnet=self.testnet,
                        dry_run=False,
                    )
                except Exception as e:
                    print(f"[DataService] Could not create BinanceClient for price fetching: {e}")

                filtered_positions = []
                for pos in positions:
                    # DataFetcher returns positions with: symbol, size_usdt, entry_price, direction, contracts
                    contracts = float(pos.get("contracts", 0))

                    if contracts == 0:
                        continue

                    side = pos.get("direction", "LONG").upper()
                    entry_price = float(pos.get("entry_price", 0))
                    symbol = pos.get("symbol", "")

                    # Futures position notional in USD (from data_fetcher if available, else derive)
                    size_usd = float(pos.get("size_usdt", 0)) if pos.get("size_usdt") is not None else 0.0
                    if not size_usd and entry_price:
                        size_usd = abs(contracts * entry_price)

                    # Leverage, liquidation price, margin (for Position Details GUI)
                    try:
                        leverage = int(pos.get("leverage", 1))
                    except (TypeError, ValueError):
                        leverage = 1
                    liq_price = pos.get("liquidation_price")
                    liquidation_price = float(liq_price) if liq_price is not None else None
                    margin_used = float(pos.get("margin_used", 0)) if pos.get("margin_used") is not None else (size_usd / leverage if leverage else 0.0)

                    # Fetch current mark price from exchange for accurate P&L
                    current_price = entry_price  # Fallback
                    if client:
                        try:
                            ticker = client.exchange.fetch_ticker(symbol)
                            # Use mark price for futures (more accurate than last price)
                            current_price = float(ticker.get("info", {}).get("markPrice") or ticker.get("last") or entry_price)
                        except Exception as e:
                            print(f"[DataService] Could not fetch mark price for {symbol}: {e}")

                    # Calculate PnL with current market price
                    pnl = 0.0
                    if side == "LONG":
                        pnl = (current_price - entry_price) * abs(contracts)
                    else:
                        pnl = (entry_price - current_price) * abs(contracts)

                    # Fetch TP/SL/BE from Binance and sync to DB
                    take_profit = None
                    stop_loss = None
                    break_even = None

                    if client and self.database_manager:
                        try:
                            from modules.auto_trade.gui.utils.tp_sl_sync import TPSLSyncService

                            # Fetch from Binance and sync to DB in one call
                            with self.database_manager.session_scope() as session:
                                result = TPSLSyncService.sync_position_tp_sl(
                                    client=client,
                                    session=session,
                                    symbol=symbol,
                                    side=side,
                                    entry_price=entry_price
                                )

                                take_profit = result.get("take_profit")
                                stop_loss = result.get("stop_loss")
                                break_even = result.get("break_even")

                                # If TP or SL missing on Binance, push them now (using config default_tp / default_sl)
                                sm = getattr(self, "settings_manager", None)
                                if (take_profit is None or stop_loss is None) and client and sm is not None and self.mode != "DRY_RUN":
                                    try:
                                        tp_sl_cfg = sm.get("tp_sl", {}) or {}
                                        default_tp = float(tp_sl_cfg.get("default_tp", 5.0))
                                        default_sl = float(tp_sl_cfg.get("default_sl", 2.5))
                                        if default_tp > 0 and default_sl > 0:
                                            pushed = TPSLSyncService.ensure_tp_sl_on_binance(
                                                client=client,
                                                symbol=symbol,
                                                side=side,
                                                entry_price=entry_price,
                                                default_tp_pct=default_tp,
                                                default_sl_pct=default_sl,
                                            )
                                            if pushed.get("take_profit") is not None:
                                                take_profit = pushed["take_profit"]
                                            if pushed.get("stop_loss") is not None:
                                                stop_loss = pushed["stop_loss"]
                                                break_even = TPSLSyncService.detect_break_even(entry_price, stop_loss, side)
                                            if take_profit is not None or stop_loss is not None:
                                                TPSLSyncService.sync_to_database(session, symbol, take_profit, stop_loss)
                                                print(f"[DataService] Pushed missing TP/SL for {symbol}: TP=${take_profit}, SL=${stop_loss}")
                                    except Exception as push_err:
                                        print(f"[DataService] Could not push TP/SL for {symbol}: {push_err}")

                                print(f"[DataService] Synced TP/SL for {symbol}: TP=${take_profit}, SL=${stop_loss}, BE=${break_even}")

                        except Exception as e:
                            print(f"[DataService] Could not sync TP/SL for {symbol}: {e}")

                            # Fallback to DB-only if sync fails
                            if self.database_manager:
                                try:
                                    from modules.auto_trade.database.models import Order
                                    with self.database_manager.session_scope() as session:
                                        db_orders = session.query(Order).filter(
                                            Order.symbol == symbol,
                                            Order.status == "OPEN"
                                        ).order_by(Order.created_at.desc()).all()

                                        if db_orders:
                                            order = db_orders[0]
                                            take_profit = order.take_profit
                                            stop_loss = order.stop_loss
                                            be_moved_flag = getattr(order, 'be_moved', False)
                                            if be_moved_flag is True and stop_loss is not None:
                                                break_even = stop_loss
                                            print(f"[DataService]   Fallback to DB: TP={take_profit}, SL={stop_loss}")
                                except Exception as db_err:
                                    print(f"[DataService]   DB fallback failed: {db_err}")

                    filtered_positions.append(
                        {
                            "symbol": pos.get("symbol", ""),
                            "side": side,
                            # For UI we treat size as notional (USD)
                            "size": size_usd,
                            "contracts": abs(contracts),
                            "entry_price": entry_price,
                            "current_price": current_price,
                            "pnl": pnl,
                            "take_profit": take_profit,
                            "stop_loss": stop_loss,
                            "break_even": break_even,
                            "leverage": leverage,
                            "margin_used": margin_used,
                            "liquidation_price": liquidation_price,
                        }
                    )

                print(f"[DataService] After filtering: {len(filtered_positions)} positions to display")
                return filtered_positions
            return []
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []
