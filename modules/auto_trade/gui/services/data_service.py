"""
Data Service Module

Unified data access layer that abstracts exchange data fetching,
database operations via DynamoDB, and mock data for dry-run mode.
"""

import os
from datetime import datetime as _datetime
from typing import Any, Dict, List, Optional, TypedDict, Union, cast

from modules.common.ui.logging import log_error, log_info, log_warn

# Cooldown (seconds) before pushing TP/SL again for the same symbol (avoids duplicate orders)
TP_SL_PUSH_COOLDOWN_SEC = 300  # 5 min cooldown to avoid duplicate conditional orders

# Local imports
from modules.auto_trade.gui.utils.mock_price_feed import MockPriceFeed


class TpSlResult(TypedDict):
    take_profit: Optional[float]
    stop_loss: Optional[float]
    break_even: Optional[float]


class DataService:
    """
    Unified data service for managing exchange and database operations.

    Supports three modes:
    - DRY_RUN: Simulated trading with mock data
    - DEMO: Testnet trading with real API
    - PRODUCTION: Live trading with real API
    """

    def __init__(
        self, mode: str = "DRY_RUN", settings_manager: Optional[Any] = None, event_bus: Optional[Any] = None
    ) -> None:
        """
        Initialize DataService.

        Args:
            mode: Operating mode ("DRY_RUN", "DEMO", or "PRODUCTION")
            settings_manager: Optional settings manager for TP/SL config
        """
        self.mode: str = mode
        self.settings_manager: Optional[Any] = settings_manager
        self.data_fetcher: Optional[Any] = None
        self.repo_context: Optional[Any] = None
        self.exchange_manager: Optional[Any] = None
        self._tp_sl_push_last: Dict[str, float] = {}  # symbol -> monotonic time of last push

        self._binance_client = None
        self._tpsl_cache: Dict[str, TpSlResult] = {}
        self._tpsl_cache_time: Dict[str, float] = {}
        self._credentials_loaded = False
        self._event_bus: Optional[Any] = None

        # Initialize MockPriceFeed (always available as fallback)
        self.mock_price_feed: Optional[MockPriceFeed] = self._initialize_mock_price_feed()

        # Load API credentials from environment
        self.api_key: str = os.getenv("BINANCE_API_KEY", "")
        self.api_secret: str = os.getenv("BINANCE_API_SECRET", "")
        self.testnet: bool = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

        # Initialize exchange components only if not DRY_RUN
        if mode != "DRY_RUN":
            self._initialize_exchange_components()

        # Initialize repository context
        self._initialize_database_manager()

        self.set_event_bus(event_bus)

    def set_event_bus(self, event_bus: Optional[Any]) -> None:
        """Attach event bus and subscribe to settings save events for credential cache invalidation."""
        if event_bus is None:
            return

        self._event_bus = event_bus
        try:
            from modules.auto_trade.monitoring.event_system import EventType

            event_bus.subscribe(EventType.SETTINGS_SAVED, self._on_settings_saved)
        except Exception as error:
            log_warn(f"Could not subscribe to SETTINGS_SAVED event: {error}")

    def _on_settings_saved(self, event: Any) -> None:
        """Invalidate cached credentials/client when settings are saved."""
        self._credentials_loaded = False
        self._binance_client = None
        log_info("[DataService] Received SETTINGS_SAVED event, invalidated credential/client cache")

    def _initialize_mock_price_feed(self) -> Optional[MockPriceFeed]:
        """Initialize MockPriceFeed (always available as fallback)."""
        try:
            return MockPriceFeed()
        except Exception as e:
            log_warn(f"Could not initialize MockPriceFeed: {e}")
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
            log_warn(f"Could not initialize DataFetcher: {e}")

    def _initialize_database_manager(self) -> None:
        """Initialize RepositoryContext for storing signals and trades."""
        try:
            from modules.auto_trade.database.repository.context import RepositoryContext

            self.repo_context = RepositoryContext.from_env()
        except Exception as e:
            log_warn(f"Could not initialize RepositoryContext: {e}")

    def _get_mock_price_feed(self) -> MockPriceFeed:
        """Get MockPriceFeed instance (creates if not exists)."""
        if self.mock_price_feed is None:
            self.mock_price_feed = MockPriceFeed()
        return self.mock_price_feed

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            if self.mode == "DRY_RUN":
                return self._get_mock_price_feed().get_current_price(symbol)

            if self.data_fetcher:
                normalized_symbol = symbol.replace("/", "")
                ticker = self.data_fetcher.fetch_ticker(normalized_symbol)
                if ticker and "last" in ticker:
                    return float(ticker["last"])

            return self._get_mock_price_feed().get_current_price(symbol)

        except Exception as e:
            log_error(f"Error fetching current price for {symbol}: {e}")
            return self._get_mock_price_feed().get_current_price(symbol)

    def _reload_credentials(self) -> None:
        """Reload API credentials from .env (e.g. after user saves in Settings)."""
        if self._credentials_loaded:
            return

        try:
            from modules.auto_trade.gui.services.credential_manager import CredentialManager

            cm = CredentialManager()
            exchange = "binance"
            creds = cm.load_credentials(exchange)
            new_key = (creds.get("api_key") or "").strip()
            new_secret = (creds.get("api_secret") or "").strip()

            if new_key != self.api_key or new_secret != self.api_secret:
                self.api_key = new_key
                self.api_secret = new_secret
                self._binance_client = None  # Invalidate cache

            self._credentials_loaded = True
        except Exception as e:
            log_warn(f"Could not reload credentials: {e}")

    def _get_or_create_client(self):
        """Get cached BinanceClient or create a new one."""
        if self._binance_client is not None:
            return self._binance_client

        try:
            from modules.auto_trade.execution.binance_client import BinanceClient

            if not self.api_key or not self.api_secret:
                return None

            self._binance_client = BinanceClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
                dry_run=False,
            )
            return self._binance_client
        except Exception as e:
            log_error(f"[DataService] Could not create BinanceClient: {e}")
            return None

    def get_cached_tpsl(self, symbol: str, ttl_seconds: int = 30) -> TpSlResult:
        import time

        now = time.monotonic()

        # Check cache
        if symbol in self._tpsl_cache and symbol in self._tpsl_cache_time:
            if now - self._tpsl_cache_time[symbol] < ttl_seconds:
                return self._tpsl_cache[symbol]

        # Cache miss or expired, fetch it
        client = self._get_or_create_client()
        result: TpSlResult = {"take_profit": None, "stop_loss": None, "break_even": None}

        if client and self.repo_context:
            try:
                from modules.auto_trade.gui.services.tp_sl_sync import TPSLSyncService

                side = None
                entry_price = None
                db_orders = self.repo_context.orders.get_open_positions(symbol=symbol)
                if not db_orders:
                    symbol_normalized = symbol.replace("/", "").split(":")[0]
                    db_orders = self.repo_context.orders.get_open_positions(symbol=symbol_normalized)

                if db_orders:
                    order = db_orders[0]
                    raw_side = order.get("side")
                    raw_entry_price = order.get("entry_price")
                    if raw_side is not None:
                        side = str(raw_side)
                    if raw_entry_price is not None:
                        entry_price = float(raw_entry_price)

                if side and entry_price is not None:
                    sync_result = TPSLSyncService.sync_position_tp_sl(
                        client=client,
                        repo_context=self.repo_context,
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                    )
                    result["take_profit"] = sync_result.get("take_profit")
                    result["stop_loss"] = sync_result.get("stop_loss")
                    result["break_even"] = sync_result.get("break_even")
                else:
                    take_profit, stop_loss, _ = TPSLSyncService.fetch_tp_sl_from_binance(client=client, symbol=symbol)
                    result["take_profit"] = take_profit
                    result["stop_loss"] = stop_loss
            except Exception as e:
                log_error(f"[DataService] Could not sync TP/SL for {symbol}: {e}")

                # Fallback to DynamoDB
                try:
                    db_orders = self.repo_context.orders.get_open_positions(symbol=symbol)
                    if db_orders:
                        order = db_orders[0]
                        result["take_profit"] = order.get("take_profit")
                        result["stop_loss"] = order.get("stop_loss")
                        if order.get("be_moved") and result["stop_loss"] is not None:
                            result["break_even"] = result["stop_loss"]
                except Exception as db_err:
                    log_error(f"[DataService] DB fallback failed: {db_err}")

        # Save to cache
        self._tpsl_cache[symbol] = result
        self._tpsl_cache_time[symbol] = now

        return result

    def get_account_data(self) -> Optional[Dict]:
        try:
            if self.mode == "DRY_RUN":
                return self._get_dry_run_account_data()

            self._reload_credentials()

            if self.data_fetcher and self.api_key and self.api_secret:
                balance = self.data_fetcher.fetch_binance_account_balance(
                    api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet, currency="USDT"
                )

                positions = self.data_fetcher.fetch_binance_futures_positions(
                    api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet
                )

                margin_used = 0.0
                unrealized_pnl = 0.0

                if positions:
                    client = self._get_or_create_client()

                    for pos in positions:
                        contracts = float(pos.get("contracts", 0))
                        if contracts == 0:
                            continue

                        margin_used += float(pos.get("size_usdt", 0))

                        symbol = pos.get("symbol", "")
                        entry_price = float(pos.get("entry_price", 0))
                        direction = pos.get("direction", "LONG").upper()

                        if client:
                            try:
                                ticker = client.exchange.fetch_ticker(symbol)
                                mark_price = float(
                                    ticker.get("info", {}).get("markPrice") or ticker.get("last") or entry_price
                                )

                                if direction == "LONG":
                                    pos_pnl = (mark_price - entry_price) * abs(contracts)
                                else:
                                    pos_pnl = (entry_price - mark_price) * abs(contracts)

                                unrealized_pnl += pos_pnl
                            except Exception as e:
                                log_error(f"[DataService] Could not calc P&L for {symbol}: {e}")

                return {
                    "balance": balance if balance else 0.0,
                    "available": balance if balance else 0.0,
                    "margin_used": margin_used,
                    "unrealized_pnl": unrealized_pnl,
                    "daily_pnl": 0.0,
                    "daily_pnl_percent": 0.0,
                }
            return self._get_demo_account_data()
        except Exception as e:
            log_error(f"Error fetching account data: {e}")
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
                if positions:
                    open_positions = len([p for p in positions if float(p.get("contracts", 0)) != 0])
                else:
                    open_positions = 0
            elif self.mode == "DRY_RUN":
                try:
                    from modules.auto_trade.gui.services.dry_run.dry_run_db import DryRunDB

                    db = DryRunDB()
                    positions = db.get_open_positions()
                    open_positions = len(positions)
                except ImportError:
                    pass

            # Not fetching daily stats correctly for now since get_daily_stats is omitted or complex in dynamodb without queries.
            # You can adapt it to read from self.repo_context directly if you rebuild `get_daily_stats`.
            today_trades = 0
            win_rate = 0.0

            return {
                "open_positions": open_positions,
                "today_trades": today_trades,
                "win_rate": win_rate,
                "mode": self.mode,
            }
        except Exception as e:
            log_error(f"Error fetching stats: {e}")
            return {"open_positions": 0, "today_trades": 0, "win_rate": 0.0, "mode": self.mode}

    @staticmethod
    def _parse_datetime(value: object) -> "Optional[_datetime]":
        """Coerce a DynamoDB value to datetime.

        DynamoDB stores datetimes as ISO-8601 strings. ``from_dynamo_item``
        does not convert them back, so this helper handles both cases:
        - already a ``datetime`` object  → returned as-is
        - ISO string                     → parsed with ``datetime.fromisoformat``
        - anything else / None           → returns None
        """
        if value is None:
            return None
        if isinstance(value, _datetime):
            return value
        if isinstance(value, str):
            try:
                return _datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    def get_signals(
        self,
        min_score: float = 0.7,
        signal_types: Optional[List[str]] = None,
        max_age_hours: float = 24.0,
    ) -> List[Dict]:
        """Fetch signals from DB, filtered by score and age.

        Args:
            min_score: Minimum confidence/final_score (default 0.7)
            signal_types: Optional allowlist of signal types (e.g. ["LONG", "SHORT"])
            max_age_hours: Discard signals older than this many hours (default 24h).
                           Pass 0 or a negative value to disable age filtering.
        """
        try:
            if self.repo_context:
                signals = self.repo_context.signals.get_recent_signals(limit=100)
                filtered = []
                stale_count = 0

                cutoff: Optional[_datetime] = None
                if max_age_hours > 0:
                    from datetime import timedelta as _td
                    from datetime import timezone as _tz

                    cutoff = _datetime.now(_tz.utc) - _td(hours=max_age_hours)

                for signal in signals:
                    raw = signal.get("final_score") or signal.get("confidence", 0.0)
                    score = float(cast(Union[float, int], raw))
                    if score < min_score:
                        continue

                    signal_type = signal.get("signal_type", "").upper()
                    if signal_types is not None and signal_type not in signal_types:
                        continue

                    # DynamoDB returns created_at as an ISO string, not a datetime object.
                    # _parse_datetime coerces either type safely.
                    created_at_dt = self._parse_datetime(signal.get("created_at"))

                    # Age filter – skip signals that are too old to be actionable
                    if cutoff is not None and created_at_dt is not None:
                        # Make created_at_dt timezone-aware if it isn't already
                        if created_at_dt.tzinfo is None:
                            from datetime import timezone as _tz

                            created_at_dt = created_at_dt.replace(tzinfo=_tz.utc)
                        if created_at_dt < cutoff:
                            stale_count += 1
                            continue

                    filtered.append(
                        {
                            "symbol": signal.get("symbol", ""),
                            "signal": signal_type,
                            "score": score,
                            "time": created_at_dt.strftime("%Y-%m-%d %H:%M") if created_at_dt else "",
                            "created_at": created_at_dt.isoformat() if created_at_dt else "",
                            "created_at_ts": created_at_dt.timestamp() if created_at_dt else 0.0,
                        }
                    )

                if stale_count:
                    log_warn(
                        f"[Signals] Dropped {stale_count} stale signal(s) older than {max_age_hours:.0f}h "
                        f"({len(filtered)} fresh signal(s) remain)"
                    )

                if filtered:
                    return filtered
            return self._get_demo_signals() if self.mode == "DRY_RUN" else []
        except Exception as e:
            log_error(f"Error fetching signals: {e}")
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
        ]

    def get_positions(self) -> List[Dict]:
        try:
            if self.mode == "DRY_RUN":
                try:
                    from modules.auto_trade.gui.services.dry_run.dry_run_db import DryRunDB

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

                        notional_usd = abs(contracts * entry_price)

                        if side == "LONG":
                            pnl = (current_price - entry_price) * contracts
                        else:
                            pnl = (entry_price - current_price) * contracts

                        filtered_positions.append(
                            {
                                "symbol": symbol,
                                "side": side,
                                "size": notional_usd,
                                "contracts": abs(contracts),
                                "entry_price": entry_price,
                                "current_price": current_price,
                                "pnl": pnl,
                                "take_profit": pos.get("take_profit"),
                                "stop_loss": pos.get("stop_loss"),
                                "break_even": pos.get("break_even"),
                            }
                        )
                    return filtered_positions
                except ImportError:
                    return []

            if self.data_fetcher and self.api_key and self.api_secret:
                positions = self.data_fetcher.fetch_binance_futures_positions(
                    api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet
                )

                client = self._get_or_create_client()

                filtered_positions = []
                for pos in positions or []:
                    contracts = float(pos.get("contracts", 0))

                    if contracts == 0:
                        continue

                    side = pos.get("direction", "LONG").upper()
                    entry_price = float(pos.get("entry_price", 0))
                    symbol = pos.get("symbol", "")

                    size_usd = float(pos.get("size_usdt", 0)) if pos.get("size_usdt") is not None else 0.0
                    if not size_usd and entry_price:
                        size_usd = abs(contracts * entry_price)

                    try:
                        leverage = int(pos.get("leverage", 1))
                    except (TypeError, ValueError):
                        leverage = 1
                    liq_price = pos.get("liquidation_price")
                    liquidation_price = float(liq_price) if liq_price is not None else None
                    margin_used = (
                        float(pos.get("margin_used", 0))
                        if pos.get("margin_used") is not None
                        else (size_usd / leverage if leverage else 0.0)
                    )

                    current_price = entry_price
                    if client:
                        try:
                            ticker = client.exchange.fetch_ticker(symbol)
                            current_price = float(
                                ticker.get("info", {}).get("markPrice") or ticker.get("last") or entry_price
                            )
                        except Exception as e:
                            log_error(f"[DataService] Could not fetch mark price for {symbol}: {e}")

                    pnl = 0.0
                    if side == "LONG":
                        pnl = (current_price - entry_price) * abs(contracts)
                    else:
                        pnl = (entry_price - current_price) * abs(contracts)

                    take_profit = None
                    stop_loss = None
                    break_even = None

                    tpsl = self.get_cached_tpsl(symbol)
                    take_profit = tpsl.get("take_profit")
                    stop_loss = tpsl.get("stop_loss")
                    break_even = tpsl.get("break_even")

                    filtered_positions.append(
                        {
                            "symbol": pos.get("symbol", ""),
                            "side": side,
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
                return filtered_positions
            return []
        except Exception as e:
            log_error(f"Error fetching positions: {e}")
            return []
