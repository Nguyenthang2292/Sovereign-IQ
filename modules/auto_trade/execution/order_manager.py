"""
Order Manager Module

Orchestrates the complete order execution flow.
Integrates all components: builder, validator, risk manager, Binance client,
and optionally RecoveryManager for gradual recovery parameters.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from modules.auto_trade.core.signal_selector import FinalSignal
from modules.auto_trade.execution.adaptive_close_calculator import AdaptiveCloseCalculator, AdaptiveCloseResult
from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.execution.order_builder import OrderBuilder, OrderTicket
from modules.auto_trade.execution.order_validator import OrderValidator
from modules.auto_trade.execution.risk_manager import RiskManager
from modules.auto_trade.security.secret_string import SecretString
from modules.common.core.data_fetcher import DataFetcher
from modules.common.domain.symbol_codec import SymbolCodec
from modules.common.domain.symbol_types import DbSymbol, FuturesSymbol
from modules.common.ui.logging import log_error, log_info, log_warn

_SYMBOL_CODEC = SymbolCodec()


class OrderManager:
    """
    Order Manager orchestrates the complete order execution flow.

    Flow:
        1. Check if any positions are open (via DataFetcher)
        2. If no positions → proceed with order execution
        3. Calculate position size (RiskManager)
        4. Check for recovery parameters (RecoveryManager)
        5. Build order ticket (OrderBuilder)
        6. Fetch current price
        7. Pre-order validation (OrderValidator)
        8. Execute order (BinanceClient)
        9. Post-order validation (OrderValidator)
        10. Return order result

    Example:
        >>> manager = OrderManager(data_fetcher, api_key, api_secret)
        >>> result = manager.execute_signal(signal)
    """

    def __init__(
        self,
        data_fetcher: DataFetcher,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        dry_run: bool = False,
        balance_percentage: float = 0.95,
        default_leverage: int = 2,
        default_tp_pct: float = 5.0,
        default_sl_pct: float = 50.0,
        recovery_manager=None,  # Optional RecoveryManager for gradual recovery
    ):
        """
        Initialize OrderManager.

        Args:
            data_fetcher: DataFetcher instance
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet if True
            dry_run: Simulate orders without execution
            balance_percentage: Percentage of balance to use (default: 0.95)
            default_leverage: Default leverage (default: 2x)
            default_tp_pct: Default take profit percentage (default: 5%)
            default_sl_pct: Default stop loss percentage (default: 50%)
            recovery_manager: Optional RecoveryManager for gradual recovery parameters
        """
        self.data_fetcher = data_fetcher
        self.api_key = SecretString(api_key)
        self.api_secret = SecretString(api_secret)
        self.testnet = testnet
        self.dry_run = dry_run
        self.recovery_manager = recovery_manager

        # Initialize components
        self.risk_manager = RiskManager(
            data_fetcher=data_fetcher,
            balance_percentage=balance_percentage,
            default_leverage=default_leverage,
        )

        self.order_builder = OrderBuilder(
            default_tp_pct=default_tp_pct,
            default_sl_pct=default_sl_pct,
            default_leverage=default_leverage,
        )

        self.order_validator = OrderValidator()

        self.binance_client = BinanceClient(
            api_key=self.api_key.get_secret_value(),
            api_secret=self.api_secret.get_secret_value(),
            testnet=testnet,
            dry_run=dry_run,
        )
        self._adaptive_close_calculator = self._init_adaptive_close_calculator()

        log_info(
            f"OrderManager initialized ({'DRY RUN' if dry_run else 'LIVE'} mode, {'testnet' if testnet else 'mainnet'})"
        )

    @staticmethod
    def _extract_opened_at_utc(market_order: dict) -> datetime:
        """Best-effort extraction of order open time from exchange payload."""
        timestamp_ms = market_order.get("timestamp")
        if timestamp_ms is not None:
            try:
                return datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=timezone.utc)
            except (TypeError, ValueError, OSError, OverflowError):
                pass
        return datetime.now(timezone.utc)

    def _init_adaptive_close_calculator(self) -> Optional[AdaptiveCloseCalculator]:
        """Initialize adaptive calculator once; fail open if settings are unavailable."""
        try:
            from modules.auto_trade.gui.services.settings_manager import SettingsManager

            settings_manager = SettingsManager()
            settings_manager.load()
            return AdaptiveCloseCalculator(settings_manager)
        except Exception as e:
            log_warn(f"Adaptive close calculator unavailable: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _fetch_open_positions(self) -> Optional[list]:
        return self.data_fetcher.fetch_binance_futures_positions(
            api_key=self.api_key.get_secret_value(), api_secret=self.api_secret.get_secret_value(), testnet=self.testnet
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _fetch_ticker(self, symbol: FuturesSymbol) -> Any:
        return self.binance_client.exchange.fetch_ticker(symbol)

    def _create_market_order(self, order: OrderTicket) -> Optional[dict]:
        return self.binance_client.create_market_order(order)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _fetch_account_balance(self) -> Optional[float]:
        return self.risk_manager.fetch_account_balance(
            api_key=self.api_key.get_secret_value(), api_secret=self.api_secret.get_secret_value(), testnet=self.testnet
        )

    def check_open_positions(self) -> Optional[list]:
        """
        Check if there are any open positions.

        Returns:
            List of open positions or None if error/no positions
        """
        try:
            positions: Optional[list] = self._fetch_open_positions()

            if not positions:
                log_info("No open positions found")
                return None

            # fetch_binance_futures_positions() normalises positions and returns
            # "contracts" (not the raw Binance "positionAmt").  Fall back to
            # "positionAmt" so the check still works if the caller ever passes
            # raw Binance dicts directly.
            def _nonzero(pos: dict) -> bool:
                contracts = pos.get("contracts")
                if contracts is not None:
                    try:
                        return float(contracts) != 0
                    except (TypeError, ValueError):
                        pass
                # fallback for raw Binance format
                position_amt = pos.get("positionAmt", 0)
                try:
                    return float(position_amt) != 0
                except (TypeError, ValueError):
                    return False

            open_positions: list = [p for p in positions if _nonzero(p)]

            if open_positions:
                log_info(f"Found {len(open_positions)} open position(s)")
                for pos in open_positions:
                    symbol: str = str(pos.get("symbol", "Unknown"))
                    amount: float = float(pos.get("contracts") or pos.get("positionAmt", 0))
                    entry_price: float = float(pos.get("entry_price") or pos.get("entryPrice", 0))
                    direction: str = str(pos.get("direction") or pos.get("side", "?"))
                    log_info(f"  - {symbol} [{direction}]: contracts={amount}, entry=${entry_price}")
            else:
                log_info("No active positions (all positions have zero contracts)")
                return None

            return open_positions

        except Exception as e:
            log_error(f"Failed to check open positions: {e}", exc_info=True)
            return None

    def execute_signal(
        self,
        signal: FinalSignal,
        force_execution: bool = False,
        leverage_override: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Execute a trading signal.

        Args:
            signal: Final signal from signal pipeline
            force_execution: Force execution even if position exists
            leverage_override: Override default leverage

        Returns:
            Order result dict or None if failed/skipped

        Flow:
            1. Check open positions
            2. Calculate position size
            3. Check for recovery parameters
            4. Build order ticket
            5. Fetch current price
            6. Pre-order validation
            7. Execute order
            8. Post-order validation
            9. Return result
        """
        log_info(f"🚀 Executing signal: {signal.symbol} {signal.signal_type}")

        # Step 1: Check for open positions
        if not force_execution:
            open_positions = self.check_open_positions()
            if open_positions:
                log_warn("Open position(s) detected, skipping new order execution")
                log_warn("Use force_execution=True to override this check")
                return None
        else:
            log_warn("Force execution enabled, skipping position check")

        # Step 2: Calculate position size
        position_size: Optional[float] = self.risk_manager.calculate_position_size(
            api_key=self.api_key.get_secret_value(), api_secret=self.api_secret.get_secret_value(), testnet=self.testnet
        )

        if not position_size:
            log_error("Failed to calculate position size, aborting")
            return None

        # Step 3: Check for recovery parameters (Gradual Recovery)
        effective_leverage: Optional[int] = leverage_override
        effective_position_size: float = position_size

        if self.recovery_manager and self.recovery_manager.is_active:
            recovery_params: dict = self.recovery_manager.get_recovery_parameters()
            if recovery_params.get("active"):
                recovery_leverage: Optional[int] = recovery_params.get("leverage")
                recovery_position_size: Optional[float] = recovery_params.get("position_size")

                if recovery_leverage:
                    effective_leverage = recovery_leverage
                    log_info(f"🔄 Recovery mode: Using leverage {effective_leverage}x")

                if recovery_position_size:
                    # Use the smaller of calculated size or recovery-recommended size
                    effective_position_size = min(position_size, recovery_position_size)
                    log_info(f"🔄 Recovery mode: Position size ${effective_position_size:.2f}")

                remaining: float = recovery_params.get("remaining_loss", 0)
                pct: float = recovery_params.get("recovery_percentage", 0)
                log_info(f"🔄 Recovery status: {pct:.1f}% complete, ${remaining:.2f} remaining")

        # Step 4: Build order ticket
        order: OrderTicket = self.order_builder.build_order(
            signal=signal,
            position_size=effective_position_size,
            leverage=effective_leverage,
        )
        # Set client_order_id (AT_ prefix) for Binance and DB sync
        from modules.auto_trade.execution.order_tagging import OrderTagger

        symbol_ccxt = _SYMBOL_CODEC.to_db(order.symbol or "")
        order.client_order_id = OrderTagger.generate_client_order_id(symbol_ccxt)

        log_info(f"Built order ticket: {order.symbol} {order.side} ${order.amount:.2f} @ {order.leverage}x")

        # Step 5: Fetch current price
        try:
            # Use DataFetcher or directly from CCXT
            ticker: dict = self._fetch_ticker(FuturesSymbol(signal.symbol))
            current_price: float = ticker["last"]
            log_info(f"Current price for {signal.symbol}: ${current_price:,.2f}")
        except Exception as e:
            log_error(f"Failed to fetch current price: {e}", exc_info=True)
            return None

        # Step 6: Pre-order validation
        balance: Optional[float] = self._fetch_account_balance()

        if balance is None:
            log_error("Failed to fetch balance for validation, aborting")
            return None

        if not self.order_validator.validate_pre_order(order, balance, current_price):
            log_error("Pre-order validation failed, aborting")
            return None

        # Step 7: Execute order
        log_info("Executing order on Binance...")
        order_result: Optional[dict] = self._create_market_order(order)

        if not order_result:
            log_error("Order execution failed")
            return None

        # Step 8: Post-order validation
        if not self.order_validator.validate_post_order(order_result, order):
            log_warn("Post-order validation failed, but order was executed")

        # Step 8.5: Persist order to DB (sync Binance -> DB)
        if not self.dry_run and order_result.get("market_order"):
            try:
                from modules.auto_trade.database.repository.context import RepositoryContext

                market: dict = order_result["market_order"]
                order_id_binance: str = str(market.get("id", ""))
                client_order_id: Optional[str] = (
                    market.get("clientOrderId")
                    or getattr(order, "client_order_id", None)
                    or (order.client_order_id if hasattr(order, "client_order_id") else None)
                )
                entry_price: float = float(order_result.get("entry_price") or order.entry_price or 0)
                ticket: dict = order_result.get("order_ticket") or order.to_dict()
                side_long_short: str = "LONG" if (order.side or "").upper() == "BUY" else "SHORT"
                _codec = SymbolCodec()
                symbol_db: DbSymbol = DbSymbol(_codec.to_db(order.symbol or ""))
                opened_at = self._extract_opened_at_utc(market)
                order_data: dict = {
                    "order_id": order_id_binance,
                    "client_order_id": client_order_id
                    or (order.client_order_id if hasattr(order, "client_order_id") else None),
                    "symbol": symbol_db,
                    "side": side_long_short,
                    "entry_price": entry_price,
                    "amount": float(order.amount),
                    "leverage": int(order.leverage),
                    "stop_loss": getattr(order, "stop_loss_price", None) or ticket.get("stop_loss_price"),
                    "take_profit": getattr(order, "take_profit_price", None) or ticket.get("take_profit_price"),
                    "status": "OPEN",
                    "order_source": "PROGRAMMATIC",
                    "execution_mode": "AUTO",
                }

                if self._adaptive_close_calculator is not None:
                    adaptive_result = self._adaptive_close_calculator.compute_adaptive_deadline_with_meta(
                        symbol=order.symbol,
                        opened_at=opened_at,
                    )
                    if adaptive_result.deadline_utc is not None:
                        order_data["auto_close_deadline_utc"] = adaptive_result.deadline_utc.isoformat().replace("+00:00", "Z")
                        order_data["auto_close_deadline_source"] = adaptive_result.source
                        order_data["adaptive_close_duration_hours"] = adaptive_result.duration_hours
                        if adaptive_result.pelt_hours is not None:
                            order_data["adaptive_close_pelt_hours"] = adaptive_result.pelt_hours
                        if adaptive_result.hmm_hours is not None:
                            order_data["adaptive_close_hmm_hours"] = adaptive_result.hmm_hours

                ctx = RepositoryContext.from_env()
                ctx.orders.create_order(order_data)

                log_info(f"Order {order_id_binance} persisted to DB (client_order_id={client_order_id})")
            except Exception as db_err:
                log_error(f"Failed to persist order to DB: {db_err}", exc_info=True)
                try:
                    fallback_path = Path.home() / ".auto_trade" / "fallback_orders.jsonl"
                    fallback_path.parent.mkdir(parents=True, exist_ok=True)
                    with fallback_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(order_data) + "\n")
                    log_warn(f"Order written to fallback file: {fallback_path}")
                except Exception as file_err:
                    log_error(f"Failed to write fallback order to disk: {file_err}")

        # Step 9: Return result
        log_info(f"✅ Order executed successfully for {signal.symbol}")
        return order_result

    def emergency_stop(self, reason: str):
        """
        Trigger emergency stop to halt all trading.

        Args:
            reason: Reason for emergency stop
        """
        self.risk_manager.trigger_emergency_stop(reason)

    def reset_emergency_stop(self):
        """Reset emergency stop."""
        self.risk_manager.reset_emergency_stop()

    @property
    def is_emergency_stop_active(self) -> bool:
        """Check if emergency stop is active."""
        return self.risk_manager.is_emergency_stop_active
