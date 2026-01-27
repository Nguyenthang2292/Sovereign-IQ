"""Binance Futures operations including positions and balance fetching."""

import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from modules.common.domain import normalize_symbol
from modules.common.ui.logging import log_debug, log_error, log_warn

if TYPE_CHECKING:
    from .base import DataFetcherBase


class BinanceFuturesFetcher:
    """Handles Binance Futures positions and balance operations."""

    def __init__(self, base: "DataFetcherBase"):
        """
        Initialize BinanceFuturesFetcher.

        Args:
            base: DataFetcherBase instance for accessing exchange_manager and state
        """
        self.base = base

    def fetch_binance_futures_positions(
        self,
        api_key: str = None,
        api_secret: str = None,
        testnet: bool = False,
        debug: bool = False,
    ) -> List[Dict]:
        """
        Fetches open positions from Binance Futures USDT-M.

        Args:
            api_key: API Key from Binance. Priority:
                1. This parameter (if provided)
                2. Environment variable BINANCE_API_KEY
                3. From ExchangeManager's default credentials
            api_secret: API Secret from Binance. Priority:
                1. This parameter (if provided)
                2. Environment variable BINANCE_API_SECRET
                3. From ExchangeManager's default credentials
            testnet: Use testnet if True (default: False)
            debug: Show debug info if True (default: False)

        Returns:
            List of dictionaries containing position information with keys:
            - symbol: Normalized symbol (e.g., 'BTC/USDT')
            - size_usdt: Position size in USDT
            - entry_price: Entry price
            - direction: 'LONG' or 'SHORT'
            - contracts: Absolute number of contracts
        """
        api_key, api_secret = self._resolve_binance_credentials(api_key, api_secret)
        exchange = self._connect_binance_futures(api_key, api_secret, testnet)

        try:
            # Fetch all positions using throttled_call
            positions = self.base.exchange_manager.authenticated.throttled_call(exchange.fetch_positions)

            # Filter only open positions (size != 0) and USDT-M
            open_positions = []
            for pos in positions:
                contracts = self._extract_position_contracts(pos)
                if contracts is None or contracts == 0:
                    continue

                normalized_symbol = self._normalize_position_symbol(pos.get("symbol", ""))
                if not self._is_usdtm_symbol(normalized_symbol):
                    continue

                entry_price = float(pos.get("entryPrice", 0) or 0)
                if debug:
                    self._debug_position(pos, normalized_symbol, contracts)

                direction = self._determine_position_direction(pos, contracts)
                size_usdt = self._calculate_position_size(pos, contracts, entry_price, exchange)

                if size_usdt <= 0:
                    continue

                open_positions.append(
                    {
                        "symbol": normalized_symbol,
                        "size_usdt": size_usdt,
                        "entry_price": entry_price,
                        "direction": direction,
                        "contracts": abs(contracts),
                    }
                )

            return open_positions

        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api" in error_msg.lower():
                raise ValueError(f"Lỗi xác thực API: {e}\nVui lòng kiểm tra lại API Key và Secret")
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                raise ValueError(f"Lỗi kết nối mạng: {e}")
            else:
                raise ValueError(f"Lỗi khi lấy positions: {e}")

    def _resolve_binance_credentials(self, api_key: Optional[str], api_secret: Optional[str]) -> Tuple[str, str]:
        """Resolve Binance API credentials from multiple sources."""
        resolved_key = (
            api_key or os.getenv("BINANCE_API_KEY") or self.base.exchange_manager.authenticated.default_api_key
        )
        resolved_secret = (
            api_secret or os.getenv("BINANCE_API_SECRET") or self.base.exchange_manager.authenticated.default_api_secret
        )

        if not resolved_key or not resolved_secret:
            raise ValueError(
                "API Key và API Secret là bắt buộc!\n"
                "Cung cấp qua một trong các cách sau:\n"
                "  1. Tham số method: api_key và api_secret\n"
                "  2. Biến môi trường: BINANCE_API_KEY và BINANCE_API_SECRET\n"
                "  3. ExchangeManager credentials (khi khởi tạo ExchangeManager với api_key/api_secret)"
            )
        return resolved_key, resolved_secret

    def _connect_binance_futures(self, api_key: str, api_secret: str, testnet: bool):
        """Connect to Binance Futures with credentials."""
        try:
            return self.base.exchange_manager.authenticated.connect_to_exchange_with_credentials(
                "binance",
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                contract_type="future",
            )
        except ValueError as exc:
            raise ValueError(f"Error connecting to Binance: {exc}")

    def fetch_binance_account_balance(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        currency: str = "USDT",
    ) -> Optional[float]:
        """
        Fetch account balance from Binance Futures USDT-M.

        Args:
            api_key: API Key from Binance (optional, uses default if not provided)
            api_secret: API Secret from Binance (optional, uses default if not provided)
            testnet: Use testnet if True (default: False)
            currency: Currency to fetch balance for (default: "USDT")

        Returns:
            Account balance in USDT as float, or None if error or no credentials
        """
        try:
            api_key, api_secret = self._resolve_binance_credentials(api_key, api_secret)
            exchange = self._connect_binance_futures(api_key, api_secret, testnet)

            # Fetch balance using throttled_call
            balance = self.base.exchange_manager.authenticated.throttled_call(exchange.fetch_balance)

            # Extract USDT balance from futures account
            # Binance futures balance structure can vary, try multiple formats
            usdt_balance = None

            # Format 1: Direct currency key with total
            if currency in balance:
                currency_info = balance[currency]
                if isinstance(currency_info, dict):
                    total = currency_info.get("total", 0) or 0
                    if total > 0:
                        usdt_balance = float(total)
                elif isinstance(currency_info, (int, float)):
                    if currency_info > 0:
                        usdt_balance = float(currency_info)

            # Format 2: Check in 'info' -> 'assets' (Binance futures API format)
            if usdt_balance is None and "info" in balance:
                info = balance["info"]
                if "assets" in info:
                    for asset in info["assets"]:
                        if asset.get("asset") == currency:
                            wallet_balance = asset.get("walletBalance", 0) or 0
                            if wallet_balance > 0:
                                usdt_balance = float(wallet_balance)
                                break
                # Also check direct currency in info
                if usdt_balance is None and currency in info:
                    wallet_balance = info[currency].get("walletBalance", 0) or info[currency].get("total", 0) or 0
                    if wallet_balance > 0:
                        usdt_balance = float(wallet_balance)

            # Format 3: Check in 'total' dict
            if usdt_balance is None and "total" in balance:
                if currency in balance["total"]:
                    total = balance["total"][currency]
                    if total > 0:
                        usdt_balance = float(total)

            # Format 4: Calculate from free + used
            if usdt_balance is None and currency in balance:
                currency_info = balance[currency]
                if isinstance(currency_info, dict):
                    free = float(currency_info.get("free", 0) or 0)
                    used = float(currency_info.get("used", 0) or 0)
                    total = free + used
                    if total > 0:
                        usdt_balance = total

            if usdt_balance is not None and usdt_balance > 0:
                return usdt_balance

            log_warn(f"No {currency} balance found in Binance account")
            return None

        except ValueError as e:
            # No credentials or connection error
            log_warn(f"Cannot fetch balance from Binance: {e}")
            return None
        except Exception as e:
            log_error(f"Error fetching balance from Binance: {e}")
            return None

    @staticmethod
    def _extract_position_contracts(position: Dict) -> Optional[float]:
        """Extract contracts amount from position data."""
        contracts = position.get("contracts")
        if contracts is not None:
            try:
                value = float(contracts or 0)
                if value != 0:
                    return value
            except (ValueError, TypeError):
                pass

        position_amt = position.get("positionAmt", 0)
        if position_amt:
            try:
                value = float(position_amt)
                if value != 0:
                    return value
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _normalize_position_symbol(symbol: str) -> str:
        """Normalize position symbol to standard format."""
        if ":" in symbol:
            symbol = symbol.split(":")[0]
        return normalize_symbol(symbol, quote="USDT")

    @staticmethod
    def _is_usdtm_symbol(symbol: str) -> bool:
        """Check if symbol is a USDT-M futures symbol."""
        return "/USDT" in symbol or symbol.endswith("USDT")

    def _determine_position_direction(self, position: Dict, contracts: float) -> str:
        """Determine position direction (LONG/SHORT)."""
        candidates = [
            position.get("positionSide"),
            position.get("side"),
            ((position.get("info") or {}).get("positionSide") if isinstance(position.get("info"), dict) else None),
        ]

        for candidate in candidates:
            if candidate:
                upper = str(candidate).upper()
                if upper in ["LONG", "SHORT"]:
                    return upper

        # Inspect raw position amount if available
        info = position.get("info")
        if isinstance(info, dict):
            raw_amt = info.get("positionAmt")
            if raw_amt:
                try:
                    amt = float(raw_amt)
                    if amt != 0:
                        return "LONG" if amt > 0 else "SHORT"
                except (ValueError, TypeError):
                    pass

        return "LONG" if contracts > 0 else "SHORT"

    def _calculate_position_size(self, position: Dict, contracts: float, entry_price: float, exchange) -> float:
        """Calculate position size in USDT."""
        notional = position.get("notional")
        if notional is not None:
            try:
                value = float(notional)
                if value != 0:
                    return abs(value)
            except (ValueError, TypeError):
                pass

        size_usdt = abs(contracts * entry_price)

        if size_usdt == 0 and entry_price > 0:
            try:
                pos_detail = self.base.exchange_manager.authenticated.throttled_call(
                    exchange.fetch_position, position.get("symbol", "")
                )
                notional = pos_detail.get("notional", None)
                if notional is not None and notional != 0:
                    size_usdt = abs(float(notional))
            except Exception:
                pass
        return size_usdt

    @staticmethod
    def _debug_position(position: Dict, symbol: str, contracts: float):
        """Debug output for position data."""
        info = position.get("info", {})
        log_debug(f"Position data for {symbol}:")
        log_debug(f"  contracts: {contracts}")
        log_debug(f"  positionSide: {position.get('positionSide')}")
        log_debug(f"  side: {position.get('side')}")
        log_debug(f"  info.positionSide: {info.get('positionSide', 'N/A') if isinstance(info, dict) else 'N/A'}")
        log_debug(f"  info.positionAmt: {info.get('positionAmt', 'N/A') if isinstance(info, dict) else 'N/A'}")
