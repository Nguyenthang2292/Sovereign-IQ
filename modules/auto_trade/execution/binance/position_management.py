"""
Position Management Module

Handles position operations (get, close).
"""

from typing import Optional, cast

import ccxt

from modules.common.ui.logging import log_error, log_info, log_warn


class PositionManagement:
    """
    Handles position management operations.
    """

    def __init__(self, exchange: ccxt.binance, dry_run: bool = False):
        """
        Initialize PositionManagement.

        Args:
            exchange: CCXT exchange instance
            dry_run: Simulate operations without executing
        """
        self.exchange = exchange
        self.dry_run = dry_run

    def get_position(self, symbol: str) -> Optional[dict]:
        """
        Fetch current position for a symbol.

        Args:
            symbol: Trading symbol (any format: BTCUSDT, BTC/USDT, BTC/USDT:USDT)

        Returns:
            Position dict or None if not found
        """
        if self.dry_run:
            return {"symbol": symbol, "contracts": 0, "side": "long", "notional": 0}

        try:
            from modules.common.domain.symbols import normalize_symbol_key

            # Normalize input symbol for comparison (e.g. SKL/USDT -> SKLUSDT)
            normalized_input = normalize_symbol_key(symbol)

            def _position_key_for_compare(key: str) -> str:
                # CCXT futures returns symbol "SKL/USDT:USDT" -> normalize_symbol_key -> "SKLUSDTUSDT".
                # Collapse trailing duplicate quote so it matches "SKLUSDT".
                if not key:
                    return key
                if key.endswith("USDTUSDT"):
                    return key[:-4]  # SKLUSDTUSDT -> SKLUSDT
                return key

            # Fetch all positions
            positions: list = self.exchange.fetch_positions()
            seen_symbols: list = []

            for pos in positions:
                # CCXT may put symbol in top-level or in info
                pos_symbol = pos.get("symbol") or (pos.get("info") or {}).get("symbol") if isinstance(pos.get("info"), dict) else ""
                pos_symbol = pos_symbol or ""
                pos_key = normalize_symbol_key(pos_symbol)
                seen_symbols.append(pos_key)

                if _position_key_for_compare(pos_key) != _position_key_for_compare(normalized_input):
                    continue

                # Contracts: top-level "contracts", or "positionAmt", or info.positionAmt
                raw_contracts = pos.get("contracts")
                if raw_contracts is None:
                    raw_contracts = pos.get("positionAmt")
                if raw_contracts is None and isinstance(pos.get("info"), dict):
                    raw_contracts = pos.get("info", {}).get("positionAmt")
                try:
                    contracts = abs(float(raw_contracts or 0))
                except (TypeError, ValueError):
                    contracts = 0.0

                if contracts != 0:
                    return pos

            log_warn(
                f"No open position found for {symbol} (normalized: {normalized_input}); "
                f"fetch_positions returned {len(positions)} position(s): {seen_symbols[:15]}"
            )
            return None
        except Exception as e:
            log_error(f"Failed to fetch position for {symbol}: {e}")
            return None

    def close_position(
        self, symbol: str, side: str, size: float, order_type: str = "market", limit_price: Optional[float] = None
    ) -> Optional[dict]:
        """
        Close a position (full or partial).

        Args:
            symbol: Trading symbol
            side: Position side ('long' or 'short')
            size: Amount to close
            order_type: 'market' or 'limit'
            limit_price: Limit price (only for limit orders)

        Returns:
            Order result dict or None if failed
        """
        if self.dry_run:
            log_info(f"[DRY RUN] Would close {size} of {symbol} {side} position ({order_type})")
            if order_type == "limit" and limit_price:
                log_info(f"  Limit price: ${limit_price:,.2f}")
            return {
                "dry_run": True,
                "symbol": symbol,
                "side": side,
                "size": size,
                "type": order_type,
            }

        # Calculate order side (opposite to position side)
        close_side: str = "sell" if side.lower() == "long" else "buy"

        # Get current price for limit orders
        if order_type == "limit" and not limit_price:
            log_error("Limit price required for limit orders")
            return None

        try:
            log_info(f"Closing {size} of {symbol} {side} position ({order_type})")

            if order_type == "market":
                # Market order
                result = cast(
                    dict,
                    self.exchange.create_order(
                        symbol=symbol, type="market", side=close_side, amount=size, params={"reduceOnly": True}
                    ),
                )
            else:
                # Limit order
                result = cast(
                    dict,
                    self.exchange.create_order(
                        symbol=symbol,
                        type="limit",
                        side=close_side,
                        amount=size,
                        price=limit_price,
                        params={"reduceOnly": True},
                    ),
                )

            log_info(f"✅ Position close order executed: {result.get('id')}")
            return result

        except Exception as e:
            log_error(f"Failed to close position: {e}")
            return None

    def modify_margin(self, symbol: str, amount: float, type: int = 1, position_side: str = "BOTH") -> Optional[dict]:
        """
        Modify position margin (for Isolated Margin).

        Args:
            symbol: Trading symbol
            amount: Amount of margin to add (or remove)
            type: 1 = Add Position Margin, 2 = Reduce Position Margin
            position_side: 'BOTH', 'LONG', or 'SHORT'

        Returns:
            Result dict or None if failed
        """
        if self.dry_run:
            action: str = "Add" if type == 1 else "Reduce"
            log_info(f"[DRY RUN] Would {action} margin for {symbol} by ${amount:,.2f}")
            return {"dry_run": True, "symbol": symbol, "amount": amount, "type": type}

        try:
            log_info(f"Modifying margin for {symbol}: amount=${amount}, type={type}")

            # Note: CCXT may not have a unified method for this
            params: dict = {
                "symbol": self.exchange.market_id(symbol),
                "amount": amount,
                "type": type,
                "positionSide": position_side,
            }

            response: dict = self.exchange.fapiPrivatePostPositionMargin(params)
            log_info(f"✅ Margin modified for {symbol}. New amount: {response.get('amount')}")
            return response

        except Exception as e:
            log_error(f"Failed to modify margin: {e}")
            return None
