"""
Binance Client Module (Legacy Compatibility Layer)

This module provides backward compatibility by importing from the new sub-module structure.
All functionality has been refactored into:
  - binance.exchange_setup: CCXT exchange initialization
  - binance.order_execution: Market orders with TP/SL
  - binance.position_management: Position operations
  - binance.order_management: TP/SL modification, order cancellation

For new code, please import from:
    from modules.auto_trade.execution.binance import BinanceClient
"""

# Import from new sub-module for backward compatibility
from modules.auto_trade.execution.binance import BinanceClient

__all__ = ["BinanceClient"]
