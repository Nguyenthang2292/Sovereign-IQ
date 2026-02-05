"""
WebSocket Module for Auto Trade System

Provides real-time data streams using ccxt.pro WebSocket connections:
- Position updates (real-time P&L tracking)
- Balance updates (account changes)
- Order updates (fills, cancellations)
- Mark price updates (for break-even calculations)
"""

from modules.auto_trade.websocket.client import BinanceWebSocketClient

__all__ = ["BinanceWebSocketClient"]
