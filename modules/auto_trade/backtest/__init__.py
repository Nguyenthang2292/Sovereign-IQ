"""
Auto-Trade Backtesting Module.

Adapter module to integrate the existing backtester for auto-trade system testing.
"""

from .adapter import AutoTradeBacktester
from .strategy_simulator import AutoTradeStrategySimulator

__all__ = ["AutoTradeBacktester", "AutoTradeStrategySimulator"]
