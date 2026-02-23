"""Compatibility shim for legacy config import path.

Legacy tests import:
    from config.position_sizing import BACKTEST_RISK_PER_TRADE
"""

from config.modules.position_sizing import *  # noqa: F401,F403
