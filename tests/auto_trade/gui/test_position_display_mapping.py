from datetime import datetime

import pytest

from modules.auto_trade.gui.main_window.websocket_handler import WebSocketHandler
from modules.auto_trade.monitoring.position_monitor import PositionSnapshot


class DummyParent:
    """Minimal parent stub to construct WebSocketHandler for mapping tests."""

    def __init__(self):
        self.settings_manager = None


def make_snapshot() -> PositionSnapshot:
    return PositionSnapshot(
        symbol="SKL/USDT",
        side="long",
        position_amt=3121.0,
        entry_price=0.00663,
        mark_price=0.00610,
        liquidation_price=None,
        unrealized_pnl=-1.65,
        unrealized_pnl_percent=-8.0,
        margin_type="isolated",
        leverage=2,
        notional=19.0381,
        timestamp=datetime.now(),
    )


def test_websocket_handler_converts_snapshot_to_usd_size():
    """Size in UI mapping should use notional (USD), not contracts."""
    handler = WebSocketHandler(DummyParent())
    snap = make_snapshot()

    result = handler._convert_positions_to_dicts([snap])
    assert len(result) == 1
    mapped = result[0]

    # Size should come from notional
    assert pytest.approx(mapped["size"], rel=1e-6) == 19.0381
    # Contracts should preserve raw amount
    assert pytest.approx(mapped["contracts"], rel=1e-6) == 3121.0
    # Leverage should be passed through correctly
    assert mapped["leverage"] == 2

