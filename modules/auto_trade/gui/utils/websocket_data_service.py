"""Compatibility adapter for legacy ``gui.utils`` import paths.

The real implementation lives in
``modules.auto_trade.gui.services.websocket_data_service``.
"""

import sys

from modules.auto_trade.gui.services import websocket_data_service as _impl

# Keep legacy import path behavior identical to the implementation module.
sys.modules[__name__] = _impl

