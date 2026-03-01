import sys

from modules.auto_trade.gui.services import websocket_data_service as _impl

sys.modules[__name__] = _impl
