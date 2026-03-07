"""Compatibility adapter for legacy ``gui.utils`` import paths."""

import sys

from modules.auto_trade.gui.services.dry_run import dry_run_executor as _impl

# Keep legacy import path behavior identical to the implementation module.
sys.modules[__name__] = _impl
