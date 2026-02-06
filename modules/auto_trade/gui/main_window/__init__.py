"""Auto Trade Dashboard Main Window Sub-package.

Keep this package import-light.

Some test environments import submodules (e.g. `auto_trade.py`) without having
all GUI dependencies available. Importing `main_window` at package import time
would pull in the full GUI stack and can fail. We expose `AutoTradeDashboard`
via lazy attribute access instead.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main_window import AutoTradeDashboard as AutoTradeDashboard

__all__ = ["AutoTradeDashboard"]


def __getattr__(name: str):
    if name == "AutoTradeDashboard":
        from .main_window import AutoTradeDashboard

        return AutoTradeDashboard
    raise AttributeError(name)
