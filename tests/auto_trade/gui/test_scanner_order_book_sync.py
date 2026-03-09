from types import SimpleNamespace


class _FakeSettingsManager:
    def __init__(self):
        self._store = {
            "order_book_imbalance": {
                "enabled": False,
                "threshold": 0.15,
                "retry_wait_seconds": 30,
                "max_retries": 2,
                "depth_limit": 20,
                "delta_window_minutes": 5,
            },
            "scanner": {},
        }

    def get(self, key, default=None):
        return self._store.get(key, default)

    def set(self, key, value):
        self._store[key] = value

    def save(self):
        return True


def test_scanner_config_syncs_order_book_depth_and_threshold():
    from modules.auto_trade.gui.main_window.scanner import ScannerManager

    parent = SimpleNamespace(settings_manager=_FakeSettingsManager())
    manager = ScannerManager(parent)

    manager.handle_config_change(
        {
            "enable_order_book": True,
            "ob_depth": 75,
            "ob_imbalance_threshold": 0.33,
        }
    )

    ob_cfg = parent.settings_manager.get("order_book_imbalance")
    assert ob_cfg["enabled"] is True
    assert ob_cfg["depth_limit"] == 75
    assert ob_cfg["threshold"] == 0.33
