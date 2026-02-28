import argparse
import os
import signal
import sys
import threading
import time
from typing import Any

from modules.common.ui.logging import log_info, log_error

try:
    from modules.auto_trade.deploy.secrets_manager import fetch_secrets_to_env
except ImportError:

    def fetch_secrets_to_env():
        log_info("Mock fetch_secrets_to_env: deploy.secrets_manager not fully implemented yet.")


try:
    import yaml
except ImportError:
    pass

from modules.auto_trade.gui.utils.modes import TradingMode
from modules.auto_trade.monitoring.event_system import EventSystem
import queue


class MockSettingsManager:
    def __init__(self, config_dict):
        self._config = config_dict

    def get(self, path: str, default=None):
        keys = path.split(".")
        curr = self._config
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                return default
        return curr

    def set(self, path, value):
        pass

    def save(self):
        pass

    def set_event_bus(self, e):
        pass


class HeadlessApp:
    """Mock parent app imitating AutoTradeDashboard to satisfy ScannerManager and AutoTradeManager."""

    def __init__(self, config_dict):
        self.settings_manager = MockSettingsManager(config_dict)
        self.mode = str(self.settings_manager.get("api.mode", TradingMode.DRY_RUN))
        self.event_bus = EventSystem()

        # Load API keys to os.environ if needed, or let Config/DataService handle it.
        # Ensure we have them for DataService.
        os.environ["BINANCE_API_KEY"] = os.getenv("BINANCE_API_KEY", "")
        os.environ["BINANCE_API_SECRET"] = os.getenv("BINANCE_API_SECRET", "")
        if self.settings_manager.get("api.testnet"):
            os.environ["BINANCE_TESTNET"] = "True"

        from modules.auto_trade.gui.utils.data_service import DataService

        self.data_service = DataService(mode=self.mode, settings_manager=self.settings_manager)
        if hasattr(self.data_service, "set_event_bus"):
            self.data_service.set_event_bus(self.event_bus)

        self._update_queue = queue.Queue()
        self.log_queue = queue.Queue()
        self.ws_data_service = None
        self.recovery_manager = None

        from modules.auto_trade.gui.main_window.updaters import UpdaterManager

        self.updater_manager = UpdaterManager(self)

        self._timers = []

    def after(self, ms, callback, *args):
        # Emulate tkinter's `after` with threading.Timer
        timer = threading.Timer(ms / 1000.0, lambda: callback(*args))
        timer.daemon = True
        timer.start()
        self._timers.append(timer)
        return timer

    def after_cancel(self, timer):
        try:
            timer.cancel()
        except Exception:
            pass

    def refresh_positions(self):
        pass

    def refresh_account(self):
        pass

    # Dummy thread refreshers used by UpdaterManager in setup_updaters()
    def _thread_refresh_signals(self):
        pass

    def _thread_refresh_positions(self):
        pass

    def _thread_refresh_account(self):
        pass

    def _thread_refresh_stats(self):
        pass


def main():
    parser = argparse.ArgumentParser(description="Headless AutoTrade Bot")
    parser.add_argument(
        "--settings", type=str, default="modules/auto_trade/settings.yaml", help="Path to settings YAML file"
    )
    args = parser.parse_args()

    # 2. Fetch secrets from Cloud/AWS
    fetch_secrets_to_env()

    # 3. Load settings.yaml
    config = {}
    if os.path.exists(args.settings):
        try:
            with open(args.settings, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            log_info(f"Loaded config from {args.settings}")
        except Exception as e:
            log_error(f"Failed to load settings from {args.settings}: {e}")
    else:
        log_info(f"Settings file {args.settings} not found. Operating with empty/default settings.")

    # Create Mock Parent
    app = HeadlessApp(config)

    # 4. Khởi ScannerManager
    from modules.auto_trade.gui.main_window.scanner import ScannerManager

    scanner_manager = ScannerManager(app)
    app.scanner_manager = scanner_manager
    scanner_manager._start_scanner()
    log_info("ScannerManager started headlessly.")

    # 5. Khởi AutoTradeEngine
    from modules.auto_trade.gui.main_window.auto_trade import AutoTradeManager

    auto_trade_manager = AutoTradeManager(app)
    app.auto_trade_manager = auto_trade_manager
    auto_trade_manager.start()
    log_info("AutoTradeEngine started headlessly.")

    # Prevent immediate exit
    log_info("Bot is running. Waiting for signals...")

    # 6. signal.pause()
    try:
        if sys.platform != "win32":
            signal.pause()
        else:
            # signal.pause() is not available on Windows
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        log_info("Bot shutting down...")
        auto_trade_manager.stop()
        scanner_manager._stop_scanner()


if __name__ == "__main__":
    main()
