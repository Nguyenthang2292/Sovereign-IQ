import sys
from pathlib import Path

import customtkinter as ctk

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gui.components.account_frame import AccountFrame
from gui.components.auto_trade_control import AutoTradeControl
from gui.components.config_panel import ConfigPanel
from gui.components.positions_frame import PositionsFrame
from gui.components.scanner_control import ScannerControl
from gui.components.signals_frame import SignalsFrame
from gui.components.stats_frame import StatsFrame
from gui.components.trade_form import TradeFormFrame
from gui.utils.colors import Colors
from gui.utils.data_service import DataService
from gui.utils.settings_manager import SettingsManager
from gui.utils.modes import TradingMode
from gui.utils.threading_utils import PeriodicUpdater


class AutoTradeDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.settings_manager = SettingsManager()
        self.settings_manager.load()

        self.mode = self.settings_manager.get("api.mode", TradingMode.DRY_RUN)

        self.data_service = DataService(mode=self.mode)

        self.title(f"Auto Trade Dashboard - [{self.mode}]")
        self.geometry("1200x800")
        self.minsize(800, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._create_layout()
        self._setup_updaters()
        self._apply_settings()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _create_layout(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._create_header()

        content_frame = ctk.CTkFrame(self)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)

        # Create tabview
        self.tabview = ctk.CTkTabview(content_frame)
        self.tabview.pack(fill="both", expand=True)

        # Dashboard tab (existing components)
        dashboard_tab = self.tabview.add("Dashboard")
        self._populate_dashboard_tab(dashboard_tab)

        # Trading tab (NEW)
        trading_tab = self.tabview.add("Trading")
        self._populate_trading_tab(trading_tab)

        # Settings tab (NEW)
        settings_tab = self.tabview.add("Settings")
        self._populate_settings_tab(settings_tab)

        self._create_status_bar()
        self._update_mode_display()

    def _populate_dashboard_tab(self, parent):
        """Create dashboard interface"""
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        left_panel = ctk.CTkFrame(parent)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.account_frame = AccountFrame(left_panel)
        self.account_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.stats_frame = StatsFrame(left_panel)
        self.stats_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        right_panel = ctk.CTkFrame(parent)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=1)

        self.signals_frame = SignalsFrame(right_panel)
        self.signals_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.positions_frame = PositionsFrame(right_panel, on_action_callback=self.on_position_action)
        self.positions_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def _populate_trading_tab(self, parent):
        """Create trading interface"""
        # Configure grid
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        # Left: Manual Trade Form
        self.trade_form = TradeFormFrame(parent, on_trade_callback=self.on_trade_executed)
        self.trade_form.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Right: Auto-Trade Control
        self.auto_trade_control = AutoTradeControl(parent, on_toggle_callback=self.on_auto_trade_toggle)
        self.auto_trade_control.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

    def _populate_settings_tab(self, parent):
        """Create settings interface"""
        # Configure grid
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        # Left: Config Panel
        self.config_panel = ConfigPanel(parent, on_settings_change=self.on_settings_change)
        self.config_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Right: Scanner Control
        self.scanner_control = ScannerControl(
            parent, on_scan_toggle=self.on_scan_toggle, on_config_change=self.on_scanner_config_change
        )
        self.scanner_control.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

    def _create_header(self):
        header_frame = ctk.CTkFrame(self, height=60)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))

        title_label = ctk.CTkLabel(header_frame, text="Auto Trade Dashboard", font=("Arial", 20, "bold"))
        title_label.pack(side="left", padx=20)

        mode_colors = {
            TradingMode.PRODUCTION: Colors.PRODUCTION,
            TradingMode.DEMO: Colors.DEMO,
            TradingMode.DRY_RUN: Colors.DRY_RUN,
        }

        mode_color = mode_colors.get(self.mode, Colors.DRY_RUN)
        mode_text = self.mode.replace("_", " ")

        self.header_mode_label = ctk.CTkLabel(
            header_frame, text=f"[{mode_text}]", font=("Arial", 12), text_color=mode_color
        )
        self.header_mode_label.pack(side="right", padx=20)

    def _create_status_bar(self):
        status_frame = ctk.CTkFrame(self, height=30)
        status_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.status_label = ctk.CTkLabel(status_frame, text="Ready", font=("Arial", 10), text_color="gray")
        self.status_label.pack(side="left", padx=10)

        self.last_update_label = ctk.CTkLabel(
            status_frame, text="Last update: --", font=("Arial", 10), text_color="gray"
        )
        self.last_update_label.pack(side="right", padx=10)

    def _update_mode_display(self):
        """Update mode indicator in stats frame and header"""
        mode_colors = {
            TradingMode.PRODUCTION: Colors.PRODUCTION,
            TradingMode.DEMO: Colors.DEMO,
            TradingMode.DRY_RUN: Colors.DRY_RUN,
        }

        mode_color = mode_colors.get(self.mode, Colors.DRY_RUN)
        mode_text = self.mode.replace("_", " ")

        if hasattr(self, "stats_frame"):
            self.stats_frame.mode_indicator.destroy()
            from gui.components.stats_frame import ModeIndicator

            self.stats_frame.mode_indicator = ModeIndicator(self.stats_frame, self.mode)
            self.stats_frame.mode_indicator.pack(pady=(0, 10))

        if hasattr(self, "header_mode_label"):
            self.header_mode_label.configure(text=f"[{mode_text}]", text_color=mode_color)

    def _setup_updaters(self):
        def refresh_all():
            self.refresh_signals()
            self.refresh_positions()
            self.refresh_account()
            self.refresh_stats()
            self._update_timestamp()

        self.signal_updater = PeriodicUpdater(self.refresh_signals, interval=30)
        self.position_updater = PeriodicUpdater(self.refresh_positions, interval=10)
        self.account_updater = PeriodicUpdater(self.refresh_account, interval=60)
        self.stats_updater = PeriodicUpdater(self.refresh_stats, interval=60)

        refresh_all()

        self.signal_updater.start()
        self.position_updater.start()
        self.account_updater.start()
        self.stats_updater.start()

    def refresh_signals(self):
        signals = self.data_service.get_signals()
        self.after(0, lambda: self.signals_frame.update_signals(signals))

    def refresh_positions(self):
        positions = self.data_service.get_positions()
        self.after(0, lambda: self.positions_frame.update_positions(positions))

    def refresh_account(self):
        data = self.data_service.get_account_data()
        if data:
            self.after(0, lambda: self.account_frame.update_data(data))

    def refresh_stats(self):
        stats = self.data_service.get_quick_stats()
        if stats:
            self.after(0, lambda: self.stats_frame.update_data(stats))

    def _update_timestamp(self):
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.after(0, lambda: self.last_update_label.configure(text=f"Last update: {timestamp}"))

    def on_trade_executed(self):
        """Callback when manual trade is executed"""
        print("Trade executed! Refreshing positions...")
        self.refresh_positions()
        self.refresh_account()

    def on_auto_trade_toggle(self, enabled: bool):
        """Callback when auto-trade is toggled"""
        print(f"Auto-trade {'enabled' if enabled else 'disabled'}")

        if enabled:
            self._start_auto_trading()
        else:
            self._stop_auto_trading()

    def _start_auto_trading(self):
        """Start auto-trading loop"""
        from gui.utils.threading_utils import PeriodicUpdater

        self.auto_trade_updater = PeriodicUpdater(
            self._auto_trade_cycle,
            interval=60,  # Check for signals every 60s
        )
        self.auto_trade_updater.start()
        print("Auto-trading started")

    def _stop_auto_trading(self):
        """Stop auto-trading loop"""
        if hasattr(self, "auto_trade_updater"):
            self.auto_trade_updater.stop()
            print("Auto-trading stopped")

    def _auto_trade_cycle(self):
        """
        Auto-trading cycle:
        1. Check for new qualifying signals
        2. Validate against risk rules
        3. Execute trade if conditions met
        """
        try:
            from modules.auto_trade.order_executor import OrderExecutor
            from modules.auto_trade.signal_selector import SignalSelector

            # Get recent signals
            signals = self.data_service.get_signals(min_score=0.7)

            # Filter and select best signal
            selector = SignalSelector()
            selected_signal = selector.select_best_signal(signals)

            if not selected_signal:
                print("No qualifying signals for auto-trade")
                return

            # Check risk limits
            if not self._check_risk_limits():
                print("Risk limits exceeded, skipping trade")
                return

            # Execute trade
            executor = OrderExecutor()
            result = executor.execute_from_signal(selected_signal)

            if result and result.get("success"):
                print(f"Auto-trade executed: {selected_signal['symbol']}")
                # Refresh UI on main thread
                self.after(0, self.refresh_positions)
                self.after(0, self.refresh_account)

        except Exception as e:
            print(f"Error in auto-trade cycle: {e}")

    def _check_risk_limits(self) -> bool:
        """
        Check if trading within risk limits:
        - Max open positions
        - Max daily loss
        - Max position size
        """
        try:
            positions = self.data_service.get_positions()

            # Get settings
            max_positions = self.settings_manager.get("risk.max_open_positions", 3)

            # Max open positions from settings
            if len(positions) >= max_positions:
                return False

            # TODO: Check daily loss limit
            # TODO: Check max position size

            return True
        except:
            return False

    def _apply_settings(self):
        """Apply loaded settings to application"""
        try:
            # Apply UI preferences
            theme = self.settings_manager.get("ui.theme", "dark")
            font_size = self.settings_manager.get("ui.font_size", 12)

            if theme == "light":
                ctk.set_appearance_mode("light")
            else:
                ctk.set_appearance_mode("dark")

            # Set default font size (note: this would require more complex implementation)
            print(f"Applied settings: Theme={theme}, Font Size={font_size}")

            # Load settings into components
            all_settings = self.settings_manager.get_all()

            if hasattr(self, "config_panel"):
                self.config_panel.load_settings(all_settings)

            if hasattr(self, "scanner_control"):
                scanner_settings = all_settings.get("scanner", {})
                self.scanner_control.load_config(scanner_settings)

        except Exception as e:
            print(f"Error applying settings: {e}")

    def on_settings_change(self, setting_type: str, value=None):
        """Handle settings change from ConfigPanel"""
        try:
            print(f"Settings changed: {setting_type} = {value}")

            # Update settings manager
            if hasattr(self, "config_panel"):
                current_settings = self.config_panel.get_settings()
                self.settings_manager.settings.update(current_settings)
                self.settings_manager.save()

                # Check if mode changed
                new_mode = current_settings.get("api", {}).get("mode")
                if new_mode and new_mode != self.mode:
                    self.mode = new_mode
                    self.title(f"Auto Trade Dashboard - [{self.mode}]")
                    self._update_mode_display()

        except Exception as e:
            print(f"Error handling settings change: {e}")

    def on_scan_toggle(self, action):
        """Handle scanner start/stop from ScannerControl"""
        try:
            print(f"Scanner action: {action}")

            if action == True:
                self._start_scanner()
            elif action == False:
                self._stop_scanner()
            elif action == "manual":
                self._manual_scan()

        except Exception as e:
            print(f"Error handling scanner toggle: {e}")

    def on_scanner_config_change(self, config: dict):
        """Handle scanner configuration change"""
        try:
            print(f"Scanner config changed: {config}")

            # Update settings manager
            self.settings_manager.set("scanner", config)
            self.settings_manager.save()

        except Exception as e:
            print(f"Error handling scanner config change: {e}")

    def _start_scanner(self):
        """Start scanner loop"""
        from gui.utils.threading_utils import PeriodicUpdater

        config = self.settings_manager.get("scanner", {})
        interval = config.get("scan_interval", 5) * 60  # Convert to seconds

        self.scanner_updater = PeriodicUpdater(self._scanner_cycle, interval=interval)
        self.scanner_updater.start()
        print("Scanner started")

    def _stop_scanner(self):
        """Stop scanner loop"""
        if hasattr(self, "scanner_updater"):
            self.scanner_updater.stop()
            print("Scanner stopped")

    def _manual_scan(self):
        """Trigger manual scan"""
        self._scanner_cycle()
        self.scanner_control.update_last_scan_time()

    def _scanner_cycle(self):
        """Scanner cycle"""
        try:
            print("Running scanner cycle...")
            # TODO: Implement actual scanning logic
            # For now, just refresh signals
            self.refresh_signals()
            self.scanner_control.update_last_scan_time()
        except Exception as e:
            print(f"Error in scanner cycle: {e}")

    def on_position_action(self, action_data: dict):
        """Handle position actions from GUI"""
        print(f"Position action received: {action_data}")

        if not self.data_service.exchange_manager:
            print("Error: Exchange manager not initialized")
            return {"success": False, "error": "Exchange manager unavailable"}

        # Try to get the underlying BinanceClient
        mgr = self.data_service.exchange_manager
        target = mgr
        if hasattr(mgr, "client") and hasattr(mgr.client, "close_position"):
            target = mgr.client

        action = action_data.get("action")
        symbol = action_data.get("symbol")

        try:
            if action == "close_position":
                side = action_data.get("side")
                size = action_data.get("size")
                close_type = action_data.get("type", "market")
                limit_price = action_data.get("limit_price")
                return target.close_position(symbol, side, size, close_type, limit_price)

            elif action == "partial_close":
                side = action_data.get("side")
                size = action_data.get("size")
                # Partial close is just a market close of a specific size
                return target.close_position(symbol, side, size, "market")

            elif action == "modify_tp_sl":
                position_id = action_data.get("position_id")
                tp = action_data.get("take_profit")
                sl = action_data.get("stop_loss")
                return target.modify_tp_sl(symbol, position_id, tp, sl)

            elif action == "add_margin":
                amount = action_data.get("amount")
                # type 1 = add margin
                return target.modify_margin(symbol, amount, type=1)

            elif action == "cancel_orders":
                return target.cancel_open_orders(symbol)

        except AttributeError:
            print(f"Error: Target {target} does not support action {action}")
            return {"success": False, "error": f"Method not supported: {action}"}
        except Exception as e:
            print(f"Error executing {action}: {e}")
            return {"success": False, "error": str(e)}

        return {"success": False, "error": "Unknown action"}

    def on_closing(self):
        # Save settings before closing
        try:
            if hasattr(self, "settings_manager"):
                self.settings_manager.save()
                print("Settings saved on exit")
        except Exception as e:
            print(f"Error saving settings: {e}")

        # Stop all updaters
        self.signal_updater.stop()
        self.position_updater.stop()
        self.account_updater.stop()
        self.stats_updater.stop()
        if hasattr(self, "auto_trade_updater"):
            self.auto_trade_updater.stop()
        if hasattr(self, "scanner_updater"):
            self.scanner_updater.stop()

        self.destroy()
