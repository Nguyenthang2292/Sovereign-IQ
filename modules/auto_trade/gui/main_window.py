import queue
import sys
from pathlib import Path
from typing import Optional

import customtkinter as ctk

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gui.components.account_frame import AccountFrame
from gui.components.auto_trade_control import AutoTradeControl
from gui.components.config_panel import ConfigPanel
from gui.components.positions_frame import PositionsFrame
from gui.components.recovery_panel import RecoveryPanel
from gui.components.scanner_control import ScannerControl
from gui.components.signals_frame import SignalsFrame
from gui.components.stats_frame import StatsFrame
from gui.components.trade_form import TradeFormFrame
from gui.utils.colors import Colors
from gui.utils.data_service import DataService
from gui.utils.modes import TradingMode
from gui.utils.settings_manager import SettingsManager
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

        self._update_queue = queue.Queue()
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

        # Settings tab (includes Recovery in compact mode)
        settings_tab = self.tabview.add("Settings")
        self._populate_settings_tab(settings_tab)

        # Database tab
        database_tab = self.tabview.add("Database")
        self._populate_database_tab(database_tab)

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
        """Create settings interface with Recovery panel"""
        # Configure grid - 60/40 split
        parent.grid_columnconfigure(0, weight=3)
        parent.grid_columnconfigure(1, weight=2)
        parent.grid_rowconfigure(0, weight=1)

        # Left: Config Panel (60%)
        self.config_panel = ConfigPanel(parent, on_settings_change=self.on_settings_change)
        self.config_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Right: Scanner Control + Recovery Panel (40%)
        right_panel = ctk.CTkFrame(parent, fg_color="transparent")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        # Scanner Control (top)
        self.scanner_control = ScannerControl(
            right_panel, on_scan_toggle=self.on_scan_toggle, on_config_change=self.on_scanner_config_change
        )
        self.scanner_control.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        # Recovery Panel (bottom - compact mode)
        self.recovery_panel = RecoveryPanel(
            right_panel,
            on_config_change=self.on_recovery_config_change,
            mode=self.mode,
            compact=True,
        )
        self.recovery_panel.grid(row=1, column=0, sticky="nsew", pady=(5, 0))

    def _populate_database_tab(self, parent):
        """Create database testing interface"""
        from gui.components.database_panel import DatabasePanel

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        self.database_panel = DatabasePanel(parent, self.settings_manager)
        self.database_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

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

        # PeriodicUpdater runs in a background thread; callbacks must not call self.after().
        # They put (kind, data) in _update_queue; main thread drains via _drain_update_queue.
        self.signal_updater = PeriodicUpdater(self._thread_refresh_signals, interval=30)
        self.position_updater = PeriodicUpdater(self._thread_refresh_positions, interval=10)
        self.account_updater = PeriodicUpdater(self._thread_refresh_account, interval=60)
        self.stats_updater = PeriodicUpdater(self._thread_refresh_stats, interval=60)

        refresh_all()

        self.signal_updater.start()
        self.position_updater.start()
        self.account_updater.start()
        self.stats_updater.start()
        self.after(100, self._drain_update_queue)

    def _drain_update_queue(self):
        """Process UI updates from background thread (must run on main thread)."""
        try:
            while True:
                kind, data = self._update_queue.get_nowait()
                if kind == "signals":
                    self.signals_frame.update_signals(data)
                elif kind == "positions":
                    self.positions_frame.update_positions(data)
                elif kind == "account" and data:
                    self.account_frame.update_data(data)
                elif kind == "stats" and data:
                    self.stats_frame.update_data(data)
                elif kind == "scanner_done":
                    if hasattr(self, "scanner_control"):
                        self.scanner_control.update_last_scan_time()
        except queue.Empty:
            pass
        self.after(100, self._drain_update_queue)

    def _thread_refresh_signals(self):
        signals = self.data_service.get_signals()
        self._update_queue.put(("signals", signals))

    def _thread_refresh_positions(self):
        positions = self.data_service.get_positions()
        self._update_queue.put(("positions", positions))

    def _thread_refresh_account(self):
        data = self.data_service.get_account_data()
        self._update_queue.put(("account", data))

    def _thread_refresh_stats(self):
        stats = self.data_service.get_quick_stats()
        self._update_queue.put(("stats", stats))

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
            from modules.auto_trade.core.atc_scanner import SignalResult
            from modules.auto_trade.core.signal_selector import SignalSelector
            from modules.auto_trade.execution.order_executor import OrderExecutor

            # Get recent signals (list of dicts: symbol, signal, score, time)
            signals = self.data_service.get_signals(min_score=0.7)
            xb_signals = []
            for s in signals:
                sym = (s.get("symbol") or "").strip()
                if sym and "/" not in sym:
                    sym = f"{sym.replace('USDT', '')}/USDT"
                if not sym:
                    continue
                xb_signals.append(
                    SignalResult(
                        symbol=sym,
                        score=float(s.get("score", 0)),
                        signal_type=(s.get("signal") or "LONG").upper(),
                        details={"time": s.get("time", "")},
                        strengths={},
                    )
                )

            # Filter and select best signal (no Gemini data in GUI auto-trade)
            selector = SignalSelector()
            selected_signal = selector.select_best_signal(xb_signals, gemini_signals={})

            if not selected_signal:
                print("No qualifying signals for auto-trade")
                return

            # Get position sizing parameters from settings
            default_position_size = self.settings_manager.get("trading.default_position_size", 100.0)
            default_leverage = self.settings_manager.get("trading.default_leverage", 2)

            # Check risk limits with symbol, position size, and leverage
            if not self._check_risk_limits(
                symbol=selected_signal.symbol, position_size=default_position_size, leverage=default_leverage
            ):
                print("Risk limits exceeded, skipping trade")
                return

            # Execute trade (OrderExecutor expects a dict)
            sig_dict = {
                "symbol": selected_signal.symbol,
                "signal": selected_signal.signal_type,
                "score": selected_signal.final_score
                if hasattr(selected_signal, "final_score")
                else selected_signal.score,
            }
            executor = OrderExecutor()
            result = executor.execute_from_signal(sig_dict)

            if result and result.get("success"):
                print(f"Auto-trade executed: {sig_dict['symbol']}")
                # Refresh UI on main thread
                self.after(0, self.refresh_positions)
                self.after(0, self.refresh_account)

        except Exception as e:
            print(f"Error in auto-trade cycle: {e}")

    def _check_risk_limits(
        self,
        symbol: Optional[str] = None,
        position_size: Optional[float] = None,
        leverage: Optional[int] = None,
    ) -> bool:
        """
        Check if trading within risk limits:
        - Max open positions
        - Max daily loss
        - Max position size
        - Total exposure
        - Per-symbol position limits
        - Leverage limits
        - Account balance

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            position_size: Intended position size in USDT
            leverage: Intended leverage

        Returns:
            True if within all risk limits, False otherwise
        """
        try:
            # 1. Get positions and validate (Priority 3)
            positions = self.data_service.get_positions()
            if positions is None:
                print("Warning: Could not fetch positions for risk check")
                return False

            if not isinstance(positions, (list, tuple)):
                print(f"Error: Invalid positions type: {type(positions)}")
                return False

            # 2. Get account data
            account_data = self.data_service.get_account_data()
            if not account_data:
                print("Warning: Could not fetch account data for risk check")
                return False

            balance = account_data.get("balance", 0)
            if balance <= 0:
                print("Error: Invalid account balance")
                return False

            # 3. Max open positions check (Priority 1 - with logging)
            max_positions = self.settings_manager.get("risk.max_open_positions", 3)

            # Type validation (Priority 6)
            if not isinstance(max_positions, int) or max_positions <= 0:
                print(f"Error: Invalid max_positions setting: {max_positions}, using default 3")
                max_positions = 3

            if len(positions) >= max_positions:
                print(f"Risk limit exceeded: Max positions reached ({len(positions)}/{max_positions})")
                return False

            # 4. Daily loss limit check (Priority 4)
            max_daily_loss_pct = self.settings_manager.get("risk.max_daily_loss_pct", 5.0)

            # Type validation (Priority 6)
            if not isinstance(max_daily_loss_pct, (int, float)) or max_daily_loss_pct <= 0:
                print(f"Error: Invalid max_daily_loss_pct setting: {max_daily_loss_pct}, using default 5.0")
                max_daily_loss_pct = 5.0

            daily_pnl_pct = account_data.get("daily_pnl_pct", 0)
            if daily_pnl_pct <= -max_daily_loss_pct:
                print(f"Risk limit exceeded: Daily loss limit hit ({daily_pnl_pct:.2f}% / -{max_daily_loss_pct}%)")
                return False

            # 5. Total exposure check (Priority 5)
            max_exposure_pct = self.settings_manager.get("risk.max_exposure_pct", 30.0)

            # Type validation (Priority 6)
            if not isinstance(max_exposure_pct, (int, float)) or max_exposure_pct <= 0:
                print(f"Error: Invalid max_exposure_pct setting: {max_exposure_pct}, using default 30.0")
                max_exposure_pct = 30.0

            # Calculate current total exposure
            total_exposure = sum(abs(float(p.get("notional", 0))) for p in positions)

            # Add intended position to exposure if provided
            if position_size is not None and leverage is not None:
                total_exposure += position_size * leverage

            exposure_pct = (total_exposure / balance) * 100
            if exposure_pct >= max_exposure_pct:
                print(f"Risk limit exceeded: Max exposure reached ({exposure_pct:.1f}% / {max_exposure_pct}%)")
                return False

            # 6. Max position size check (Priority 7)
            if position_size is not None:
                max_position_size_pct = self.settings_manager.get("risk.max_position_size_pct", 10.0)

                # Type validation (Priority 6)
                if not isinstance(max_position_size_pct, (int, float)) or max_position_size_pct <= 0:
                    print(f"Error: Invalid max_position_size_pct: {max_position_size_pct}, using default 10.0")
                    max_position_size_pct = 10.0

                max_position_size = balance * (max_position_size_pct / 100)
                if position_size > max_position_size:
                    print(
                        f"Risk limit exceeded: Position size too large "
                        f"({position_size:.2f} USDT > {max_position_size:.2f} USDT / "
                        f"{max_position_size_pct}% of balance)"
                    )
                    return False

            # 7. Per-symbol position limit (Priority 8)
            if symbol is not None:
                max_per_symbol = self.settings_manager.get("risk.max_positions_per_symbol", 1)

                # Type validation (Priority 6)
                if not isinstance(max_per_symbol, int) or max_per_symbol <= 0:
                    print(f"Error: Invalid max_positions_per_symbol: {max_per_symbol}, using default 1")
                    max_per_symbol = 1

                # Count existing positions for this symbol
                symbol_positions = [
                    p
                    for p in positions
                    if p.get("symbol", "").replace("USDT", "/USDT") == symbol or p.get("symbol") == symbol
                ]

                if len(symbol_positions) >= max_per_symbol:
                    print(
                        f"Risk limit exceeded: Max positions for {symbol} reached "
                        f"({len(symbol_positions)}/{max_per_symbol})"
                    )
                    return False

            # 8. Leverage limit check (Priority 9)
            if leverage is not None:
                max_leverage = self.settings_manager.get("risk.max_leverage", 5)

                # Type validation (Priority 6)
                if not isinstance(max_leverage, int) or max_leverage <= 0:
                    print(f"Error: Invalid max_leverage setting: {max_leverage}, using default 5")
                    max_leverage = 5

                if leverage > max_leverage:
                    print(f"Risk limit exceeded: Leverage too high ({leverage}x > {max_leverage}x)")
                    return False

            # 9. Minimum account balance check (additional safety)
            min_balance = self.settings_manager.get("risk.min_account_balance", 10.0)

            # Type validation (Priority 6)
            if not isinstance(min_balance, (int, float)) or min_balance < 0:
                print(f"Error: Invalid min_account_balance: {min_balance}, using default 10.0")
                min_balance = 10.0

            if balance < min_balance:
                print(
                    f"Risk limit exceeded: Account balance too low "
                    f"({balance:.2f} USDT < {min_balance:.2f} USDT minimum)"
                )
                return False

            # All checks passed
            return True

        except Exception as e:
            # Priority 1 & 2: Add logging to exception handler
            print(f"Error checking risk limits: {e}")
            import traceback

            traceback.print_exc()
            return False  # Fail-safe: reject trade on error

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

                # Check if theme changed
                new_theme = current_settings.get("ui", {}).get("theme")
                if new_theme:
                    self._refresh_theme_colors()

        except Exception as e:
            print(f"Error handling settings change: {e}")

    def _refresh_theme_colors(self):
        """Refresh all component colors when theme changes (light/dark)."""
        from gui.utils.colors import Colors

        def _update_frame_colors(widget):
            """Recursively set card-like frames to current theme card bg."""
            try:
                if isinstance(widget, ctk.CTkFrame):
                    current_fg = widget.cget("fg_color")
                    if current_fg and current_fg != "transparent":
                        widget.configure(fg_color=Colors.get_card_bg())
                for child in widget.winfo_children():
                    _update_frame_colors(child)
            except Exception:
                pass

        try:
            for name in [
                "account_frame",
                "stats_frame",
                "positions_frame",
                "trade_form",
                "auto_trade_control",
                "scanner_control",
                "config_panel",
            ]:
                if hasattr(self, name):
                    _update_frame_colors(getattr(self, name))

            if hasattr(self, "signals_frame"):
                self.signals_frame._configure_table_tags()

            if hasattr(self, "recovery_panel"):
                _update_frame_colors(self.recovery_panel)

            print("Theme colors refreshed")
        except Exception as e:
            print(f"Error refreshing theme colors: {e}")

    def on_scan_toggle(self, action):
        """Handle scanner start/stop from ScannerControl"""
        try:
            print(f"Scanner action: {action}")

            if action:
                self._start_scanner()
            elif not action:
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

    def on_recovery_config_change(self, event_type: str, data):
        """Handle recovery configuration change"""
        try:
            print(f"Recovery {event_type}: {data}")

            if event_type == "recovery_started":
                # Save recovery config to settings
                self.settings_manager.set("recovery.enabled", True)
                self.settings_manager.set("recovery.config", data)
                self.settings_manager.save()
            elif event_type == "recovery_reset":
                self.settings_manager.set("recovery.enabled", False)
                self.settings_manager.save()
            elif event_type == "recovery_alert":
                # Show alert notification in status bar
                if hasattr(self, "status_label"):
                    self.status_label.configure(text=f"Recovery: {data}")

        except Exception as e:
            print(f"Error handling recovery config change: {e}")

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
        """Scanner cycle (runs in background thread; enqueues UI updates)."""
        try:
            print("Running scanner cycle...")
            # TODO: Implement actual scanning logic
            # For now, refresh signals and notify UI via queue (no self.after from thread)
            signals = self.data_service.get_signals()
            self._update_queue.put(("signals", signals))
            self._update_queue.put(("scanner_done", None))
        except Exception as e:
            print(f"Error in scanner cycle: {e}")

    def on_position_action(self, action_data: dict):
        """Handle position actions from GUI"""
        print(f"Position action received: {action_data}")

        if not self.data_service.exchange_manager:
            print("Error: Exchange manager not initialized")
            return {"success": False, "error": "Exchange manager unavailable"}

        # Try to get the underlying trading client (e.g. BinanceClient) that supports position actions.
        # ExchangeManager does not define "client"; use getattr for optional attribute.
        mgr = self.data_service.exchange_manager
        client = getattr(mgr, "client", None)
        if client is None or not hasattr(client, "close_position"):
            return {"success": False, "error": "Position actions not available (no trading client)"}
        target = client

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
