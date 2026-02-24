from typing import Dict


def export_settings(panel):
    """Export settings to file."""
    try:
        from tkinter import filedialog

        from modules.auto_trade.gui.utils.settings_manager import SettingsManager

        manager = SettingsManager()
        file_path = filedialog.asksaveasfilename(
            title="Export Settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if file_path:
            if manager.export(file_path):
                print(f"Settings exported to {file_path}")
    except Exception as e:
        print(f"Error exporting settings: {e}")


def import_settings(panel):
    """Import settings from file."""
    try:
        from tkinter import filedialog

        from modules.auto_trade.gui.utils.settings_manager import SettingsManager

        manager = SettingsManager()
        file_path = filedialog.askopenfilename(
            title="Import Settings",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if file_path:
            if manager.import_settings(file_path):
                print(f"Settings imported from {file_path}")
                settings = manager.get_all()
                panel.load_settings(settings)
    except Exception as e:
        print(f"Error importing settings: {e}")


def reset_settings(panel):
    """Reset settings to defaults."""
    try:
        from tkinter import messagebox

        confirm = messagebox.askyesno(
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?\n\nThis cannot be undone.",
        )

        if confirm:
            from modules.auto_trade.gui.utils.settings_manager import SettingsManager

            manager = SettingsManager()
            if manager.reset_to_defaults():
                print("Settings reset to defaults")
                settings = manager.get_all()
                panel.load_settings(settings)
    except Exception as e:
        print(f"Error resetting settings: {e}")


def get_settings(panel) -> Dict:
    """
    Get current settings (excluding API credentials for security).

    Returns:
        Dictionary with risk and tp_sl settings.
        API credentials are NOT included and must be loaded separately.
    """
    try:
        try:
            max_position_size = float(panel.max_pos_size_entry.get())
            if max_position_size <= 0:
                raise ValueError("Max position size must be positive")
        except ValueError as e:
            print(f"Invalid max position size: {e}, using default 100.00")
            max_position_size = 100.0

        try:
            max_open_positions = int(panel.max_positions_entry.get())
            if max_open_positions <= 0:
                raise ValueError("Max open positions must be positive")
        except ValueError as e:
            print(f"Invalid max open positions: {e}, using default 3")
            max_open_positions = 3

        try:
            max_daily_loss = float(panel.max_daily_loss_entry.get())
            if max_daily_loss <= 0:
                raise ValueError("Max daily loss must be positive")
        except ValueError as e:
            print(f"Invalid max daily loss: {e}, using default 50.00")
            max_daily_loss = 50.0

        try:
            default_tp = float(panel.default_tp_entry.get())
            if default_tp <= 0 or default_tp > 100:
                raise ValueError("Default TP must be between 0 and 100")
        except ValueError as e:
            print(f"Invalid default TP: {e}, using default 5.0")
            default_tp = 5.0

        try:
            default_sl = float(panel.default_sl_entry.get())
            if default_sl <= 0 or default_sl > 100:
                raise ValueError("Default SL must be between 0 and 100")
        except ValueError as e:
            print(f"Invalid default SL: {e}, using default 2.5")
            default_sl = 2.5

        try:
            trailing_step_pct = float(panel.trailing_step_pct_entry.get())
            if trailing_step_pct <= 0 or trailing_step_pct > 50:
                raise ValueError("Trailing step must be between 0 and 50")
        except ValueError as e:
            print(f"Invalid trailing step: {e}, using default 2.0")
            trailing_step_pct = 2.0

        try:
            trailing_max_steps = int(panel.max_steps_entry.get())
            if trailing_max_steps < 1:
                raise ValueError("Max steps must be at least 1")
        except ValueError as e:
            print(f"Invalid max steps: {e}, using default 5")
            trailing_max_steps = 5

        try:
            negative_be_threshold = float(panel.negative_be_threshold_entry.get())
            if negative_be_threshold <= 0 or negative_be_threshold > 100:
                raise ValueError("Negative BE threshold must be between 0 and 100")
        except ValueError as e:
            print(f"Invalid negative BE threshold: {e}, using default 2.0")
            negative_be_threshold = 2.0

        return {
            "risk": {
                "limits_enabled": panel.risk_limits_enabled_var.get(),
                "max_position_size": max_position_size,
                "max_open_positions": max_open_positions,
                "max_daily_loss": max_daily_loss,
                "default_leverage": panel.default_leverage_var.get(),
            },
            "api": {
                "exchange": panel.exchange_var.get(),
                "mode": panel.mode_var.get(),
            },
            "tp_sl": {
                "default_tp": default_tp,
                "default_sl": default_sl,
                "trailing_stop": panel.trailing_stop_var.get(),
                "trailing_step_pct": trailing_step_pct,
                "trailing_limit_steps": panel.limit_trailing_steps_var.get(),
                "trailing_max_steps": trailing_max_steps,
                "mode": panel.tp_sl_mode_var.get(),
                "negative_be_enabled": panel.negative_be_var.get(),
                "negative_be_threshold_pct": negative_be_threshold,
            },
        }
    except Exception as e:
        print(f"Error getting settings: {e}")
        return {
            "risk": {
                "limits_enabled": True,
                "max_position_size": 100.0,
                "max_open_positions": 3,
                "max_daily_loss": 50.0,
                "default_leverage": "10x",
            },
            "api": {
                "exchange": "Binance",
            },
            "tp_sl": {
                "default_tp": 5.0,
                "default_sl": 2.5,
                "trailing_stop": False,
                "trailing_step_pct": 2.0,
                "trailing_limit_steps": False,
                "trailing_max_steps": 5,
                "mode": "Percentage",
            },
        }


def load_settings(panel, settings: Dict):
    """Load settings into UI."""
    if "risk" in settings:
        risk = settings["risk"]
        if hasattr(panel, "risk_limits_enabled_var"):
            panel.risk_limits_enabled_var.set(risk.get("limits_enabled", True))
        panel.max_pos_size_entry.delete(0, "end")
        panel.max_pos_size_entry.insert(0, str(risk.get("max_position_size", 100.0)))
        panel.max_positions_entry.delete(0, "end")
        panel.max_positions_entry.insert(0, str(risk.get("max_open_positions", 3)))
        panel.max_daily_loss_entry.delete(0, "end")
        panel.max_daily_loss_entry.insert(0, str(risk.get("max_daily_loss", 50.0)))
        panel.default_leverage_var.set(risk.get("default_leverage", "10x"))

    if "api" in settings:
        api = settings["api"]
        panel._suppress_mode_notify = True
        panel.mode_var.set(api.get("mode", "DRY_RUN"))
        panel.exchange_var.set(api.get("exchange", "Binance"))
        panel.api_key_entry.delete(0, "end")
        panel.api_secret_entry.delete(0, "end")
        panel._editing_credentials = False
        panel._on_mode_change(show_warning=False)
        panel._refresh_credentials_display()
        panel._suppress_mode_notify = False

    if "tp_sl" in settings:
        tp_sl = settings["tp_sl"]
        panel.default_tp_entry.delete(0, "end")
        panel.default_tp_entry.insert(0, str(tp_sl.get("default_tp", 5.0)))
        panel.default_sl_entry.delete(0, "end")
        panel.default_sl_entry.insert(0, str(tp_sl.get("default_sl", 2.5)))
        panel.trailing_stop_var.set(tp_sl.get("trailing_stop", False))
        panel.trailing_step_pct_entry.delete(0, "end")
        panel.trailing_step_pct_entry.insert(0, str(tp_sl.get("trailing_step_pct", 2.0)))
        panel.limit_trailing_steps_var.set(tp_sl.get("trailing_limit_steps", False))
        panel.max_steps_entry.delete(0, "end")
        panel.max_steps_entry.insert(0, str(tp_sl.get("trailing_max_steps", 5)))
        panel._on_limit_steps_toggle()
        panel.tp_sl_mode_var.set(tp_sl.get("mode", "Percentage"))
        panel.negative_be_var.set(tp_sl.get("negative_be_enabled", False))
        panel.negative_be_threshold_entry.delete(0, "end")
        panel.negative_be_threshold_entry.insert(0, str(tp_sl.get("negative_be_threshold_pct", 2.0)))

    if "recovery" in settings and hasattr(panel, "recovery_panel"):
        panel.recovery_panel.load_config(settings["recovery"])
