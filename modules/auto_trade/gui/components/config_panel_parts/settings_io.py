from typing import Dict

from modules.auto_trade.gui.components.config_panel_parts.auto_close_settings import (
    extract_auto_close_settings,
    load_auto_close_settings,
)


def _safe_get(panel, attr: str, fallback):
    widget = getattr(panel, attr, None)
    if widget is None:
        return fallback
    getter = getattr(widget, "get", None)
    if callable(getter):
        return getter()
    return fallback


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
            trailing_step_pct = float(_safe_get(panel, "trailing_step_pct_entry", "2.0"))
            if trailing_step_pct <= 0 or trailing_step_pct > 50:
                raise ValueError("Trailing step must be between 0 and 50")
        except ValueError as e:
            print(f"Invalid trailing step: {e}, using default 2.0")
            trailing_step_pct = 2.0

        try:
            trailing_max_steps = int(_safe_get(panel, "max_steps_entry", "5"))
            if trailing_max_steps < 1:
                raise ValueError("Max steps must be at least 1")
        except ValueError as e:
            print(f"Invalid max steps: {e}, using default 5")
            trailing_max_steps = 5

        try:
            negative_be_threshold = float(_safe_get(panel, "negative_be_threshold_entry", "2.0"))
            if negative_be_threshold <= 0 or negative_be_threshold > 100:
                raise ValueError("Negative BE threshold must be between 0 and 100")
        except ValueError as e:
            print(f"Invalid negative BE threshold: {e}, using default 2.0")
            negative_be_threshold = 2.0

        try:
            min_volume = float(_safe_get(panel, "min_volume_entry", "50"))
            if min_volume < 0:
                raise ValueError("Min volume must be >= 0")
        except ValueError:
            min_volume = 50.0

        try:
            min_signal_score = float(_safe_get(panel, "min_score_var", 0.7))
        except (TypeError, ValueError):
            min_signal_score = 0.7

        enable_xgboost = bool(_safe_get(panel, "enable_xgboost_var", True))
        whitelist_raw = str(_safe_get(panel, "whitelist_entry", "")).strip()
        whitelist_symbols = [s.strip() for s in whitelist_raw.split(",") if s.strip()] if whitelist_raw else []

        return {
            "risk": {
                "limits_enabled": bool(_safe_get(panel, "risk_limits_enabled_var", True)),
                "max_position_size": max_position_size,
                "max_open_positions": max_open_positions,
                "max_daily_loss": max_daily_loss,
                "default_leverage": _safe_get(panel, "default_leverage_var", "10x"),
            },
            "filters": {
                "min_volume": min_volume,
                "min_signal_score": min_signal_score,
                "enable_xgboost": enable_xgboost,
                "whitelist_symbols": whitelist_symbols,
            },
            "api": {
                "exchange": _safe_get(panel, "exchange_var", "Binance"),
                "mode": _safe_get(panel, "mode_var", "DRY_RUN"),
            },
            "tp_sl": {
                "default_tp": default_tp,
                "default_sl": default_sl,
                "trailing_stop": bool(_safe_get(panel, "trailing_stop_var", False)),
                "trailing_step_pct": trailing_step_pct,
                "trailing_limit_steps": bool(_safe_get(panel, "limit_trailing_steps_var", False)),
                "trailing_max_steps": trailing_max_steps,
                "mode": _safe_get(panel, "tp_sl_mode_var", "Percentage"),
                "negative_be_enabled": bool(_safe_get(panel, "negative_be_var", False)),
                "negative_be_threshold_pct": negative_be_threshold,
            },
            "auto_close": extract_auto_close_settings(panel),
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
            "filters": {
                "min_volume": 50.0,
                "min_signal_score": 0.7,
                "enable_xgboost": True,
                "whitelist_symbols": [],
            },
            "api": {
                "exchange": "Binance",
                "mode": "DRY_RUN",
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
            "auto_close": {
                "enabled": False,
                "max_duration_enabled": True,
                "max_duration_hours": 4.0,
                "daily_close_enabled": True,
                "daily_close_time": "22:00",
                "daily_close_days": "1234567",
                "grace_period_minutes": 5,
                "tp_offset_pct": 0.05,
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

    load_auto_close_settings(panel, settings)

    if "recovery" in settings and hasattr(panel, "recovery_panel"):
        panel.recovery_panel.load_config(settings["recovery"])
