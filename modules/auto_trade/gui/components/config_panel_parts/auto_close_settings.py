from typing import Any, Dict

import customtkinter as ctk
from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.fonts import Fonts


_ADAPTIVE_MANAGED_KEYS = {
    "enabled",
    "min_duration_hours",
    "max_duration_hours",
    "lookback_days",
    "timeframe",
}


def build_auto_close_section(panel, parent_frame, *, show_separator: bool = True, show_title: bool = True) -> None:
    """Build Auto-Close settings UI section inside TP/SL area."""
    if show_separator:
        separator = ctk.CTkLabel(parent_frame, text="-------------------------", text_color=Colors.TEXT_MUTED)
        separator.pack(anchor="w", pady=(12, 5))

    if show_title:
        ctk.CTkLabel(parent_frame, text="Auto-Close Timer", font=Fonts.H2).pack(anchor="w", pady=(0, 8))

    panel.auto_close_enabled_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(
        parent_frame,
        text="Enable Auto-Close Timer",
        variable=panel.auto_close_enabled_var,
    ).pack(anchor="w", pady=(0, 8))

    panel.auto_close_max_duration_enabled_var = ctk.BooleanVar(value=True)
    ctk.CTkCheckBox(
        parent_frame,
        text="Enable max duration timeout",
        variable=panel.auto_close_max_duration_enabled_var,
    ).pack(anchor="w", pady=(0, 4))

    ctk.CTkLabel(parent_frame, text="Max duration (hours):", font=Fonts.INPUT).pack(anchor="w", pady=(5, 2))
    panel.auto_close_max_duration_hours_entry = ctk.CTkEntry(parent_frame, placeholder_text="4.0")
    panel.auto_close_max_duration_hours_entry.pack(fill="x", pady=(2, 8))
    panel.auto_close_max_duration_hours_entry.insert(0, "4.0")

    panel.auto_close_daily_enabled_var = ctk.BooleanVar(value=True)
    ctk.CTkCheckBox(
        parent_frame,
        text="Enable daily close window",
        variable=panel.auto_close_daily_enabled_var,
    ).pack(anchor="w", pady=(0, 4))

    ctk.CTkLabel(parent_frame, text="Daily close time (UTC HH:MM):", font=Fonts.INPUT).pack(anchor="w", pady=(5, 2))
    panel.auto_close_daily_time_entry = ctk.CTkEntry(parent_frame, placeholder_text="22:00")
    panel.auto_close_daily_time_entry.pack(fill="x", pady=(2, 8))
    panel.auto_close_daily_time_entry.insert(0, "22:00")

    ctk.CTkLabel(parent_frame, text="Daily close days (1=Mon...7=Sun):", font=Fonts.INPUT).pack(anchor="w", pady=(5, 2))
    panel.auto_close_daily_days_entry = ctk.CTkEntry(parent_frame, placeholder_text="1234567")
    panel.auto_close_daily_days_entry.pack(fill="x", pady=(2, 8))
    panel.auto_close_daily_days_entry.insert(0, "1234567")

    ctk.CTkLabel(parent_frame, text="Grace period (minutes):", font=Fonts.INPUT).pack(anchor="w", pady=(5, 2))
    panel.auto_close_grace_minutes_entry = ctk.CTkEntry(parent_frame, placeholder_text="5")
    panel.auto_close_grace_minutes_entry.pack(fill="x", pady=(2, 8))
    panel.auto_close_grace_minutes_entry.insert(0, "5")

    ctk.CTkLabel(parent_frame, text="TP offset (%) for quasi-market close:", font=Fonts.INPUT).pack(anchor="w", pady=(5, 2))
    panel.auto_close_tp_offset_pct_entry = ctk.CTkEntry(parent_frame, placeholder_text="0.05")
    panel.auto_close_tp_offset_pct_entry.pack(fill="x", pady=(2, 8))
    panel.auto_close_tp_offset_pct_entry.insert(0, "0.05")

    separator_adaptive = ctk.CTkLabel(parent_frame, text="-------------------------", text_color=Colors.TEXT_MUTED)
    separator_adaptive.pack(anchor="w", pady=(12, 5))

    ctk.CTkLabel(parent_frame, text="Adaptive Close", font=Fonts.H2).pack(anchor="w", pady=(0, 8))

    panel.adaptive_close_enabled_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(
        parent_frame,
        text="Enable adaptive deadline from regime analysis",
        variable=panel.adaptive_close_enabled_var,
    ).pack(anchor="w", pady=(0, 8))

    ctk.CTkLabel(parent_frame, text="Adaptive min duration (hours):", font=Fonts.INPUT).pack(anchor="w", pady=(5, 2))
    panel.adaptive_close_min_duration_hours_entry = ctk.CTkEntry(parent_frame, placeholder_text="1.0")
    panel.adaptive_close_min_duration_hours_entry.pack(fill="x", pady=(2, 8))
    panel.adaptive_close_min_duration_hours_entry.insert(0, "1.0")

    ctk.CTkLabel(parent_frame, text="Adaptive max duration (hours):", font=Fonts.INPUT).pack(anchor="w", pady=(5, 2))
    panel.adaptive_close_max_duration_hours_entry = ctk.CTkEntry(parent_frame, placeholder_text="12.0")
    panel.adaptive_close_max_duration_hours_entry.pack(fill="x", pady=(2, 8))
    panel.adaptive_close_max_duration_hours_entry.insert(0, "12.0")

    ctk.CTkLabel(parent_frame, text="Lookback days:", font=Fonts.INPUT).pack(anchor="w", pady=(5, 2))
    panel.adaptive_close_lookback_days_entry = ctk.CTkEntry(parent_frame, placeholder_text="60")
    panel.adaptive_close_lookback_days_entry.pack(fill="x", pady=(2, 8))
    panel.adaptive_close_lookback_days_entry.insert(0, "60")

    ctk.CTkLabel(parent_frame, text="Adaptive timeframe:", font=Fonts.INPUT).pack(anchor="w", pady=(5, 2))
    panel.adaptive_close_timeframe_entry = ctk.CTkEntry(parent_frame, placeholder_text="15m")
    panel.adaptive_close_timeframe_entry.pack(fill="x", pady=(2, 8))
    panel.adaptive_close_timeframe_entry.insert(0, "15m")

    panel._adaptive_close_extra = {}


def extract_auto_close_settings(panel) -> Dict[str, Any]:
    """Extract and validate auto_close settings from UI widgets."""
    try:
        max_duration_hours = float(panel.auto_close_max_duration_hours_entry.get())
        if max_duration_hours <= 0:
            raise ValueError("max_duration_hours")
    except ValueError:
        max_duration_hours = 4.0

    daily_time = str(panel.auto_close_daily_time_entry.get() or "22:00").strip() or "22:00"
    parts = daily_time.split(":")
    valid_daily_time = (
        len(parts) == 2
        and parts[0].isdigit()
        and parts[1].isdigit()
        and 0 <= int(parts[0]) <= 23
        and 0 <= int(parts[1]) <= 59
    )
    if not valid_daily_time:
        daily_time = "22:00"

    daily_days = str(panel.auto_close_daily_days_entry.get() or "1234567").strip() or "1234567"
    if any(ch not in "1234567" for ch in daily_days):
        daily_days = "1234567"

    try:
        grace_period_minutes = int(panel.auto_close_grace_minutes_entry.get())
        if grace_period_minutes < 0:
            raise ValueError("grace_period_minutes")
    except ValueError:
        grace_period_minutes = 5

    try:
        tp_offset_pct = float(panel.auto_close_tp_offset_pct_entry.get())
        if tp_offset_pct <= 0 or tp_offset_pct > 5:
            raise ValueError("tp_offset_pct")
    except ValueError:
        tp_offset_pct = 0.05

    try:
        adaptive_min = float(panel.adaptive_close_min_duration_hours_entry.get())
        if adaptive_min <= 0:
            raise ValueError("adaptive_min")
    except ValueError:
        adaptive_min = 1.0

    try:
        adaptive_max = float(panel.adaptive_close_max_duration_hours_entry.get())
        if adaptive_max <= 0:
            raise ValueError("adaptive_max")
    except ValueError:
        adaptive_max = 12.0

    if adaptive_max < adaptive_min:
        adaptive_max = adaptive_min

    try:
        adaptive_lookback_days = int(panel.adaptive_close_lookback_days_entry.get())
        if adaptive_lookback_days < 1:
            raise ValueError("adaptive_lookback_days")
    except ValueError:
        adaptive_lookback_days = 60

    adaptive_timeframe = str(panel.adaptive_close_timeframe_entry.get() or "15m").strip() or "15m"
    adaptive_extra = getattr(panel, "_adaptive_close_extra", {})
    if not isinstance(adaptive_extra, dict):
        adaptive_extra = {}

    adaptive_cfg = {
        **adaptive_extra,
        "enabled": panel.adaptive_close_enabled_var.get(),
        "min_duration_hours": adaptive_min,
        "max_duration_hours": adaptive_max,
        "lookback_days": adaptive_lookback_days,
        "timeframe": adaptive_timeframe,
    }

    return {
        "enabled": panel.auto_close_enabled_var.get(),
        "max_duration_enabled": panel.auto_close_max_duration_enabled_var.get(),
        "max_duration_hours": max_duration_hours,
        "daily_close_enabled": panel.auto_close_daily_enabled_var.get(),
        "daily_close_time": daily_time,
        "daily_close_days": daily_days,
        "grace_period_minutes": grace_period_minutes,
        "tp_offset_pct": tp_offset_pct,
        "adaptive": adaptive_cfg,
    }


def load_auto_close_settings(panel, settings: Dict[str, Any]) -> None:
    """Load auto_close settings into UI widgets."""
    auto_close = settings.get("auto_close", {}) if isinstance(settings, dict) else {}

    panel.auto_close_enabled_var.set(auto_close.get("enabled", False))
    panel.auto_close_max_duration_enabled_var.set(auto_close.get("max_duration_enabled", True))
    panel.auto_close_max_duration_hours_entry.delete(0, "end")
    panel.auto_close_max_duration_hours_entry.insert(0, str(auto_close.get("max_duration_hours", 4.0)))

    panel.auto_close_daily_enabled_var.set(auto_close.get("daily_close_enabled", True))
    panel.auto_close_daily_time_entry.delete(0, "end")
    panel.auto_close_daily_time_entry.insert(0, str(auto_close.get("daily_close_time", "22:00")))

    panel.auto_close_daily_days_entry.delete(0, "end")
    panel.auto_close_daily_days_entry.insert(0, str(auto_close.get("daily_close_days", "1234567")))

    panel.auto_close_grace_minutes_entry.delete(0, "end")
    panel.auto_close_grace_minutes_entry.insert(0, str(auto_close.get("grace_period_minutes", 5)))

    panel.auto_close_tp_offset_pct_entry.delete(0, "end")
    panel.auto_close_tp_offset_pct_entry.insert(0, str(auto_close.get("tp_offset_pct", 0.05)))

    adaptive = auto_close.get("adaptive", {})
    if not isinstance(adaptive, dict):
        adaptive = {}

    panel._adaptive_close_extra = {
        key: value for key, value in adaptive.items() if key not in _ADAPTIVE_MANAGED_KEYS
    }

    panel.adaptive_close_enabled_var.set(bool(adaptive.get("enabled", False)))

    panel.adaptive_close_min_duration_hours_entry.delete(0, "end")
    panel.adaptive_close_min_duration_hours_entry.insert(0, str(adaptive.get("min_duration_hours", 1.0)))

    panel.adaptive_close_max_duration_hours_entry.delete(0, "end")
    panel.adaptive_close_max_duration_hours_entry.insert(0, str(adaptive.get("max_duration_hours", 12.0)))

    panel.adaptive_close_lookback_days_entry.delete(0, "end")
    panel.adaptive_close_lookback_days_entry.insert(0, str(adaptive.get("lookback_days", 60)))

    panel.adaptive_close_timeframe_entry.delete(0, "end")
    panel.adaptive_close_timeframe_entry.insert(0, str(adaptive.get("timeframe", "15m")))
