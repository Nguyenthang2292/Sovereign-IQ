from typing import Any, Dict

import customtkinter as ctk


def build_auto_close_section(panel, parent_frame, *, show_separator: bool = True, show_title: bool = True) -> None:
    """Build Auto-Close settings UI section inside TP/SL area."""
    if show_separator:
        separator = ctk.CTkLabel(parent_frame, text="─────────────────────────", text_color="gray")
        separator.pack(anchor="w", pady=(12, 5))

    if show_title:
        ctk.CTkLabel(parent_frame, text="Auto-Close Timer", font=("Arial", 13, "bold")).pack(anchor="w", pady=(0, 8))

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

    ctk.CTkLabel(parent_frame, text="Max duration (hours):", font=("Arial", 12)).pack(anchor="w", pady=(5, 2))
    panel.auto_close_max_duration_hours_entry = ctk.CTkEntry(parent_frame, placeholder_text="4.0")
    panel.auto_close_max_duration_hours_entry.pack(fill="x", pady=(2, 8))
    panel.auto_close_max_duration_hours_entry.insert(0, "4.0")

    panel.auto_close_daily_enabled_var = ctk.BooleanVar(value=True)
    ctk.CTkCheckBox(
        parent_frame,
        text="Enable daily close window",
        variable=panel.auto_close_daily_enabled_var,
    ).pack(anchor="w", pady=(0, 4))

    ctk.CTkLabel(parent_frame, text="Daily close time (UTC HH:MM):", font=("Arial", 12)).pack(anchor="w", pady=(5, 2))
    panel.auto_close_daily_time_entry = ctk.CTkEntry(parent_frame, placeholder_text="22:00")
    panel.auto_close_daily_time_entry.pack(fill="x", pady=(2, 8))
    panel.auto_close_daily_time_entry.insert(0, "22:00")

    ctk.CTkLabel(parent_frame, text="Daily close days (1=Mon...7=Sun):", font=("Arial", 12)).pack(anchor="w", pady=(5, 2))
    panel.auto_close_daily_days_entry = ctk.CTkEntry(parent_frame, placeholder_text="1234567")
    panel.auto_close_daily_days_entry.pack(fill="x", pady=(2, 8))
    panel.auto_close_daily_days_entry.insert(0, "1234567")

    ctk.CTkLabel(parent_frame, text="Grace period (minutes):", font=("Arial", 12)).pack(anchor="w", pady=(5, 2))
    panel.auto_close_grace_minutes_entry = ctk.CTkEntry(parent_frame, placeholder_text="5")
    panel.auto_close_grace_minutes_entry.pack(fill="x", pady=(2, 8))
    panel.auto_close_grace_minutes_entry.insert(0, "5")

    ctk.CTkLabel(parent_frame, text="TP offset (%) for quasi-market close:", font=("Arial", 12)).pack(anchor="w", pady=(5, 2))
    panel.auto_close_tp_offset_pct_entry = ctk.CTkEntry(parent_frame, placeholder_text="0.05")
    panel.auto_close_tp_offset_pct_entry.pack(fill="x", pady=(2, 8))
    panel.auto_close_tp_offset_pct_entry.insert(0, "0.05")


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

    return {
        "enabled": panel.auto_close_enabled_var.get(),
        "max_duration_enabled": panel.auto_close_max_duration_enabled_var.get(),
        "max_duration_hours": max_duration_hours,
        "daily_close_enabled": panel.auto_close_daily_enabled_var.get(),
        "daily_close_time": daily_time,
        "daily_close_days": daily_days,
        "grace_period_minutes": grace_period_minutes,
        "tp_offset_pct": tp_offset_pct,
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
