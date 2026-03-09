from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import customtkinter as ctk
except Exception:  # pragma: no cover - fallback in headless environments
    class _CTKFallback:
        @staticmethod
        def get_appearance_mode() -> str:
            return "Dark"

    ctk = _CTKFallback()  # type: ignore[assignment]


def _load_theme_data() -> dict[str, Any]:
    theme_path = Path(__file__).resolve().parent.parent / "config" / "matrix_theme.json"
    try:
        with theme_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_THEME_DATA: dict[str, Any] = _load_theme_data()
_APP_COLORS: dict[str, Any] = (
    _THEME_DATA.get("AppColors", {}) if isinstance(_THEME_DATA.get("AppColors"), dict) else {}
)


def _app_color(name: str) -> str:
    value = _APP_COLORS.get(name)
    return value if isinstance(value, str) else ""


def _theme_color(widget: str, key: str) -> str:
    widget_data = _THEME_DATA.get(widget, {})
    if not isinstance(widget_data, dict):
        return ""

    value = widget_data.get(key)
    if isinstance(value, list) and value:
        return value[0] if isinstance(value[0], str) else ""
    return value if isinstance(value, str) else ""


class Colors:
    """Theme color registry loaded from matrix_theme.json."""

    TRANSPARENT: str = _app_color("TRANSPARENT")
    BLACK: str = _app_color("BLACK")
    WHITE: str = _app_color("WHITE")

    TEXT_MUTED: str = _app_color("TEXT_MUTED")
    TEXT_MUTED_DARK: str = _app_color("TEXT_MUTED_DARK")
    TEXT_DISABLED: str = _app_color("TEXT_DISABLED")
    TEXT_SECONDARY_ALT: str = _app_color("TEXT_SECONDARY_ALT")
    TEXT_FAINT: str = _app_color("TEXT_FAINT")

    SEPARATOR_DARK: str = _app_color("SEPARATOR_DARK")
    SEPARATOR_LIGHT: str = _app_color("SEPARATOR_LIGHT")

    CARD_MUTED: str = _app_color("CARD_MUTED")
    CARD_ELEVATED: str = _app_color("CARD_ELEVATED")
    CARD_DIALOG: str = _app_color("CARD_DIALOG")
    BORDER_SUBTLE: str = _app_color("BORDER_SUBTLE")
    WARNING_BG: str = _app_color("WARNING_BG")

    TAB_SELECTED_HOVER: str = _app_color("TAB_SELECTED_HOVER")
    INFO: str = _app_color("INFO")
    SUCCESS_BRIGHT: str = _app_color("SUCCESS_BRIGHT")
    SUCCESS_DIM: str = _app_color("SUCCESS_DIM")
    WARNING_DIM: str = _app_color("WARNING_DIM")
    DANGER_ALT_HOVER: str = _app_color("DANGER_ALT_HOVER")
    DANGER_CRITICAL: str = _app_color("DANGER_CRITICAL")
    LOSS_SOFT: str = _app_color("LOSS_SOFT")
    WARNING_ORANGE: str = _app_color("WARNING_ORANGE")
    WARNING_BRIGHT: str = _app_color("WARNING_BRIGHT")
    TOAST_INFO: str = _app_color("TOAST_INFO")
    TOAST_ERROR: str = _app_color("TOAST_ERROR")
    TOAST_WARNING: str = _app_color("TOAST_WARNING")
    SUCCESS_ALT: str = _app_color("SUCCESS_ALT")

    LONG: str = _app_color("LONG")
    SHORT: str = _app_color("SHORT")
    # Keep neutral gray stable for tests and legacy UI expectations.
    NEUTRAL: str = "#888888"
    PROFIT: str = _app_color("PROFIT")
    LOSS: str = _app_color("LOSS")
    PRODUCTION: str = _app_color("PRODUCTION")
    DEMO: str = _app_color("DEMO")
    DRY_RUN: str = "#4488ff"

    BTN_SUCCESS: str = _app_color("BTN_SUCCESS")
    BTN_SUCCESS_HOVER: str = _app_color("BTN_SUCCESS_HOVER")
    BTN_SUCCESS_TEXT: str = _app_color("BTN_SUCCESS_TEXT")
    BTN_DANGER: str = _app_color("BTN_DANGER")
    BTN_DANGER_HOVER: str = _app_color("BTN_DANGER_HOVER")
    BTN_DANGER_ALT: str = _app_color("BTN_DANGER_ALT")
    BTN_DANGER_ALT_TEXT: str = _app_color("BTN_DANGER_ALT_TEXT")
    BTN_DANGER_ALT_HOVER: str = _app_color("BTN_DANGER_ALT_HOVER")
    BTN_DANGER_ALT_HOVER_TEXT: str = _app_color("BTN_DANGER_ALT_HOVER_TEXT")
    BTN_PRIMARY: str = _theme_color("CTkButton", "fg_color") or _app_color("BTN_PRIMARY")
    BTN_PRIMARY_HOVER: str = _theme_color("CTkButton", "hover_color") or _app_color("BTN_PRIMARY_HOVER")
    BTN_PRIMARY_TEXT: str = _theme_color("CTkButton", "text_color") or _app_color("BTN_PRIMARY_TEXT")
    BTN_NEUTRAL: str = _app_color("BTN_NEUTRAL")
    BTN_NEUTRAL_TEXT: str = _app_color("BTN_NEUTRAL_TEXT")
    BTN_NEUTRAL_HOVER: str = _app_color("BTN_NEUTRAL_HOVER")
    BTN_NEUTRAL_HOVER_TEXT: str = _app_color("BTN_NEUTRAL_HOVER_TEXT")
    BTN_WARNING: str = _app_color("BTN_WARNING")
    BTN_WARNING_HOVER: str = _app_color("BTN_WARNING_HOVER")

    BG_DARK: str = _theme_color("CTk", "fg_color") or _app_color("BG_DARK")
    BG_LIGHT: str = _app_color("BG_LIGHT") or "#f2f2f2"
    BG_CARD_DARK: str = _theme_color("CTkFrame", "fg_color") or _app_color("BG_CARD_DARK")
    BG_CARD_LIGHT: str = _app_color("BG_CARD_LIGHT") or "#ffffff"
    BG_HEADER_DARK: str = _app_color("BG_HEADER_DARK")
    BG_HEADER_LIGHT: str = _app_color("BG_HEADER_LIGHT") or "#e8e8e8"
    TEXT_PRIMARY_DARK: str = _theme_color("CTkLabel", "text_color") or _app_color("TEXT_PRIMARY_DARK")
    TEXT_PRIMARY_LIGHT: str = _app_color("TEXT_PRIMARY_LIGHT") or "#111111"
    TEXT_SECONDARY_DARK: str = _app_color("TEXT_SECONDARY_DARK")
    TEXT_SECONDARY_LIGHT: str = _app_color("TEXT_SECONDARY_LIGHT") or "#444444"
    BG_HIGHLIGHT: str = _app_color("BG_HIGHLIGHT")
    BG_INPUT: str = _theme_color("CTkEntry", "fg_color") or _app_color("BG_INPUT")
    TEXT_DIM: str = _app_color("TEXT_DIM")
    TEXT_BRIGHT: str = _app_color("TEXT_BRIGHT")
    BORDER_NEON: str = _theme_color("CTkFrame", "border_color") or _app_color("BORDER_NEON")
    BORDER_ACTIVE: str = _app_color("BORDER_ACTIVE")
    ACCENT: str = _app_color("ACCENT")
    ACCENT_DIM: str = _app_color("ACCENT_DIM")
    ICON_ON_DARK: str = _app_color("ICON_ON_DARK")
    ICON_ON_LIGHT: str = _app_color("ICON_ON_LIGHT")

    @classmethod
    def get_current_theme(cls) -> str:
        try:
            return str(ctk.get_appearance_mode())
        except Exception:
            return "Dark"

    @classmethod
    def is_dark_mode(cls) -> bool:
        return cls.get_current_theme().lower() == "dark"

    @classmethod
    def get_bg(cls) -> str:
        return cls.BG_DARK if cls.is_dark_mode() else cls.BG_LIGHT

    @classmethod
    def get_card_bg(cls) -> str:
        return cls.BG_CARD_DARK if cls.is_dark_mode() else cls.BG_CARD_LIGHT

    @classmethod
    def get_header_bg(cls) -> str:
        return cls.BG_HEADER_DARK if cls.is_dark_mode() else cls.BG_HEADER_LIGHT

    @classmethod
    def get_text_primary(cls) -> str:
        return cls.TEXT_PRIMARY_DARK if cls.is_dark_mode() else cls.TEXT_PRIMARY_LIGHT

    @classmethod
    def get_text_secondary(cls) -> str:
        return cls.TEXT_SECONDARY_DARK if cls.is_dark_mode() else cls.TEXT_SECONDARY_LIGHT

    @classmethod
    def get_hover_bg(cls) -> str:
        return cls.BG_HIGHLIGHT

    @classmethod
    def get_accent(cls) -> str:
        return cls.ACCENT

    @property
    def BG_CARD(self) -> str:
        return self.get_card_bg()

    @property
    def BG_HEADER(self) -> str:
        return self.get_header_bg()

    @property
    def TEXT_PRIMARY(self) -> str:
        return self.get_text_primary()

    @property
    def TEXT_SECONDARY(self) -> str:
        return self.get_text_secondary()
