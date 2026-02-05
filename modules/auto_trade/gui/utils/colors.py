import customtkinter as ctk


class Colors:
    """Theme-aware color system for Auto Trade GUI"""

    # Static colors (theme-independent)
    LONG = "#00ff88"
    SHORT = "#ff4444"
    NEUTRAL = "#888888"
    PROFIT = "#00ff88"
    LOSS = "#ff4444"
    PRODUCTION = "#ff4444"
    DEMO = "#ffaa00"
    DRY_RUN = "#4488ff"

    # Dark theme colors
    BG_DARK = "#1a1a1a"
    BG_CARD_DARK = "#2b2b2b"
    BG_HEADER_DARK = "#1e1e1e"
    TEXT_PRIMARY_DARK = "#ffffff"
    TEXT_SECONDARY_DARK = "#888888"

    # Light theme colors
    BG_LIGHT = "#f0f0f0"
    BG_CARD_LIGHT = "#ffffff"
    BG_HEADER_LIGHT = "#e8e8e8"
    TEXT_PRIMARY_LIGHT = "#000000"
    TEXT_SECONDARY_LIGHT = "#666666"

    @staticmethod
    def get_current_theme() -> str:
        """Get current CustomTkinter appearance mode"""
        return ctk.get_appearance_mode()

    @staticmethod
    def is_dark_mode() -> bool:
        """Check if current theme is dark mode"""
        return Colors.get_current_theme().lower() == "dark"

    @classmethod
    def get_bg(cls) -> str:
        """Get background color for current theme"""
        return cls.BG_DARK if cls.is_dark_mode() else cls.BG_LIGHT

    @classmethod
    def get_card_bg(cls) -> str:
        """Get card background color for current theme"""
        return cls.BG_CARD_DARK if cls.is_dark_mode() else cls.BG_CARD_LIGHT

    @classmethod
    def get_header_bg(cls) -> str:
        """Get header background color for current theme"""
        return cls.BG_HEADER_DARK if cls.is_dark_mode() else cls.BG_HEADER_LIGHT

    @classmethod
    def get_text_primary(cls) -> str:
        """Get primary text color for current theme"""
        return cls.TEXT_PRIMARY_DARK if cls.is_dark_mode() else cls.TEXT_PRIMARY_LIGHT

    @classmethod
    def get_text_secondary(cls) -> str:
        """Get secondary text color for current theme"""
        return cls.TEXT_SECONDARY_DARK if cls.is_dark_mode() else cls.TEXT_SECONDARY_LIGHT

    @classmethod
    def get_hover_bg(cls) -> str:
        """Get hover/section background color for current theme"""
        # Slightly lighter than card background for visual separation
        return "#333333" if cls.is_dark_mode() else "#f8f8f8"

    @classmethod
    def get_accent(cls) -> str:
        """Get accent color for headers and highlights"""
        return "#4488ff"  # Blue accent color that works in both themes

    # Legacy properties for backward compatibility
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
