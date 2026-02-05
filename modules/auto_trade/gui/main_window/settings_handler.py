"""Settings management and theme handling."""

from typing import TYPE_CHECKING

import customtkinter as ctk

from gui.utils.colors import Colors

if TYPE_CHECKING:
    from .main_window import AutoTradeDashboard


class SettingsHandler:
    """Manages application settings and theme changes."""

    def __init__(self, parent: "AutoTradeDashboard"):
        self.parent = parent

    def apply_settings(self):
        """Apply loaded settings to application."""
        try:
            theme = self.parent.settings_manager.get("ui.theme", "dark")
            font_size = self.parent.settings_manager.get("ui.font_size", 12)

            if theme == "light":
                ctk.set_appearance_mode("light")
            else:
                ctk.set_appearance_mode("dark")

            print(f"Applied settings: Theme={theme}, Font Size={font_size}")

            all_settings = self.parent.settings_manager.get_all()

            config_panel = getattr(self.parent, "config_panel", None)
            if config_panel is not None:
                config_panel.load_settings(all_settings)

            scanner_control = getattr(self.parent, "scanner_control", None)
            if scanner_control is not None:
                scanner_settings = all_settings.get("scanner", {})
                scanner_control.load_config(scanner_settings)

            auto_trade_control = getattr(self.parent, "auto_trade_control", None)
            if auto_trade_control is not None and hasattr(auto_trade_control, "update_from_settings"):
                status = (
                    self.parent._get_current_status()
                    if hasattr(self.parent, "_get_current_status")
                    else None
                )
                auto_trade_control.update_from_settings(
                    self.parent.settings_manager.settings, status=status
                )

        except Exception as e:
            print(f"Error applying settings: {e}")

    def handle_settings_change(self, setting_type: str, value=None):
        """Handle settings change from ConfigPanel."""
        try:
            print(f"Settings changed: {setting_type} = {value}")

            config_panel = getattr(self.parent, "config_panel", None)
            if config_panel is not None:
                current_settings = config_panel.get_settings()
                self.parent.settings_manager.settings.update(current_settings)
                self.parent.settings_manager.save()

                new_mode = current_settings.get("api", {}).get("mode")
                if new_mode and new_mode != self.parent.mode:
                    self.parent.mode = new_mode
                    self.parent.title(f"Auto Trade Dashboard - [{self.parent.mode}]")
                    self.parent._update_mode_display()
                    self.parent._restart_websocket_service()

                new_theme = current_settings.get("ui", {}).get("theme")
                if new_theme:
                    self.refresh_theme_colors()

            if setting_type == "save_credentials" and value:
                print("Credentials updated, restarting WebSocket service...")
                self.parent._restart_websocket_service()

        except Exception as e:
            print(f"Error handling settings change: {e}")

    def refresh_theme_colors(self):
        """Refresh all component colors when theme changes."""

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
                widget = getattr(self.parent, name, None)
                if widget is not None:
                    _update_frame_colors(widget)

            signals_frame = getattr(self.parent, "signals_frame", None)
            if signals_frame is not None:
                signals_frame._configure_table_tags()

            config_panel = getattr(self.parent, "config_panel", None)
            if config_panel is not None and hasattr(config_panel, "recovery_panel"):
                _update_frame_colors(config_panel.recovery_panel)

            print("Theme colors refreshed")
        except Exception as e:
            print(f"Error refreshing theme colors: {e}")
