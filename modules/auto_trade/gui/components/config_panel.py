from typing import Callable, Dict, Optional

import customtkinter as ctk

from modules.auto_trade.gui.components.config_panel_parts.credentials import (
    on_cancel_credentials,
    on_change_credentials,
    refresh_credentials_display,
    save_credentials,
    test_connection,
)
from modules.auto_trade.gui.components.config_panel_parts.event_handlers import (
    on_limit_steps_toggle,
    on_mode_change,
    on_negative_be_toggle,
)
from modules.auto_trade.gui.components.config_panel_parts.settings_io import (
    get_settings,
    load_settings,
)
from modules.auto_trade.gui.components.config_panel_parts.tab_builders import (
    create_api_keys_tab,
    create_auto_close_timer_tab,
    create_gradual_recovery_tab,
    create_risk_settings_tab,
)


class ConfigPanel(ctk.CTkFrame):
    """Configuration panel (facade) split into sub-modules for maintainability."""

    def __init__(
        self,
        parent,
        on_settings_change: Optional[Callable] = None,
        mode: str = "DRY_RUN",
        on_recovery_config_change: Optional[Callable] = None,
    ):
        super().__init__(parent)

        self.on_settings_change = on_settings_change
        self.mode = mode
        self.on_recovery_config_change = on_recovery_config_change
        self._editing_credentials = False
        self._suppress_mode_notify = True

        title = ctk.CTkLabel(self, text="⚙️ Configuration", font=("Arial", 16, "bold"))
        title.pack(pady=(10, 15))

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        self._create_risk_settings_tab()
        self._create_auto_close_timer_tab()
        self._create_gradual_recovery_tab()
        self._create_api_keys_tab()
        self._suppress_mode_notify = False

    def _create_risk_settings_tab(self):
        create_risk_settings_tab(self)

    def _create_api_keys_tab(self):
        create_api_keys_tab(self)

    def _create_auto_close_timer_tab(self):
        create_auto_close_timer_tab(self)

    def _create_gradual_recovery_tab(self):
        create_gradual_recovery_tab(self)

    def _create_signal_filters_tab(self):
        return None

    def _create_tp_sl_tab(self):
        return None

    def _create_ui_preferences_tab(self):
        return None



    def _test_connection(self):
        test_connection(self)

    def _save_credentials(self):
        save_credentials(self)

    def _on_change_credentials(self):
        on_change_credentials(self)

    def _on_cancel_credentials(self):
        on_cancel_credentials(self)

    def _refresh_credentials_display(self):
        refresh_credentials_display(self)

    def _on_limit_steps_toggle(self):
        on_limit_steps_toggle(self)

    def _on_negative_be_toggle(self):
        on_negative_be_toggle(self)

    def _on_mode_change(self, show_warning: bool = True):
        on_mode_change(self, show_warning=show_warning)

    def get_settings(self) -> Dict:
        return get_settings(self)

    def load_settings(self, settings: Dict):
        load_settings(self, settings)
