import customtkinter as ctk

from modules.auto_trade.gui.components.config_panel_parts.auto_close_settings import build_auto_close_section
from modules.auto_trade.gui.utils.svg_icons import get_icon


def create_risk_settings_tab(panel):
    """Create Risk Settings tab (Risk + TP/SL)."""
    tab = panel.tabview.add("Risk Settings")

    container = ctk.CTkFrame(tab, fg_color="transparent")
    container.pack(fill="both", expand=True, padx=5, pady=5)
    container.grid_columnconfigure(0, weight=1)
    container.grid_columnconfigure(1, weight=1)

    from modules.auto_trade.gui.utils.colors import Colors

    risk_box = ctk.CTkFrame(container, fg_color=Colors.get_card_bg(), corner_radius=10)
    risk_box.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
    risk_frame = ctk.CTkFrame(risk_box, fg_color="transparent")
    risk_frame.pack(fill="both", expand=True, padx=15, pady=15)

    ctk.CTkLabel(risk_frame, text="Risk Management", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 15))

    panel.risk_limits_enabled_var = ctk.BooleanVar(value=True)
    ctk.CTkCheckBox(
        risk_frame,
        text="Enable risk limits (max exposure, daily loss, positions, etc.)",
        variable=panel.risk_limits_enabled_var,
    ).pack(anchor="w", pady=(0, 10))

    ctk.CTkLabel(risk_frame, text="Max Position Size ($):", font=("Arial", 12)).pack(anchor="w", pady=(5, 2))
    panel.max_pos_size_entry = ctk.CTkEntry(risk_frame, placeholder_text="100.00")
    panel.max_pos_size_entry.pack(fill="x", pady=(2, 8))
    panel.max_pos_size_entry.insert(0, "100.00")

    ctk.CTkLabel(risk_frame, text="Max Open Positions:", font=("Arial", 12)).pack(anchor="w", pady=(5, 2))
    panel.max_positions_entry = ctk.CTkEntry(risk_frame, placeholder_text="3")
    panel.max_positions_entry.pack(fill="x", pady=(2, 8))
    panel.max_positions_entry.insert(0, "3")

    ctk.CTkLabel(risk_frame, text="Max Daily Loss ($):", font=("Arial", 12)).pack(anchor="w", pady=(5, 2))
    panel.max_daily_loss_entry = ctk.CTkEntry(risk_frame, placeholder_text="50.00")
    panel.max_daily_loss_entry.pack(fill="x", pady=(2, 8))
    panel.max_daily_loss_entry.insert(0, "50.00")

    ctk.CTkLabel(risk_frame, text="Default Leverage:", font=("Arial", 12)).pack(anchor="w", pady=(5, 2))
    panel.default_leverage_var = ctk.StringVar(value="10x")
    leverage_dropdown = ctk.CTkComboBox(
        risk_frame,
        values=["1x", "2x", "3x", "5x", "10x", "20x", "50x", "100x"],
        variable=panel.default_leverage_var,
    )
    leverage_dropdown.pack(fill="x", pady=(2, 8))

    tp_sl_box = ctk.CTkFrame(container, fg_color=Colors.get_card_bg(), corner_radius=10)
    tp_sl_box.grid(row=0, column=1, sticky="nsew", padx=5)
    tp_sl_inner = ctk.CTkFrame(tp_sl_box, fg_color="transparent")
    tp_sl_inner.pack(fill="both", expand=True, padx=15, pady=15)

    ctk.CTkLabel(tp_sl_inner, text="TP/SL Configuration", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 15))

    ctk.CTkLabel(tp_sl_inner, text="Default Take Profit (%):", font=("Arial", 12)).pack(anchor="w", pady=(5, 2))
    panel.default_tp_entry = ctk.CTkEntry(tp_sl_inner, placeholder_text="5.0")
    panel.default_tp_entry.pack(fill="x", pady=(2, 8))
    panel.default_tp_entry.insert(0, "5.0")

    ctk.CTkLabel(tp_sl_inner, text="Default Stop Loss (%):", font=("Arial", 12)).pack(anchor="w", pady=(5, 2))
    panel.default_sl_entry = ctk.CTkEntry(tp_sl_inner, placeholder_text="2.5")
    panel.default_sl_entry.pack(fill="x", pady=(2, 8))
    panel.default_sl_entry.insert(0, "2.5")

    ctk.CTkLabel(tp_sl_inner, text="TP/SL Mode:", font=("Arial", 12)).pack(anchor="w", pady=(5, 2))
    panel.tp_sl_mode_var = ctk.StringVar(value="Percentage")
    mode_dropdown = ctk.CTkComboBox(
        tp_sl_inner,
        values=["Percentage", "Price", "ATR", "ROI"],
        variable=panel.tp_sl_mode_var,
    )
    mode_dropdown.pack(fill="x", pady=(2, 8))

    panel.trailing_stop_var = ctk.BooleanVar(value=False)
    trailing_checkbox = ctk.CTkCheckBox(tp_sl_inner, text="Enable Trailing Stop", variable=panel.trailing_stop_var)
    trailing_checkbox.pack(anchor="w", pady=(15, 2))

    ctk.CTkLabel(tp_sl_inner, text="Trailing Step (%):", font=("Arial", 12)).pack(anchor="w", pady=(10, 2))
    panel.trailing_step_pct_entry = ctk.CTkEntry(tp_sl_inner, placeholder_text="2.0")
    panel.trailing_step_pct_entry.pack(fill="x", pady=(2, 8))
    panel.trailing_step_pct_entry.insert(0, "2.0")

    panel.limit_trailing_steps_var = ctk.BooleanVar(value=False)
    limit_steps_checkbox = ctk.CTkCheckBox(
        tp_sl_inner,
        text="Limit trailing steps",
        variable=panel.limit_trailing_steps_var,
        command=panel._on_limit_steps_toggle,
    )
    limit_steps_checkbox.pack(anchor="w", pady=(5, 2))

    panel.max_steps_label = ctk.CTkLabel(tp_sl_inner, text="Max Steps:", font=("Arial", 12))
    panel.max_steps_entry = ctk.CTkEntry(tp_sl_inner, placeholder_text="5")
    panel.max_steps_entry.insert(0, "5")

    separator = ctk.CTkLabel(tp_sl_inner, text="─────────────────────────", text_color="gray")
    separator.pack(anchor="w", pady=(15, 5))

    panel.negative_be_var = ctk.BooleanVar(value=False)
    negative_be_checkbox = ctk.CTkCheckBox(
        tp_sl_inner,
        text="Enable Negative Breakeven",
        variable=panel.negative_be_var,
        command=panel._on_negative_be_toggle,
    )
    negative_be_checkbox.pack(anchor="w", pady=(5, 2))

    panel.negative_be_threshold_label = ctk.CTkLabel(
        tp_sl_inner,
        text="Negative BE threshold (%):",
        font=("Arial", 12),
    )
    panel.negative_be_threshold_label.pack(anchor="w", pady=(10, 2))

    panel.negative_be_threshold_entry = ctk.CTkEntry(tp_sl_inner, placeholder_text="2.0")
    panel.negative_be_threshold_entry.pack(fill="x", pady=(2, 8))
    panel.negative_be_threshold_entry.insert(0, "2.0")


def create_gradual_recovery_tab(panel):
    """Create dedicated Gradual Recovery tab."""
    tab = panel.tabview.add("Gradual Recovery")

    container = ctk.CTkFrame(tab, fg_color="transparent")
    container.pack(fill="both", expand=True, padx=10, pady=10)
    container.grid_columnconfigure(0, weight=1)

    from modules.auto_trade.gui.components.recovery_panel import RecoveryPanel

    panel.recovery_panel = RecoveryPanel(
        container,
        on_config_change=panel.on_recovery_config_change,
        mode=panel.mode,
        compact=True,
    )
    panel.recovery_panel.grid(row=0, column=0, sticky="nsew")


def create_auto_close_timer_tab(panel):
    """Create dedicated Auto-Close Timer tab."""
    tab = panel.tabview.add("Auto-Close Timer")

    container = ctk.CTkFrame(tab, fg_color="transparent")
    container.pack(fill="both", expand=True, padx=10, pady=10)
    container.grid_columnconfigure(0, weight=1)

    from modules.auto_trade.gui.utils.colors import Colors

    auto_close_box = ctk.CTkFrame(container, fg_color=Colors.get_card_bg(), corner_radius=10)
    auto_close_box.grid(row=0, column=0, sticky="nsew")

    auto_close_inner = ctk.CTkFrame(auto_close_box, fg_color="transparent")
    auto_close_inner.pack(fill="both", expand=True, padx=15, pady=15)

    build_auto_close_section(panel, auto_close_inner, show_separator=False, show_title=True)


def create_api_keys_tab(panel):
    """Create API Keys tab."""
    from modules.auto_trade.gui.utils.colors import Colors

    tab = panel.tabview.add("API Keys")
    api_frame = ctk.CTkFrame(tab, fg_color="transparent")
    api_frame.pack(fill="both", expand=True, padx=10, pady=10)

    ctk.CTkLabel(api_frame, text="Trading Mode:", font=("Arial", 12)).pack(anchor="w", pady=(5, 2))
    panel.mode_var = ctk.StringVar(value="DRY_RUN")

    mode_frame = ctk.CTkFrame(api_frame, fg_color="transparent")
    mode_frame.pack(anchor="w", pady=2)

    panel.mode_production_radio = ctk.CTkRadioButton(
        mode_frame,
        text="Production",
        variable=panel.mode_var,
        value="PRODUCTION",
        command=panel._on_mode_change,
    )
    panel.mode_production_radio.pack(side="left", padx=(0, 10))

    panel.mode_demo_radio = ctk.CTkRadioButton(
        mode_frame,
        text="Demo",
        variable=panel.mode_var,
        value="DEMO",
        command=panel._on_mode_change,
    )
    panel.mode_demo_radio.pack(side="left", padx=(0, 10))

    panel.mode_dry_run_radio = ctk.CTkRadioButton(
        mode_frame,
        text="Dry Run",
        variable=panel.mode_var,
        value="DRY_RUN",
        command=panel._on_mode_change,
    )
    panel.mode_dry_run_radio.pack(side="left")

    panel.mode_description_label = ctk.CTkLabel(
        api_frame,
        text="  Safe local simulation",
        font=("Arial", 10),
        text_color="gray",
        image=get_icon("shield_check", size=(16, 16), light_color="gray", dark_color="gray"),
        compound="left",
    )
    panel.mode_description_label.pack(anchor="w", pady=(2, 10))

    ctk.CTkLabel(api_frame, text="Exchange:", font=("Arial", 12)).pack(anchor="w", pady=(5, 2))
    panel.exchange_var = ctk.StringVar(value="Binance")
    exchange_dropdown = ctk.CTkComboBox(
        api_frame,
        values=["Binance", "Demo"],
        variable=panel.exchange_var,
        width=200,
        command=lambda _: panel._refresh_credentials_display(),
    )
    exchange_dropdown.pack(anchor="w", pady=2)

    panel.api_key_frame = ctk.CTkFrame(api_frame, fg_color="transparent")
    panel.api_key_frame.pack(fill="x")

    panel.credentials_masked_frame = ctk.CTkFrame(panel.api_key_frame, fg_color="transparent")
    ctk.CTkLabel(panel.credentials_masked_frame, text="API Key:", font=("Arial", 12)).pack(anchor="w", pady=(10, 2))
    panel.api_key_masked_label = ctk.CTkLabel(panel.credentials_masked_frame, text="—", font=("Arial", 12, "bold"))
    panel.api_key_masked_label.pack(anchor="w", pady=2)
    ctk.CTkLabel(panel.credentials_masked_frame, text="API Secret:", font=("Arial", 12)).pack(anchor="w", pady=(10, 2))
    panel.api_secret_masked_label = ctk.CTkLabel(panel.credentials_masked_frame, text="—", font=("Arial", 12, "bold"))
    panel.api_secret_masked_label.pack(anchor="w", pady=2)

    change_btn = ctk.CTkButton(
        panel.credentials_masked_frame,
        text="  Change Credentials",
        fg_color="#ffcc00",
        text_color="black",
        hover_color="#e6b800",
        command=panel._on_change_credentials,
        image=get_icon("pencil", size=(16, 16), light_color="black", dark_color="black"),
        compound="left",
    )
    change_btn.pack(anchor="w", pady=(20, 5))

    panel.credentials_entry_frame = ctk.CTkFrame(panel.api_key_frame, fg_color="transparent")
    panel.credentials_entry_frame.pack(fill="x")

    ctk.CTkLabel(panel.credentials_entry_frame, text="API Key:", font=("Arial", 12)).pack(anchor="w", pady=(10, 2))
    panel.api_key_entry = ctk.CTkEntry(
        panel.credentials_entry_frame,
        placeholder_text="Enter your API key",
        show="•",
        width=300,
    )
    panel.api_key_entry.pack(anchor="w", pady=2)

    ctk.CTkLabel(panel.credentials_entry_frame, text="API Secret:", font=("Arial", 12)).pack(anchor="w", pady=(10, 2))
    panel.api_secret_entry = ctk.CTkEntry(
        panel.credentials_entry_frame,
        placeholder_text="Enter your API secret",
        show="•",
        width=300,
    )
    panel.api_secret_entry.pack(anchor="w", pady=2)

    entry_buttons_frame = ctk.CTkFrame(panel.credentials_entry_frame, fg_color="transparent")
    entry_buttons_frame.pack(anchor="w", pady=(20, 5), fill="x")

    test_btn = ctk.CTkButton(
        entry_buttons_frame,
        text="  Test Connection",
        fg_color=Colors.BTN_SUCCESS,
        hover_color=Colors.BTN_SUCCESS_HOVER,
        command=panel._test_connection,
        image=get_icon("link", size=(16, 16), light_color="black", dark_color="black"),
        compound="left",
    )
    test_btn.pack(side="left", padx=(0, 10))

    save_btn = ctk.CTkButton(
        entry_buttons_frame,
        text="  Save Credentials",
        fg_color=Colors.BTN_PRIMARY,
        hover_color=Colors.BTN_PRIMARY_HOVER,
        command=panel._save_credentials,
        image=get_icon("save", size=(16, 16)),
        compound="left",
    )
    save_btn.pack(side="left", padx=(0, 10))

    panel.cancel_credentials_btn = ctk.CTkButton(
        entry_buttons_frame,
        text="  Cancel",
        fg_color=Colors.BTN_DANGER,
        hover_color="#cc3333",
        command=panel._on_cancel_credentials,
        image=get_icon("x", size=(16, 16)),
        compound="left",
    )

    panel._on_mode_change()
