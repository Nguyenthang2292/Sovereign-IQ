import customtkinter as ctk
from typing import Dict, Optional, Callable


class ConfigPanel(ctk.CTkFrame):
    """
    Configuration Panel with tabbed interface
    Contains Risk Settings, Signal Filters, API Keys, and Default TP/SL
    """

    def __init__(self, parent, on_settings_change: Callable = None):
        super().__init__(parent)

        self.on_settings_change = on_settings_change

        # Title
        title = ctk.CTkLabel(self, text="⚙️ Configuration", font=("Arial", 16, "bold"))
        title.pack(pady=(10, 15))

        # Create tabbed interface
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Add tabs
        self._create_risk_settings_tab()
        self._create_signal_filters_tab()
        self._create_api_keys_tab()
        self._create_tp_sl_tab()
        self._create_ui_preferences_tab()

    def _create_risk_settings_tab(self):
        """Create Risk Settings tab"""
        tab = self.tabview.add("Risk Settings")

        # Risk settings frame
        risk_frame = ctk.CTkFrame(tab, fg_color="transparent")
        risk_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Max position size
        label = ctk.CTkLabel(risk_frame, text="Max Position Size ($):", font=("Arial", 12))
        label.pack(anchor="w", pady=(5, 2))

        self.max_pos_size_entry = ctk.CTkEntry(risk_frame, placeholder_text="100.00", width=200)
        self.max_pos_size_entry.pack(anchor="w", pady=2)
        self.max_pos_size_entry.insert(0, "100.00")

        # Max open positions
        label = ctk.CTkLabel(risk_frame, text="Max Open Positions:", font=("Arial", 12))
        label.pack(anchor="w", pady=(10, 2))

        self.max_positions_entry = ctk.CTkEntry(risk_frame, placeholder_text="3", width=200)
        self.max_positions_entry.pack(anchor="w", pady=2)
        self.max_positions_entry.insert(0, "3")

        # Max daily loss
        label = ctk.CTkLabel(risk_frame, text="Max Daily Loss ($):", font=("Arial", 12))
        label.pack(anchor="w", pady=(10, 2))

        self.max_daily_loss_entry = ctk.CTkEntry(risk_frame, placeholder_text="50.00", width=200)
        self.max_daily_loss_entry.pack(anchor="w", pady=2)
        self.max_daily_loss_entry.insert(0, "50.00")

        # Default leverage
        label = ctk.CTkLabel(risk_frame, text="Default Leverage:", font=("Arial", 12))
        label.pack(anchor="w", pady=(10, 2))

        self.default_leverage_var = ctk.StringVar(value="10x")
        leverage_dropdown = ctk.CTkComboBox(
            risk_frame,
            values=["1x", "2x", "3x", "5x", "10x", "20x", "50x", "100x"],
            variable=self.default_leverage_var,
            width=200,
        )
        leverage_dropdown.pack(anchor="w", pady=2)

    def _create_signal_filters_tab(self):
        """Create Signal Filters tab"""
        tab = self.tabview.add("Signal Filters")

        # Signal filters frame
        filters_frame = ctk.CTkFrame(tab, fg_color="transparent")
        filters_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Min signal score
        label = ctk.CTkLabel(filters_frame, text="Min Signal Score:", font=("Arial", 12))
        label.pack(anchor="w", pady=(5, 2))

        self.min_score_var = ctk.DoubleVar(value=0.7)
        score_slider = ctk.CTkSlider(
            filters_frame, from_=0.0, to=1.0, number_of_steps=100, variable=self.min_score_var, width=300
        )
        score_slider.pack(anchor="w", pady=2)

        score_label = ctk.CTkLabel(filters_frame, text=f"Current: {self.min_score_var.get():.2f}", font=("Arial", 10))
        score_label.pack(anchor="w", pady=2)

        # XGBoost checkbox
        self.enable_xgboost_var = ctk.BooleanVar(value=True)
        xgboost_checkbox = ctk.CTkCheckBox(filters_frame, text="Enable XGBoost Model", variable=self.enable_xgboost_var)
        xgboost_checkbox.pack(anchor="w", pady=(10, 2))

        # Symbol whitelist
        label = ctk.CTkLabel(filters_frame, text="Symbol Whitelist (comma-separated):", font=("Arial", 12))
        label.pack(anchor="w", pady=(10, 2))

        self.whitelist_entry = ctk.CTkTextbox(filters_frame, height=80, width=300)
        self.whitelist_entry.pack(anchor="w", pady=2)
        self.whitelist_entry.insert("0.0", "BTC/USDT\nETH/USDT\nSOL/USDT")

        # Min volume filter
        label = ctk.CTkLabel(filters_frame, text="Min 24h Volume (M):", font=("Arial", 12))
        label.pack(anchor="w", pady=(10, 2))

        self.min_volume_entry = ctk.CTkEntry(filters_frame, placeholder_text="50", width=200)
        self.min_volume_entry.pack(anchor="w", pady=2)
        self.min_volume_entry.insert(0, "50")

    def _create_api_keys_tab(self):
        """Create API Keys tab"""
        tab = self.tabview.add("API Keys")

        # API keys frame
        api_frame = ctk.CTkFrame(tab, fg_color="transparent")
        api_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Mode selector
        label = ctk.CTkLabel(api_frame, text="Trading Mode:", font=("Arial", 12))
        label.pack(anchor="w", pady=(5, 2))

        self.mode_var = ctk.StringVar(value="DRY_RUN")

        mode_frame = ctk.CTkFrame(api_frame, fg_color="transparent")
        mode_frame.pack(anchor="w", pady=2)

        self.mode_production_radio = ctk.CTkRadioButton(
            mode_frame, text="Production", variable=self.mode_var, value="PRODUCTION", command=self._on_mode_change
        )
        self.mode_production_radio.pack(side="left", padx=(0, 10))

        self.mode_demo_radio = ctk.CTkRadioButton(
            mode_frame, text="Demo", variable=self.mode_var, value="DEMO", command=self._on_mode_change
        )
        self.mode_demo_radio.pack(side="left", padx=(0, 10))

        self.mode_dry_run_radio = ctk.CTkRadioButton(
            mode_frame, text="Dry Run", variable=self.mode_var, value="DRY_RUN", command=self._on_mode_change
        )
        self.mode_dry_run_radio.pack(side="left")

        # Mode description
        self.mode_description_label = ctk.CTkLabel(
            api_frame, text="✅ Safe local simulation", font=("Arial", 10), text_color="gray"
        )
        self.mode_description_label.pack(anchor="w", pady=(2, 10))

        # Exchange selector
        label = ctk.CTkLabel(api_frame, text="Exchange:", font=("Arial", 12))
        label.pack(anchor="w", pady=(5, 2))

        self.exchange_var = ctk.StringVar(value="Binance")
        exchange_dropdown = ctk.CTkComboBox(
            api_frame, values=["Binance", "Demo"], variable=self.exchange_var, width=200
        )
        exchange_dropdown.pack(anchor="w", pady=2)

        # API key frame (for disabling when DRY_RUN)
        self.api_key_frame = ctk.CTkFrame(api_frame, fg_color="transparent")
        self.api_key_frame.pack(fill="x")

        # API Key
        label = ctk.CTkLabel(self.api_key_frame, text="API Key:", font=("Arial", 12))
        label.pack(anchor="w", pady=(10, 2))

        self.api_key_entry = ctk.CTkEntry(
            self.api_key_frame, placeholder_text="Enter your API key", show="•", width=300
        )
        self.api_key_entry.pack(anchor="w", pady=2)

        # API Secret
        label = ctk.CTkLabel(self.api_key_frame, text="API Secret:", font=("Arial", 12))
        label.pack(anchor="w", pady=(10, 2))

        self.api_secret_entry = ctk.CTkEntry(
            self.api_key_frame, placeholder_text="Enter your API secret", show="•", width=300
        )
        self.api_secret_entry.pack(anchor="w", pady=2)

        # Test connection button
        test_btn = ctk.CTkButton(
            self.api_key_frame,
            text="🔗 Test Connection",
            fg_color="#00ff88",
            hover_color="#00cc66",
            command=self._test_connection,
        )
        test_btn.pack(anchor="w", pady=(20, 5))

        # Save credentials button
        save_btn = ctk.CTkButton(
            self.api_key_frame,
            text="💾 Save Credentials",
            fg_color="#4488ff",
            hover_color="#0066ff",
            command=self._save_credentials,
        )
        save_btn.pack(anchor="w", pady=2)

    def _create_tp_sl_tab(self):
        """Create Default TP/SL tab"""
        tab = self.tabview.add("TP/SL Settings")

        # TP/SL frame
        tp_sl_frame = ctk.CTkFrame(tab, fg_color="transparent")
        tp_sl_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Default TP percentage
        label = ctk.CTkLabel(tp_sl_frame, text="Default Take Profit (%):", font=("Arial", 12))
        label.pack(anchor="w", pady=(5, 2))

        self.default_tp_entry = ctk.CTkEntry(tp_sl_frame, placeholder_text="5.0", width=200)
        self.default_tp_entry.pack(anchor="w", pady=2)
        self.default_tp_entry.insert(0, "5.0")

        # Default SL percentage
        label = ctk.CTkLabel(tp_sl_frame, text="Default Stop Loss (%):", font=("Arial", 12))
        label.pack(anchor="w", pady=(10, 2))

        self.default_sl_entry = ctk.CTkEntry(tp_sl_frame, placeholder_text="2.5", width=200)
        self.default_sl_entry.pack(anchor="w", pady=2)
        self.default_sl_entry.insert(0, "2.5")

        # Trailing stop checkbox
        self.trailing_stop_var = ctk.BooleanVar(value=False)
        trailing_checkbox = ctk.CTkCheckBox(tp_sl_frame, text="Enable Trailing Stop", variable=self.trailing_stop_var)
        trailing_checkbox.pack(anchor="w", pady=(10, 2))

        # TP/SL mode selector
        label = ctk.CTkLabel(tp_sl_frame, text="TP/SL Mode:", font=("Arial", 12))
        label.pack(anchor="w", pady=(10, 2))

        self.tp_sl_mode_var = ctk.StringVar(value="Percentage")
        mode_dropdown = ctk.CTkComboBox(
            tp_sl_frame, values=["Percentage", "Price", "ATR"], variable=self.tp_sl_mode_var, width=200
        )
        mode_dropdown.pack(anchor="w", pady=2)

    def _create_ui_preferences_tab(self):
        """Create UI Preferences tab"""
        tab = self.tabview.add("UI Preferences")

        # UI preferences frame
        ui_frame = ctk.CTkFrame(tab, fg_color="transparent")
        ui_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Theme toggle
        label = ctk.CTkLabel(ui_frame, text="Theme:", font=("Arial", 12))
        label.pack(anchor="w", pady=(5, 2))

        self.theme_var = ctk.StringVar(value="Dark")
        theme_switch = ctk.CTkSegmentedButton(
            ui_frame, values=["Dark", "Light"], variable=self.theme_var, width=200, command=self._on_theme_change
        )
        theme_switch.pack(anchor="w", pady=2)

        # Font size
        label = ctk.CTkLabel(ui_frame, text="Font Size:", font=("Arial", 12))
        label.pack(anchor="w", pady=(10, 2))

        self.font_size_var = ctk.IntVar(value=12)
        font_size_slider = ctk.CTkSlider(
            ui_frame,
            from_=10,
            to=16,
            number_of_steps=7,
            variable=self.font_size_var,
            width=300,
            command=self._on_font_size_change,
        )
        font_size_slider.pack(anchor="w", pady=2)

        self.font_size_label = ctk.CTkLabel(ui_frame, text=f"Current: {self.font_size_var.get()}pt", font=("Arial", 10))
        self.font_size_label.pack(anchor="w", pady=2)

        # Auto-refresh intervals
        label = ctk.CTkLabel(ui_frame, text="Auto-Refresh Intervals (seconds):", font=("Arial", 12))
        label.pack(anchor="w", pady=(15, 2))

        refresh_frame = ctk.CTkFrame(ui_frame, fg_color="transparent")
        refresh_frame.pack(anchor="w", pady=2)

        # Signals refresh
        label1 = ctk.CTkLabel(refresh_frame, text="Signals:", font=("Arial", 10), text_color="gray")
        label1.pack(side="left", padx=(0, 5))

        self.signal_refresh_entry = ctk.CTkEntry(refresh_frame, placeholder_text="30", width=80)
        self.signal_refresh_entry.pack(side="left", padx=5)
        self.signal_refresh_entry.insert(0, "30")

        # Positions refresh
        label2 = ctk.CTkLabel(refresh_frame, text="Positions:", font=("Arial", 10), text_color="gray")
        label2.pack(side="left", padx=(10, 5))

        self.position_refresh_entry = ctk.CTkEntry(refresh_frame, placeholder_text="10", width=80)
        self.position_refresh_entry.pack(side="left", padx=5)
        self.position_refresh_entry.insert(0, "10")

        # Import/Export/Reset buttons
        button_frame = ctk.CTkFrame(ui_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=(20, 0))

        # Export button
        export_btn = ctk.CTkButton(
            button_frame,
            text="📤 Export Settings",
            fg_color="#4488ff",
            hover_color="#0066ff",
            command=self._export_settings,
        )
        export_btn.pack(side="left", padx=5)

        # Import button
        import_btn = ctk.CTkButton(
            button_frame,
            text="📥 Import Settings",
            fg_color="#44aaff",
            hover_color="#0088ff",
            command=self._import_settings,
        )
        import_btn.pack(side="left", padx=5)

        # Reset button
        reset_btn = ctk.CTkButton(
            button_frame,
            text="🔄 Reset to Defaults",
            fg_color="#ff6644",
            hover_color="#cc4422",
            command=self._reset_settings,
        )
        reset_btn.pack(side="right", padx=5)

    def _on_theme_change(self, choice):
        """Handle theme change"""
        try:
            print(f"Theme changed to: {choice}")
            if choice == "Light":
                ctk.set_appearance_mode("light")
            else:
                ctk.set_appearance_mode("dark")

            if self.on_settings_change:
                self.on_settings_change("theme", choice)
        except Exception as e:
            print(f"Error changing theme: {e}")

    def _on_font_size_change(self, value):
        """Handle font size change"""
        try:
            self.font_size_label.configure(text=f"Current: {int(value)}pt")
            if self.on_settings_change:
                self.on_settings_change("font_size", int(value))
        except Exception as e:
            print(f"Error changing font size: {e}")

    def _export_settings(self):
        """Export settings to file"""
        try:
            from gui.utils.settings_manager import SettingsManager
            from tkinter import filedialog

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

    def _import_settings(self):
        """Import settings from file"""
        try:
            from gui.utils.settings_manager import SettingsManager
            from tkinter import filedialog

            manager = SettingsManager()
            file_path = filedialog.askopenfilename(
                title="Import Settings", filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if file_path:
                if manager.import_settings(file_path):
                    print(f"Settings imported from {file_path}")
                    # Reload UI
                    settings = manager.get_all()
                    self.load_settings(settings)
        except Exception as e:
            print(f"Error importing settings: {e}")

    def _reset_settings(self):
        """Reset settings to defaults"""
        try:
            from tkinter import messagebox

            confirm = messagebox.askyesno(
                "Reset Settings", "Are you sure you want to reset all settings to defaults?\n\nThis cannot be undone."
            )

            if confirm:
                from gui.utils.settings_manager import SettingsManager

                manager = SettingsManager()
                if manager.reset_to_defaults():
                    print("Settings reset to defaults")
                    # Reload UI
                    settings = manager.get_all()
                    self.load_settings(settings)
        except Exception as e:
            print(f"Error resetting settings: {e}")

    def _test_connection(self):
        """Test API connection"""
        try:
            from gui.utils.credential_manager import CredentialManager
            from tkinter import messagebox

            # Get credentials from UI
            exchange = self.exchange_var.get().lower()
            api_key = self.api_key_entry.get().strip()
            api_secret = self.api_secret_entry.get().strip()

            if not api_key or not api_secret:
                messagebox.showwarning("Missing Credentials", "Please enter both API Key and API Secret")
                return

            # Test connection
            manager = CredentialManager()
            result = manager.test_connection(exchange, api_key, api_secret)

            if result["success"]:
                balance_info = result.get("balance", {})
                balance_str = "\n".join([f"{k}: {v}" for k, v in list(balance_info.items())[:5]])
                messagebox.showinfo(
                    "Connection Successful",
                    f"{result['message']}\n\nSample Balance:\n{balance_str if balance_str else 'No balance data'}",
                )

                if self.on_settings_change:
                    self.on_settings_change("connection_test", True)
            else:
                messagebox.showerror("Connection Failed", result["message"])

                if self.on_settings_change:
                    self.on_settings_change("connection_test", False)

        except Exception as e:
            from tkinter import messagebox

            messagebox.showerror("Error", f"Connection test failed: {e}")

    def _save_credentials(self):
        """Save API credentials"""
        try:
            from gui.utils.credential_manager import CredentialManager
            from tkinter import messagebox

            # Get credentials from UI
            exchange = self.exchange_var.get().lower()
            api_key = self.api_key_entry.get().strip()
            api_secret = self.api_secret_entry.get().strip()

            if not api_key or not api_secret:
                messagebox.showwarning("Missing Credentials", "Please enter both API Key and API Secret")
                return

            # Confirm before saving
            confirm = messagebox.askyesno(
                "Save Credentials",
                f"Save API credentials for {exchange}?\n\n"
                "Credentials will be stored securely in the .env file.\n\n"
                "⚠️ Make sure .env is in your .gitignore!",
            )

            if not confirm:
                return

            # Save credentials
            manager = CredentialManager()
            success = manager.save_credentials(exchange, api_key, api_secret)

            if success:
                messagebox.showinfo(
                    "Success", f"Credentials saved successfully for {exchange}!\n\nThey are stored in the .env file."
                )

                # Clear UI fields for security
                self.api_key_entry.delete(0, "end")
                self.api_secret_entry.delete(0, "end")

                if self.on_settings_change:
                    self.on_settings_change("save_credentials", True)
            else:
                messagebox.showerror("Error", "Failed to save credentials")

                if self.on_settings_change:
                    self.on_settings_change("save_credentials", False)

        except Exception as e:
            from tkinter import messagebox

            messagebox.showerror("Error", f"Failed to save credentials: {e}")

    def _on_mode_change(self):
        """Handle mode radio button change"""
        try:
            from gui.utils.colors import Colors
            from tkinter import messagebox

            mode = self.mode_var.get()

            mode_descriptions = {
                "PRODUCTION": ("⚠️ Real money at risk", Colors.PRODUCTION),
                "DEMO": ("Testnet - Requires API keys", Colors.DEMO),
                "DRY_RUN": ("✅ Safe local simulation", Colors.DRY_RUN),
            }

            description, color = mode_descriptions.get(mode, ("✅ Safe local simulation", Colors.DRY_RUN))
            self.mode_description_label.configure(text=description, text_color=color)

            if mode == "PRODUCTION":
                messagebox.showwarning(
                    "Production Mode",
                    "⚠️ WARNING: You are about to use PRODUCTION mode!\n\n"
                    "This will execute REAL trades with REAL money.\n"
                    "Make sure you understand the risks involved.",
                )

            elif mode == "DRY_RUN":
                self.api_key_frame.pack_forget()

            else:
                if not self.api_key_frame.winfo_ismapped():
                    self.api_key_frame.pack(fill="x", after=self.mode_description_label)

            if self.on_settings_change:
                self.on_settings_change("mode", mode)

        except Exception as e:
            print(f"Error handling mode change: {e}")

    def get_settings(self) -> Dict:
        """
        Get current settings (excluding API credentials for security)

        Returns:
            Dictionary with risk, filters, and tp_sl settings
            Note: API credentials are NOT included and must be loaded separately
                  using CredentialManager
        """
        try:
            # Validate and parse numeric values
            try:
                max_position_size = float(self.max_pos_size_entry.get())
                if max_position_size <= 0:
                    raise ValueError("Max position size must be positive")
            except ValueError as e:
                print(f"Invalid max position size: {e}, using default 100.00")
                max_position_size = 100.0

            try:
                max_open_positions = int(self.max_positions_entry.get())
                if max_open_positions <= 0:
                    raise ValueError("Max open positions must be positive")
            except ValueError as e:
                print(f"Invalid max open positions: {e}, using default 3")
                max_open_positions = 3

            try:
                max_daily_loss = float(self.max_daily_loss_entry.get())
                if max_daily_loss <= 0:
                    raise ValueError("Max daily loss must be positive")
            except ValueError as e:
                print(f"Invalid max daily loss: {e}, using default 50.00")
                max_daily_loss = 50.0

            try:
                min_volume = float(self.min_volume_entry.get())
                if min_volume < 0:
                    raise ValueError("Min volume cannot be negative")
            except ValueError as e:
                print(f"Invalid min volume: {e}, using default 50")
                min_volume = 50.0

            try:
                default_tp = float(self.default_tp_entry.get())
                if default_tp <= 0 or default_tp > 100:
                    raise ValueError("Default TP must be between 0 and 100")
            except ValueError as e:
                print(f"Invalid default TP: {e}, using default 5.0")
                default_tp = 5.0

            try:
                default_sl = float(self.default_sl_entry.get())
                if default_sl <= 0 or default_sl > 100:
                    raise ValueError("Default SL must be between 0 and 100")
            except ValueError as e:
                print(f"Invalid default SL: {e}, using default 2.5")
                default_sl = 2.5

            return {
                "risk": {
                    "max_position_size": max_position_size,
                    "max_open_positions": max_open_positions,
                    "max_daily_loss": max_daily_loss,
                    "default_leverage": self.default_leverage_var.get(),
                },
                "filters": {
                    "min_signal_score": self.min_score_var.get(),
                    "enable_xgboost": self.enable_xgboost_var.get(),
                    "symbol_whitelist": self.whitelist_entry.get("0.0", "end-1c"),
                    "min_volume": min_volume,
                },
                "api": {
                    "exchange": self.exchange_var.get(),
                    "mode": self.mode_var.get(),
                    # SECURITY: API credentials are NOT returned here
                    # Use CredentialManager.load_credentials() to retrieve them
                },
                "tp_sl": {
                    "default_tp": default_tp,
                    "default_sl": default_sl,
                    "trailing_stop": self.trailing_stop_var.get(),
                    "mode": self.tp_sl_mode_var.get(),
                },
            }
        except Exception as e:
            print(f"Error getting settings: {e}")
            # Return safe defaults
            return {
                "risk": {
                    "max_position_size": 100.0,
                    "max_open_positions": 3,
                    "max_daily_loss": 50.0,
                    "default_leverage": "10x",
                },
                "filters": {
                    "min_signal_score": 0.7,
                    "enable_xgboost": True,
                    "symbol_whitelist": "",
                    "min_volume": 50.0,
                },
                "api": {
                    "exchange": "Binance",
                },
                "tp_sl": {
                    "default_tp": 5.0,
                    "default_sl": 2.5,
                    "trailing_stop": False,
                    "mode": "Percentage",
                },
            }

    def load_settings(self, settings: Dict):
        """Load settings into UI"""
        if "risk" in settings:
            risk = settings["risk"]
            self.max_pos_size_entry.delete(0, "end")
            self.max_pos_size_entry.insert(0, str(risk.get("max_position_size", 100.0)))
            self.max_positions_entry.delete(0, "end")
            self.max_positions_entry.insert(0, str(risk.get("max_open_positions", 3)))
            self.max_daily_loss_entry.delete(0, "end")
            self.max_daily_loss_entry.insert(0, str(risk.get("max_daily_loss", 50.0)))
            self.default_leverage_var.set(risk.get("default_leverage", "10x"))

        if "filters" in settings:
            filters = settings["filters"]
            self.min_score_var.set(filters.get("min_signal_score", 0.7))
            self.enable_xgboost_var.set(filters.get("enable_xgboost", True))
            self.whitelist_entry.delete("0.0", "end")
            self.whitelist_entry.insert("0.0", filters.get("symbol_whitelist", ""))
            self.min_volume_entry.delete(0, "end")
            self.min_volume_entry.insert(0, str(filters.get("min_volume", 50)))

        if "api" in settings:
            api = settings["api"]
            self.mode_var.set(api.get("mode", "DRY_RUN"))
            self.exchange_var.set(api.get("exchange", "Binance"))
            self.api_key_entry.delete(0, "end")
            self.api_key_entry.insert(0, api.get("api_key", ""))
            self.api_secret_entry.delete(0, "end")
            self.api_secret_entry.insert(0, api.get("api_secret", ""))
            self._on_mode_change()

        if "tp_sl" in settings:
            tp_sl = settings["tp_sl"]
            self.default_tp_entry.delete(0, "end")
            self.default_tp_entry.insert(0, str(tp_sl.get("default_tp", 5.0)))
            self.default_sl_entry.delete(0, "end")
            self.default_sl_entry.insert(0, str(tp_sl.get("default_sl", 2.5)))
            self.trailing_stop_var.set(tp_sl.get("trailing_stop", False))
            self.tp_sl_mode_var.set(tp_sl.get("mode", "Percentage"))
