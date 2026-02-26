def on_limit_steps_toggle(panel):
    """Show/hide max steps field based on checkbox."""
    try:
        if panel.limit_trailing_steps_var.get():
            panel.max_steps_label.pack(anchor="w", pady=(5, 2))
            panel.max_steps_entry.pack(fill="x", pady=(2, 8))
        else:
            panel.max_steps_label.pack_forget()
            panel.max_steps_entry.pack_forget()
    except Exception as e:
        print(f"Error toggling limit steps: {e}")


def on_negative_be_toggle(panel):
    """Show/hide negative breakeven threshold based on checkbox."""
    try:
        if panel.negative_be_var.get():
            panel.negative_be_threshold_label.pack(anchor="w", pady=(10, 2))
            panel.negative_be_threshold_entry.pack(fill="x", pady=(2, 8))
        else:
            panel.negative_be_threshold_label.pack_forget()
            panel.negative_be_threshold_entry.pack_forget()
    except Exception as e:
        print(f"Error toggling negative breakeven: {e}")


def on_mode_change(panel, show_warning: bool = True):
    """Handle mode radio button change."""
    try:
        from tkinter import messagebox

        from modules.auto_trade.gui.utils.colors import Colors
        from modules.auto_trade.gui.utils.svg_icons import get_icon

        mode = panel.mode_var.get()

        mode_descriptions = {
            "PRODUCTION": ("  Real money at risk", Colors.PRODUCTION, "alert_triangle"),
            "DEMO": ("  Testnet - Requires API keys", Colors.DEMO, "database"),
            "DRY_RUN": ("  Safe local simulation", Colors.DRY_RUN, "shield_check"),
        }

        description, color, icon_key = mode_descriptions.get(
            mode, ("  Safe local simulation", Colors.DRY_RUN, "shield_check")
        )
        icon = get_icon(icon_key, size=(16, 16), light_color=color, dark_color=color)

        panel.mode_description_label.configure(text=description, text_color=color, image=icon, compound="left")

        if mode == "DRY_RUN":
            panel.api_key_frame.pack_forget()
        else:
            if not panel.api_key_frame.winfo_ismapped():
                panel.api_key_frame.pack(fill="x", after=panel.mode_description_label)
            panel._refresh_credentials_display()

        if show_warning and mode == "PRODUCTION":
            messagebox.showwarning(
                "Production Mode",
                "⚠️ WARNING: You are about to use PRODUCTION mode!\n\n"
                "This will execute REAL trades with REAL money.\n"
                "Make sure you understand the risks involved.",
            )

        if panel.on_settings_change and not panel._suppress_mode_notify:
            panel.on_settings_change("mode", mode)

    except Exception as e:
        print(f"Error handling mode change: {e}")
