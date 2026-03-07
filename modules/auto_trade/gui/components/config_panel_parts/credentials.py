from modules.auto_trade.gui.utils.mask_utils import mask_api_key, mask_secret
from modules.common.ui.logging import log_error


def test_connection(panel):
    """Test API connection."""
    try:
        from tkinter import messagebox

        from modules.auto_trade.gui.services.credential_manager import CredentialManager

        exchange = panel.exchange_var.get().lower()
        api_key = panel.api_key_entry.get().strip()
        api_secret = panel.api_secret_entry.get().strip()

        if not api_key or not api_secret:
            messagebox.showwarning("Missing Credentials", "Please enter both API Key and API Secret")
            return

        manager = CredentialManager()
        result = manager.test_connection(exchange, api_key, api_secret)

        if result["success"]:
            balance_info = result.get("balance", {})
            balance_str = "\n".join([f"{k}: {v}" for k, v in list(balance_info.items())[:5]])
            messagebox.showinfo(
                "Connection Successful",
                f"{result['message']}\n\nSample Balance:\n{balance_str if balance_str else 'No balance data'}",
            )

            if panel.on_settings_change:
                panel.on_settings_change("connection_test", True)
        else:
            messagebox.showerror("Connection Failed", result["message"])

            if panel.on_settings_change:
                panel.on_settings_change("connection_test", False)

    except Exception as e:
        from tkinter import messagebox

        messagebox.showerror("Error", f"Connection test failed: {e}")


def save_credentials(panel):
    """Save API credentials."""
    try:
        from tkinter import messagebox

        from modules.auto_trade.gui.services.credential_manager import CredentialManager

        exchange = panel.exchange_var.get().lower()
        api_key = panel.api_key_entry.get().strip()
        api_secret = panel.api_secret_entry.get().strip()

        if not api_key or not api_secret:
            messagebox.showwarning("Missing Credentials", "Please enter both API Key and API Secret")
            return

        confirm = messagebox.askyesno(
            "Save Credentials",
            f"Save API credentials for {exchange}?\n\n"
            "Credentials will be stored securely in the .env file.\n\n"
            "⚠️ Make sure .env is in your .gitignore!",
        )

        if not confirm:
            return

        manager = CredentialManager()
        success = manager.save_credentials(exchange, api_key, api_secret)

        if success:
            messagebox.showinfo(
                "Success",
                f"Credentials saved successfully for {exchange}!\n\nThey are stored in the .env file.",
            )

            panel.api_key_entry.delete(0, "end")
            panel.api_secret_entry.delete(0, "end")
            panel._editing_credentials = False
            panel._refresh_credentials_display()

            if panel.on_settings_change:
                panel.on_settings_change("save_credentials", True)
        else:
            messagebox.showerror("Error", "Failed to save credentials")

            if panel.on_settings_change:
                panel.on_settings_change("save_credentials", False)

    except Exception as e:
        from tkinter import messagebox

        messagebox.showerror("Error", f"Failed to save credentials: {e}")


def on_change_credentials(panel):
    """Handle Change Credentials button click."""
    panel._editing_credentials = True
    panel._refresh_credentials_display()


def on_cancel_credentials(panel):
    """Handle Cancel Credentials button click."""
    panel._editing_credentials = False
    panel.api_key_entry.delete(0, "end")
    panel.api_secret_entry.delete(0, "end")
    panel._refresh_credentials_display()


def refresh_credentials_display(panel):
    """Refresh credentials display (masked vs entry)."""
    try:
        from modules.auto_trade.gui.services.credential_manager import CredentialManager

        exchange = panel.exchange_var.get().lower()
        manager = CredentialManager()

        if manager.has_credentials(exchange) and not panel._editing_credentials:
            panel.credentials_entry_frame.pack_forget()

            creds = manager.load_credentials(exchange)
            panel.api_key_masked_label.configure(text=mask_api_key(creds.get("api_key") or ""))
            panel.api_secret_masked_label.configure(text=mask_secret(creds.get("api_secret") or ""))

            panel.credentials_masked_frame.pack(fill="x")
        else:
            panel.credentials_masked_frame.pack_forget()

            if not panel._editing_credentials:
                panel.api_key_entry.delete(0, "end")
                panel.api_secret_entry.delete(0, "end")

            if panel._editing_credentials:
                panel.cancel_credentials_btn.pack(side="left", padx=(0, 10))
            else:
                panel.cancel_credentials_btn.pack_forget()

            panel.credentials_entry_frame.pack(fill="x")

    except Exception as e:
        log_error("Error refreshing credentials display: %s", e)
