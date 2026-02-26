"""Lifecycle and action callbacks for Auto Trade Dashboard."""

from typing import TYPE_CHECKING, Any, Callable

from modules.common.ui.logging import log_error, log_info


class LifecycleActionsMixin:
    """Provide lifecycle and action callback handlers."""

    # --- Attribute stubs satisfied at class body level so Pylance/mypy
    #     can resolve them in all methods of this mixin.  The real values
    #     are injected by the concrete subclass (MainWindow) at runtime.
    auto_trade_manager: Any
    position_action_handler: Any
    updater_manager: Any
    scanner_manager: Any
    recovery_manager: Any
    settings_manager: Any
    ws_data_service: Any
    status_bar: Any
    _original_stdout: Any

    if TYPE_CHECKING:
        # Method stubs: satisfied by tkinter.Tk / the concrete subclass.
        def after(self, ms: int, func: Callable) -> str: ...
        def destroy(self) -> None: ...
        def refresh_positions(self) -> None: ...
        def refresh_account(self) -> None: ...
        def _update_connection_status(self) -> None: ...

    def on_trade_executed(self):
        """Callback when manual trade is executed."""
        log_info("Trade executed! Refreshing positions...")
        self.refresh_positions()
        self.refresh_account()

    def on_auto_trade_toggle(self, enabled: bool):
        """Callback when auto-trade is toggled."""
        log_info(f"Auto-trade {'enabled' if enabled else 'disabled'}")
        if enabled:
            self.auto_trade_manager.start()
        else:
            self.auto_trade_manager.stop()

    def on_position_action(self, action_data: dict):
        """Handle position actions from GUI."""
        return self.position_action_handler.handle_action(action_data)

    def on_sync_positions(self):
        """Handle manual sync of Binance positions to database."""
        import threading
        from tkinter import messagebox

        def sync_thread():
            try:
                self.after(0, lambda: self.status_bar.set_connection_status(True, "🔄 Syncing positions..."))

                result = self.position_action_handler.sync_positions_from_binance()

                if result.get("success"):
                    stats = result.get("stats", {})
                    message = (
                        f"✅ Sync completed!\n\n"
                        f"Found: {stats.get('fetched', 0)} positions\n"
                        f"Synced: {stats.get('synced', 0)} new\n"
                        f"Existing: {stats.get('existing', 0)} already in DB\n"
                        f"Closed: {stats.get('closed', 0)} stale\n"
                        f"Failed: {stats.get('failed', 0)}"
                    )
                    self.after(0, lambda: messagebox.showinfo("Position Sync", message))
                    self.after(0, lambda: self.status_bar.set_connection_status(True, "✅ Sync completed"))
                    self.after(100, self.refresh_positions)
                    self.after(3000, lambda: self._update_connection_status())
                else:
                    error_msg = result.get("message", "Unknown error")
                    self.after(0, lambda: messagebox.showerror("Position Sync Failed", f"Error: {error_msg}"))
                    self.after(0, lambda: self.status_bar.set_connection_status(False, "❌ Sync failed"))
                    self.after(3000, lambda: self._update_connection_status())

            except Exception as e:
                log_error(f"Error during position sync: {e}", exc_info=True)
                err_str = str(e)
                self.after(
                    0, lambda error=err_str: messagebox.showerror("Position Sync Error", f"Fatal error: {error}")
                )
                self.after(0, lambda: self.status_bar.set_connection_status(False, "❌ Sync error"))
                self.after(3000, lambda: self._update_connection_status())

        thread = threading.Thread(target=sync_thread, daemon=True)
        thread.start()

    def on_closing(self):
        """Handle application shutdown.

        Strategy: destroy the window immediately for a snappy UX, then let
        background daemon threads handle slow teardown (WebSocket close,
        listen-key DELETE, DB flushes). Daemon threads are killed automatically
        when the process exits, so nothing leaks.
        """
        import threading

        # 1. Save settings synchronously – this is fast (disk write only).
        try:
            if hasattr(self, "settings_manager"):
                self.settings_manager.save()
                log_info("Settings saved on exit")
        except Exception as e:
            log_error(f"Error saving settings: {e}")

        # 2. Signal all updaters/managers to stop (non-blocking flag sets).
        try:
            self.updater_manager.stop_all()
        except Exception:
            pass
        try:
            self.auto_trade_manager.stop()
        except Exception:
            pass
        try:
            self.scanner_manager._stop_scanner()
        except Exception:
            pass
        try:
            if hasattr(self, "recovery_manager"):
                self.recovery_manager.stop()
        except Exception:
            pass

        # 3. Restore original stdout before destroying the window.
        import sys

        if hasattr(self, "_original_stdout"):
            sys.stdout = self._original_stdout

        # 4. Destroy the window NOW so the user sees the GUI close instantly.
        #    The actual slow cleanup (WebSocket teardown, REST calls) happens
        #    in a daemon thread that will be killed when the process exits.
        self.destroy()

        # 4. Kick off slow cleanup in a daemon thread.
        def _background_cleanup():
            try:
                if hasattr(self, "ws_data_service"):
                    self.ws_data_service.stop()
            except Exception as e:
                log_error(f"Background WS cleanup error: {e}")

        t = threading.Thread(target=_background_cleanup, daemon=True, name="GUICleanup")
        t.start()
