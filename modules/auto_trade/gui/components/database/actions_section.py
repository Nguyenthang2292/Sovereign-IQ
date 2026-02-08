"""Actions Section Component for Database Panel."""

import logging
import os
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
from typing import Any, Callable, Optional

import customtkinter as ctk

from modules.auto_trade.database import get_open_positions, session_scope
from modules.auto_trade.database.models import AuditLog, MartingaleChain, Order, Signal
from modules.auto_trade.gui.components.loading_overlay import LoadingOverlay
from modules.auto_trade.gui.config.database_panel_config import DatabasePanelConfig
from modules.auto_trade.gui.services.database_service import (
    DatabaseService,
    ReconciliationService,
    DataViewerService,
)

logger = logging.getLogger(__name__)


class ActionsSection:
    """Quick actions section component."""

    def __init__(
        self,
        parent: ctk.CTkFrame,
        log_callback: Callable,
        refresh_callback: Callable,
        get_current_table_callback: Callable[[], str],
        settings_manager: Optional[Any] = None,
    ):
        self.parent = parent
        self.log_callback = log_callback
        self.refresh_callback = refresh_callback
        self.get_current_table = get_current_table_callback
        self.settings_manager = settings_manager
        self._create_ui()

    def _create_ui(self):
        """Create the actions section UI."""
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(frame, text="⚡ Quick Actions", font=DatabasePanelConfig.TITLE_FONT).pack(
            anchor="w",
            padx=DatabasePanelConfig.PADX_MEDIUM,
            pady=(DatabasePanelConfig.PADX_MEDIUM, DatabasePanelConfig.PADY_SMALL),
        )

        actions = [
            ("💾 Create Backup", self._create_backup),
            ("🔄 Run Migrations", self._run_migrations),
            ("🔄 Reconcile with Binance", self._reconcile_with_binance),
            ("🗑️ Remove All Open Orders in DB", self._remove_all_open_orders),
            ("🧹 Cleanup Old Records", self._cleanup_records),
            ("📤 Export to CSV", self._export_csv),
            ("📋 View Audit Log", self._view_audit_log),
            ("🔍 Check Integrity", self._check_integrity),
        ]

        for text, command in actions:
            ctk.CTkButton(frame, text=text, command=command).pack(fill="x", padx=10, pady=2)

    def _create_backup(self):
        """Create database backup."""
        try:
            backup_path = DatabaseService.create_backup()
            if backup_path:
                self.log_callback(f"Backup created at: {backup_path}", "SUCCESS")
                self.refresh_callback()
            else:
                self.log_callback("Backup failed (check logs)", "ERROR")
        except Exception as e:
            self.log_callback(f"Backup failed: {e}", "ERROR")

    def _run_migrations(self):
        """Run database migrations."""
        try:
            success, msg = DatabaseService.run_migrations()
            level = "INFO" if success else "WARNING"
            self.log_callback(msg, level)
        except Exception as e:
            self.log_callback(f"Migration run failed: {e}", "ERROR")

    def _cleanup_records(self):
        """Cleanup old records."""
        if not messagebox.askyesno(
            "Confirm Cleanup",
            f"Are you sure you want to delete old records (>{DatabasePanelConfig.DEFAULT_DAYS_TO_KEEP} days)?",
        ):
            return

        try:
            success, msg = DatabaseService.cleanup_old_records()
            if success:
                self.log_callback(msg, "SUCCESS")
                messagebox.showinfo("Cleanup Complete", msg)
                self.refresh_callback()
            else:
                self.log_callback(f"Cleanup failed: {msg}", "ERROR")
        except Exception as e:
            self.log_callback(f"Cleanup failed: {e}", "ERROR")

    def _export_csv(self):
        """Export current table to CSV."""
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            from modules.auto_trade.database.utils import DataExporter

            table_map = {
                "Orders": Order,
                "Signals": Signal,
                "Martingale Chains": MartingaleChain,
                "Audit Log": AuditLog,
            }

            model_class = table_map.get(self.get_current_table())
            if not model_class:
                self.log_callback(f"Unknown table selected for export: {self.get_current_table()}", "ERROR")
                return

            with session_scope() as session:
                success = DataExporter.export_to_csv(session, model_class, file_path)

                if success:
                    self.log_callback(f"Exported {self.get_current_table()} to {file_path}", "SUCCESS")
                else:
                    self.log_callback("Export failed (check logs)", "ERROR")

        except Exception as e:
            self.log_callback(f"Export failed: {e}", "ERROR")

    def _view_audit_log(self):
        """View audit logs."""
        try:
            logs = DataViewerService.get_audit_logs(limit=100)

            output = "Recent Audit Logs:\n"
            output += "-" * 80 + "\n"
            for log in logs:
                output += f"[{log.timestamp}] [{log.severity}] {log.event_type}: {log.event_summary}\n"

            self._show_in_data_viewer(output)
            self.log_callback("Retrieved audit logs", "INFO")

        except Exception as e:
            self.log_callback(f"Failed to view audit log: {e}", "ERROR")

    def _check_integrity(self):
        """Check database integrity."""
        try:
            is_ok, status = DatabaseService.check_integrity()
            self.log_callback(f"Integrity Check: {status}", "INFO" if is_ok else "ERROR")
            messagebox.showinfo("Integrity Check", f"Database Integrity: {status}")

        except Exception as e:
            self.log_callback(f"Integrity check failed: {e}", "ERROR")

    def _reconcile_with_binance(self):
        """Fetch AT_* orders from Binance and insert any missing into DB."""
        api_key = os.getenv("BINANCE_API_KEY", "").strip()
        api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
        if not api_key or not api_secret:
            self.log_callback("Reconcile skipped: BINANCE_API_KEY or BINANCE_API_SECRET not set", "WARNING")
            messagebox.showwarning(
                "Reconcile",
                "Set BINANCE_API_KEY and BINANCE_API_SECRET to reconcile with Binance.",
            )
            return

        if not self.settings_manager:
            self.log_callback("Reconcile skipped: settings_manager not available", "WARNING")
            return

        testnet = bool(self.settings_manager.get("api.testnet", False))
        symbols = self.settings_manager.get("filters.symbol_whitelist") or None

        # Show loading overlay
        loading = LoadingOverlay(self.parent)
        loading.show("Reconciling with Binance...")

        self.log_callback(f"Reconciling with Binance (last {DatabasePanelConfig.DEFAULT_RECONCILE_HOURS}h)...", "INFO")
        try:
            result = ReconciliationService.reconcile_with_binance(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                symbols=symbols,
                since_hours=DatabasePanelConfig.DEFAULT_RECONCILE_HOURS,
            )
            inserted = result.get("inserted", 0)
            skipped = result.get("skipped", 0)
            closed_stale = result.get("closed_stale", 0)
            errors = result.get("errors", [])
            self.log_callback(
                f"Reconcile done: inserted={inserted}, skipped={skipped}, closed_stale={closed_stale}", "SUCCESS"
            )
            for err in errors[: DatabasePanelConfig.MAX_RECONCILE_ERRORS_SHOWN]:
                self.log_callback(err, "ERROR")
            if len(errors) > DatabasePanelConfig.MAX_RECONCILE_ERRORS_SHOWN:
                self.log_callback(
                    f"... and {len(errors) - DatabasePanelConfig.MAX_RECONCILE_ERRORS_SHOWN} more errors", "ERROR"
                )
            self.refresh_callback()
            messagebox.showinfo(
                "Reconcile",
                f"Inserted: {inserted}, Skipped (already in DB): {skipped}, Closed stale: {closed_stale}. Errors: {len(errors)}",
            )
        except Exception as e:
            self.log_callback(f"Reconcile failed: {e}", "ERROR")
            messagebox.showerror("Reconcile", str(e))
        finally:
            # Always hide overlay to prevent stuck UI
            loading.hide()

    def _remove_all_open_orders(self):
        """Delete all open (programmatic) orders from the database after clearing FK references."""
        if not messagebox.askyesno(
            "Confirm Remove",
            "Remove all open orders from the database? This only deletes records in DB; it does not cancel orders on the exchange.",
        ):
            return
        try:
            with session_scope() as session:
                positions = get_open_positions(session)
                count = len(positions)
                if count == 0:
                    self.log_callback("No open orders in DB to remove", "INFO")
                    messagebox.showinfo("Remove Open Orders", "No open orders in database.")
                    return
                order_ids = [o.order_id for o in positions]
                # Clear FK references so we can delete orders
                session.query(Signal).filter(Signal.execution_order_id.in_(order_ids)).update(
                    {Signal.execution_order_id: None}, synchronize_session=False
                )
                from sqlalchemy import or_

                chains = (
                    session.query(MartingaleChain)
                    .filter(
                        or_(
                            MartingaleChain.initial_order_id.in_(order_ids),
                            MartingaleChain.latest_order_id.in_(order_ids),
                            MartingaleChain.recovery_order_id.in_(order_ids),
                        )
                    )
                    .all()
                )
                for ch in chains:
                    if ch.initial_order_id in order_ids:
                        ch.initial_order_id = None
                    if ch.latest_order_id in order_ids:
                        ch.latest_order_id = None
                    if ch.recovery_order_id in order_ids:
                        ch.recovery_order_id = None
                session.query(Order).filter(Order.parent_order_id.in_(order_ids)).update(
                    {Order.parent_order_id: None}, synchronize_session=False
                )
                for o in positions:
                    session.delete(o)
            self.log_callback(f"Removed {count} open order(s) from DB", "SUCCESS")
            messagebox.showinfo("Remove Open Orders", f"Removed {count} open order(s) from database.")
            self.refresh_callback()
        except Exception as e:
            self.log_callback(f"Remove open orders failed: {e}", "ERROR")
            messagebox.showerror("Remove Open Orders", str(e))

    def _show_in_data_viewer(self, content: str):
        """Show content in data viewer."""
        if hasattr(self.parent, "data_viewer_callback"):
            self.parent.data_viewer_callback(content)
