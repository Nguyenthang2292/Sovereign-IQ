"""Actions Section Component for Database Panel."""

import os
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
from collections.abc import Mapping
from typing import Any, Callable, Optional

import customtkinter as ctk

from modules.auto_trade.database.repository.context import RepositoryContext
from modules.auto_trade.gui.components.loading_overlay import LoadingOverlay
from modules.auto_trade.gui.config.database_panel_config import DatabasePanelConfig
from modules.auto_trade.gui.services.database_service import (
    DatabaseService,
    DataViewerService,
)
from modules.auto_trade.gui.utils.svg_icons import get_button_icon
from modules.common.ui.logging import log_warn


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

        # (label, callback, svg_icon_key)
        actions = [
            ("Create Backup", self._create_backup, "save"),
            ("Run Migrations", self._run_migrations, "refresh"),
            ("Reconcile with Binance", self._reconcile_with_binance, "git_compare"),
            ("Remove All Open Orders in DB", self._remove_all_open_orders, "trash"),
            ("Cleanup Old Records", self._cleanup_records, "sparkles"),
            ("Export to CSV", self._export_csv, "file_up"),
            ("View Audit Log", self._view_audit_log, "scroll_text"),
            ("Check Integrity", self._check_integrity, "shield_check"),
        ]

        for text, command, icon_key in actions:
            icon = get_button_icon(icon_key, size=(16, 16), variant="primary")
            ctk.CTkButton(
                frame,
                text=f"  {text}",
                command=command,
                image=icon,
                compound="left",
                anchor="w",
            ).pack(fill="x", padx=10, pady=2)

    def _create_backup(self):
        """Create database backup."""
        try:
            backup_path = DatabaseService.create_backup()
            if backup_path:
                self.log_callback(f"Backup: {backup_path}", "SUCCESS")
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

    def _export_csv(self) -> None:
        """Export current data to CSV using RepositoryContext data."""
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            import csv  # noqa: PLC0415
            csv_module: Any = csv

            table_name = self.get_current_table()
            rows: list[Any] = DataViewerService.get_table_data(table_name, limit=99999)

            if not rows:
                self.log_callback(f"No data found for table: {table_name}", "WARNING")
                return

            # rows are dicts — write header from first row keys
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                first_row = rows[0]
                if isinstance(first_row, Mapping):
                    dict_writer = csv_module.DictWriter(f, fieldnames=first_row.keys())
                    dict_writer.writeheader()
                    dict_writer.writerows(rows)
                else:
                    # ORM objects — convert attributes
                    row_writer = csv_module.writer(f)
                    cols = [c.key for c in first_row.__table__.columns]
                    row_writer.writerow(cols)
                    for row in rows:
                        row_writer.writerow([getattr(row, c) for c in cols])

            self.log_callback(f"Exported {table_name} ({len(rows)} rows) to {file_path}", "SUCCESS")

        except Exception as e:
            self.log_callback(f"Export failed: {e}", "ERROR")

    def _view_audit_log(self):
        """View audit logs."""
        try:
            logs = DataViewerService.get_audit_logs(limit=100)

            output = "Recent Audit Logs:\n"
            output += "-" * 80 + "\n"
            for log in logs:
                ts = log.get("timestamp", log.get("created_at", ""))
                severity = log.get("severity", "")
                event_type = log.get("event_type", "")
                summary = log.get("event_summary", log.get("summary", ""))
                output += f"[{ts}] [{severity}] {event_type}: {summary}\n"

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

        loading = LoadingOverlay(self.parent)
        loading.show("Reconciling with Binance...")

        self.log_callback(
            f"Reconciling with Binance (last {DatabasePanelConfig.DEFAULT_RECONCILE_HOURS}h)...",
            "INFO",
        )
        try:
            from modules.auto_trade.execution.binance_client import BinanceClient
            from modules.auto_trade.gui.services.position_sync_service import PositionSyncService

            client = BinanceClient(api_key=api_key, api_secret=api_secret, testnet=testnet, dry_run=False)

            stats = PositionSyncService.sync_all_positions(client)

            inserted = stats.get("synced", 0)
            skipped = stats.get("existing", 0)
            errors_count = stats.get("failed", 0)

            self.log_callback(
                f"Reconcile done: inserted={inserted}, skipped={skipped}, errors={errors_count}",
                "SUCCESS",
            )
            self.refresh_callback()
            messagebox.showinfo(
                "Reconcile",
                f"Inserted: {inserted}, Skipped (already in DB): {skipped}. Errors: {errors_count}",
            )
        except Exception as e:
            self.log_callback(f"Reconcile failed: {e}", "ERROR")
            messagebox.showerror("Reconcile", str(e))
        finally:
            loading.hide()

    def _remove_all_open_orders(self):
        """Cancel all open (programmatic) orders in the database."""
        if not messagebox.askyesno(
            "Confirm Remove",
            "Remove all open orders from the database? This only updates records in DB; it does not cancel orders on the exchange.",
        ):
            return
        try:
            ctx = RepositoryContext.from_env()
            positions = ctx.orders.get_open_positions()
            count = len(positions)
            if count == 0:
                self.log_callback("No open orders in DB to remove", "INFO")
                messagebox.showinfo("Remove Open Orders", "No open orders in database.")
                return

            # DynamoDB: update status to CANCELLED
            removed = 0
            for pos in positions:
                order_id = pos.get("order_id")
                if not isinstance(order_id, str) or not order_id:
                    log_warn(f"Skipping open order without valid order_id: {pos}")
                    continue
                try:
                    ctx.orders.update_order_status(order_id, "CANCELLED")
                    removed += 1
                except Exception as update_err:
                    log_warn(f"Could not cancel order {order_id}: {update_err}")
            self.log_callback(f"Cancelled {removed} open order(s) in DB", "SUCCESS")
            messagebox.showinfo("Remove Open Orders", f"Cancelled {removed} open order(s) in database.")

            self.refresh_callback()
        except Exception as e:
            self.log_callback(f"Remove open orders failed: {e}", "ERROR")
            messagebox.showerror("Remove Open Orders", str(e))

    def _show_in_data_viewer(self, content: str):
        """Show content in data viewer."""
        callback = getattr(self.parent, "data_viewer_callback", None)
        if callable(callback):
            callback(content)
