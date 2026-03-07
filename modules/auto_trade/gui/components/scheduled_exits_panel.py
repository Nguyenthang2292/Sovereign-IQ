from __future__ import annotations

from datetime import datetime, timedelta, timezone
from tkinter import messagebox, simpledialog
from typing import Any, Callable, Dict, List, Optional

import customtkinter as ctk

from modules.auto_trade.database import RepositoryContext
from modules.auto_trade.execution.auto_close_timer import compute_deadline_utc, get_order_id, parse_utc_datetime
from modules.auto_trade.gui.utils.colors import Colors
from modules.common.ui.logging import log_error, log_info, log_warn


class ScheduledExitsPanel(ctk.CTkFrame):
    """Panel displaying pending auto-close exits and close history."""

    def __init__(
        self,
        parent: Any,
        settings_manager: Any,
        on_open_settings: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.on_open_settings = on_open_settings
        self.repo_context: Optional[RepositoryContext] = None

        self._pending_rows: List[Dict[str, Any]] = []
        self._build_ui()
        self._schedule_refresh()
        self._schedule_countdown_tick()

    def _get_repo_context(self) -> Optional[RepositoryContext]:
        if self.repo_context is None:
            try:
                self.repo_context = RepositoryContext.from_env()
            except Exception as exc:
                log_warn(f"[ScheduledExits] Could not initialize repository context: {exc}")
                return None
        return self.repo_context

    def _build_ui(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        top.grid_columnconfigure(1, weight=1)

        self.enabled_var = ctk.BooleanVar(value=bool(self.settings_manager.get("auto_close.enabled", False)))
        self.enabled_checkbox = ctk.CTkCheckBox(
            top,
            text="Enable Auto-Close",
            variable=self.enabled_var,
            command=self._on_toggle_enabled,
        )
        self.enabled_checkbox.grid(row=0, column=0, sticky="w")

        self.status_label = ctk.CTkLabel(top, text="", text_color="gray")
        self.status_label.grid(row=0, column=1, sticky="w", padx=(12, 0))

        open_settings_btn = ctk.CTkButton(top, text="Open Settings", width=120, command=self._open_settings)
        open_settings_btn.grid(row=0, column=2, sticky="e")

        content = ctk.CTkFrame(self)
        content.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        pending_wrap = ctk.CTkFrame(content)
        pending_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        pending_wrap.grid_rowconfigure(1, weight=1)
        pending_wrap.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(pending_wrap, text="Pending Exits", font=("Arial", 14, "bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 5)
        )

        self.pending_scroll = ctk.CTkScrollableFrame(pending_wrap)
        self.pending_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        history_wrap = ctk.CTkFrame(content)
        history_wrap.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        history_wrap.grid_rowconfigure(1, weight=1)
        history_wrap.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(history_wrap, text="Close History", font=("Arial", 14, "bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 5)
        )

        self.history_scroll = ctk.CTkScrollableFrame(history_wrap)
        self.history_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def _open_settings(self) -> None:
        if callable(self.on_open_settings):
            self.on_open_settings()

    def _on_toggle_enabled(self) -> None:
        enabled = bool(self.enabled_var.get())
        self.settings_manager.set("auto_close.enabled", enabled)
        self.settings_manager.save()
        self._set_status(f"Auto-close {'enabled' if enabled else 'disabled'}")

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _schedule_refresh(self) -> None:
        self._refresh_data()
        self.after(10000, self._schedule_refresh)

    def _schedule_countdown_tick(self) -> None:
        self._refresh_countdowns()
        self.after(1000, self._schedule_countdown_tick)

    def _refresh_data(self) -> None:
        try:
            self.enabled_var.set(bool(self.settings_manager.get("auto_close.enabled", False)))
            self._render_pending()
            self._render_history()
        except Exception as exc:
            log_error(f"[ScheduledExits] Refresh error: {exc}")

    def _get_auto_close_cfg(self) -> Dict[str, Any]:
        return self.settings_manager.get("auto_close", {}) or {}

    def _fetch_open_orders(self) -> List[Dict[str, Any]]:
        ctx = self._get_repo_context()
        if ctx is None:
            return []
        try:
            return ctx.orders.get_open_positions() or []
        except Exception as exc:
            log_warn(f"[ScheduledExits] Could not fetch open orders: {exc}")
            return []

    def _fetch_closed_orders(self) -> List[Dict[str, Any]]:
        ctx = self._get_repo_context()
        if ctx is None:
            return []
        try:
            rows = ctx.orders.get_all_programmatic_orders(status="CLOSED", limit=50, offset=0)
            filtered = [row for row in rows if row.get("auto_close_reason")]
            return filtered[:30]
        except Exception as exc:
            log_warn(f"[ScheduledExits] Could not fetch closed orders: {exc}")
            return []

    def _format_countdown(self, deadline: Optional[datetime]) -> str:
        if deadline is None:
            return "—"
        now = datetime.now(timezone.utc)
        remaining = int((deadline - now).total_seconds())
        if remaining <= 0:
            return "due"
        hours, rem = divmod(remaining, 3600)
        mins, secs = divmod(rem, 60)
        return f"{hours}h {mins:02d}m {secs:02d}s"

    def _render_pending(self) -> None:
        for widget in self.pending_scroll.winfo_children():
            widget.destroy()

        cfg = self._get_auto_close_cfg()
        max_duration_enabled = bool(cfg.get("max_duration_enabled", True))
        max_duration_hours = float(cfg.get("max_duration_hours", 4.0) or 4.0)

        open_orders = self._fetch_open_orders()
        pending_rows: List[Dict[str, Any]] = []

        for order in open_orders:
            if bool(order.get("auto_close_triggered", False)):
                continue

            order_id = get_order_id(order)
            if not order_id:
                continue

            deadline = compute_deadline_utc(
                order=order,
                max_duration_enabled=max_duration_enabled,
                max_duration_hours=max_duration_hours,
            )
            trigger = "timer"
            if order.get("auto_close_deadline_utc") is None and bool(cfg.get("daily_close_enabled", True)):
                trigger = "daily"

            pending_rows.append(
                {
                    "order_id": order_id,
                    "symbol": str(order.get("symbol", "")),
                    "side": str(order.get("side", "")),
                    "pnl": order.get("pnl", "—"),
                    "deadline": deadline,
                    "trigger": trigger,
                }
            )

        self._pending_rows = pending_rows

        if not pending_rows:
            ctk.CTkLabel(self.pending_scroll, text="No pending scheduled exits", text_color="gray").pack(
                anchor="w", padx=4, pady=4
            )
            return

        for row in pending_rows:
            item = ctk.CTkFrame(self.pending_scroll)
            item.pack(fill="x", pady=(0, 6))
            item.grid_columnconfigure(0, weight=1)

            title = (
                f"{row['symbol']} {row['side']} | trigger={row['trigger']} | "
                f"countdown={self._format_countdown(row['deadline'])}"
            )
            label = ctk.CTkLabel(item, text=title, anchor="w")
            label.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
            row["_countdown_label"] = label

            meta = ctk.CTkLabel(
                item,
                text=f"deadline_utc={row['deadline'].isoformat() if row['deadline'] else '—'} | pnl={row['pnl']}",
                text_color="gray",
                anchor="w",
                font=("Arial", 11),
            )
            meta.grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))

            btns = ctk.CTkFrame(item, fg_color="transparent")
            btns.grid(row=0, column=1, rowspan=2, sticky="e", padx=8)
            ctk.CTkButton(
                btns,
                text="Override Deadline",
                width=130,
                command=lambda oid=row["order_id"]: self._override_deadline(oid),
            ).pack(side="top", pady=(0, 4))
            ctk.CTkButton(
                btns,
                text="Cancel Auto-Close",
                width=130,
                fg_color=Colors.BTN_DANGER_ALT,
                hover_color=Colors.BTN_DANGER_ALT_HOVER,
                command=lambda oid=row["order_id"]: self._cancel_auto_close(oid),
            ).pack(side="top")

    def _render_history(self) -> None:
        for widget in self.history_scroll.winfo_children():
            widget.destroy()

        rows = self._fetch_closed_orders()
        if not rows:
            ctk.CTkLabel(self.history_scroll, text="No auto-close history", text_color="gray").pack(
                anchor="w", padx=4, pady=4
            )
            return

        for order in rows:
            symbol = str(order.get("symbol", ""))
            reason = str(order.get("auto_close_reason", ""))
            pnl = order.get("pnl", "—")
            triggered_at = str(order.get("auto_close_triggered_at", "—"))

            item = ctk.CTkFrame(self.history_scroll)
            item.pack(fill="x", pady=(0, 6))

            ctk.CTkLabel(
                item,
                text=f"{symbol} | reason={reason} | pnl={pnl}",
                anchor="w",
            ).pack(anchor="w", padx=8, pady=(8, 2))
            ctk.CTkLabel(
                item,
                text=f"triggered_at={triggered_at}",
                text_color="gray",
                anchor="w",
                font=("Arial", 11),
            ).pack(anchor="w", padx=8, pady=(0, 8))

    def _refresh_countdowns(self) -> None:
        for row in self._pending_rows:
            label = row.get("_countdown_label")
            if label is None:
                continue
            countdown = self._format_countdown(row.get("deadline"))
            text = f"{row['symbol']} {row['side']} | trigger={row['trigger']} | countdown={countdown}"
            label.configure(text=text)

    def _override_deadline(self, order_id: str) -> None:
        value = simpledialog.askstring(
            "Override Deadline",
            "Enter UTC deadline (ISO8601), e.g. 2026-02-28T22:00:00Z",
            parent=self,
        )
        if not value:
            return

        parsed = parse_utc_datetime(value)
        if parsed is None:
            messagebox.showerror("Invalid deadline", "Could not parse datetime. Use ISO8601 UTC format.")
            return

        ctx = self._get_repo_context()
        if ctx is None:
            return

        ok = ctx.orders.update(
            order_id,
            {
                "auto_close_deadline_utc": parsed.isoformat().replace("+00:00", "Z"),
                "auto_close_triggered": False,
            },
        )
        if not ok:
            messagebox.showerror("Update failed", f"Could not update deadline for {order_id}")
            return

        log_info(f"[ScheduledExits] Overrode deadline for {order_id} -> {parsed.isoformat()}")
        self._set_status("Deadline updated")
        self._refresh_data()

    def _cancel_auto_close(self, order_id: str) -> None:
        if not messagebox.askyesno("Cancel Auto-Close", "Cancel auto-close for this order?"):
            return

        ctx = self._get_repo_context()
        if ctx is None:
            return

        # Keep deadline in the future to effectively disable auto close per-order
        far_future = datetime.now(timezone.utc) + timedelta(days=3650)
        ok = ctx.orders.update(
            order_id,
            {
                "auto_close_deadline_utc": far_future.isoformat().replace("+00:00", "Z"),
                "auto_close_triggered": False,
            },
        )
        if not ok:
            messagebox.showerror("Update failed", f"Could not cancel auto-close for {order_id}")
            return

        log_info(f"[ScheduledExits] Cancelled auto-close for {order_id}")
        self._set_status("Auto-close cancelled for selected order")
        self._refresh_data()
