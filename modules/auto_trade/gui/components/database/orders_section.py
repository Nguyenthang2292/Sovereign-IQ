"""Orders Section Component for Database Panel."""

import uuid
from datetime import datetime
from typing import Any, Callable

import customtkinter as ctk

from modules.auto_trade.database.repository.context import RepositoryContext
from modules.auto_trade.gui.config.database_panel_config import DatabasePanelConfig
from modules.auto_trade.gui.utils.svg_icons import get_icon


class OrdersSection:
    """Orders testing section component."""

    def __init__(
        self,
        parent: ctk.CTkFrame | ctk.CTkScrollableFrame,
        log_callback: Callable[[str, str], None],
        refresh_callback: Callable[[], None],
    ) -> None:
        self.parent = parent
        self.log_callback = log_callback
        self.refresh_callback = refresh_callback
        self._create_ui()

    def _create_ui(self) -> None:
        """Create the orders section UI."""
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(
            frame,
            text="  Orders Testing",
            font=DatabasePanelConfig.TITLE_FONT,
            image=get_icon("clipboard_list", size=(20, 20)),
            compound="left",
        ).pack(
            anchor="w",
            padx=DatabasePanelConfig.PADX_MEDIUM,
            pady=(DatabasePanelConfig.PADX_MEDIUM, DatabasePanelConfig.PADY_SMALL),
        )

        input_frame = ctk.CTkFrame(frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(input_frame, text="Symbol:").pack(side="left", padx=(0, 5))
        self.order_symbol = ctk.CTkEntry(input_frame, width=100)
        self.order_symbol.pack(side="left", padx=(0, 10))
        self.order_symbol.insert(0, "BTCUSDT")

        ctk.CTkLabel(input_frame, text="Side:").pack(side="left", padx=(0, 5))
        self.order_side = ctk.CTkOptionMenu(input_frame, values=["LONG", "SHORT"], width=100)
        self.order_side.pack(side="left", padx=(0, 10))

        ctk.CTkButton(input_frame, text="Create Test Order", command=self._create_test_order).pack(side="right")

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(
            btn_frame,
            text="  Query Open Positions",
            command=self._query_open_positions,
            image=get_icon("database", size=(16, 16)),
            compound="left",
        ).pack(side="left", padx=(0, 5), fill="x", expand=True)

        ctk.CTkButton(
            btn_frame,
            text="  Get Overall Stats",
            command=self._get_overall_stats,
            image=get_icon("bar_chart_2", size=(16, 16)),
            compound="left",
        ).pack(side="left", padx=5, fill="x", expand=True)

        ctk.CTkButton(
            btn_frame,
            text="  Get Daily Stats (30d)",
            command=self._get_daily_stats,
            image=get_icon("calendar", size=(16, 16)),
            compound="left",
        ).pack(side="left", padx=(5, 0), fill="x", expand=True)

    def _create_test_order(self) -> None:
        """Create a test order via RepositoryContext."""
        symbol = self.order_symbol.get()
        side = self.order_side.get()

        try:
            ctx = RepositoryContext.from_env()
            order_data = {
                "order_id": f"TEST_{uuid.uuid4().hex[:8]}",
                "client_order_id": f"AT_{int(datetime.now().timestamp())}_{symbol}",
                "symbol": symbol,
                "side": side,
                "entry_price": 50000.0,
                "amount": 0.01,
                "leverage": 2,
                "status": "OPEN",
                "order_source": "PROGRAMMATIC",
                "execution_mode": "AUTO",
            }
            ctx.orders.create_order(order_data)
            self.log_callback(f"Created test order for {symbol} ({side})", "SUCCESS")
            self.refresh_callback()
        except Exception as e:
            self.log_callback(f"Failed to create test order: {e}", "ERROR")

    def _query_open_positions(self) -> None:
        """Query open positions via RepositoryContext."""
        try:
            ctx = RepositoryContext.from_env()
            positions = ctx.orders.get_open_positions()

            output = "Open Positions:\n"
            output += "-" * 50 + "\n"
            for pos in positions:
                if isinstance(pos, dict):
                    output += (
                        f"ID: {pos.get('order_id')} | {pos.get('symbol')} | "
                        f"{pos.get('side')} | Entry: {pos.get('entry_price')}\n"
                    )
                else:
                    output += f"ID: {pos.order_id} | {pos.symbol} | {pos.side} | Entry: {pos.entry_price}\n"

            self._show_in_data_viewer(output)
            self.log_callback(f"Queried {len(positions)} open positions", "INFO")

        except Exception as e:
            self.log_callback(f"Failed to query open positions: {e}", "ERROR")

    def _get_overall_stats(self) -> None:
        """Get overall trading statistics via RepositoryContext."""
        try:
            ctx = RepositoryContext.from_env()

            all_orders = ctx.orders.get_all_programmatic_orders(limit=99999)
            open_pos = ctx.orders.get_open_positions()
            signals = ctx.signals.get_recent_signals(limit=99999)

            total = len(all_orders)
            open_count = len(open_pos)
            closed_count = 0
            win_count = 0
            for order in all_orders:
                status = str(order.get("status", "") if isinstance(order, dict) else getattr(order, "status", ""))
                if status in ("CLOSED", "CANCELLED"):
                    closed_count += 1

                raw_pnl: Any
                if isinstance(order, dict):
                    raw_pnl = order.get("realized_pnl")
                else:
                    raw_pnl = getattr(order, "realized_pnl", 0.0)

                try:
                    pnl_value = float(0.0 if raw_pnl is None else raw_pnl)
                except (TypeError, ValueError):
                    pnl_value = 0.0

                if pnl_value > 0:
                    win_count += 1
            win_rate = (win_count / closed_count * 100) if closed_count > 0 else 0.0

            stats = {
                "total_orders": total,
                "open_positions": open_count,
                "closed_orders": closed_count,
                "winning_orders": win_count,
                "win_rate_pct": f"{win_rate:.2f}%",
                "total_signals": len(signals),
            }

            output = "Overall Trading Statistics:\n"
            output += "=" * 30 + "\n"
            for key, value in stats.items():
                output += f"{key.replace('_', ' ').title()}: {value}\n"

            self._show_in_data_viewer(output)
            self.log_callback("Retrieved overall stats", "INFO")

        except Exception as e:
            self.log_callback(f"Failed to get overall stats: {e}", "ERROR")

    def _get_daily_stats(self) -> None:
        """Get daily statistics for last 30 days via RepositoryContext."""
        try:
            ctx = RepositoryContext.from_env()

            # Fetch all orders and group by date
            all_orders = ctx.orders.get_all_programmatic_orders(limit=99999)
            from datetime import timedelta, timezone

            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(days=30)

            daily: dict[str, dict[str, float | int]] = {}
            for o in all_orders:
                if isinstance(o, dict):
                    created_raw = o.get("created_at") or o.get("updated_at") or ""
                    pnl = float(o.get("realized_pnl") or 0)
                else:
                    created_raw = getattr(o, "created_at", None)
                    pnl = float(getattr(o, "realized_pnl", 0) or 0)

                try:
                    if isinstance(created_raw, str):
                        from datetime import datetime as _dt

                        dt = _dt.fromisoformat(created_raw.replace("Z", "+00:00"))
                    else:
                        dt = created_raw
                    if dt and dt >= cutoff:
                        date_key = dt.strftime("%Y-%m-%d")
                        if date_key not in daily:
                            daily[date_key] = {"total_orders": 0, "realized_pnl": 0.0}
                        daily[date_key]["total_orders"] += 1
                        daily[date_key]["realized_pnl"] += pnl
                except Exception:
                    pass

            output = "Daily Statistics (Last 30 Days):\n"
            output += f"{'Date':<12} | {'Orders':<8} | {'PnL':<10}\n"
            output += "-" * 35 + "\n"

            for date_key in sorted(daily.keys(), reverse=True):
                d = daily[date_key]
                output += f"{date_key:<12} | {d['total_orders']:<8} | {d['realized_pnl']:<10.2f}\n"

            self._show_in_data_viewer(output)
            self.log_callback("Retrieved daily stats", "INFO")

        except Exception as e:
            self.log_callback(f"Failed to get daily stats: {e}", "ERROR")

    def _show_in_data_viewer(self, content: str) -> None:
        """Show content in data viewer. To be connected by parent."""
        callback = getattr(self.parent, "data_viewer_callback", None)
        if callable(callback):
            callback(content)
