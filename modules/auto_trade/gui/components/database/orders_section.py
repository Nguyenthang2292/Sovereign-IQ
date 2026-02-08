"""Orders Section Component for Database Panel."""

import logging
import uuid
from datetime import datetime
from typing import Callable

import customtkinter as ctk

from modules.auto_trade.database import (
    create_order,
    get_daily_stats,
    get_open_positions,
    get_overall_stats,
    session_scope,
)
from modules.auto_trade.gui.config.database_panel_config import DatabasePanelConfig

logger = logging.getLogger(__name__)


class OrdersSection:
    """Orders testing section component."""

    def __init__(self, parent: ctk.CTkFrame, log_callback: Callable, refresh_callback: Callable):
        self.parent = parent
        self.log_callback = log_callback
        self.refresh_callback = refresh_callback
        self._create_ui()

    def _create_ui(self):
        """Create the orders section UI."""
        # Frame
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=5, pady=5)

        # Title
        ctk.CTkLabel(frame, text="📋 Orders Testing", font=DatabasePanelConfig.TITLE_FONT).pack(
            anchor="w",
            padx=DatabasePanelConfig.PADX_MEDIUM,
            pady=(DatabasePanelConfig.PADX_MEDIUM, DatabasePanelConfig.PADY_SMALL),
        )

        # Inputs Frame
        input_frame = ctk.CTkFrame(frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=5)

        # Symbol
        ctk.CTkLabel(input_frame, text="Symbol:").pack(side="left", padx=(0, 5))
        self.order_symbol = ctk.CTkEntry(input_frame, width=100)
        self.order_symbol.pack(side="left", padx=(0, 10))
        self.order_symbol.insert(0, "BTCUSDT")

        # Side
        ctk.CTkLabel(input_frame, text="Side:").pack(side="left", padx=(0, 5))
        self.order_side = ctk.CTkOptionMenu(input_frame, values=["LONG", "SHORT"], width=100)
        self.order_side.pack(side="left", padx=(0, 10))

        # Create Button
        ctk.CTkButton(input_frame, text="Create Test Order", command=self._create_test_order).pack(side="right")

        # Query Buttons Frame
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(btn_frame, text="📊 Query Open Positions", command=self._query_open_positions).pack(
            side="left", padx=(0, 5), fill="x", expand=True
        )
        ctk.CTkButton(btn_frame, text="📈 Get Overall Stats", command=self._get_overall_stats).pack(
            side="left", padx=5, fill="x", expand=True
        )
        ctk.CTkButton(btn_frame, text="📅 Get Daily Stats (30d)", command=self._get_daily_stats).pack(
            side="left", padx=(5, 0), fill="x", expand=True
        )

    def _create_test_order(self):
        """Create a test order."""
        symbol = self.order_symbol.get()
        side = self.order_side.get()

        try:
            with session_scope() as session:
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

                create_order(session, order_data)

                self.log_callback(f"Created test order for {symbol} ({side})", "SUCCESS")
                self.refresh_callback()

        except Exception as e:
            self.log_callback(f"Failed to create test order: {e}", "ERROR")

    def _query_open_positions(self):
        """Query open positions."""
        try:
            with session_scope() as session:
                positions = get_open_positions(session)

                output = "Open Positions:\n"
                output += "-" * 50 + "\n"
                for pos in positions:
                    output += f"ID: {pos.order_id} | {pos.symbol} | {pos.side} | Entry: {pos.entry_price}\n"

                self._show_in_data_viewer(output)
                self.log_callback(f"Queried {len(positions)} open positions", "INFO")

        except Exception as e:
            self.log_callback(f"Failed to query open positions: {e}", "ERROR")

    def _get_overall_stats(self):
        """Get overall trading statistics."""
        try:
            with session_scope() as session:
                stats = get_overall_stats(session)

                output = "Overall Trading Statistics:\n"
                output += "=" * 30 + "\n"
                for key, value in stats.items():
                    output += f"{key.replace('_', ' ').title()}: {value}\n"

                self._show_in_data_viewer(output)
                self.log_callback("Retrieved overall stats", "INFO")

        except Exception as e:
            self.log_callback(f"Failed to get overall stats: {e}", "ERROR")

    def _get_daily_stats(self):
        """Get daily statistics for last 30 days."""
        try:
            with session_scope() as session:
                stats = get_daily_stats(session, days=30)

                output = "Daily Statistics (Last 30 Days):\n"
                output += f"{'Date':<12} | {'Orders':<8} | {'PnL':<10}\n"
                output += "-" * 35 + "\n"

                for day in stats:
                    date_str = day.get("date", "N/A")
                    orders = day.get("total_orders", 0)
                    pnl = day.get("realized_pnl", 0.0)
                    output += f"{str(date_str):<12} | {orders:<8} | {pnl:<10.2f}\n"

                self._show_in_data_viewer(output)
                self.log_callback("Retrieved daily stats", "INFO")

        except Exception as e:
            self.log_callback(f"Failed to get daily stats: {e}", "ERROR")

    def _show_in_data_viewer(self, content: str):
        """Show content in data viewer. To be connected by parent."""
        # This will be overridden by the parent to show in the data viewer
        if hasattr(self.parent, "data_viewer_callback"):
            self.parent.data_viewer_callback(content)
