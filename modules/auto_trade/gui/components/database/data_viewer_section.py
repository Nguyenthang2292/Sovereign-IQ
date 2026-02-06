"""Data Viewer Section Component for Database Panel."""

import logging
from typing import Any, Callable, List, Optional

import customtkinter as ctk

from modules.auto_trade.database import (
    get_audit_log_cursor,
    get_martingale_chains_cursor,
    get_orders_cursor,
    get_signals_cursor,
    session_scope,
)
from modules.auto_trade.database.models import AuditLog, MartingaleChain, Order, Signal

logger = logging.getLogger(__name__)


class DataViewerSection:
    """Data viewer section component with cursor-based pagination."""

    def __init__(self, parent: ctk.CTkFrame, log_callback: Callable):
        self.parent = parent
        self.log_callback = log_callback
        self.current_page = 1
        self.total_pages = 1
        self.page_size = 20
        self.current_table = "Orders"
        # Cursor stack for cursor-based pagination: page_cursors[k] = last_id for page k+1
        self._page_cursors: List[Optional[int]] = [None]
        self._last_page_count = 0  # number of items on last fetch (to detect end)
        self._create_ui()

    def _create_ui(self):
        """Create the data viewer section UI."""
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        header_frame = ctk.CTkFrame(frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(header_frame, text="📂 Data Viewer", font=("Roboto", 14, "bold")).pack(side="left")

        self.table_selector = ctk.CTkOptionMenu(
            header_frame, values=["Orders", "Signals", "Martingale Chains", "Audit Log"], command=self._on_table_changed
        )
        self.table_selector.pack(side="right")

        self.data_viewer = ctk.CTkTextbox(frame, height=200, font=("Consolas", 12))
        self.data_viewer.pack(fill="both", expand=True, padx=10, pady=5)

        pagination_frame = ctk.CTkFrame(frame, fg_color="transparent")
        pagination_frame.pack(fill="x", padx=10, pady=5)

        self.prev_btn = ctk.CTkButton(pagination_frame, text="< Prev", width=80, command=self._prev_page)
        self.prev_btn.pack(side="left")

        self.page_label = ctk.CTkLabel(pagination_frame, text=f"Page {self.current_page}/{self.total_pages}")
        self.page_label.pack(side="left", fill="x", expand=True)

        self.next_btn = ctk.CTkButton(pagination_frame, text="Next >", width=80, command=self._next_page)
        self.next_btn.pack(side="right")

    def _on_table_changed(self, value):
        """Handle table selection change."""
        self.current_table = value
        self.current_page = 1
        self._page_cursors = [None]
        self.refresh()

    def _prev_page(self):
        """Go to previous page."""
        if self.current_page > 1:
            self.current_page -= 1
            self.refresh()

    def _next_page(self):
        """Go to next page."""
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.refresh()

    def refresh(self):
        """Refresh the data viewer with current page data (cursor-based pagination)."""
        try:
            table_name = self.table_selector.get()
            last_id: Optional[int] = (
                self._page_cursors[self.current_page - 1]
                if self.current_page <= len(self._page_cursors)
                else None
            )

            with session_scope() as session:
                total_count = self._get_table_count(session, table_name)
                self.total_pages = max(1, (total_count + self.page_size - 1) // self.page_size)
                self.page_label.configure(text=f"Page {self.current_page}/{self.total_pages}")

                data = self._query_table_data_cursor(session, table_name, self.page_size, last_id)
                self._last_page_count = len(data)
                if data and self.current_page == len(self._page_cursors):
                    last_item = data[-1]
                    cursor_id = getattr(last_item, "id", None)
                    if cursor_id is not None:
                        self._page_cursors.append(cursor_id)

                # Format output
                if not data:
                    self.data_viewer.delete("1.0", "end")
                    self.data_viewer.insert("1.0", f"No data found in {table_name}")
                    return

                # Create basic table view
                output = f"Table: {table_name} (Total: {total_count})\n"
                output += "-" * 80 + "\n"

                if data:
                    # Get columns from first item if it's a dict, or attributes if object
                    first = data[0]
                    if hasattr(first, "to_dict"):
                        first_dict = first.to_dict()
                    elif hasattr(first, "__dict__"):
                        first_dict = {k: v for k, v in first.__dict__.items() if not k.startswith("_")}
                    elif isinstance(first, dict):
                        first_dict = first
                    else:
                        first_dict = {"value": str(first)}

                    columns = list(first_dict.keys())
                    # Limit columns for display
                    display_cols = columns[:5]  # Show first 5 columns to fit

                    # Header
                    header = " | ".join([f"{col:<15}" for col in display_cols])
                    output += header + "\n"
                    output += "-" * len(header) + "\n"

                    for item in data:
                        if hasattr(item, "to_dict"):
                            item_dict = item.to_dict()
                        elif hasattr(item, "__dict__"):
                            item_dict = {k: v for k, v in item.__dict__.items() if not k.startswith("_")}
                        elif isinstance(item, dict):
                            item_dict = item
                        else:
                            item_dict = {"value": str(item)}

                        row = " | ".join([f"{str(item_dict.get(col, ''))[:15]:<15}" for col in display_cols])
                        output += row + "\n"

                self.data_viewer.delete("1.0", "end")
                self.data_viewer.insert("1.0", output)

        except Exception as e:
            self.log_callback(f"Failed to refresh data viewer: {e}", "ERROR")

    def _get_table_count(self, session, table_name: str) -> int:
        """Get total count for a table."""
        if table_name == "Orders":
            return session.query(Order).count()
        elif table_name == "Signals":
            return session.query(Signal).count()
        elif table_name == "Martingale Chains":
            return session.query(MartingaleChain).count()
        elif table_name == "Audit Log":
            return session.query(AuditLog).count()
        return 0

    def _query_table_data_cursor(
        self, session: Any, table_name: str, limit: int, last_id: Optional[int]
    ) -> List[Any]:
        """Query data from a table using cursor-based pagination."""
        if table_name == "Orders":
            return get_orders_cursor(session, last_id=last_id, limit=limit)
        if table_name == "Signals":
            return get_signals_cursor(session, last_id=last_id, limit=limit)
        if table_name == "Martingale Chains":
            return get_martingale_chains_cursor(session, last_id=last_id, limit=limit)
        if table_name == "Audit Log":
            return get_audit_log_cursor(session, last_id=last_id, limit=limit)
        return []

    def set_content(self, content: str):
        """Set content in the data viewer."""
        self.data_viewer.delete("1.0", "end")
        self.data_viewer.insert("1.0", content)

    def get_current_table(self) -> str:
        """Get currently selected table name."""
        return self.current_table
