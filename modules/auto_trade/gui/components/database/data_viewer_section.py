"""Data Viewer Section Component for Database Panel."""

import threading
from typing import Any, Callable, Dict, List, Optional

import customtkinter as ctk
from modules.auto_trade.gui.utils.colors import Colors

from modules.auto_trade.gui.components.empty_state import EmptyState
from modules.auto_trade.gui.config.database_panel_config import DatabasePanelConfig
from modules.auto_trade.gui.services.database_service import DataViewerService
from modules.auto_trade.gui.utils.svg_icons import get_icon


class DataViewerSection:
    """Data viewer section component with cursor-based pagination."""

    def __init__(self, parent: ctk.CTkFrame, log_callback: Callable):
        self.parent = parent
        self.log_callback = log_callback
        self.current_page = DatabasePanelConfig.INITIAL_PAGE
        self.total_pages = 1
        self.page_size = DatabasePanelConfig.DEFAULT_PAGE_SIZE
        self.current_table = DatabasePanelConfig.TABLE_ORDERS
        # Cursor stack for cursor-based pagination: page_cursors[k] = last_id for page k+1
        self._page_cursors: List[Optional[int]] = [None]
        self._last_page_count = 0  # number of items on last fetch (to detect end)
        self._empty_state_widget: Optional[EmptyState] = None
        self._refresh_in_flight = False
        self._refresh_requested = False
        self._create_ui()
        # Load initial data after startup settles.
        # Early DB queries can block first window render on slow networks.
        self.parent.after(2500, self.refresh)

    def _create_ui(self):
        """Create the data viewer section UI."""
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        header_frame = ctk.CTkFrame(frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            header_frame,
            text="  Data Viewer",
            font=DatabasePanelConfig.TITLE_FONT,
            image=get_icon("folder_open", size=(20, 20)),
            compound="left",
        ).pack(side="left")

        self.table_selector = ctk.CTkOptionMenu(
            header_frame, values=list(DatabasePanelConfig.AVAILABLE_TABLES), command=self._on_table_changed
        )
        self.table_selector.set(self.current_table)  # Set initial value to match current_table
        self.table_selector.pack(side="right")

        self.content_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.data_viewer = ctk.CTkTextbox(
            self.content_frame,
            height=DatabasePanelConfig.DATA_VIEWER_HEIGHT,
            font=DatabasePanelConfig.TEXTBOX_FONT,
            wrap="none",  # Disable word wrap for table display
            state="normal",  # Ensure textbox is editable/writable
        )
        self.data_viewer.pack(fill="both", expand=True)
        print(f"[DataViewer] Textbox created with font={DatabasePanelConfig.TEXTBOX_FONT}")

        # Test textbox is working
        try:
            test_msg = "📂 Data Viewer initialized. Loading data...\n"
            self.data_viewer.insert("1.0", test_msg)
            print(f"[DataViewer] Initial test insert successful: {len(test_msg)} chars")
        except Exception as e:
            print(f"[DataViewer] ERROR in initial test insert: {e}")

        pagination_frame = ctk.CTkFrame(frame, fg_color="transparent")
        pagination_frame.pack(fill="x", padx=10, pady=5)

        self.prev_btn = ctk.CTkButton(
            pagination_frame,
            text=" Prev",
            width=80,
            command=self._prev_page,
            image=get_icon("chevron_left", size=(16, 16)),
            compound="left",
        )
        self.prev_btn.pack(side="left")

        self.page_label = ctk.CTkLabel(pagination_frame, text=f"Page {self.current_page}/{self.total_pages}")
        self.page_label.pack(side="left", padx=10)

        # Force Reload button for debugging
        self.reload_btn = ctk.CTkButton(
            pagination_frame,
            text=" Force Reload",
            width=120,
            command=self._force_reload,
            fg_color=Colors.BTN_PRIMARY,
            hover_color=Colors.BTN_PRIMARY_HOVER,
            image=get_icon("refresh", size=(16, 16)),
            compound="left",
        )
        self.reload_btn.pack(side="left", padx=5)

        self.next_btn = ctk.CTkButton(
            pagination_frame,
            text="Next ",
            width=80,
            command=self._next_page,
            image=get_icon("chevron_right", size=(16, 16)),
            compound="right",
        )
        self.next_btn.pack(side="right")

    def _on_table_changed(self, value):
        """Handle table selection change."""
        print(f"[DataViewer] Table changed to: {value}")
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

    def _force_reload(self):
        """Force reload with detailed diagnostics."""
        print("\n" + "=" * 80)
        print("[DataViewer] FORCE RELOAD TRIGGERED")
        print("=" * 80)

        # Check textbox state
        print(f"[DataViewer] Textbox widget: {self.data_viewer}")
        print(f"[DataViewer] Textbox is mapped: {self.data_viewer.winfo_ismapped()}")
        print(f"[DataViewer] Textbox is visible: {self.data_viewer.winfo_viewable()}")
        print(
            f"[DataViewer] Textbox width x height: {self.data_viewer.winfo_width()} x {self.data_viewer.winfo_height()}"
        )

        # Check current content
        current_content = self.data_viewer.get("1.0", "end")
        print(f"[DataViewer] Current textbox content length: {len(current_content)} chars")
        print(f"[DataViewer] First 200 chars: {current_content[:200]}")

        # Test insert
        print("[DataViewer] Testing textbox insert...")
        try:
            test_text = "=== FORCE RELOAD TEST ===\nThis is a test message.\n"
            self.data_viewer.delete("1.0", "end")
            self.data_viewer.insert("1.0", test_text)
            print("[DataViewer] Test insert successful")

            # Verify insert
            verify_content = self.data_viewer.get("1.0", "end")
            print(f"[DataViewer] Verify content length: {len(verify_content)} chars")
            print(f"[DataViewer] Verify content: {verify_content[:100]}")
        except Exception as test_err:
            print(f"[DataViewer] ERROR in test insert: {test_err}")
            import traceback

            traceback.print_exc()

        # Now try real refresh
        print("[DataViewer] Calling refresh()...")
        self.refresh()
        print("=" * 80 + "\n")

    def refresh(self) -> None:
        """Refresh data asynchronously to avoid blocking the Tkinter main thread."""
        if self._refresh_in_flight:
            self._refresh_requested = True
            return

        try:
            table_name = self.table_selector.get()
            current_page = self.current_page
            last_id: Optional[int] = (
                self._page_cursors[current_page - 1] if current_page <= len(self._page_cursors) else None
            )
        except Exception as e:
            self.log_callback(f"Failed to schedule data viewer refresh: {e}", "ERROR")
            return

        self._refresh_in_flight = True
        self._set_loading_state(True)

        worker = threading.Thread(
            target=self._refresh_worker,
            args=(table_name, current_page, last_id),
            daemon=True,
            name=f"data-viewer-refresh-{table_name}-{current_page}",
        )
        worker.start()

    def _refresh_worker(self, table_name: str, page: int, last_id: Optional[int]) -> None:
        """Background worker for DB I/O; marshals results back to Tk thread."""
        result: Dict[str, Any] = {
            "table_name": table_name,
            "page": page,
            "last_id": last_id,
            "total_count": 0,
            "data": [],
            "error": None,
        }

        try:
            print(f"[DataViewer] Refreshing table: {table_name}, page: {page}")
            total_count = self._get_table_count(None, table_name)
            data = self._query_table_data_cursor(None, table_name, self.page_size, last_id)
            result["total_count"] = total_count
            result["data"] = data
        except Exception as e:
            result["error"] = e

        self.parent.after(0, lambda: self._apply_refresh_result(result))

    def _apply_refresh_result(self, result: Dict[str, Any]) -> None:
        """Apply worker result on Tk main thread."""
        try:
            table_name = str(result.get("table_name", self.current_table))
            page = int(result.get("page", self.current_page))
            error = result.get("error")

            if error is not None:
                error_msg = f"Failed to refresh data viewer: {error}"
                print(f"[DataViewer] ERROR: {error_msg}")
                self.log_callback(error_msg, "ERROR")
                try:
                    error_display = (
                        f"\n❌ ERROR LOADING DATA ❌\n\n{error_msg}\n\n"
                        "Check terminal for full traceback.\n"
                    )
                    self.data_viewer.delete("1.0", "end")
                    self.data_viewer.insert("1.0", error_display)
                except Exception as display_err:
                    print(f"[DataViewer] Cannot even display error in textbox: {display_err}")
                return

            # If user changed table/page while request was in-flight, skip stale result.
            if table_name != self.current_table or page != self.current_page:
                return

            total_count = int(result.get("total_count", 0))
            data: List[Any] = list(result.get("data", []))
            print(f"[DataViewer] Total count for {table_name}: {total_count}")
            print(f"[DataViewer] Query returned {len(data)} items")

            self.total_pages = max(1, (total_count + self.page_size - 1) // self.page_size)
            self.page_label.configure(text=f"Page {self.current_page}/{self.total_pages}")

            self._last_page_count = len(data)
            if data and self.current_page == len(self._page_cursors):
                last_item = data[-1]
                cursor_id = last_item.get("id") if isinstance(last_item, dict) else getattr(last_item, "id", None)
                print(f"[DataViewer] Extracted cursor_id: {cursor_id} from {type(last_item)}")
                if cursor_id is not None:
                    self._page_cursors.append(cursor_id)

            if not data:
                self.data_viewer.pack_forget()
                if self._empty_state_widget:
                    self._empty_state_widget.destroy()
                self._empty_state_widget = EmptyState(
                    self.content_frame,
                    icon="📂",
                    message=f"No data found in {table_name}",
                    hint="There are no records to display for this table.",
                )
                self._empty_state_widget.pack(fill="both", expand=True)
                return

            if self._empty_state_widget:
                self._empty_state_widget.destroy()
                self._empty_state_widget = None

            if not self.data_viewer.winfo_ismapped():
                self.data_viewer.pack(fill="both", expand=True)

            output = self._format_table_output(table_name, total_count, data)
            print(f"[DataViewer] Updating textbox with {len(output)} chars")
            self.data_viewer.delete("1.0", "end")
            self.data_viewer.insert("1.0", output)
            print("[DataViewer] Textbox updated successfully")

        finally:
            self._refresh_in_flight = False
            self._set_loading_state(False)
            if self._refresh_requested:
                self._refresh_requested = False
                self.refresh()

    def _set_loading_state(self, loading: bool) -> None:
        """Update controls while refresh is in progress."""
        state = "disabled" if loading else "normal"
        try:
            self.prev_btn.configure(state=state)
            self.next_btn.configure(state=state)
            self.reload_btn.configure(state=state)
            self.table_selector.configure(state=state)
        except Exception:
            pass

    def _format_table_output(self, table_name: str, total_count: int, data: List[Any]) -> str:
        """Format table rows for textbox rendering."""
        output = f"Table: {table_name} (Total: {total_count})\n"
        output += "-" * 80 + "\n"

        try:
            first = data[0]
            print(f"[DataViewer] First item type: {type(first)}")
            print(f"[DataViewer] First item keys: {list(first.keys()) if isinstance(first, dict) else dir(first)}")

            if isinstance(first, dict):
                first_dict = first
                print("[DataViewer] Item is already dict")
            elif hasattr(first, "to_dict") and callable(getattr(first, "to_dict", None)):
                first_dict = first.to_dict()
                print("[DataViewer] Using to_dict() method")
            elif hasattr(first, "__dict__"):
                first_dict = {k: v for k, v in first.__dict__.items() if not k.startswith("_")}
                print("[DataViewer] Using __dict__ attribute")
            else:
                first_dict = {"value": str(first)}
                print("[DataViewer] Converting to string dict")

            _DYNAMO_INTERNAL_KEYS = {
                "pk",
                "sk",
                "gsi1pk",
                "gsi1sk",
                "gsi2pk",
                "gsi2sk",
                "gsi3pk",
                "gsi3sk",
                "entity_type",
            }

            _PREFERRED_COLUMNS = {
                "Orders": [
                    "order_id",
                    "symbol",
                    "side",
                    "status",
                    "entry_price",
                    "amount",
                    "pnl",
                    "created_at",
                    "take_profit",
                    "stop_loss",
                    "closed_at",
                    "order_source",
                ],
                "Signals": [
                    "correlation_id",
                    "symbol",
                    "direction",
                    "confidence",
                    "executed",
                    "created_at",
                    "outcome",
                    "outcome_pnl",
                ],
                "Martingale Chains": [
                    "chain_id",
                    "symbol",
                    "status",
                    "step",
                    "total_invested",
                    "created_at",
                ],
                "Audit Log": [
                    "event_type",
                    "order_id",
                    "symbol",
                    "description",
                    "created_at",
                ],
            }

            preferred = _PREFERRED_COLUMNS.get(table_name, [])
            all_cols = list(first_dict.keys())
            display_cols = [c for c in preferred if c in first_dict]
            remaining = [c for c in all_cols if c not in _DYNAMO_INTERNAL_KEYS and c not in display_cols]
            display_cols += remaining
            display_cols = display_cols[:8]

            print(f"[DataViewer] Display columns: {display_cols}")

            col_width = 18
            header = " | ".join([f"{col:<{col_width}}" for col in display_cols])
            output += header + "\n"
            output += "-" * len(header) + "\n"

            for idx, item in enumerate(data):
                try:
                    if hasattr(item, "to_dict"):
                        item_dict = item.to_dict()
                    elif hasattr(item, "__dict__"):
                        item_dict = {k: v for k, v in item.__dict__.items() if not k.startswith("_")}
                    elif isinstance(item, dict):
                        item_dict = item
                    else:
                        item_dict = {"value": str(item)}

                    row = " | ".join([f"{str(item_dict.get(col, ''))[:col_width]:<{col_width}}" for col in display_cols])
                    output += row + "\n"
                except Exception as row_err:
                    print(f"[DataViewer] Error formatting row {idx}: {row_err}")
                    output += f"[Error formatting row {idx}]\n"

            print(f"[DataViewer] Formatted output length: {len(output)} chars")
        except Exception as format_err:
            print(f"[DataViewer] Error formatting data: {format_err}")
            import traceback

            traceback.print_exc()
            output += f"\n[Error formatting data: {format_err}]\n"

        return output

    def _get_table_count(self, session, table_name: str) -> int:
        """Get total count for a table using DataViewerService."""
        return DataViewerService.get_table_count(table_name)

    def _query_table_data_cursor(self, session: Any, table_name: str, limit: int, last_id: Optional[int]) -> List[Any]:
        """Query data from a table using cursor-based pagination via DataViewerService."""
        return DataViewerService.get_table_data(table_name, limit=limit, last_id=last_id)

    def set_content(self, content: str):
        """Set content in the data viewer."""
        self.data_viewer.delete("1.0", "end")
        self.data_viewer.insert("1.0", content)

    def get_current_table(self) -> str:
        """Get currently selected table name."""
        return self.current_table
