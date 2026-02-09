import importlib
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))


def _set_module(name, module, originals):
    if name not in originals:
        originals[name] = sys.modules.get(name)
    sys.modules[name] = module


class TestDatabasePanel(unittest.TestCase):
    _original_modules: dict = {}
    DatabasePanel: type = type(None)  # set in setUpClass

    @classmethod
    def setUpClass(cls):
        cls._original_modules = {}

        # Ensure any previously loaded customtkinter is removed so we can mock it
        for name in list(sys.modules):
            if name == "customtkinter" or name.startswith("customtkinter."):
                if name not in cls._original_modules:
                    cls._original_modules[name] = sys.modules.get(name)
                sys.modules.pop(name, None)

        # Ensure database panel modules are reloaded with the mocked customtkinter
        for name in list(sys.modules):
            if name == "modules.auto_trade.gui.components.database_panel" or name.startswith(
                "modules.auto_trade.gui.components.database"
            ):
                if name not in cls._original_modules:
                    cls._original_modules[name] = sys.modules.get(name)
                sys.modules.pop(name, None)

        # Mock customtkinter before importing the panel
        customtk_mock = MagicMock()
        _set_module("customtkinter", customtk_mock, cls._original_modules)

        # Create a mock Tk root that terminates the widget hierarchy
        class MockTkRoot:
            def __init__(self):
                self.tk = self  # Root's tk attribute points to itself
                self.master = None
                self._last_child_ids = None
                self.children = {}
                self._w = '.'
                self.widgetName = 'tk'

            def call(self, *args, **kwargs):
                """Mock Tcl/Tk interpreter call - just return empty string."""
                return ""

            def winfo_pathname(self, *args, **kwargs):
                """Mock winfo_pathname."""
                return "."

        mock_root = MockTkRoot()

        class MockCTkFrame:
            def __init__(self, *args, **kwargs):
                # Add tkinter attributes to avoid AttributeError when used as parent
                self.tk = mock_root
                self._last_child_ids = None
                self.children = {}
                self._w = '.mock'
                self.widgetName = 'frame'
                self.master = mock_root  # Point to root to terminate traversal

            def pack(self, *args, **kwargs):
                pass

            def grid(self, *args, **kwargs):
                pass

            def grid_rowconfigure(self, *args, **kwargs):
                pass

            def grid_columnconfigure(self, *args, **kwargs):
                pass

            def after(self, *args, **kwargs):
                pass

        customtk_mock.CTkFrame = MockCTkFrame
        customtk_mock.CTkScrollableFrame = MockCTkFrame
        customtk_mock.CTkButton = MagicMock

        # Configure Label mock
        label_mock = MagicMock()
        label_mock.return_value.configure = MagicMock()
        customtk_mock.CTkLabel = label_mock

        # Configure Entry mock to support insert
        entry_mock = MagicMock()
        entry_mock.return_value.insert = MagicMock()
        entry_mock.return_value.get = MagicMock(return_value="test")
        customtk_mock.CTkEntry = entry_mock

        # Configure Textbox mock
        textbox_mock = MagicMock()
        textbox_mock.return_value.insert = MagicMock()
        textbox_mock.return_value.delete = MagicMock()
        textbox_mock.return_value.see = MagicMock()
        customtk_mock.CTkTextbox = textbox_mock

        # Configure OptionMenu mock
        option_menu_mock = MagicMock()
        option_menu_mock.return_value.get = MagicMock(return_value="test")
        customtk_mock.CTkOptionMenu = option_menu_mock

        # Mock database module
        db_mock = MagicMock()
        _set_module("modules.auto_trade.database", db_mock, cls._original_modules)
        _set_module("modules.auto_trade.database.models", MagicMock(), cls._original_modules)
        _set_module("modules.auto_trade.database.utils", MagicMock(), cls._original_modules)
        _set_module("modules.auto_trade.database.config", MagicMock(), cls._original_modules)

        # Mock tkinter for messagebox/filedialog
        _set_module("tkinter", MagicMock(), cls._original_modules)
        _set_module("tkinter.messagebox", MagicMock(), cls._original_modules)
        _set_module("tkinter.filedialog", MagicMock(), cls._original_modules)

        # Import with mocked customtkinter (reload in case it was loaded earlier)
        module = importlib.import_module("modules.auto_trade.gui.components.database_panel")
        cls.DatabasePanel = module.DatabasePanel

    @classmethod
    def tearDownClass(cls):
        for name, original in cls._original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    def setUp(self):
        self.parent = MagicMock()
        self.settings = MagicMock()
        # Suppress logging during tests
        with patch("logging.getLogger"):
            self.panel = self.DatabasePanel(self.parent, self.settings)

    @patch("modules.auto_trade.gui.components.database.stats_section.DatabaseService.get_stats")
    @patch("modules.auto_trade.gui.components.database.stats_section.DatabaseService.get_last_backup_time")
    def test_refresh_stats(self, mock_get_last_backup, mock_get_stats):
        # Setup mock returns
        mock_get_stats.return_value = {
            "total_orders": 10,
            "open_positions": 5,
            "total_signals": 20,
            "active_chains": 3,
            "audit_logs": 100,
        }
        mock_get_last_backup.return_value = "2024-01-01 12:00"

        self.panel.data_viewer_section.refresh = MagicMock()
        self.panel._refresh_stats()

        # Verify stats updated (labels configured)
        if "total_orders" in self.panel.stats_section.stats_labels:
            self.assertTrue(self.panel.stats_section.stats_labels["total_orders"].configure.called)
            self.assertTrue(self.panel.stats_section.stats_labels["open_positions"].configure.called)

    def test_pagination(self):
        self.panel.data_viewer_section.refresh = MagicMock()
        self.panel.data_viewer_section.current_page = 1
        self.panel.data_viewer_section.total_pages = 5

        self.panel.data_viewer_section._next_page()
        self.assertEqual(self.panel.data_viewer_section.current_page, 2)

        self.panel.data_viewer_section._prev_page()
        self.assertEqual(self.panel.data_viewer_section.current_page, 1)

    def test_table_changed(self):
        self.panel.data_viewer_section.refresh = MagicMock()
        self.panel.data_viewer_section._on_table_changed("Signals")

        self.assertEqual(self.panel.data_viewer_section.current_table, "Signals")
        self.assertEqual(self.panel.data_viewer_section.current_page, 1)
        self.panel.data_viewer_section.refresh.assert_called_once()

    @patch("modules.auto_trade.gui.components.database.orders_section.create_order")
    @patch("modules.auto_trade.gui.components.database.orders_section.session_scope")
    def test_create_test_order(self, mock_session_scope, mock_create_order):
        # Mock UI inputs
        self.panel.orders_section.order_symbol = MagicMock()
        self.panel.orders_section.order_symbol.get.return_value = "BTCUSDT"
        self.panel.orders_section.order_side = MagicMock()
        self.panel.orders_section.order_side.get.return_value = "LONG"

        self.panel.orders_section.refresh_callback = MagicMock()

        self.panel.orders_section._create_test_order()

        mock_create_order.assert_called_once()
        self.panel.orders_section.refresh_callback.assert_called_once()

    @patch("sqlalchemy.or_")
    @patch("modules.auto_trade.gui.components.database.actions_section.messagebox")
    @patch("modules.auto_trade.gui.components.database.actions_section.get_open_positions")
    @patch("modules.auto_trade.gui.components.database.actions_section.session_scope")
    def test_remove_all_open_orders_confirmed_calls_session_and_get_positions(
        self, mock_session_scope, mock_get_open_positions, mock_messagebox, mock_or_
    ):
        mock_or_.return_value = MagicMock()
        mock_messagebox.askyesno.return_value = True
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session
        mock_session_scope.return_value.__exit__.return_value = None
        mock_order = MagicMock()
        mock_order.order_id = "ORD_001"
        mock_get_open_positions.return_value = [mock_order]
        mock_session.query.return_value.filter.return_value.update.return_value = None
        mock_session.query.return_value.filter.return_value.all.return_value = []

        self.panel.actions_section.log_callback = MagicMock()
        self.panel.actions_section.refresh_callback = MagicMock()

        self.panel.actions_section._remove_all_open_orders()

        mock_messagebox.askyesno.assert_called_once()
        mock_get_open_positions.assert_called_once_with(mock_session)
        self.panel.actions_section.log_callback.assert_any_call("Removed 1 open order(s) from DB", "SUCCESS")
        self.panel.actions_section.refresh_callback.assert_called_once()

    @patch("modules.auto_trade.gui.components.database.actions_section.messagebox")
    @patch("modules.auto_trade.gui.components.database.actions_section.get_open_positions")
    @patch("modules.auto_trade.gui.components.database.actions_section.session_scope")
    def test_remove_all_open_orders_cancelled_does_nothing(
        self, mock_session_scope, mock_get_open_positions, mock_messagebox
    ):
        mock_messagebox.askyesno.return_value = False

        self.panel.actions_section._remove_all_open_orders()

        mock_messagebox.askyesno.assert_called_once()
        mock_get_open_positions.assert_not_called()

    @patch("modules.auto_trade.gui.components.database.actions_section.messagebox")
    @patch("modules.auto_trade.gui.components.database.actions_section.get_open_positions")
    @patch("modules.auto_trade.gui.components.database.actions_section.session_scope")
    def test_remove_all_open_orders_no_orders_shows_info(
        self, mock_session_scope, mock_get_open_positions, mock_messagebox
    ):
        mock_messagebox.askyesno.return_value = True
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session
        mock_session_scope.return_value.__exit__.return_value = None
        mock_get_open_positions.return_value = []

        self.panel.actions_section.log_callback = MagicMock()

        self.panel.actions_section._remove_all_open_orders()

        mock_messagebox.showinfo.assert_called_once()
        call_args = mock_messagebox.showinfo.call_args[0]
        self.assertIn("No open orders", call_args[1])
        mock_session.delete.assert_not_called()

    def test_refresh_data_viewer_on_table_switch(self):
        """Test data viewer refresh when table is changed (table switch triggers refresh)."""
        self.panel.data_viewer_section.refresh = MagicMock()
        self.panel.data_viewer_section._on_table_changed("Orders")

        self.assertEqual(
            self.panel.data_viewer_section.current_table,
            "Orders",
            msg="current_table should be set to Orders after table change",
        )
        self.assertEqual(
            self.panel.data_viewer_section.current_page,
            1,
            msg="current_page should be reset to 1 on table change",
        )
        self.panel.data_viewer_section.refresh.assert_called_once_with()

    @patch("modules.auto_trade.database.utils.DataExporter")
    @patch("modules.auto_trade.gui.components.database.actions_section.session_scope")
    @patch("modules.auto_trade.gui.components.database.actions_section.filedialog")
    def test_export_to_csv_success(self, mock_filedialog, mock_session_scope, mock_data_exporter):
        """Test successful export of current table to CSV."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            export_path = f.name
        try:
            mock_filedialog.asksaveasfilename.return_value = export_path
            mock_session = MagicMock()
            mock_session_scope.return_value.__enter__.return_value = mock_session
            mock_session_scope.return_value.__exit__.return_value = None
            mock_data_exporter.export_to_csv.return_value = True

            self.panel.actions_section.get_current_table = MagicMock(return_value="Orders")
            self.panel.actions_section.log_callback = MagicMock()

            self.panel.actions_section._export_csv()

            self.panel.actions_section.log_callback.assert_called()
            success_calls = [
                c
                for c in self.panel.actions_section.log_callback.call_args_list
                if len(c[0]) >= 2 and c[0][1] == "SUCCESS"
            ]
            self.assertTrue(
                len(success_calls) >= 1,
                msg="Expected log_callback with SUCCESS for export; got %s"
                % self.panel.actions_section.log_callback.call_args_list,
            )
            self.assertIn("Exported", success_calls[0][0][0])
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)


if __name__ == "__main__":
    unittest.main()
