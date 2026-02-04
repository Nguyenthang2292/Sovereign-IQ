import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

# Mock customtkinter before importing the panel
sys.modules["customtkinter"] = MagicMock()


class MockCTkFrame:
    def __init__(self, *args, **kwargs):
        pass

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


sys.modules["customtkinter"].CTkFrame = MockCTkFrame
sys.modules["customtkinter"].CTkScrollableFrame = MockCTkFrame
sys.modules["customtkinter"].CTkButton = MagicMock

# Configure Label mock
label_mock = MagicMock()
label_mock.return_value.configure = MagicMock()
sys.modules["customtkinter"].CTkLabel = label_mock

# Configure Entry mock to support insert
entry_mock = MagicMock()
entry_mock.return_value.insert = MagicMock()
entry_mock.return_value.get = MagicMock(return_value="test")
sys.modules["customtkinter"].CTkEntry = entry_mock

# Configure Textbox mock
textbox_mock = MagicMock()
textbox_mock.return_value.insert = MagicMock()
textbox_mock.return_value.delete = MagicMock()
textbox_mock.return_value.see = MagicMock()
sys.modules["customtkinter"].CTkTextbox = textbox_mock

# Configure OptionMenu mock
option_menu_mock = MagicMock()
option_menu_mock.return_value.get = MagicMock(return_value="test")
sys.modules["customtkinter"].CTkOptionMenu = option_menu_mock

# Mock database module
db_mock = MagicMock()
sys.modules["modules.auto_trade.database"] = db_mock
sys.modules["modules.auto_trade.database.models"] = MagicMock()
sys.modules["modules.auto_trade.database.utils"] = MagicMock()
sys.modules["modules.auto_trade.database.config"] = MagicMock()

# Mock tkinter for messagebox/filedialog
sys.modules["tkinter"] = MagicMock()
sys.modules["tkinter.messagebox"] = MagicMock()
sys.modules["tkinter.filedialog"] = MagicMock()

from modules.auto_trade.gui.components.database_panel import DatabasePanel


class TestDatabasePanel(unittest.TestCase):
    def setUp(self):
        self.parent = MagicMock()
        self.settings = MagicMock()
        # Suppress logging during tests
        with patch("logging.getLogger"):
            self.panel = DatabasePanel(self.parent, self.settings)

    @patch("modules.auto_trade.gui.components.database_panel.session_scope")
    def test_refresh_stats(self, mock_session_scope):
        # Setup mock session
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Setup query returns
        # The chain of calls is query().count() or query().filter().count()
        # We need to be generic as multiple queries happen
        mock_session.query.return_value.count.return_value = 10
        mock_session.query.return_value.filter.return_value.count.return_value = 5

        self.panel._refresh_stats()

        # Verify stats updated (labels configured)
        # Note: In the implementation we used: if 'total_orders' in self.stats_labels:
        # Since __init__ calls _create_layout which calls _create_stats_section, labels should exist
        if "total_orders" in self.panel.stats_labels:
            self.assertTrue(self.panel.stats_labels["total_orders"].configure.called)
            self.assertTrue(self.panel.stats_labels["open_positions"].configure.called)

    def test_pagination(self):
        self.panel._refresh_data_viewer = MagicMock()
        self.panel.current_page = 1
        self.panel.total_pages = 5

        self.panel._next_page()
        self.assertEqual(self.panel.current_page, 2)

        self.panel._prev_page()
        self.assertEqual(self.panel.current_page, 1)

    def test_table_changed(self):
        self.panel._refresh_data_viewer = MagicMock()
        self.panel._on_table_changed("Signals")

        self.assertEqual(self.panel.current_table, "Signals")
        self.assertEqual(self.panel.current_page, 1)
        self.panel._refresh_data_viewer.assert_called_once()

    @patch("modules.auto_trade.gui.components.database_panel.create_order")
    @patch("modules.auto_trade.gui.components.database_panel.session_scope")
    def test_create_test_order(self, mock_session_scope, mock_create_order):
        # Mock UI inputs
        self.panel.order_symbol = MagicMock()
        self.panel.order_symbol.get.return_value = "BTCUSDT"
        self.panel.order_side = MagicMock()
        self.panel.order_side.get.return_value = "LONG"

        self.panel._refresh_stats = MagicMock()
        self.panel._refresh_data_viewer = MagicMock()

        self.panel._create_test_order()

        mock_create_order.assert_called_once()
        self.panel._refresh_stats.assert_called_once()


if __name__ == "__main__":
    unittest.main()
