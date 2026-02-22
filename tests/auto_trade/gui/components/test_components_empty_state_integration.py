import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
# Current file: tests/auto_trade/gui/components/test_components_empty_state_integration.py
# Root: ../../../../
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# NOTE: Do NOT patch sys.modules at module level — it permanently poisons
# tkinter/customtkinter for all tests collected in the same process.

# Keys that will be temporarily replaced with mocks during this test class.
_MOCKED_MODULE_KEYS = [
    "customtkinter",
    "tkinter",
    "tkinter.ttk",
    "tkinter.messagebox",
    "tkinter.filedialog",
    "tkinter.simpledialog",
]


class TestComponentsEmptyStateIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Save original modules and inject mocks
        cls._saved_modules = {k: sys.modules.get(k) for k in _MOCKED_MODULE_KEYS}  # type: ignore[attr-defined,misc]
        for k in _MOCKED_MODULE_KEYS:
            sys.modules[k] = MagicMock()

        # Configure customtkinter mock
        import customtkinter as ctk

        # Mock CTkFrame (avoid passing parent as spec so .pack, .pack_forget exist)
        class MockFrame(MagicMock):
            def __init__(self, *args, **kwargs):
                MagicMock.__init__(self)

            def winfo_children(self):
                return []

        ctk.CTkFrame = MockFrame

        # Mock CTkScrollableFrame (no spec so .pack etc. work when parent is Mock)
        class MockScrollableFrame(MagicMock):
            def __init__(self, *args, **kwargs):
                MagicMock.__init__(self)

            def winfo_children(self):
                return []

        ctk.CTkScrollableFrame = MockScrollableFrame

        # Mock CTkTabview
        class MockTabview(MagicMock):
            def __init__(self, *args, **kwargs):
                MagicMock.__init__(self)

            def add(self, name):
                return MagicMock()

            def set(self, name):
                pass

        ctk.CTkTabview = MockTabview

        # Mock CTkCheckBox
        class MockCheckBox(MagicMock):
            def __init__(self, *args, **kwargs):
                MagicMock.__init__(self)

            def select(self):
                pass

            def deselect(self):
                pass

            def get(self):
                return 1

        ctk.CTkCheckBox = MockCheckBox

        # Other widgets
        ctk.CTkLabel = MagicMock()
        ctk.CTkButton = MagicMock()
        ctk.CTkEntry = MagicMock()
        ctk.CTkProgressBar = MagicMock()
        ctk.CTkComboBox = MagicMock()
        ctk.CTkTextbox = MagicMock()
        ctk.CTkOptionMenu = MagicMock()
        ctk.CTkToplevel = MagicMock()

        # Mock tkinter.ttk
        import tkinter.ttk as ttk

        ttk.Treeview = MagicMock()  # type: ignore[misc]
        ttk.Style = MagicMock()  # type: ignore[misc]
        ttk.Label = MagicMock()  # type: ignore[misc]
        ttk.Button = MagicMock()  # type: ignore[misc]

        # Store configured mock ctk for tests that run after real ctk was imported (e.g. in suite)
        cls._mock_ctk = ctk  # type: ignore[attr-defined,misc]

    @classmethod
    def tearDownClass(cls):
        # Restore original modules so subsequent tests get real tkinter/customtkinter
        for k, orig in cls._saved_modules.items():  # type: ignore[attr-defined]
            if orig is not None:
                sys.modules[k] = orig
            else:
                sys.modules.pop(k, None)

    def setUp(self):
        self.parent = MagicMock()

    def test_positions_frame_empty_state(self):

        # Force fresh import so PositionsFrame uses the configured MockFrame base class
        mod_key = "modules.auto_trade.gui.components.positions_frame"
        old_module = sys.modules.pop(mod_key, None)
        try:
            import modules.auto_trade.gui.components.positions_frame as positions_frame_module

            PositionsFrame = positions_frame_module.PositionsFrame
            mock_empty_state = MagicMock()
            original_empty_state = positions_frame_module.EmptyState
            positions_frame_module.EmptyState = mock_empty_state  # type: ignore[misc]

            frame = PositionsFrame(self.parent)

            # Ensure scroll_frame is our mock type
            frame.scroll_frame = MagicMock()  # type: ignore[method-assign,misc]
            frame.scroll_frame.winfo_children = MagicMock(return_value=[])

            # Test update with empty positions
            frame.update_positions([])

            # Verify EmptyState was instantiated and packed
            mock_empty_state.assert_called()
            mock_empty_state.return_value.pack.assert_called()

            # Prepare for non-empty update
            frame.scroll_frame.winfo_children.return_value = [mock_empty_state.return_value]  # type: ignore[misc]

            positions = [
                {
                    "symbol": "BTC/USDT",
                    "side": "LONG",
                    "size": 1.0,
                    "entry_price": 50000,
                    "current_price": 51000,
                    "pnl": 100,
                }
            ]

            mock_card = MagicMock()
            positions_frame_module.PositionCard = mock_card  # type: ignore[misc]
            frame.update_positions(positions)

            # Verify EmptyState was destroyed
            mock_empty_state.return_value.destroy.assert_called()

            # Verify PositionCard created
            mock_card.assert_called()

            # Restore
            positions_frame_module.EmptyState = original_empty_state  # type: ignore[misc]
        finally:
            # Restore module cache
            if old_module is not None:
                sys.modules[mod_key] = old_module
            else:
                sys.modules.pop(mod_key, None)

    def test_signals_frame_empty_state(self):
        # Force fresh import so SignalsFrame uses configured MockFrame base class
        mod_key = "modules.auto_trade.gui.components.signals_frame"
        old_module = sys.modules.pop(mod_key, None)
        try:
            import modules.auto_trade.gui.components.signals_frame as signals_frame_module

            SignalsFrame = signals_frame_module.SignalsFrame

            mock_empty_state = MagicMock()
            original_empty_state = signals_frame_module.EmptyState
            signals_frame_module.EmptyState = mock_empty_state  # type: ignore[misc]

            frame = SignalsFrame(self.parent)

            # Verify EmptyState was instantiated
            mock_empty_state.assert_called()
            empty_state_instance = frame.empty_state

            # Test update with empty signals
            frame.update_signals([])

            # Should be packed
            empty_state_instance.pack.assert_called()

            # Test update with signals
            signals = [{"symbol": "BTC/USDT", "signal": "LONG", "score": 0.9, "time": "12:00"}]
            frame.update_signals(signals)

            # Should be hidden
            empty_state_instance.pack_forget.assert_called()

            # Restore
            signals_frame_module.EmptyState = original_empty_state  # type: ignore[misc]
        finally:
            if old_module is not None:
                sys.modules[mod_key] = old_module
            else:
                sys.modules.pop(mod_key, None)

    def test_recovery_panel_empty_state(self):
        # Force fresh import so RecoveryPanel uses configured MockFrame base class
        mod_key = "modules.auto_trade.gui.components.recovery_panel"
        old_module = sys.modules.pop(mod_key, None)
        try:
            import modules.auto_trade.gui.components.recovery_panel as recovery_panel_module

            RecoveryPanel = recovery_panel_module.RecoveryPanel

            mock_empty_state = MagicMock()
            original_empty_state = recovery_panel_module.EmptyState
            recovery_panel_module.EmptyState = mock_empty_state  # type: ignore[misc]

            panel = RecoveryPanel(self.parent)

            # Verify EmptyState was instantiated
            mock_empty_state.assert_called()
            empty_state_instance = panel.empty_state_widget

            # Should be packed initially
            empty_state_instance.pack.assert_called()

            # Simulate active recovery
            panel.recovery_strategy = MagicMock()  # type: ignore[misc]
            panel.recovery_strategy.get_state.return_value = MagicMock(
                initial_loss=100,
                remaining_loss=100,
                recovery_percentage=0,
                trades_count=0,
                win_streak=0,
                estimated_trades_remaining=10,
                is_complete=False,
            )
            panel.recovery_strategy.calculate_next_position_size.return_value = 100.0
            panel.recovery_strategy.calculate_next_leverage.return_value = 2
            panel.recovery_strategy.should_stop.return_value = False
            panel._update_status_display()

            # Should be hidden
            empty_state_instance.pack_forget.assert_called()

            # Simulate stop recovery
            panel.recovery_strategy = None  # type: ignore[misc]
            panel._update_status_display()

            # Should be shown
            empty_state_instance.pack.assert_called()

            # Restore
            recovery_panel_module.EmptyState = original_empty_state  # type: ignore[misc]
        finally:
            if old_module is not None:
                sys.modules[mod_key] = old_module
            else:
                sys.modules.pop(mod_key, None)

    @patch("modules.auto_trade.gui.components.empty_state.EmptyState")
    def test_data_viewer_empty_state(self, mock_empty_state):
        # Patch ctk in data_viewer_section so we use mock ctk when module was already
        # loaded with real ctk (e.g. when run after test_data_viewer_section in suite).
        with patch("modules.auto_trade.gui.components.database.data_viewer_section.ctk", self._mock_ctk):  # type: ignore[attr-defined]
            from modules.auto_trade.gui.components.database import data_viewer_section
            from modules.auto_trade.gui.components.database.data_viewer_section import DataViewerSection

            section = DataViewerSection(self.parent, log_callback=MagicMock())

        # When run in suite, data_viewer_section may already have real EmptyState; patch it so refresh() uses mock
        with patch.object(data_viewer_section, "EmptyState", mock_empty_state):
            section._get_table_count = MagicMock(return_value=0)  # type: ignore[method-assign]
            section._query_table_data_cursor = MagicMock(return_value=[])  # type: ignore[method-assign]

            section.refresh()

            mock_empty_state.assert_called()
            mock_empty_state.return_value.pack.assert_called()
            section.data_viewer.pack_forget.assert_called()

            section._get_table_count.return_value = 5
            section._query_table_data_cursor.return_value = [MagicMock(id=1, to_dict=lambda: {"id": 1})]
            section._empty_state_widget = mock_empty_state.return_value

            section.refresh()

            mock_empty_state.return_value.destroy.assert_called()


if __name__ == "__main__":
    unittest.main()
