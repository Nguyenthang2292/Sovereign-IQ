import sys
import os
import unittest
import tkinter as tk
from unittest.mock import MagicMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

# Mock customtkinter if not available or if TclError occurs
try:
    import customtkinter as ctk
    ctk.CTk() # Try to init
except (ImportError, tk.TclError, Exception):
    # Mock customtkinter
    ctk = MagicMock()
    ctk.CTkFrame = MagicMock
    ctk.CTkLabel = MagicMock
    ctk.CTkButton = MagicMock
    ctk.CTkEntry = MagicMock
    ctk.CTkCheckBox = MagicMock
    ctk.CTkScrollbar = MagicMock
    ctk.CTk = MagicMock
    # Inject into sys.modules
    sys.modules['customtkinter'] = ctk

from modules.auto_trade.gui.components.signals_frame import SignalsFrame

class TestSignalsFrameIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            if hasattr(ctk, 'CTk') and isinstance(ctk.CTk, type):
                cls.root = ctk.CTk()
            else:
                cls.root = MagicMock()
        except Exception:
            cls.root = MagicMock()

    def setUp(self):
        self.mock_callback = MagicMock()
        self.signals_frame = SignalsFrame(self.root, on_run_scanner_callback=self.mock_callback)

    def tearDown(self):
        if hasattr(self, 'signals_frame'):
            # If it's a real widget, destroy it
            if hasattr(self.signals_frame, 'destroy') and not isinstance(self.signals_frame.destroy, MagicMock):
                self.signals_frame.destroy()

    def test_empty_state_visibility(self):
        # Force empty signals
        self.signals_frame.update_signals([])

        if isinstance(self.root, MagicMock):
            # Verify empty state pack was called
            self.signals_frame.empty_state.pack.assert_called()
            # Verify table frame pack_forget was called
            self.signals_frame.table_frame.pack_forget.assert_called()
        else:
            self.root.update()
            # Check if empty state is packed
            try:
                self.signals_frame.empty_state.pack_info()
                is_empty_packed = True
            except tk.TclError:
                is_empty_packed = False
            self.assertTrue(is_empty_packed, "EmptyState should be visible")

            # Check if table frame is NOT packed
            try:
                self.signals_frame.table_frame.pack_info()
                is_table_packed = True
            except tk.TclError:
                is_table_packed = False
            self.assertFalse(is_table_packed, "Table should be hidden")

    def test_signals_visibility(self):
        signals = [
            {"symbol": "BTC/USDT", "signal": "LONG", "score": 0.85, "time": "12:00"}
        ]
        self.signals_frame.update_signals(signals)

        if isinstance(self.root, MagicMock):
            # Verify empty state pack_forget was called
            self.signals_frame.empty_state.pack_forget.assert_called()
            # Verify table frame pack was called
            self.signals_frame.table_frame.pack.assert_called()
        else:
            self.root.update()
            # Check if empty state is hidden
            try:
                self.signals_frame.empty_state.pack_info()
                is_empty_packed = True
            except tk.TclError:
                is_empty_packed = False
            self.assertFalse(is_empty_packed, "EmptyState should be hidden")

            # Check if table frame is packed
            try:
                self.signals_frame.table_frame.pack_info()
                is_table_packed = True
            except tk.TclError:
                is_table_packed = False
            self.assertTrue(is_table_packed, "Table should be visible")

    def test_callback_wiring(self):
        # Call the callback stored in empty_state
        self.signals_frame.empty_state.action_callback()
        self.mock_callback.assert_called_once()

if __name__ == '__main__':
    unittest.main()
