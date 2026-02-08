import unittest
from unittest.mock import MagicMock
import sys
import os
import tkinter as tk

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

# Mock customtkinter
try:
    import customtkinter as ctk
    ctk.CTk()
except (ImportError, tk.TclError, Exception):
    ctk = MagicMock()
    ctk.CTkFrame = MagicMock
    ctk.CTkLabel = MagicMock
    ctk.CTkButton = MagicMock
    ctk.CTkEntry = MagicMock
    ctk.CTkCheckBox = MagicMock
    ctk.CTkScrollbar = MagicMock
    ctk.CTkScrollableFrame = MagicMock
    ctk.CTkTabview = MagicMock
    ctk.CTkProgressBar = MagicMock
    ctk.CTkComboBox = MagicMock
    ctk.CTkToplevel = MagicMock
    ctk.CTkTextbox = MagicMock
    ctk.CTk = MagicMock
    sys.modules['customtkinter'] = ctk

from modules.auto_trade.gui.components.recovery_panel import RecoveryPanel

class TestRecoveryPanelEmptyState(unittest.TestCase):
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
        self.panel = RecoveryPanel(self.root)
        if isinstance(self.root, MagicMock):
            self.panel.tabview = MagicMock()
            # If tabview is mocked, we need to ensure the methods used exist
            self.panel.tabview.set = MagicMock()
        else:
            # If real, patch the set method to verify call
            self.panel.tabview.set = MagicMock(wraps=self.panel.tabview.set)

    def tearDown(self):
        if hasattr(self, 'panel'):
            if hasattr(self.panel, 'destroy') and not isinstance(self.panel.destroy, MagicMock):
                self.panel.destroy()

    def test_empty_state_callback_switches_tab(self):
        """Test that the EmptyState action button switches to the Config tab."""
        # Get the callback
        callback = self.panel.empty_state_widget.action_callback

        # Execute it
        callback()

        # Verify it switched to "Config" tab
        # We mocked or wrapped the tabview.set method
        self.panel.tabview.set.assert_called_with("Config")

if __name__ == '__main__':
    unittest.main()
