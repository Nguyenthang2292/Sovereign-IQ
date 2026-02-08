#!/usr/bin/env python
"""Debug script to reproduce the issue with the EmptyState test."""

import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock imports BEFORE importing modules that use them
sys.modules["customtkinter"] = MagicMock()
sys.modules["tkinter"] = MagicMock()
sys.modules["tkinter.ttk"] = MagicMock()
sys.modules["tkinter.messagebox"] = MagicMock()
sys.modules["tkinter.filedialog"] = MagicMock()
sys.modules["tkinter.simpledialog"] = MagicMock()

class MockFrame(MagicMock):
    def __init__(self, *args, **kwargs):
        MagicMock.__init__(self)
    def winfo_children(self):
        return []

def debug_positions_frame_test():
    print("Setting up mocks...")

    # Configure customtkinter mock
    import customtkinter as ctk
    ctk.CTkFrame = MockFrame
    ctk.CTkScrollableFrame = MockFrame
    ctk.CTkLabel = MagicMock()
    ctk.CTkButton = MagicMock()
    ctk.CTkEntry = MagicMock()

    print("Mocks configured.")

    # Now test the actual functionality
    with patch("modules.auto_trade.gui.components.positions_frame.EmptyState") as mock_empty_state:
        print(f"Mock EmptyState created: {mock_empty_state}")

        from modules.auto_trade.gui.components.positions_frame import PositionsFrame

        parent = MagicMock()
        print("Creating PositionsFrame...")
        frame = PositionsFrame(parent)
        print(f"PositionsFrame created: {frame}")

        print("Calling update_positions([])...")
        frame.update_positions([])

        print(f"Mock EmptyState call count: {mock_empty_state.call_count}")
        print(f"Was mock called? {mock_empty_state.called}")

        if mock_empty_state.called:
            print("Mock was called - checking if pack was called on return value...")
            instance = mock_empty_state.return_value
            print(f"Instance: {instance}")
            print(f"Pack called? {instance.pack.called}")
        else:
            print("Mock was NOT called!")

        # This is what the test is checking
        try:
            mock_empty_state.assert_called()
            print("assert_called() passed!")
        except AssertionError as e:
            print(f"assert_called() failed: {e}")

if __name__ == "__main__":
    debug_positions_frame_test()
