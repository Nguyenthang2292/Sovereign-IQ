import sys
import os
import pytest
from unittest.mock import Mock, MagicMock
import tkinter as tk

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

# Mock customtkinter if not available
try:
    import customtkinter as ctk
except ImportError:
    ctk = MagicMock()
    ctk.CTk = MagicMock
    ctk.CTkFrame = MagicMock
    ctk.CTkLabel = MagicMock
    ctk.CTkButton = MagicMock
    ctk.CTkScrollbar = MagicMock
    ctk.CTkScrollableFrame = MagicMock
    sys.modules['customtkinter'] = ctk

from modules.auto_trade.gui.components.positions_frame import PositionsFrame
from modules.auto_trade.gui.components.empty_state import EmptyState

@pytest.fixture
def root():
    """Create a hidden CTk root window for testing."""
    try:
        # Try to create a real root if possible
        if hasattr(ctk, 'CTk') and isinstance(ctk.CTk, type):
             # It's the real class
            root = ctk.CTk()
            root.withdraw()
            yield root
            root.destroy()
        else:
            # It's a mock
            yield MagicMock()
    except Exception:
        # Fallback to mock if creation fails (headless)
        yield MagicMock()

def test_positions_frame_empty_state(root):
    """Test that EmptyState is shown when positions list is empty."""
    frame = PositionsFrame(root)

    # Update with empty positions
    frame.update_positions([])

    # Check if EmptyState is present in scroll_frame children
    # If using Mock, we need to inspect calls or children differently
    if isinstance(frame.scroll_frame, MagicMock):
        # Mock behavior verification
        # Assuming PositionsFrame stores the empty state in self._empty_state
        assert hasattr(frame, '_empty_state'), "Expected PositionsFrame to track empty state"
        assert frame._empty_state is not None, "Expected empty state to be created for empty positions"
        # Verify it was packed
        frame._empty_state.pack.assert_called()
    else:
        # Real Tkinter verification
        children = frame.scroll_frame.winfo_children()
        has_empty_state = any(isinstance(child, EmptyState) for child in children)
        assert has_empty_state, "EmptyState should be present when positions are empty"

        # Verify EmptyState properties
        empty_state = next(child for child in children if isinstance(child, EmptyState))
        assert empty_state.message == "No open positions", "Unexpected empty-state message"
        assert empty_state.icon == "📭", "Unexpected empty-state icon"

def test_positions_frame_with_positions(root):
    """Test that EmptyState is NOT shown when positions exist."""
    frame = PositionsFrame(root)

    positions = [
        {
            "symbol": "BTC/USDT",
            "side": "LONG",
            "size": 0.1,
            "entry_price": 50000.0,
            "current_price": 51000.0,
            "pnl": 100.0,
            "id": "123"
        }
    ]

    # Update with positions
    frame.update_positions(positions)

    if isinstance(frame.scroll_frame, MagicMock):
        # Mock verification
        # Ideally _empty_state is None if never called with empty list
        # But implementation sets it to None in __init__
        # If update_positions calls destroy() on it if it exists
        # We can check if it's None or not packed
        assert frame._empty_state is None, "EmptyState should not be set when positions exist"
    else:
        # Check children
        children = frame.scroll_frame.winfo_children()
        # Should be PositionCard instances, not EmptyState
        has_empty_state = any(isinstance(child, EmptyState) for child in children)
        assert not has_empty_state, "EmptyState should not be present when positions exist"

        # Should have 1 child (the position card)
        assert len(children) == 1, "Expected a single position card"

def test_positions_frame_callback_integration(root):
    """Test that on_open_trade_callback is passed to EmptyState."""
    mock_callback = Mock()
    frame = PositionsFrame(root, on_open_trade_callback=mock_callback)

    frame.update_positions([])

    if isinstance(frame.scroll_frame, MagicMock):
        assert frame._empty_state is not None, "Expected empty state when positions list is empty"
        assert frame._empty_state.action_callback == mock_callback, "Expected callback wiring on empty state"
        assert frame._empty_state.action_text == "Open Trade", "Expected Open Trade action text"

        # Simulate click
        frame._empty_state._on_action_button_click()
        mock_callback.assert_called_once()
    else:
        children = frame.scroll_frame.winfo_children()
        empty_state = next(child for child in children if isinstance(child, EmptyState))

        assert empty_state.action_callback == mock_callback, "Expected callback wiring on empty state"
        assert empty_state.action_text == "Open Trade", "Expected Open Trade action text"

        # Simulate click
        empty_state._on_action_button_click()
        mock_callback.assert_called_once()
