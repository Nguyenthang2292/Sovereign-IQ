
import sys
import os
import pytest
from unittest.mock import MagicMock, patch
import customtkinter as ctk
import tkinter as tk

# Add project root to path
# Current file: tests/auto_trade/gui/components/database/test_data_viewer_section.py
# Root: ../../../../../
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../'))
sys.path.insert(0, project_root)

from modules.auto_trade.gui.components.database.data_viewer_section import DataViewerSection
from modules.auto_trade.gui.components.empty_state import EmptyState

@pytest.fixture
def mock_data_viewer_service():
    with patch("modules.auto_trade.gui.components.database.data_viewer_section.DataViewerService") as mock_service:
        mock_service.get_table_count.return_value = 0
        mock_service.get_table_data.return_value = []
        yield mock_service

@pytest.fixture
def root():
    root = ctk.CTk()
    root.withdraw()
    yield root
    root.destroy()

def test_data_viewer_empty_state(root, mock_data_viewer_service):
    """Test that EmptyState is shown when no data is returned."""
    # Setup mocks to return empty list
    mock_data_viewer_service.get_table_count.return_value = 0
    mock_data_viewer_service.get_table_data.return_value = []

    # Initialize DataViewerSection
    log_callback = MagicMock()
    data_viewer = DataViewerSection(root, log_callback)

    # Call refresh manually
    data_viewer.refresh()

    # Verify EmptyState is created
    assert data_viewer._empty_state_widget is not None, "Expected empty state widget when no data"
    assert isinstance(data_viewer._empty_state_widget, EmptyState), "Expected EmptyState instance"
    assert data_viewer._empty_state_widget.message == "No data found in Orders", "Unexpected empty message"

    # Verify we can switch tables and get appropriate empty message
    data_viewer.table_selector.set("Signals")
    # Simulate table change
    data_viewer._on_table_changed("Signals")
    mock_data_viewer_service.get_table_data.return_value = []
    data_viewer.refresh()

    assert data_viewer._empty_state_widget.message == "No data found in Signals", "Unexpected empty message"
    mock_data_viewer_service.get_table_data.assert_called()

def test_data_viewer_with_data(root, mock_data_viewer_service):
    """Test that EmptyState is removed when data is returned."""
    # Setup mocks to return some data
    mock_order = MagicMock()
    mock_order.to_dict.return_value = {"id": 1, "symbol": "BTC/USDT", "status": "closed"}
    # Mock __dict__ for safety as logic checks it
    mock_order.__dict__ = {"id": 1, "symbol": "BTC/USDT", "status": "closed"}

    mock_data_viewer_service.get_table_data.return_value = [mock_order]
    mock_data_viewer_service.get_table_count.return_value = 1

    log_callback = MagicMock()
    data_viewer = DataViewerSection(root, log_callback)

    # First force empty state
    mock_data_viewer_service.get_table_data.return_value = []
    data_viewer.refresh()
    assert data_viewer._empty_state_widget is not None, "Expected empty state before data load"

    # Now provide data
    mock_data_viewer_service.get_table_data.return_value = [mock_order]
    data_viewer.refresh()

    # Verify EmptyState is destroyed/None
    assert data_viewer._empty_state_widget is None, "Expected empty state to be cleared when data exists"

    # Verify data_viewer has content
    content = data_viewer.data_viewer.get("1.0", "end")
    assert "BTC/USDT" in content, "Expected symbol to appear in data viewer output"
    assert "Table: Orders" in content, "Expected table header in data viewer output"
