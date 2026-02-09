"""
Integration tests for DataViewerSection empty state behavior.

Tests refresh() logic with mocked DataViewerService. Uses real Tk only when
DISPLAY is available; otherwise skips to avoid CI hang.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Skip entire module if running in headless CI (no display)
_has_display = os.environ.get("DISPLAY") or sys.platform == "win32"
pytestmark = pytest.mark.skipif(not _has_display, reason="No display available")


@pytest.fixture(scope="module")
def _tk_root():
    """Single CTk root for all tests (customtkinter needs its own root)."""
    import customtkinter as ctk

    root = ctk.CTk()
    root.withdraw()
    yield root
    try:
        root.quit()
        root.destroy()
    except Exception:
        pass


@pytest.fixture
def viewer(_tk_root):
    """Create DataViewerSection with mocked DataViewerService."""
    import customtkinter as ctk

    parent = ctk.CTkFrame(_tk_root)
    log_callback = MagicMock()

    with patch("modules.auto_trade.gui.components.database.data_viewer_section.DataViewerService") as mock_svc:
        mock_svc.get_table_count.return_value = 0
        mock_svc.get_table_data.return_value = []

        from modules.auto_trade.gui.components.database.data_viewer_section import DataViewerSection

        v = DataViewerSection(parent, log_callback)
        v.mock_service = mock_svc  # type: ignore[attr-defined]
        yield v

    try:
        parent.destroy()
    except Exception:
        pass


@pytest.mark.timeout(10)
def test_empty_state_shown_when_no_data(viewer):
    """Empty state is shown when no data is available."""
    from modules.auto_trade.gui.components.empty_state import EmptyState

    viewer.mock_service.get_table_count.return_value = 0
    viewer.mock_service.get_table_data.return_value = []

    viewer.refresh()

    assert viewer._empty_state_widget is not None
    assert isinstance(viewer._empty_state_widget, EmptyState)
    assert viewer._empty_state_widget.message == "No data found in Orders"


@pytest.mark.timeout(10)
def test_empty_state_hidden_when_data_exists(viewer):
    """Empty state is removed when data exists."""
    viewer.mock_service.get_table_count.return_value = 10
    mock_order = MagicMock()
    mock_order.to_dict.return_value = {"id": 1, "symbol": "BTC/USDT"}
    mock_order.id = 1
    viewer.mock_service.get_table_data.return_value = [mock_order]

    viewer._empty_state_widget = MagicMock()

    viewer.refresh()

    assert viewer._empty_state_widget is None
