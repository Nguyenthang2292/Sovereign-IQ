"""
Unit tests for Toast notification system
"""
import pytest
from unittest.mock import MagicMock, patch
from gui.utils.toast import ToastNotification, show_toast


class TestToastNotification:
    """Test cases for ToastNotification widget"""

    @pytest.fixture
    def mock_parent(self):
        """Create a mock parent widget"""
        parent = MagicMock()
        parent.winfo_rootx.return_value = 100
        parent.winfo_rooty.return_value = 100
        parent.winfo_width.return_value = 800
        parent.winfo_height.return_value = 600
        return parent

    @patch('customtkinter.CTkToplevel.__init__', return_value=None)
    def test_toast_creation(self, mock_toplevel, mock_parent):
        """Test toast notification creation"""
        with patch.object(ToastNotification, 'overrideredirect'):
            with patch.object(ToastNotification, 'geometry'):
                with patch.object(ToastNotification, 'configure'):
                    with patch.object(ToastNotification, 'attributes'):
                        with patch.object(ToastNotification, 'after'):
                            with patch.object(ToastNotification, 'bind'):
                                with patch('customtkinter.CTkLabel', return_value=MagicMock()):
                                    toast = ToastNotification(mock_parent, "Test message")

                                    # Verify initialization called
                                    mock_toplevel.assert_called_once()

    def test_show_toast_info(self, mock_parent):
        """Test show_toast with info type"""
        with patch('gui.utils.toast.ToastNotification') as mock_toast:
            show_toast(mock_parent, "Info message", type="info")

            mock_toast.assert_called_once_with(
                mock_parent,
                "Info message",
                3000,
                fg_color="#333333"
            )

    def test_show_toast_success(self, mock_parent):
        """Test show_toast with success type"""
        with patch('gui.utils.toast.ToastNotification') as mock_toast:
            show_toast(mock_parent, "Success message", type="success")

            mock_toast.assert_called_once_with(
                mock_parent,
                "Success message",
                3000,
                fg_color="#228822"
            )

    def test_show_toast_error(self, mock_parent):
        """Test show_toast with error type"""
        with patch('gui.utils.toast.ToastNotification') as mock_toast:
            show_toast(mock_parent, "Error message", type="error")

            mock_toast.assert_called_once_with(
                mock_parent,
                "Error message",
                3000,
                fg_color="#aa2222"
            )

    def test_show_toast_warning(self, mock_parent):
        """Test show_toast with warning type"""
        with patch('gui.utils.toast.ToastNotification') as mock_toast:
            show_toast(mock_parent, "Warning message", type="warning")

            mock_toast.assert_called_once_with(
                mock_parent,
                "Warning message",
                3000,
                fg_color="#aa8822"
            )

    def test_show_toast_custom_duration(self, mock_parent):
        """Test show_toast with custom duration"""
        with patch('gui.utils.toast.ToastNotification') as mock_toast:
            show_toast(mock_parent, "Message", type="info", duration=5000)

            mock_toast.assert_called_once_with(
                mock_parent,
                "Message",
                5000,
                fg_color="#333333"
            )

    def test_show_toast_unknown_type(self, mock_parent):
        """Test show_toast with unknown type defaults to info"""
        with patch('gui.utils.toast.ToastNotification') as mock_toast:
            show_toast(mock_parent, "Message", type="unknown")

            # Should default to info color
            mock_toast.assert_called_once_with(
                mock_parent,
                "Message",
                3000,
                fg_color="#333333"
            )
