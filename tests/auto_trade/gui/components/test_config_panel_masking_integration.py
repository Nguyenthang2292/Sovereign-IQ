"""
Integration tests for API key masking in ConfigPanel.
Verifies the interaction between CredentialManager and the UI states.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock customtkinter BEFORE importing any GUI components
# Create proper mock classes that can be inherited from
class MockCTkFrame:
    def __init__(self, *args, **kwargs):
        pass

mock_ctk = MagicMock()
mock_ctk.CTkFrame = MockCTkFrame
mock_ctk.CTkLabel = MagicMock()
mock_ctk.CTkButton = MagicMock()
mock_ctk.CTkEntry = MagicMock()
mock_ctk.StringVar = MagicMock()
mock_ctk.CTkTabview = MagicMock()
mock_ctk.CTkOptionMenu = MagicMock()
mock_ctk.CTkCheckBox = MagicMock()
mock_ctk.CTkSlider = MagicMock()
mock_ctk.CTkComboBox = MagicMock()


class TestConfigPanelMaskingIntegration:
    @pytest.fixture
    def config_panel(self):
        """Create a mock ConfigPanel with just enough state to test the masking logic."""
        # Save original modules before patching
        _patched_keys = [
            "customtkinter",
            "tkinter",
            "tkinter.ttk",
            "tkinter.messagebox",
            "tkinter.filedialog",
        ]
        _saved = {k: sys.modules.get(k) for k in _patched_keys}

        # Temporarily replace with mocks
        sys.modules["customtkinter"] = mock_ctk
        sys.modules["tkinter"] = MagicMock()
        sys.modules["tkinter.ttk"] = MagicMock()
        sys.modules["tkinter.messagebox"] = MagicMock()
        sys.modules["tkinter.filedialog"] = MagicMock()

        # Force fresh import to get the real ConfigPanel class (now with mocked ctk)
        modules_to_clear = [
            "modules.auto_trade.gui.components.config_panel",
            "modules.auto_trade.gui.components",
            "modules.auto_trade.gui",
        ]
        old_modules = {}
        for mod_key in modules_to_clear:
            old_modules[mod_key] = sys.modules.pop(mod_key, None)
        
        try:
            from modules.auto_trade.gui.components.config_panel import ConfigPanel
            
            # We don't call __init__ to avoid customtkinter issues
            panel = MagicMock(spec=ConfigPanel)

            # Attach real methods we want to test
            panel._refresh_credentials_display = ConfigPanel._refresh_credentials_display.__get__(panel, ConfigPanel)
            panel._on_change_credentials = ConfigPanel._on_change_credentials.__get__(panel, ConfigPanel)
            panel._on_cancel_credentials = ConfigPanel._on_cancel_credentials.__get__(panel, ConfigPanel)

            # Setup mock state
            panel._editing_credentials = False
            panel.exchange_var = MagicMock()

            # Mock widgets
            panel.credentials_entry_frame = MagicMock()
            panel.credentials_masked_frame = MagicMock()
            panel.api_key_masked_label = MagicMock()
            panel.api_secret_masked_label = MagicMock()
            panel.api_key_entry = MagicMock()
            panel.api_secret_entry = MagicMock()
            panel.cancel_credentials_btn = MagicMock()

            yield panel
        finally:
            # Restore module cache
            for mod_key, old_module in old_modules.items():
                if old_module is not None:
                    sys.modules[mod_key] = old_module
                else:
                    sys.modules.pop(mod_key, None)
            # Restore patched stdlib/ctk modules
            for k, orig in _saved.items():
                if orig is not None:
                    sys.modules[k] = orig
                else:
                    sys.modules.pop(k, None)

    @patch('modules.auto_trade.gui.utils.credential_manager.CredentialManager')
    def test_refresh_display_saved_state(self, MockManager, config_panel):
        """Test that saved credentials result in masked display."""
        # Setup mocks
        mock_manager = MockManager.return_value
        mock_manager.has_credentials.return_value = True
        mock_manager.load_credentials.return_value = {
            "api_key": "abcd1234wxyz9",
            "api_secret": "secret12345678"
        }

        config_panel.exchange_var.get.return_value = "Binance"
        config_panel._editing_credentials = False

        # Execute
        config_panel._refresh_credentials_display()

        # Verify
        config_panel.credentials_entry_frame.pack_forget.assert_called_once()
        config_panel.api_key_masked_label.configure.assert_called_with(text="abcd*****xyz9")
        config_panel.api_secret_masked_label.configure.assert_called_with(text="secr******5678")
        config_panel.credentials_masked_frame.pack.assert_called_with(fill="x")
        config_panel.exchange_var.get.assert_called_once()
        mock_manager.has_credentials.assert_called_once_with("binance")
        mock_manager.load_credentials.assert_called_once_with("binance")

    @patch('modules.auto_trade.gui.utils.credential_manager.CredentialManager')
    def test_refresh_display_not_saved_state(self, MockManager, config_panel):
        """Test that no credentials result in entry display."""
        # Setup mocks
        mock_manager = MockManager.return_value
        mock_manager.has_credentials.return_value = False

        config_panel.exchange_var.get.return_value = "Binance"
        config_panel._editing_credentials = False

        # Execute
        config_panel._refresh_credentials_display()

        # Verify
        config_panel.credentials_masked_frame.pack_forget.assert_called_once()
        config_panel.api_key_entry.delete.assert_called_with(0, "end")
        config_panel.api_secret_entry.delete.assert_called_with(0, "end")
        config_panel.cancel_credentials_btn.pack_forget.assert_called_once()
        config_panel.credentials_entry_frame.pack.assert_called_with(fill="x")
        config_panel.exchange_var.get.assert_called_once()
        mock_manager.has_credentials.assert_called_once_with("binance")

    def test_on_change_credentials(self, config_panel):
        """Test switching to editing mode."""
        # Ensure we don't call the real refresh display during this test
        with patch.object(config_panel, '_refresh_credentials_display'):
            config_panel._editing_credentials = False
            config_panel._on_change_credentials()

            assert config_panel._editing_credentials is True, "Expected editing mode to be enabled"
            config_panel._refresh_credentials_display.assert_called_once()

    def test_on_cancel_credentials(self, config_panel):
        """Test canceling editing mode."""
        # Ensure we don't call the real refresh display during this test
        with patch.object(config_panel, '_refresh_credentials_display'):
            config_panel._editing_credentials = True
            config_panel._on_cancel_credentials()

            assert config_panel._editing_credentials is False, "Expected editing mode to be disabled"
            config_panel.api_key_entry.delete.assert_called_with(0, "end")
            config_panel.api_secret_entry.delete.assert_called_with(0, "end")
            config_panel._refresh_credentials_display.assert_called_once()
