"""
Unit tests for ConfigPanel validation and settings management
"""
import pytest
from unittest.mock import MagicMock, patch
from gui.components.config_panel import ConfigPanel


class TestConfigPanelValidation:
    """Test cases for config panel validation logic"""

    @pytest.fixture
    def mock_parent(self):
        """Create a mock parent widget"""
        parent = MagicMock()
        return parent

    @pytest.fixture
    def config_panel(self, mock_parent):
        """Create a ConfigPanel instance"""
        with patch('customtkinter.CTkFrame.__init__', return_value=None):
            with patch.object(ConfigPanel, '_create_risk_settings_tab'):
                with patch.object(ConfigPanel, '_create_signal_filters_tab'):
                    with patch.object(ConfigPanel, '_create_api_keys_tab'):
                        with patch.object(ConfigPanel, '_create_tp_sl_tab'):
                            with patch.object(ConfigPanel, '_create_ui_preferences_tab'):
                                panel = ConfigPanel.__new__(ConfigPanel)
                                panel.on_settings_change = None
                                return panel

    def test_get_settings_valid_inputs(self, config_panel):
        """Test get_settings with valid inputs"""
        # Mock entry widgets with valid values
        config_panel.max_pos_size_entry = MagicMock()
        config_panel.max_pos_size_entry.get.return_value = "100.00"

        config_panel.max_positions_entry = MagicMock()
        config_panel.max_positions_entry.get.return_value = "3"

        config_panel.max_daily_loss_entry = MagicMock()
        config_panel.max_daily_loss_entry.get.return_value = "50.00"

        config_panel.min_volume_entry = MagicMock()
        config_panel.min_volume_entry.get.return_value = "50"

        config_panel.default_tp_entry = MagicMock()
        config_panel.default_tp_entry.get.return_value = "5.0"

        config_panel.default_sl_entry = MagicMock()
        config_panel.default_sl_entry.get.return_value = "2.5"

        config_panel.default_leverage_var = MagicMock()
        config_panel.default_leverage_var.get.return_value = "10x"

        config_panel.min_score_var = MagicMock()
        config_panel.min_score_var.get.return_value = 0.7

        config_panel.enable_xgboost_var = MagicMock()
        config_panel.enable_xgboost_var.get.return_value = True

        config_panel.whitelist_entry = MagicMock()
        config_panel.whitelist_entry.get.return_value = "BTC/USDT"

        config_panel.exchange_var = MagicMock()
        config_panel.exchange_var.get.return_value = "Binance"

        config_panel.trailing_stop_var = MagicMock()
        config_panel.trailing_stop_var.get.return_value = False

        config_panel.tp_sl_mode_var = MagicMock()
        config_panel.tp_sl_mode_var.get.return_value = "Percentage"

        settings = config_panel.get_settings()

        assert settings["risk"]["max_position_size"] == 100.0
        assert settings["risk"]["max_open_positions"] == 3
        assert settings["risk"]["max_daily_loss"] == 50.0
        assert settings["risk"]["default_leverage"] == "10x"
        assert settings["filters"]["min_signal_score"] == 0.7
        assert settings["filters"]["enable_xgboost"] is True
        assert settings["tp_sl"]["default_tp"] == 5.0
        assert settings["tp_sl"]["default_sl"] == 2.5

    def test_get_settings_invalid_max_position_size(self, config_panel):
        """Test get_settings with invalid max position size"""
        # Mock entry widgets
        config_panel.max_pos_size_entry = MagicMock()
        config_panel.max_pos_size_entry.get.return_value = "invalid"

        config_panel.max_positions_entry = MagicMock()
        config_panel.max_positions_entry.get.return_value = "3"

        config_panel.max_daily_loss_entry = MagicMock()
        config_panel.max_daily_loss_entry.get.return_value = "50"

        config_panel.min_volume_entry = MagicMock()
        config_panel.min_volume_entry.get.return_value = "50"

        config_panel.default_tp_entry = MagicMock()
        config_panel.default_tp_entry.get.return_value = "5.0"

        config_panel.default_sl_entry = MagicMock()
        config_panel.default_sl_entry.get.return_value = "2.5"

        # Mock other required attributes
        config_panel.default_leverage_var = MagicMock()
        config_panel.default_leverage_var.get.return_value = "10x"
        config_panel.min_score_var = MagicMock()
        config_panel.min_score_var.get.return_value = 0.7
        config_panel.enable_xgboost_var = MagicMock()
        config_panel.enable_xgboost_var.get.return_value = True
        config_panel.whitelist_entry = MagicMock()
        config_panel.whitelist_entry.get.return_value = ""
        config_panel.exchange_var = MagicMock()
        config_panel.exchange_var.get.return_value = "Binance"
        config_panel.trailing_stop_var = MagicMock()
        config_panel.trailing_stop_var.get.return_value = False
        config_panel.tp_sl_mode_var = MagicMock()
        config_panel.tp_sl_mode_var.get.return_value = "Percentage"

        settings = config_panel.get_settings()

        # Should return default value
        assert settings["risk"]["max_position_size"] == 100.0

    def test_get_settings_negative_max_position_size(self, config_panel):
        """Test get_settings with negative max position size"""
        config_panel.max_pos_size_entry = MagicMock()
        config_panel.max_pos_size_entry.get.return_value = "-100"

        # Mock other required attributes
        config_panel.max_positions_entry = MagicMock()
        config_panel.max_positions_entry.get.return_value = "3"
        config_panel.max_daily_loss_entry = MagicMock()
        config_panel.max_daily_loss_entry.get.return_value = "50"
        config_panel.min_volume_entry = MagicMock()
        config_panel.min_volume_entry.get.return_value = "50"
        config_panel.default_tp_entry = MagicMock()
        config_panel.default_tp_entry.get.return_value = "5.0"
        config_panel.default_sl_entry = MagicMock()
        config_panel.default_sl_entry.get.return_value = "2.5"
        config_panel.default_leverage_var = MagicMock()
        config_panel.default_leverage_var.get.return_value = "10x"
        config_panel.min_score_var = MagicMock()
        config_panel.min_score_var.get.return_value = 0.7
        config_panel.enable_xgboost_var = MagicMock()
        config_panel.enable_xgboost_var.get.return_value = True
        config_panel.whitelist_entry = MagicMock()
        config_panel.whitelist_entry.get.return_value = ""
        config_panel.exchange_var = MagicMock()
        config_panel.exchange_var.get.return_value = "Binance"
        config_panel.trailing_stop_var = MagicMock()
        config_panel.trailing_stop_var.get.return_value = False
        config_panel.tp_sl_mode_var = MagicMock()
        config_panel.tp_sl_mode_var.get.return_value = "Percentage"

        settings = config_panel.get_settings()

        # Should return default value for negative input
        assert settings["risk"]["max_position_size"] == 100.0

    def test_get_settings_tp_out_of_range(self, config_panel):
        """Test get_settings with TP percentage out of valid range"""
        # Mock entry widgets
        config_panel.default_tp_entry = MagicMock()
        config_panel.default_tp_entry.get.return_value = "150"  # > 100

        # Mock other required attributes
        config_panel.max_pos_size_entry = MagicMock()
        config_panel.max_pos_size_entry.get.return_value = "100"
        config_panel.max_positions_entry = MagicMock()
        config_panel.max_positions_entry.get.return_value = "3"
        config_panel.max_daily_loss_entry = MagicMock()
        config_panel.max_daily_loss_entry.get.return_value = "50"
        config_panel.min_volume_entry = MagicMock()
        config_panel.min_volume_entry.get.return_value = "50"
        config_panel.default_sl_entry = MagicMock()
        config_panel.default_sl_entry.get.return_value = "2.5"
        config_panel.default_leverage_var = MagicMock()
        config_panel.default_leverage_var.get.return_value = "10x"
        config_panel.min_score_var = MagicMock()
        config_panel.min_score_var.get.return_value = 0.7
        config_panel.enable_xgboost_var = MagicMock()
        config_panel.enable_xgboost_var.get.return_value = True
        config_panel.whitelist_entry = MagicMock()
        config_panel.whitelist_entry.get.return_value = ""
        config_panel.exchange_var = MagicMock()
        config_panel.exchange_var.get.return_value = "Binance"
        config_panel.trailing_stop_var = MagicMock()
        config_panel.trailing_stop_var.get.return_value = False
        config_panel.tp_sl_mode_var = MagicMock()
        config_panel.tp_sl_mode_var.get.return_value = "Percentage"

        settings = config_panel.get_settings()

        # Should return default value for out-of-range TP
        assert settings["tp_sl"]["default_tp"] == 5.0

    def test_get_settings_no_api_credentials(self, config_panel):
        """Test that API credentials are NOT included in settings"""
        # Mock all required attributes
        config_panel.max_pos_size_entry = MagicMock()
        config_panel.max_pos_size_entry.get.return_value = "100"
        config_panel.max_positions_entry = MagicMock()
        config_panel.max_positions_entry.get.return_value = "3"
        config_panel.max_daily_loss_entry = MagicMock()
        config_panel.max_daily_loss_entry.get.return_value = "50"
        config_panel.min_volume_entry = MagicMock()
        config_panel.min_volume_entry.get.return_value = "50"
        config_panel.default_tp_entry = MagicMock()
        config_panel.default_tp_entry.get.return_value = "5.0"
        config_panel.default_sl_entry = MagicMock()
        config_panel.default_sl_entry.get.return_value = "2.5"
        config_panel.default_leverage_var = MagicMock()
        config_panel.default_leverage_var.get.return_value = "10x"
        config_panel.min_score_var = MagicMock()
        config_panel.min_score_var.get.return_value = 0.7
        config_panel.enable_xgboost_var = MagicMock()
        config_panel.enable_xgboost_var.get.return_value = True
        config_panel.whitelist_entry = MagicMock()
        config_panel.whitelist_entry.get.return_value = ""
        config_panel.exchange_var = MagicMock()
        config_panel.exchange_var.get.return_value = "Binance"
        config_panel.trailing_stop_var = MagicMock()
        config_panel.trailing_stop_var.get.return_value = False
        config_panel.tp_sl_mode_var = MagicMock()
        config_panel.tp_sl_mode_var.get.return_value = "Percentage"

        settings = config_panel.get_settings()

        # API section should exist but NOT contain credentials
        assert "api" in settings
        assert "api_key" not in settings["api"]
        assert "api_secret" not in settings["api"]
        assert "exchange" in settings["api"]

    def test_get_settings_exception_handling(self, config_panel):
        """Test that get_settings handles exceptions and returns defaults"""
        # Mock entry that raises exception
        config_panel.max_pos_size_entry = MagicMock()
        config_panel.max_pos_size_entry.get.side_effect = Exception("Widget error")

        settings = config_panel.get_settings()

        # Should return safe defaults
        assert settings["risk"]["max_position_size"] == 100.0
        assert settings["risk"]["max_open_positions"] == 3
        assert settings["risk"]["max_daily_loss"] == 50.0
