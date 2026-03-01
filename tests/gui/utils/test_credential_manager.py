"""
Unit tests for CredentialManager
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from modules.auto_trade.gui.utils.credential_manager import CredentialManager


class TestCredentialManager:
    """Test cases for CredentialManager"""

    @pytest.fixture
    def temp_env_file(self, tmp_path):
        """Create a temporary .env file"""
        env_file = tmp_path / ".env"
        env_file.touch()
        return env_file

    @pytest.fixture
    def manager(self, temp_env_file, monkeypatch):
        """Create a CredentialManager with temporary env file"""
        with patch.object(CredentialManager, "_find_or_create_env_file", return_value=temp_env_file):
            manager = CredentialManager()
            return manager

    def test_save_credentials(self, manager, temp_env_file):
        """Test saving credentials to .env file"""
        # Save credentials
        success = manager.save_credentials("binance", "test_key", "test_secret")

        assert success is True

        # Verify credentials were saved
        env_content = temp_env_file.read_text()
        assert "BINANCE_API_KEY=test_key" in env_content
        assert "BINANCE_API_SECRET=test_secret" in env_content

    def test_load_credentials(self, manager, temp_env_file):
        """Test loading credentials from environment"""
        # Save credentials first
        manager.save_credentials("binance", "test_key", "test_secret")

        # Load credentials
        creds = manager.load_credentials("binance")

        assert creds["api_key"] == "test_key"
        assert creds["api_secret"] == "test_secret"

    def test_load_credentials_not_set(self, manager):
        """Test loading credentials that don't exist"""
        creds = manager.load_credentials("nonexistent")

        assert creds["api_key"] is None
        assert creds["api_secret"] is None

    def test_has_credentials(self, manager):
        """Test checking if credentials exist"""
        # Initially no credentials
        assert manager.has_credentials("binance") is False

        # Save credentials
        manager.save_credentials("binance", "test_key", "test_secret")

        # Now should have credentials
        assert manager.has_credentials("binance") is True

    def test_clear_credentials(self, manager, temp_env_file):
        """Test clearing credentials"""
        # Save credentials
        manager.save_credentials("binance", "test_key", "test_secret")
        assert manager.has_credentials("binance") is True

        # Clear credentials
        success = manager.clear_credentials("binance")
        assert success is True

        # Verify cleared
        assert manager.has_credentials("binance") is False

    @patch("modules.auto_trade.gui.utils.credential_manager.ccxt.binance")
    def test_test_connection_success(self, mock_binance_class, manager):
        """Test successful connection test"""
        # Mock exchange instance
        mock_exchange = MagicMock()
        mock_exchange.fetch_balance.return_value = {"total": {"BTC": 1.5, "USDT": 1000}}
        mock_binance_class.return_value = mock_exchange

        # Test connection
        result = manager.test_connection("binance", "test_key", "test_secret")

        assert result["success"] is True
        assert "Successfully connected" in result["message"]
        assert "balance" in result

    @patch("modules.auto_trade.gui.utils.credential_manager.ccxt.binance")
    def test_test_connection_auth_error(self, mock_binance_class, manager):
        """Test connection with authentication error"""
        import ccxt

        # Mock exchange instance that raises auth error
        mock_exchange = MagicMock()
        mock_exchange.fetch_balance.side_effect = ccxt.AuthenticationError("Invalid credentials")
        mock_binance_class.return_value = mock_exchange

        # Test connection
        result = manager.test_connection("binance", "bad_key", "bad_secret")

        assert result["success"] is False
        assert "Authentication failed" in result["message"]

    @patch("modules.auto_trade.gui.utils.credential_manager.ccxt.binance")
    def test_test_connection_network_error(self, mock_binance_class, manager):
        """Test connection with network error"""
        import ccxt

        # Mock exchange instance that raises network error
        mock_exchange = MagicMock()
        mock_exchange.fetch_balance.side_effect = ccxt.NetworkError("Connection timeout")
        mock_binance_class.return_value = mock_exchange

        # Test connection
        result = manager.test_connection("binance", "test_key", "test_secret")

        assert result["success"] is False
        assert "Network error" in result["message"]

    def test_test_connection_unsupported_exchange(self, manager):
        """Test connection with unsupported exchange"""
        result = manager.test_connection("unsupported_exchange", "key", "secret")

        assert result["success"] is False
        assert "Unsupported exchange" in result["message"]

    def test_exchange_name_case_insensitive(self, manager):
        """Test that exchange names are case-insensitive"""
        # Test with different cases
        manager.save_credentials("BINANCE", "key1", "secret1")
        creds = manager.load_credentials("binance")

        assert creds["api_key"] == "key1"
        assert creds["api_secret"] == "secret1"

    def test_save_credentials_overwrites_existing(self, manager):
        """Test that saving new credentials overwrites existing ones"""
        # Save initial credentials
        manager.save_credentials("binance", "old_key", "old_secret")

        # Save new credentials
        manager.save_credentials("binance", "new_key", "new_secret")

        # Verify new credentials
        creds = manager.load_credentials("binance")
        assert creds["api_key"] == "new_key"
        assert creds["api_secret"] == "new_secret"
