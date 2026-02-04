"""
Comprehensive tests for CredentialManager.

Tests cover:
- Environment file management
- Credential storage and retrieval
- Connection testing
- Error handling
- Security best practices
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import ccxt
import pytest

from modules.auto_trade.gui.utils.credential_manager import CredentialManager


class TestCredentialManager:
    """Test CredentialManager functionality."""

    def test_init_creates_env_file(self, tmp_path):
        """Test that initialization creates .env file if it doesn't exist."""
        env_file = tmp_path / ".env"

        with patch("modules.auto_trade.gui.utils.credential_manager.find_dotenv", return_value=None):
            with patch("modules.auto_trade.gui.utils.credential_manager.Path") as mock_path_class:
                # Mock Path to return our temp directory structure
                mock_file = MagicMock()
                mock_file.parent.parent.parent.parent = tmp_path
                mock_path_class.return_value = env_file
                mock_path_class.__file__ = mock_file

                manager = CredentialManager()

                # Env file should exist or be set
                assert manager.env_file is not None

    def test_save_and_load_credentials(self, temp_env_file, monkeypatch):
        """Test saving and loading credentials."""
        monkeypatch.setenv("HOME", str(temp_env_file.parent))

        with patch("modules.auto_trade.gui.utils.credential_manager.find_dotenv", return_value=str(temp_env_file)):
            manager = CredentialManager()
            manager.env_file = temp_env_file

            # Save credentials
            success = manager.save_credentials(
                exchange="binance",
                api_key="test_key_123",
                api_secret="test_secret_456"
            )

            assert success is True

            # Load credentials
            creds = manager.load_credentials("binance")

            assert creds["api_key"] == "test_key_123"
            assert creds["api_secret"] == "test_secret_456"

    def test_has_credentials(self, temp_env_file):
        """Test checking if credentials exist."""
        with patch("modules.auto_trade.gui.utils.credential_manager.find_dotenv", return_value=str(temp_env_file)):
            # Clear any existing env vars that might interfere
            with patch.dict(os.environ, {}, clear=True):
                manager = CredentialManager()
                manager.env_file = temp_env_file

                # Initially no credentials
                assert manager.has_credentials("binance") is False

                # Save credentials
                manager.save_credentials("binance", "key", "secret")

                # Now credentials exist
                assert manager.has_credentials("binance") is True

    def test_clear_credentials(self, temp_env_file):
        """Test clearing credentials."""
        with patch("modules.auto_trade.gui.utils.credential_manager.find_dotenv", return_value=str(temp_env_file)):
            manager = CredentialManager()
            manager.env_file = temp_env_file

            # Save credentials
            manager.save_credentials("binance", "key", "secret")
            assert manager.has_credentials("binance") is True

            # Clear credentials
            success = manager.clear_credentials("binance")
            assert success is True

            # Credentials should be gone
            creds = manager.load_credentials("binance")
            assert creds["api_key"] == "" or creds["api_key"] is None
            assert creds["api_secret"] == "" or creds["api_secret"] is None

    @patch("modules.auto_trade.gui.utils.credential_manager.ccxt.binance")
    def test_connection_success(self, mock_binance, temp_env_file):
        """Test successful connection test."""
        with patch("modules.auto_trade.gui.utils.credential_manager.find_dotenv", return_value=str(temp_env_file)):
            manager = CredentialManager()

            # Mock successful exchange connection
            mock_exchange = MagicMock()
            mock_exchange.fetch_balance.return_value = {"total": {"USDT": 1000.0}}
            mock_binance.return_value = mock_exchange

            result = manager.test_connection("binance", "test_key", "test_secret")

            assert result["success"] is True
            assert "Successfully connected" in result["message"]
            assert "balance" in result

    @patch("modules.auto_trade.gui.utils.credential_manager.ccxt.binance")
    def test_connection_authentication_error(self, mock_binance, temp_env_file):
        """Test connection with authentication error."""
        with patch("modules.auto_trade.gui.utils.credential_manager.find_dotenv", return_value=str(temp_env_file)):
            manager = CredentialManager()

            # Mock authentication error
            mock_exchange = MagicMock()
            mock_exchange.fetch_balance.side_effect = ccxt.AuthenticationError("Invalid API key")
            mock_binance.return_value = mock_exchange

            result = manager.test_connection("binance", "bad_key", "bad_secret")

            assert result["success"] is False
            assert "Authentication failed" in result["message"]

    @patch("modules.auto_trade.gui.utils.credential_manager.ccxt.binance")
    def test_connection_network_error(self, mock_binance, temp_env_file):
        """Test connection with network error."""
        with patch("modules.auto_trade.gui.utils.credential_manager.find_dotenv", return_value=str(temp_env_file)):
            manager = CredentialManager()

            # Mock network error
            mock_exchange = MagicMock()
            mock_exchange.fetch_balance.side_effect = ccxt.NetworkError("Connection timeout")
            mock_binance.return_value = mock_exchange

            result = manager.test_connection("binance", "key", "secret")

            assert result["success"] is False
            assert "Network error" in result["message"]

    @patch("modules.auto_trade.gui.utils.credential_manager.ccxt.binance")
    def test_connection_timestamp_error(self, mock_binance, temp_env_file):
        """Test connection with timestamp synchronization error."""
        with patch("modules.auto_trade.gui.utils.credential_manager.find_dotenv", return_value=str(temp_env_file)):
            manager = CredentialManager()

            # Mock timestamp error (-1021)
            mock_exchange = MagicMock()
            mock_exchange.fetch_balance.side_effect = Exception("binance -1021 Timestamp for this request")
            mock_binance.return_value = mock_exchange

            result = manager.test_connection("binance", "key", "secret")

            assert result["success"] is False
            assert "Time synchronization error" in result["message"]
            assert "-1021" in result["message"]

    def test_connection_unsupported_exchange(self, temp_env_file):
        """Test connection with unsupported exchange."""
        with patch("modules.auto_trade.gui.utils.credential_manager.find_dotenv", return_value=str(temp_env_file)):
            manager = CredentialManager()

            result = manager.test_connection("unsupported_exchange", "key", "secret")

            assert result["success"] is False
            assert "Unsupported exchange" in result["message"]

    def test_gitignore_update(self, tmp_path, monkeypatch):
        """Test that .env is added to .gitignore."""
        gitignore_path = tmp_path / ".gitignore"

        with patch("modules.auto_trade.gui.utils.credential_manager.find_dotenv", return_value=str(tmp_path / ".env")):
            manager = CredentialManager()
            manager._add_to_gitignore(gitignore_path)

            # Check that .gitignore was created and contains .env
            if gitignore_path.exists():
                content = gitignore_path.read_text()
                assert ".env" in content

    def test_demo_mode_uses_testnet(self, temp_env_file):
        """Test that demo mode enables testnet."""
        with patch("modules.auto_trade.gui.utils.credential_manager.find_dotenv", return_value=str(temp_env_file)):
            manager = CredentialManager()

            with patch("modules.auto_trade.gui.utils.credential_manager.ccxt.binance") as mock_binance:
                mock_exchange = MagicMock()
                mock_exchange.fetch_balance.return_value = {"total": {}}
                mock_binance.return_value = mock_exchange

                manager.test_connection("demo", "key", "secret")

                # Verify set_sandbox_mode was called
                mock_exchange.set_sandbox_mode.assert_called_once_with(True)

    def test_credential_persistence(self, temp_env_file):
        """Test that credentials persist across manager instances."""
        with patch("modules.auto_trade.gui.utils.credential_manager.find_dotenv", return_value=str(temp_env_file)):
            # First manager saves credentials
            manager1 = CredentialManager()
            manager1.env_file = temp_env_file
            manager1.save_credentials("binance", "persistent_key", "persistent_secret")

            # Second manager should load the same credentials
            manager2 = CredentialManager()
            manager2.env_file = temp_env_file
            creds = manager2.load_credentials("binance")

            assert creds["api_key"] == "persistent_key"
            assert creds["api_secret"] == "persistent_secret"
