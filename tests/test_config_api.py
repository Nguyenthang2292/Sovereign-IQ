"""
Tests for co11111111111111111111111nfig/config_api.py.

Tests cover:
- API key reading from environment variables
- Windows Registry reading (on Windows)
- Fallback behavior when keys are not found
- Security considerations
"""

import pytest
import os
from unittest.mock import patch, Mock, MagicMock

# Import the module to test
import config.config_api as config_api


class TestAPIKeyReading:
    """Test API key reading from environment variables."""
    
    def test_binance_api_key_from_env(self):
        """Test BINANCE_API_KEY is read from environment."""
        test_key = "test_binance_key_123"
        with patch.dict(os.environ, {'BINANCE_API_KEY': test_key}, clear=False):
            # Load API keys to pick up new env var
            result = config_api.load_api_keys()
            # Verify that the function returned the correct key
            assert result['BINANCE_API_KEY'] == test_key
            # Verify that the module-level attribute was set correctly
            assert config_api.BINANCE_API_KEY == test_key
    
    def test_binance_api_secret_from_env(self):
        """Test BINANCE_API_SECRET is read from environment."""
        test_secret = "test_binance_secret_456"
        with patch.dict(os.environ, {'BINANCE_API_SECRET': test_secret}, clear=False):
            result = config_api.load_api_keys()
            assert result['BINANCE_API_SECRET'] == test_secret
            assert config_api.BINANCE_API_SECRET == test_secret
    
    def test_gemini_api_key_from_env(self):
        """Test GEMINI_API_KEY is read from environment."""
        test_key = "test_gemini_key_789"
        with patch.dict(os.environ, {'GEMINI_API_KEY': test_key}, clear=False):
            result = config_api.load_api_keys()
            assert result['GEMINI_API_KEY'] == test_key
            assert config_api.GEMINI_API_KEY == test_key
    
    def test_api_keys_none_when_not_in_env(self):
        """Test API keys are None when not in environment."""
        # Remove keys from environment temporarily
        with patch.dict(os.environ, {}, clear=True):
            with patch('config.config_api._read_from_registry', return_value=None):
                result = config_api.load_api_keys()
                # Verify module handles missing env vars appropriately
                # (either None or fallback to registry)
                assert result['BINANCE_API_KEY'] is None
                assert result['BINANCE_API_SECRET'] is None
                assert result['GEMINI_API_KEY'] is None
                assert config_api.BINANCE_API_KEY is None
                assert config_api.BINANCE_API_SECRET is None
                assert config_api.GEMINI_API_KEY is None
class TestReadFromRegistry:
    """Test _read_from_registry function."""
    
    @pytest.mark.skipif(os.name != 'nt', reason="Windows Registry only available on Windows")
    def test_read_from_registry_user_env(self):
        """Test reading from HKEY_CURRENT_USER\\Environment."""
        # On Windows, test with a non-existent key to verify function works
        # Function should return None when key doesn't exist
        result = config_api._read_from_registry("__TEST_NONEXISTENT_KEY_FOR_TESTING__")
        
        # Should return None when key doesn't exist, or a string if it does
        # Just verify function executes without error
        assert result is None or isinstance(result, str)
    
    def test_read_from_registry_behaviour(self):
        """Test _read_from_registry returns None on non-Windows systems and handles exceptions gracefully."""
        # Function should handle ImportError, OSError, FileNotFoundError gracefully
        # We can't easily mock the internal import, but we can verify the function
        # returns None on non-Windows systems and doesn't crash on Windows
        if os.name != 'nt':
            result = config_api._read_from_registry("TEST_KEY")
            assert result is None
        else:
            # On Windows, just verify function is callable and doesn't crash
            result = config_api._read_from_registry("NONEXISTENT_KEY_FOR_TEST")
            # Should return None when key doesn't exist, or a string if it does
            assert result is None or isinstance(result, str)


class TestAPIConfigSecurity:
    """Test security aspects of API configuration."""
    
    def test_no_hardcoded_keys(self):
        """Test that API keys are not hardcoded in the module."""
        import inspect
        
        # Read the source code
        source = inspect.getsource(config_api)
        
        # Check that no obvious API keys are hardcoded
        # This is a basic check - real keys would be longer and more random
        suspicious_patterns = [
            "sk_live_",
            "pk_live_",
            "AIza",
            "AKIA",  # AWS key prefix
        ]
        
        for pattern in suspicious_patterns:
            assert pattern not in source, f"Potential hardcoded key pattern found: {pattern}"
    
    def test_api_keys_use_env_vars(self):
        """Test that API keys are read from environment variables."""
        test_binance_key = "test_binance_env_key_123"
        test_gemini_key = "test_gemini_env_key_456"
        
        with patch.dict(os.environ, {
            'BINANCE_API_KEY': test_binance_key,
            'GEMINI_API_KEY': test_gemini_key
        }, clear=False):
            result = config_api.load_api_keys()
            
            # Assert that function returns the env values
            assert result['BINANCE_API_KEY'] == test_binance_key
            assert result['GEMINI_API_KEY'] == test_gemini_key
            # Assert that module-level variables reflect the env values
            assert config_api.BINANCE_API_KEY == test_binance_key
            assert config_api.GEMINI_API_KEY == test_gemini_key


class TestAPIConfigFallback:
    """Test fallback behavior when keys are not available."""
    
    def test_fallback_to_registry_when_env_empty(self):
        """Test that registry is checked when env vars are empty."""
        # Create mock for registry
        mock_registry = Mock(return_value="registry_value")
        
        # Remove API keys from environment and mock registry
        with patch.dict(os.environ, {}, clear=False):
            # Remove the specific API keys
            os.environ.pop('BINANCE_API_KEY', None)
            os.environ.pop('BINANCE_API_SECRET', None)
            os.environ.pop('GEMINI_API_KEY', None)
            
            with patch('config.config_api._read_from_registry', mock_registry):
                # Load API keys to trigger fallback to registry when env vars are empty
                result = config_api.load_api_keys()
                
                # Assert that the function returned the registry value
                assert result['BINANCE_API_KEY'] == "registry_value"
                # Assert that the module-level constant was resolved from registry
                assert config_api.BINANCE_API_KEY == "registry_value"
                
                # Assert that the registry was called with the correct key
                # (it may be called multiple times for different keys)
                assert mock_registry.call_count >= 1
                assert any(call[0][0] == "BINANCE_API_KEY" for call in mock_registry.call_args_list)
    
    def test_returns_none_when_both_empty(self):
        """Test that None is returned when both env and registry are empty."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('config.config_api._read_from_registry', return_value=None):
                result = config_api.load_api_keys()
                # Verify function returns None when both sources empty
                assert result['BINANCE_API_KEY'] is None
                assert result['BINANCE_API_SECRET'] is None
                assert result['GEMINI_API_KEY'] is None
                # Verify module attributes are None when both sources empty
                assert config_api.BINANCE_API_KEY is None
                assert config_api.BINANCE_API_SECRET is None
                assert config_api.GEMINI_API_KEY is None

class TestAPIConfigConstants:
    """Test that API configuration constants exist."""
    
    def test_binance_api_key_exists(self):
        """Test that BINANCE_API_KEY constant exists."""
        assert hasattr(config_api, 'BINANCE_API_KEY')
        # Value could be None if not set, which is acceptable
        assert config_api.BINANCE_API_KEY is None or isinstance(config_api.BINANCE_API_KEY, str)
    
    def test_binance_api_secret_exists(self):
        """Test that BINANCE_API_SECRET constant exists."""
        assert hasattr(config_api, 'BINANCE_API_SECRET')
        assert config_api.BINANCE_API_SECRET is None or isinstance(config_api.BINANCE_API_SECRET, str)
    
    def test_gemini_api_key_exists(self):
        """Test that GEMINI_API_KEY constant exists."""
        assert hasattr(config_api, 'GEMINI_API_KEY')
        assert config_api.GEMINI_API_KEY is None or isinstance(config_api.GEMINI_API_KEY, str)
    
    def test_read_from_registry_function_exists(self):
        """Test that _read_from_registry function exists."""
        assert hasattr(config_api, '_read_from_registry')
        assert callable(config_api._read_from_registry)

