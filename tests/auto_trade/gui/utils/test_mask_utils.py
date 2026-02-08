"""
Unit tests for API key and secret masking utilities.
"""
import pytest
from modules.auto_trade.gui.utils.mask_utils import mask_api_key, mask_secret

def test_mask_api_key_empty():
    """Test mask_api_key with empty or None values."""
    assert mask_api_key(None) == "—", "Expected placeholder for None API key"
    assert mask_api_key("") == "—", "Expected placeholder for empty API key"

def test_mask_api_key_short():
    """Test mask_api_key with keys of length <= 8."""
    assert mask_api_key("abc") == "***", "Short API key should be fully masked"
    assert mask_api_key("12345678") == "********", "8-char API key should be fully masked"

def test_mask_api_key_long():
    """Test mask_api_key with keys of length > 8."""
    # Length 9: 4 + 1 + 4
    assert mask_api_key("abcdE1234") == "abcd*1234", "Expected first/last 4 chars visible"
    # Length 13: 4 + 5 + 4
    assert mask_api_key("abcd1234wxyz9") == "abcd*****xyz9", "Expected middle masked for long key"

def test_mask_secret_empty():
    """Test mask_secret with empty or None values."""
    assert mask_secret(None) == "—", "Expected placeholder for None secret"
    assert mask_secret("") == "—", "Expected placeholder for empty secret"

def test_mask_secret_short():
    """Test mask_secret with secrets of length <= 8. Should return exactly 8 asterisks."""
    assert mask_secret("abc") == "********", "Short secret should be masked to 8 asterisks"
    assert mask_secret("12345678") == "********", "8-char secret should be masked to 8 asterisks"

def test_mask_secret_long():
    """Test mask_secret with secrets of length > 8."""
    # Length 9: 4 + 1 + 4
    assert mask_secret("abcdE1234") == "abcd*1234", "Expected first/last 4 chars visible"
    # Length 13: 4 + 5 + 4
    assert mask_secret("abcd1234wxyz9") == "abcd*****xyz9", "Expected middle masked for long secret"
