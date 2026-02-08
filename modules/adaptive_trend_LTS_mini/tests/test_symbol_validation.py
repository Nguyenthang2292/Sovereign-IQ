"""Tests for symbol format validation (security and input sanitization)."""

import pytest

from modules.common.domain.symbol_validation import (
    filter_valid_symbols,
    require_valid_symbol,
    validate_symbol,
)


class TestValidateSymbol:
    """Test validate_symbol() returns True only for safe formats."""

    def test_valid_common_symbols(self):
        assert validate_symbol("BTCUSDT") is True
        assert validate_symbol("ETHUSDT") is True
        assert validate_symbol("1000PEPEUSDT") is True

    def test_valid_min_length(self):
        assert validate_symbol("AB") is True

    def test_valid_max_length(self):
        assert validate_symbol("A" * 30) is True

    def test_reject_too_short(self):
        assert validate_symbol("A") is False
        assert validate_symbol("") is False

    def test_reject_too_long(self):
        assert validate_symbol("A" * 31) is False

    def test_reject_non_alphanumeric(self):
        assert validate_symbol("BTC/USDT") is False
        assert validate_symbol("BTC-USDT") is False
        assert validate_symbol("..") is False
        assert validate_symbol("path/to/symbol") is False
        assert validate_symbol("symbol;rm -rf") is False

    def test_reject_non_string(self):
        assert validate_symbol(123) is False  # type: ignore[arg-type]
        assert validate_symbol(None) is False  # type: ignore[arg-type]

    def test_strip_whitespace(self):
        assert validate_symbol("  BTCUSDT  ") is True


class TestRequireValidSymbol:
    """Test require_valid_symbol() raises for invalid input."""

    def test_accepts_valid(self):
        require_valid_symbol("BTCUSDT")

    def test_raises_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            require_valid_symbol("")
        with pytest.raises(ValueError, match="non-empty"):
            require_valid_symbol("   ")

    def test_raises_invalid_format(self):
        with pytest.raises(ValueError, match="2–30 alphanumeric"):
            require_valid_symbol("BTC/USDT")
        with pytest.raises(ValueError, match="2–30 alphanumeric"):
            require_valid_symbol("a")

    def test_raises_not_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            require_valid_symbol(123)  # type: ignore[arg-type]


class TestFilterValidSymbols:
    """Test filter_valid_symbols() filters list."""

    def test_empty_list(self):
        assert filter_valid_symbols([]) == []

    def test_all_valid(self):
        syms = ["BTCUSDT", "ETHUSDT"]
        assert filter_valid_symbols(syms) == syms

    def test_filters_invalid(self):
        syms = ["BTCUSDT", "bad/sym", "OK", "x", "ETHUSDT"]
        assert filter_valid_symbols(syms) == ["BTCUSDT", "OK", "ETHUSDT"]
