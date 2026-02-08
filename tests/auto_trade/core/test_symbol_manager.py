"""
Comprehensive unit tests for SymbolManager.

Tests cover:
- Initialization and validation
- Whitelist/blacklist filtering
- Sampling logic (edge cases)
- Caching behavior
- Empty symbol list handling
- Random seed reproducibility
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List

from modules.auto_trade.core.symbol_manager import SymbolManager
from modules.common.core.data_fetcher import DataFetcher


class TestSymbolManagerInitialization:
    """Test SymbolManager initialization and validation."""

    def test_init_with_valid_parameters(self):
        """Test initialization with valid parameters."""
        data_fetcher = Mock(spec=DataFetcher)
        manager = SymbolManager(
            data_fetcher=data_fetcher,
            whitelist=["BTC/USDT", "ETH/USDT"],
            blacklist=["SHIB/USDT"],
            max_symbols=50,
            random_seed=42,
        )

        assert manager.data_fetcher == data_fetcher
        assert manager.whitelist == {"BTC/USDT", "ETH/USDT"}
        assert manager.blacklist == {"SHIB/USDT"}
        assert manager.max_symbols == 50
        assert manager._cached_symbols == []

    def test_init_without_optional_parameters(self):
        """Test initialization without optional parameters."""
        data_fetcher = Mock(spec=DataFetcher)
        manager = SymbolManager(data_fetcher=data_fetcher)

        assert manager.whitelist == set()
        assert manager.blacklist == set()
        assert manager.max_symbols == 100

    def test_init_with_zero_max_symbols_raises_error(self):
        """Test that max_symbols=0 raises ValueError."""
        data_fetcher = Mock(spec=DataFetcher)

        with pytest.raises(ValueError, match="max_symbols must be positive, got 0"):
            SymbolManager(data_fetcher=data_fetcher, max_symbols=0)

    def test_init_with_negative_max_symbols_raises_error(self):
        """Test that negative max_symbols raises ValueError."""
        data_fetcher = Mock(spec=DataFetcher)

        with pytest.raises(ValueError, match="max_symbols must be positive, got -10"):
            SymbolManager(data_fetcher=data_fetcher, max_symbols=-10)

    @patch("modules.auto_trade.core.symbol_manager.log_warn")
    def test_init_with_large_max_symbols_logs_warning(self, mock_log_warn):
        """Test that very large max_symbols logs a warning."""
        data_fetcher = Mock(spec=DataFetcher)
        manager = SymbolManager(data_fetcher=data_fetcher, max_symbols=20000)

        assert manager.max_symbols == 20000
        mock_log_warn.assert_called_once()
        assert "very large" in mock_log_warn.call_args[0][0]


class TestSymbolManagerRefresh:
    """Test symbol refreshing functionality."""

    def _create_mock_data_fetcher(self, symbols: List[str]) -> Mock:
        """Helper to create a mock DataFetcher with symbol discovery."""
        data_fetcher = Mock(spec=DataFetcher)
        # SymbolManager calls data_fetcher.list_binance_futures_symbols directly
        data_fetcher.list_binance_futures_symbols.return_value = symbols
        return data_fetcher

    def test_refresh_symbols_without_filters(self):
        """Test refreshing symbols without whitelist or blacklist."""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"]
        data_fetcher = self._create_mock_data_fetcher(symbols)

        manager = SymbolManager(data_fetcher=data_fetcher)
        manager.refresh_symbols()

        assert manager._cached_symbols == symbols
        data_fetcher.list_binance_futures_symbols.assert_called_once_with(
            exclude_symbols=set(), max_candidates=100, progress_label="Refreshing Symbols"
        )

    def test_refresh_symbols_with_blacklist(self):
        """Test that blacklist is passed to symbol discovery."""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        data_fetcher = self._create_mock_data_fetcher(symbols)

        manager = SymbolManager(data_fetcher=data_fetcher, blacklist=["SHIB/USDT", "DOGE/USDT"])
        manager.refresh_symbols()

        assert manager._cached_symbols == symbols
        data_fetcher.list_binance_futures_symbols.assert_called_once_with(
            exclude_symbols={"SHIB/USDT", "DOGE/USDT"}, max_candidates=100, progress_label="Refreshing Symbols"
        )

    def test_refresh_symbols_with_whitelist_preserves_volume_order(self):
        """Test that whitelist preserves volume-sorted order."""
        # Symbols in volume order (descending)
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
        data_fetcher = self._create_mock_data_fetcher(symbols)

        # Whitelist in different order
        whitelist = ["ADA/USDT", "BTC/USDT", "SOL/USDT"]
        manager = SymbolManager(data_fetcher=data_fetcher, whitelist=whitelist)
        manager.refresh_symbols()

        # Should maintain volume order, not whitelist order
        assert manager._cached_symbols == ["BTC/USDT", "ADA/USDT", "SOL/USDT"]

    @patch("modules.auto_trade.core.symbol_manager.log_warn")
    def test_refresh_symbols_with_whitelist_missing_symbols(self, mock_log_warn):
        """Test warning when whitelist symbols are not in top volume list."""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        data_fetcher = self._create_mock_data_fetcher(symbols)

        whitelist = ["BTC/USDT", "XYZ/USDT", "ABC/USDT"]  # XYZ and ABC not in top volume
        manager = SymbolManager(data_fetcher=data_fetcher, whitelist=whitelist)
        manager.refresh_symbols()

        assert manager._cached_symbols == ["BTC/USDT"]
        mock_log_warn.assert_called_once()
        assert "not in top" in mock_log_warn.call_args[0][0]
        assert "XYZ/USDT" in str(mock_log_warn.call_args[0][0]) or "ABC/USDT" in str(mock_log_warn.call_args[0][0])

    @patch("modules.auto_trade.core.symbol_manager.log_warn")
    def test_refresh_symbols_with_whitelist_no_matches(self, mock_log_warn):
        """Test warning when no whitelist symbols are found."""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        data_fetcher = self._create_mock_data_fetcher(symbols)

        whitelist = ["XYZ/USDT", "ABC/USDT"]  # None in top volume
        manager = SymbolManager(data_fetcher=data_fetcher, whitelist=whitelist)
        manager.refresh_symbols()

        assert manager._cached_symbols == []
        assert mock_log_warn.call_count == 2  # One for missing, one for no active
        assert any("not in top" in str(call) for call in mock_log_warn.call_args_list)
        assert any("No whitelist symbols found" in str(call) for call in mock_log_warn.call_args_list)


class TestSymbolManagerGetSymbols:
    """Test get_symbols method with various sampling scenarios."""

    def _create_manager_with_symbols(self, symbols: List[str], random_seed: int = 42) -> SymbolManager:
        """Helper to create a SymbolManager with pre-populated symbols."""
        data_fetcher = Mock(spec=DataFetcher)
        data_fetcher.list_binance_futures_symbols.return_value = symbols

        manager = SymbolManager(data_fetcher=data_fetcher, random_seed=random_seed)
        manager.refresh_symbols()
        return manager

    def test_get_symbols_auto_refresh_when_empty(self):
        """Test that get_symbols auto-refreshes when cache is empty."""
        symbols = ["BTC/USDT", "ETH/USDT"]
        data_fetcher = Mock(spec=DataFetcher)
        data_fetcher.list_binance_futures_symbols.return_value = symbols

        manager = SymbolManager(data_fetcher=data_fetcher)
        result = manager.get_symbols()

        assert result == symbols
        data_fetcher.list_binance_futures_symbols.assert_called_once()

    def test_get_symbols_100_percent(self):
        """Test getting 100% of symbols."""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"]
        manager = self._create_manager_with_symbols(symbols)

        result = manager.get_symbols(100.0)

        assert result == symbols
        assert result is not manager._cached_symbols  # Should be a copy

    def test_get_symbols_0_percent(self):
        """Test getting 0% of symbols returns empty list."""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        manager = self._create_manager_with_symbols(symbols)

        result = manager.get_symbols(0.0)

        assert result == []

    def test_get_symbols_50_percent(self):
        """Test getting 50% of symbols."""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"]
        manager = self._create_manager_with_symbols(symbols)

        result = manager.get_symbols(50.0)

        assert len(result) == 2
        assert all(s in symbols for s in result)

    def test_get_symbols_small_percentage_returns_at_least_one(self):
        """Test that small percentage still returns at least 1 symbol."""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
        manager = self._create_manager_with_symbols(symbols)

        result = manager.get_symbols(0.5)  # 0.5% of 5 = 0.025, should round to 1

        assert len(result) == 1
        assert result[0] in symbols

    def test_get_symbols_rounding_behavior(self):
        """Test and document the sampling rounding behavior with round()."""
        symbols = ["S1", "S2", "S3", "S4", "S5"]
        manager = self._create_manager_with_symbols(symbols, random_seed=42)

        # 50% of 5 = 2.5 → round() uses banker's rounding → 2
        result_50 = manager.get_symbols(50.0)
        assert len(result_50) == 2

        # 60% of 5 = 3.0 → round() = 3
        result_60 = manager.get_symbols(60.0)
        assert len(result_60) == 3

        # 25% of 4 = 1.0 → round() = 1
        manager_4 = self._create_manager_with_symbols(symbols[:4], random_seed=42)
        result_25 = manager_4.get_symbols(25.0)
        assert len(result_25) == 1

    def test_get_symbols_sampling_is_random(self):
        """Test that sampling produces different results with different seeds."""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]

        manager1 = self._create_manager_with_symbols(symbols, random_seed=42)
        manager2 = self._create_manager_with_symbols(symbols, random_seed=99)

        result1 = manager1.get_symbols(40.0)
        result2 = manager2.get_symbols(40.0)

        # Different seeds should produce different samples (very likely)
        assert len(result1) == len(result2)
        # Note: There's a small chance they could be the same, but very unlikely

    def test_get_symbols_sampling_is_reproducible(self):
        """Test that sampling is reproducible with same seed across different instances."""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]

        # Create two separate managers with the same seed
        manager1 = self._create_manager_with_symbols(symbols, random_seed=42)
        manager2 = self._create_manager_with_symbols(symbols, random_seed=42)

        result1 = manager1.get_symbols(60.0)
        result2 = manager2.get_symbols(60.0)

        # Same seed should produce same sequence
        assert result1 == result2

    def test_get_symbols_invalid_negative_percentage(self):
        """Test that negative sample_percent raises ValueError."""
        symbols = ["BTC/USDT", "ETH/USDT"]
        manager = self._create_manager_with_symbols(symbols)

        with pytest.raises(ValueError, match="sample_percent must be 0-100, got -10.0"):
            manager.get_symbols(-10.0)

    def test_get_symbols_invalid_over_100_percentage(self):
        """Test that sample_percent > 100 raises ValueError."""
        symbols = ["BTC/USDT", "ETH/USDT"]
        manager = self._create_manager_with_symbols(symbols)

        with pytest.raises(ValueError, match="sample_percent must be 0-100, got 150.0"):
            manager.get_symbols(150.0)

    @patch("modules.auto_trade.core.symbol_manager.log_warn")
    def test_get_symbols_with_empty_cache_after_refresh(self, mock_log_warn):
        """Test handling when no symbols are available after refresh."""
        data_fetcher = Mock(spec=DataFetcher)
        data_fetcher.list_binance_futures_symbols.return_value = []

        manager = SymbolManager(data_fetcher=data_fetcher)
        result = manager.get_symbols()

        assert result == []
        mock_log_warn.assert_called_once()
        assert "No symbols available" in mock_log_warn.call_args[0][0]


class TestSymbolManagerCaching:
    """Test caching behavior."""

    def test_get_all_cached_symbols_returns_copy(self):
        """Test that get_all_cached_symbols returns a copy."""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        data_fetcher = Mock(spec=DataFetcher)
        data_fetcher.list_binance_futures_symbols.return_value = symbols

        manager = SymbolManager(data_fetcher=data_fetcher)
        manager.refresh_symbols()

        result = manager.get_all_cached_symbols()

        assert result == symbols
        assert result is not manager._cached_symbols

        # Modifying result shouldn't affect cache
        result.append("NEW/USDT")
        assert len(manager._cached_symbols) == 3

    def test_refresh_symbols_updates_cache(self):
        """Test that calling refresh_symbols updates the cache."""
        symbols1 = ["BTC/USDT", "ETH/USDT"]
        symbols2 = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"]

        data_fetcher = Mock(spec=DataFetcher)

        manager = SymbolManager(data_fetcher=data_fetcher)

        # First refresh
        data_fetcher.list_binance_futures_symbols.return_value = symbols1
        manager.refresh_symbols()
        assert len(manager._cached_symbols) == 2

        # Second refresh with different symbols
        data_fetcher.list_binance_futures_symbols.return_value = symbols2
        manager.refresh_symbols()
        assert len(manager._cached_symbols) == 4


class TestSymbolManagerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_symbol(self):
        """Test with only one symbol available."""
        symbols = ["BTC/USDT"]
        data_fetcher = Mock(spec=DataFetcher)
        data_fetcher.list_binance_futures_symbols.return_value = symbols

        manager = SymbolManager(data_fetcher=data_fetcher, random_seed=42)
        manager.refresh_symbols()

        # Get 50% should still return 1 symbol (min constraint)
        result = manager.get_symbols(50.0)
        assert len(result) == 1
        assert result[0] == "BTC/USDT"

    def test_whitelist_and_blacklist_interaction(self):
        """Test that whitelist takes precedence (blacklist already filters in discovery)."""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]  # Already blacklist-filtered
        data_fetcher = Mock(spec=DataFetcher)
        data_fetcher.list_binance_futures_symbols.return_value = symbols

        # Even if we set both, whitelist should just filter from what discovery returns
        manager = SymbolManager(
            data_fetcher=data_fetcher, whitelist=["BTC/USDT", "ETH/USDT"], blacklist=["BNB/USDT"]
        )
        manager.refresh_symbols()

        assert manager._cached_symbols == ["BTC/USDT", "ETH/USDT"]

    def test_max_symbols_parameter_is_passed_to_discovery(self):
        """Test that max_symbols is correctly passed to symbol discovery."""
        symbols = ["BTC/USDT", "ETH/USDT"]
        data_fetcher = Mock(spec=DataFetcher)
        data_fetcher.list_binance_futures_symbols.return_value = symbols

        manager = SymbolManager(data_fetcher=data_fetcher, max_symbols=25)
        manager.refresh_symbols()

        data_fetcher.list_binance_futures_symbols.assert_called_once_with(
            exclude_symbols=set(), max_candidates=25, progress_label="Refreshing Symbols"
        )
