"""
Comprehensive test suite for ExchangeManager module.

Uses pure pytest with pytest-mock for mocking.
Install: pip install pytest pytest-mock
"""

import threading
import time

import pytest

from modules.common.core.exchange_manager import (
    AuthenticatedExchangeManager,
    ExchangeManager,
    ExchangeWrapper,
    PublicExchangeManager,
)


class TestExchangeWrapper:
    """Test ExchangeWrapper reference counting."""

    def test_initial_refcount_is_zero(self):
        """Verify new wrapper starts with refcount of 0."""
        mock_exchange = object()  # Simple object for testing
        wrapper = ExchangeWrapper(mock_exchange)
        assert wrapper.get_refcount() == 0
        assert not wrapper.is_in_use()

    def test_increment_refcount(self):
        """Verify increment increases refcount."""
        mock_exchange = object()
        wrapper = ExchangeWrapper(mock_exchange)

        assert wrapper.increment_refcount() == 1
        assert wrapper.get_refcount() == 1
        assert wrapper.is_in_use()

        assert wrapper.increment_refcount() == 2
        assert wrapper.get_refcount() == 2

    def test_decrement_refcount(self):
        """Verify decrement decreases refcount."""
        mock_exchange = object()
        wrapper = ExchangeWrapper(mock_exchange)

        wrapper.increment_refcount()
        wrapper.increment_refcount()

        assert wrapper.decrement_refcount() == 1
        assert wrapper.get_refcount() == 1

        assert wrapper.decrement_refcount() == 0
        assert wrapper.get_refcount() == 0
        assert not wrapper.is_in_use()

    def test_decrement_below_zero_stops_at_zero(self):
        """Verify decrement doesn't go below 0."""
        mock_exchange = object()
        wrapper = ExchangeWrapper(mock_exchange)

        assert wrapper.decrement_refcount() == 0
        assert wrapper.get_refcount() == 0

    def test_thread_safety(self):
        """Verify refcount is thread-safe."""
        mock_exchange = object()
        wrapper = ExchangeWrapper(mock_exchange)

        def increment_many():
            for _ in range(100):
                wrapper.increment_refcount()

        threads = [threading.Thread(target=increment_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should be exactly 1000 (10 threads * 100 increments each)
        assert wrapper.get_refcount() == 1000


class TestAuthenticatedExchangeManager:
    """Test AuthenticatedExchangeManager functionality."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        return AuthenticatedExchangeManager(api_key="test_key", api_secret="test_secret", testnet=True)

    def test_initialization(self, manager):
        """Verify manager initializes correctly."""
        assert manager.default_api_key == "test_key"
        assert manager.default_api_secret == "test_secret"
        assert manager.testnet is True
        assert len(manager._authenticated_exchanges) == 0

    def test_connect_to_exchange_creates_wrapper(self, manager, mocker):
        """Verify connection creates and caches exchange."""
        mock_exchange = mocker.MagicMock()
        mock_create = mocker.patch(
            "modules.common.core.exchange_manager.connection_factory.ExchangeConnectionFactory.create_authenticated_exchange",
            return_value=mock_exchange,
        )

        result = manager.connect_to_exchange_with_credentials("binance")

        assert result == mock_exchange
        assert "binance_True_future" in manager._authenticated_exchanges
        wrapper = manager._authenticated_exchanges["binance_True_future"]
        assert wrapper.get_refcount() == 1
        mock_create.assert_called_once()

    def test_connect_twice_reuses_exchange(self, manager, mocker):
        """Verify second connection reuses cached exchange."""
        mock_exchange = mocker.MagicMock()
        mock_create = mocker.patch(
            "modules.common.core.exchange_manager.connection_factory.ExchangeConnectionFactory.create_authenticated_exchange",
            return_value=mock_exchange,
        )

        result1 = manager.connect_to_exchange_with_credentials("binance")
        result2 = manager.connect_to_exchange_with_credentials("binance")

        assert result1 == result2
        assert mock_create.call_count == 1  # Only created once
        wrapper = manager._authenticated_exchanges["binance_True_future"]
        assert wrapper.get_refcount() == 2

    def test_set_exchange_credentials(self, manager):
        """Verify per-exchange credentials can be set."""
        manager.set_exchange_credentials("okx", "okx_key", "okx_secret")

        assert "okx" in manager._exchange_credentials
        assert manager._exchange_credentials["okx"]["api_key"] == "okx_key"
        assert manager._exchange_credentials["okx"]["api_secret"] == "okx_secret"

    def test_release_exchange_decrements_refcount(self, manager, mocker):
        """Verify release decrements refcount."""
        mock_exchange = mocker.MagicMock()
        mocker.patch(
            "modules.common.core.exchange_manager.connection_factory.ExchangeConnectionFactory.create_authenticated_exchange",
            return_value=mock_exchange,
        )

        manager.connect_to_exchange_with_credentials("binance")
        wrapper = manager._authenticated_exchanges["binance_True_future"]
        assert wrapper.get_refcount() == 1

        manager.release_exchange("binance", testnet=True, contract_type="future")
        assert wrapper.get_refcount() == 0

    def test_context_manager_releases_on_exit(self, manager, mocker):
        """Verify context manager releases reference on exit."""
        mock_exchange = mocker.MagicMock()
        mocker.patch(
            "modules.common.core.exchange_manager.connection_factory.ExchangeConnectionFactory.create_authenticated_exchange",
            return_value=mock_exchange,
        )

        with manager.exchange_context("binance") as exchange:
            assert exchange == mock_exchange
            wrapper = manager._authenticated_exchanges["binance_True_future"]
            assert wrapper.get_refcount() == 1

        # After context exit, refcount should be 0
        wrapper = manager._authenticated_exchanges["binance_True_future"]
        assert wrapper.get_refcount() == 0

    def test_context_manager_releases_on_exception(self, manager, mocker):
        """Verify context manager releases reference even on exception."""
        mock_exchange = mocker.MagicMock()
        mocker.patch(
            "modules.common.core.exchange_manager.connection_factory.ExchangeConnectionFactory.create_authenticated_exchange",
            return_value=mock_exchange,
        )

        with pytest.raises(ValueError):
            with manager.exchange_context("binance") as _:
                wrapper = manager._authenticated_exchanges["binance_True_future"]
                assert wrapper.get_refcount() == 1
                raise ValueError("Test exception")

        # After exception, refcount should still be 0
        wrapper = manager._authenticated_exchanges["binance_True_future"]
        assert wrapper.get_refcount() == 0

    def test_cleanup_removes_unused_exchanges(self, manager, mocker):
        """Verify cleanup removes unused exchanges."""
        mock_exchange = mocker.MagicMock()
        mock_exchange.close = mocker.MagicMock()
        mocker.patch(
            "modules.common.core.exchange_manager.connection_factory.ExchangeConnectionFactory.create_authenticated_exchange",
            return_value=mock_exchange,
        )

        # Create and release exchange
        manager.connect_to_exchange_with_credentials("binance")
        manager.release_exchange("binance", testnet=True, contract_type="future")

        assert "binance_True_future" in manager._authenticated_exchanges

        # Cleanup should remove it
        manager.cleanup_unused_exchanges()

        assert "binance_True_future" not in manager._authenticated_exchanges
        mock_exchange.close.assert_called_once()

    def test_cleanup_keeps_in_use_exchanges(self, manager, mocker):
        """Verify cleanup doesn't remove in-use exchanges."""
        mock_exchange = mocker.MagicMock()
        mocker.patch(
            "modules.common.core.exchange_manager.connection_factory.ExchangeConnectionFactory.create_authenticated_exchange",
            return_value=mock_exchange,
        )

        # Create exchange and keep reference
        manager.connect_to_exchange_with_credentials("binance")

        assert "binance_True_future" in manager._authenticated_exchanges

        # Cleanup should NOT remove it (still in use)
        manager.cleanup_unused_exchanges()

        assert "binance_True_future" in manager._authenticated_exchanges

    def test_cleanup_with_age_filter(self, manager, mocker):
        """Verify cleanup respects max_age_hours parameter."""
        mock_exchange = mocker.MagicMock()
        mock_exchange.close = mocker.MagicMock()
        mocker.patch(
            "modules.common.core.exchange_manager.connection_factory.ExchangeConnectionFactory.create_authenticated_exchange",
            return_value=mock_exchange,
        )

        # Create and release exchange
        manager.connect_to_exchange_with_credentials("binance")
        cache_key = "binance_True_future"
        manager.release_exchange("binance", testnet=True, contract_type="future")

        # Set timestamp to 2 hours ago
        manager._exchange_timestamps[cache_key] = time.time() - (2 * 3600)

        # Cleanup with 1 hour max age should remove it
        manager.cleanup_unused_exchanges(max_age_hours=1.0)
        assert cache_key not in manager._authenticated_exchanges

        # Create another exchange
        manager.connect_to_exchange_with_credentials("binance")
        manager.release_exchange("binance", testnet=True, contract_type="future")

        # Cleanup with 3 hour max age should keep it
        manager.cleanup_unused_exchanges(max_age_hours=3.0)
        assert cache_key in manager._authenticated_exchanges

    def test_missing_credentials_raises_error(self, manager):
        """Verify missing credentials raises proper error."""
        manager_no_creds = AuthenticatedExchangeManager()
        # Explicitly clear defaults that might be picked up from environment
        manager_no_creds.default_api_key = None
        manager_no_creds.default_api_secret = None

        with pytest.raises(ValueError) as exc_info:
            manager_no_creds.connect_to_exchange_with_credentials("binance")

        assert "API Key and API Secret are required" in str(exc_info.value)
        assert "binance" in str(exc_info.value)

    def test_throttled_call_enforces_rate_limit(self, manager, mocker):
        """Verify throttled_call enforces minimum delay."""
        mock_func = mocker.MagicMock(return_value="result")

        start = time.time()
        result1 = manager.throttled_call(mock_func, "arg1")
        result2 = manager.throttled_call(mock_func, "arg2")
        elapsed = time.time() - start

        assert result1 == "result"
        assert result2 == "result"
        assert mock_func.call_count == 2
        assert elapsed >= manager.request_pause


class TestPublicExchangeManager:
    """Test PublicExchangeManager functionality."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        return PublicExchangeManager()

    def test_connect_to_exchange_creates_instance(self, manager, mocker):
        """Verify connection creates exchange instance."""
        mock_exchange = mocker.MagicMock()
        mock_binance_class = mocker.patch("ccxt.binance", return_value=mock_exchange)

        result = manager.connect_to_exchange_with_no_credentials("binance")

        assert result == mock_exchange
        assert "binance" in manager._public_exchanges
        mock_binance_class.assert_called_once()

    def test_connect_twice_reuses_exchange(self, manager, mocker):
        """Verify second connection reuses cached exchange."""
        mock_exchange = mocker.MagicMock()
        mock_binance_class = mocker.patch("ccxt.binance", return_value=mock_exchange)

        result1 = manager.connect_to_exchange_with_no_credentials("binance")
        result2 = manager.connect_to_exchange_with_no_credentials("binance")

        assert result1 == result2
        assert mock_binance_class.call_count == 1  # Only created once

    def test_unsupported_exchange_raises_error(self, manager):
        """Verify unsupported exchange raises error."""
        with pytest.raises(ValueError) as exc_info:
            manager.connect_to_exchange_with_no_credentials("fake_exchange")

        assert "not supported by ccxt" in str(exc_info.value)

    def test_cleanup_removes_exchanges(self, manager, mocker):
        """Verify cleanup removes exchanges."""
        mock_exchange = mocker.MagicMock()
        mock_exchange.close = mocker.MagicMock()
        mocker.patch("ccxt.binance", return_value=mock_exchange)

        manager.connect_to_exchange_with_no_credentials("binance")
        assert "binance" in manager._public_exchanges

        manager.cleanup_unused_exchanges()

        assert "binance" not in manager._public_exchanges
        mock_exchange.close.assert_called_once()

    def test_exchange_priority_for_fallback(self, manager):
        """Verify exchange priority can be set and retrieved."""
        priority = ["kraken", "binance", "kucoin"]
        manager.exchange_priority_for_fallback = priority

        assert manager.exchange_priority_for_fallback == priority


class TestExchangeManager:
    """Test ExchangeManager facade."""

    def test_initialization(self):
        """Verify facade initializes both managers."""
        manager = ExchangeManager(api_key="test_key", api_secret="test_secret")

        assert manager.authenticated is not None
        assert manager.public is not None
        assert isinstance(manager.authenticated, AuthenticatedExchangeManager)
        assert isinstance(manager.public, PublicExchangeManager)

    def test_normalize_symbol_removes_contract_marker(self):
        """Verify normalize_symbol handles futures notation."""
        manager = ExchangeManager()

        assert manager.normalize_symbol("BTC/USDT:USDT") == "BTC/USDT"
        assert manager.normalize_symbol("BTC/USDT") == "BTC/USDT"

    def test_cleanup_delegates_to_both_managers(self, mocker):
        """Verify cleanup is called on both managers."""
        manager = ExchangeManager()
        mock_auth_cleanup = mocker.patch.object(manager.authenticated, "cleanup_unused_exchanges")
        mock_public_cleanup = mocker.patch.object(manager.public, "cleanup_unused_exchanges")

        manager.cleanup_unused_exchanges(max_age_hours=1.0)

        mock_auth_cleanup.assert_called_once_with(1.0)
        mock_public_cleanup.assert_called_once_with(1.0)

    def test_close_exchange_delegates_to_both_managers(self, mocker):
        """Verify close_exchange is called on both managers."""
        manager = ExchangeManager()
        mock_auth_close = mocker.patch.object(manager.authenticated, "close_exchange")
        mock_public_close = mocker.patch.object(manager.public, "close_exchange")

        manager.close_exchange("binance", testnet=True, contract_type="future")

        mock_auth_close.assert_called_once_with("binance", True, "future")
        mock_public_close.assert_called_once_with("binance")


class TestThreadSafety:
    """Test concurrent access scenarios."""

    def test_concurrent_connections_are_safe(self, mocker):
        """Verify multiple threads can safely connect simultaneously."""
        mock_exchange = mocker.MagicMock()
        mocker.patch(
            "modules.common.core.exchange_manager.connection_factory.ExchangeConnectionFactory.create_authenticated_exchange",
            return_value=mock_exchange,
        )

        manager = AuthenticatedExchangeManager(api_key="test", api_secret="test")
        results = []

        def connect():
            result = manager.connect_to_exchange_with_credentials("binance")
            results.append(result)

        threads = [threading.Thread(target=connect) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same exchange instance
        assert len(results) == 10
        assert all(r == mock_exchange for r in results)

        # Refcount should be 10 (one per thread)
        wrapper = manager._authenticated_exchanges["binance_False_future"]
        assert wrapper.get_refcount() == 10


# Integration Tests (require real ccxt, but no API keys)
class TestIntegration:
    """Integration tests with real ccxt (but no API calls)."""

    @pytest.mark.integration
    def test_public_manager_creates_real_exchange(self):
        """Verify we can create real ccxt exchange (no API calls)."""
        manager = PublicExchangeManager()
        exchange = manager.connect_to_exchange_with_no_credentials("binance")

        assert exchange is not None
        assert hasattr(exchange, "fetch_ohlcv")
        assert hasattr(exchange, "fetch_ticker")

    @pytest.mark.integration
    def test_exchange_config_has_correct_defaults(self):
        """Verify exchange is configured correctly."""
        manager = PublicExchangeManager()
        exchange = manager.connect_to_exchange_with_no_credentials("binance")

        assert exchange.enableRateLimit is True
        assert "defaultType" in exchange.options
        # Should default to 'future'
        assert exchange.options["defaultType"] == "future"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
