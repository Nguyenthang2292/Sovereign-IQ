import pytest
import ccxt
import time
import threading
from unittest.mock import patch, MagicMock

# Mock pandas_ta before importing to avoid dependency issues
import sys
from unittest.mock import MagicMock as MockModule

# Create a mock pandas_ta module
mock_pandas_ta = MockModule()
sys.modules['pandas_ta'] = mock_pandas_ta

# Now import exchange_manager
from modules.common.core.exchange_manager import (
    ExchangeManager, 
    PublicExchangeManager,
    AuthenticatedExchangeManager,
    ExchangeWrapper
)


class DummyExchange:
    def __init__(self):
        self.calls = 0

    def ping(self):
        self.calls += 1
        return "pong"


def test_public_manager_caches_instances(monkeypatch):
    public = PublicExchangeManager()

    class DummyCCXTModule:
        def __init__(self, params=None):
            pass

    # Mock hasattr and getattr for ccxt module
    original_hasattr = hasattr
    original_getattr = getattr
    
    def mock_hasattr(obj, name):
        if obj is ccxt and name == "dummy":
            return True
        return original_hasattr(obj, name)
    
    def mock_getattr(obj, name, default=None):
        if obj is ccxt and name == "dummy":
            return DummyCCXTModule
        return original_getattr(obj, name, default)
    
    monkeypatch.setattr("builtins.hasattr", mock_hasattr)
    monkeypatch.setattr("builtins.getattr", mock_getattr)
    
    ex1 = public.connect_to_exchange_with_no_credentials("dummy")
    ex2 = public.connect_to_exchange_with_no_credentials("dummy")
    
    assert ex1 is ex2

    ex1 = public.connect_to_exchange_with_no_credentials("dummy")
    ex2 = public.connect_to_exchange_with_no_credentials("dummy")

    assert ex1 is ex2


def test_throttled_call_enforces_wait(monkeypatch):
    public = PublicExchangeManager(request_pause=0.01)
    exchange = DummyExchange()

    called = []

    def fake_time():
        return len(called) * 0.02

    monkeypatch.setattr(time, "time", fake_time)
    result = public.throttled_call(exchange.ping)
    called.append(1)
    result2 = public.throttled_call(exchange.ping)

    assert result == "pong" and result2 == "pong"
    assert exchange.calls == 2


def test_exchange_manager_normalizes_symbols():
    manager = ExchangeManager()
    assert manager.normalize_symbol("BTC/USDT:USDT") == "BTC/USDT"


def test_public_manager_rejects_unknown_exchange(monkeypatch):
    public = PublicExchangeManager()

    # Mock getattr to raise AttributeError for unknown exchange
    original_getattr = getattr
    def mock_getattr(obj, name, default=None):
        if obj is ccxt and name == "not_real":
            raise AttributeError(f"module 'ccxt' has no attribute '{name}'")
        return original_getattr(obj, name, default)
    
    monkeypatch.setattr("builtins.getattr", mock_getattr)

    with pytest.raises(ValueError):
        public.connect_to_exchange_with_no_credentials("not_real")


@pytest.fixture
def mock_binance_exchange(monkeypatch):
    """Fixture to mock ccxt.binance exchange for use across multiple test classes."""

    # Validate ccxt is in scope before using, avoids NameError
    try:
        import ccxt
    except ImportError as e:
        pytest.skip("ccxt must be installed for these tests")  # Or you could raise

    class MockBinance:
        def __init__(self, params):
            self.params = params

    original_hasattr = hasattr
    original_getattr = getattr

    def mock_hasattr(obj, name):
        # Check 'obj' is actually the module, prevents AttributeError
        if obj is ccxt and name == "binance":
            return True
        return original_hasattr(obj, name)

    def mock_getattr(obj, name, default=None):
        # Defensive: only mock binance when used on ccxt
        if obj is ccxt and name == "binance":
            return MockBinance
        # Use try/except to handle missing attributes
        try:
            return original_getattr(obj, name, default)
        except AttributeError:
            if default is not None:
                return default
            raise

    monkeypatch.setattr("builtins.hasattr", mock_hasattr)
    monkeypatch.setattr("builtins.getattr", mock_getattr)


class TestExchangeWrapper:
    """Test ExchangeWrapper class for reference counting."""
    
    def test_exchange_wrapper_initialization(self):
        """Test ExchangeWrapper initialization."""
        mock_exchange = MagicMock()
        wrapper = ExchangeWrapper(mock_exchange)
        
        assert wrapper.exchange is mock_exchange
        assert wrapper.get_refcount() == 0
        assert not wrapper.is_in_use()
    
    def test_increment_refcount(self):
        """Test incrementing reference count."""
        wrapper = ExchangeWrapper(MagicMock())
        
        assert wrapper.increment_refcount() == 1
        assert wrapper.increment_refcount() == 2
        assert wrapper.get_refcount() == 2
        assert wrapper.is_in_use()
    
    def test_decrement_refcount(self):
        """Test decrementing reference count."""
        wrapper = ExchangeWrapper(MagicMock())
        
        wrapper.increment_refcount()
        wrapper.increment_refcount()
        assert wrapper.decrement_refcount() == 1
        assert wrapper.decrement_refcount() == 0
        assert wrapper.decrement_refcount() == 0  # Should not go below 0
        assert wrapper.get_refcount() == 0
        assert not wrapper.is_in_use()
    
    def test_is_in_use(self):
        """Test is_in_use method."""
        wrapper = ExchangeWrapper(MagicMock())
        
        assert not wrapper.is_in_use()
        wrapper.increment_refcount()
        assert wrapper.is_in_use()
        wrapper.decrement_refcount()
        assert not wrapper.is_in_use()
    
    def test_thread_safe_refcount(self):
        """Test that reference counting is thread-safe."""
        wrapper = ExchangeWrapper(MagicMock())
        
        def increment_multiple():
            for _ in range(100):
                wrapper.increment_refcount()
        
        threads = [threading.Thread(target=increment_multiple) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Final refcount should be 500 (5 threads * 100 increments)
        assert wrapper.get_refcount() == 500


class TestAuthenticatedExchangeManagerReferenceCounting:
    """Test reference counting in AuthenticatedExchangeManager."""
    
    def test_connect_increments_refcount(self, mock_binance_exchange):
        """Test that connect_to_exchange_with_credentials increments refcount."""
        manager = AuthenticatedExchangeManager(api_key="test_key", api_secret="test_secret")
        
        exchange = manager.connect_to_exchange_with_credentials("binance")
        
        cache_key = "binance_False_future"
        assert cache_key in manager._authenticated_exchanges
        wrapper = manager._authenticated_exchanges[cache_key]
        assert wrapper.get_refcount() == 1
        assert wrapper.is_in_use()
    
    def test_multiple_connects_increment_refcount(self, mock_binance_exchange):
        """Test that multiple connects increment refcount."""
        manager = AuthenticatedExchangeManager(api_key="test_key", api_secret="test_secret")
        
        exchange1 = manager.connect_to_exchange_with_credentials("binance")
        exchange2 = manager.connect_to_exchange_with_credentials("binance")
        
        assert exchange1 is exchange2
        
        cache_key = "binance_False_future"
        wrapper = manager._authenticated_exchanges[cache_key]
        assert wrapper.get_refcount() == 2
    
    def test_release_exchange_decrements_refcount(self, mock_binance_exchange):
        """Test that release_exchange decrements refcount."""
        manager = AuthenticatedExchangeManager(api_key="test_key", api_secret="test_secret")
        
        exchange = manager.connect_to_exchange_with_credentials("binance")
        manager.connect_to_exchange_with_credentials("binance")  # Increment again
        
        cache_key = "binance_False_future"
        wrapper = manager._authenticated_exchanges[cache_key]
        assert wrapper.get_refcount() == 2
        
        manager.release_exchange("binance")
        assert wrapper.get_refcount() == 1
        
        manager.release_exchange("binance")
        assert wrapper.get_refcount() == 0
        assert not wrapper.is_in_use()
    
    def test_release_exchange_not_in_cache(self):
        """Test releasing exchange that's not in cache."""
        manager = AuthenticatedExchangeManager()
        
        # Should not raise error
        manager.release_exchange("nonexistent")
    
    def test_exchange_context_manager(self, mock_binance_exchange):
        """Test exchange_context context manager."""
        manager = AuthenticatedExchangeManager(api_key="test_key", api_secret="test_secret")
        
        cache_key = "binance_False_future"
        
        with manager.exchange_context("binance") as exchange:
            wrapper = manager._authenticated_exchanges[cache_key]
            assert wrapper.get_refcount() == 1
            assert wrapper.is_in_use()
        
        # After context exit, refcount should be decremented
        wrapper = manager._authenticated_exchanges[cache_key]
        assert wrapper.get_refcount() == 0
        assert not wrapper.is_in_use()
    
    def test_exchange_context_manager_with_exception(self, mock_binance_exchange):
        """Test exchange_context releases refcount even on exception."""
        manager = AuthenticatedExchangeManager(api_key="test_key", api_secret="test_secret")
        
        cache_key = "binance_False_future"
        
        with pytest.raises(ValueError):
            with manager.exchange_context("binance") as exchange:
                wrapper = manager._authenticated_exchanges[cache_key]
                assert wrapper.get_refcount() == 1
                raise ValueError("Test exception")
        
        # After exception, refcount should still be decremented
        wrapper = manager._authenticated_exchanges[cache_key]
        assert wrapper.get_refcount() == 0


class TestExchangeManagerCleanup:
    """Test cleanup functionality in ExchangeManager."""
    
    def test_authenticated_cleanup_unused_exchanges(self):
        """Test cleanup of unused exchange connections."""
        manager = AuthenticatedExchangeManager()
        
        # Create wrappers with refcount = 0 (unused)
        mock_exchange1 = MagicMock()
        mock_exchange2 = MagicMock()
        wrapper1 = ExchangeWrapper(mock_exchange1)
        wrapper2 = ExchangeWrapper(mock_exchange2)
        
        with manager._request_lock:
            manager._authenticated_exchanges['binance_False_future'] = wrapper1
            manager._authenticated_exchanges['kraken_False_spot'] = wrapper2
            manager._exchange_timestamps['binance_False_future'] = time.time()
            manager._exchange_timestamps['kraken_False_spot'] = time.time()
            manager._exchange_credentials['binance'] = {'api_key': 'test', 'api_secret': 'test'}
        
        # Verify exchanges exist
        assert len(manager._authenticated_exchanges) == 2
        assert len(manager._exchange_credentials) == 1
        
        # Cleanup
        manager.cleanup_unused_exchanges()
        
        # Verify unused exchanges are cleared
        assert len(manager._authenticated_exchanges) == 0
        # Credentials should NOT be cleared
        assert len(manager._exchange_credentials) == 1
    
    def test_authenticated_cleanup_preserves_in_use_exchanges(self):
        """Test that cleanup does not remove exchanges in use."""
        manager = AuthenticatedExchangeManager()
        
        # Create wrapper with refcount > 0 (in use)
        mock_exchange = MagicMock()
        wrapper = ExchangeWrapper(mock_exchange)
        wrapper.increment_refcount()  # Mark as in use
        
        with manager._request_lock:
            manager._authenticated_exchanges['binance_False_future'] = wrapper
            manager._exchange_timestamps['binance_False_future'] = time.time()
        
        # Cleanup
        manager.cleanup_unused_exchanges()
        
        # Verify exchange is NOT cleared because it's in use
        assert len(manager._authenticated_exchanges) == 1
        assert 'binance_False_future' in manager._authenticated_exchanges
    
    def test_authenticated_cleanup_with_max_age_hours(self):
        """Test cleanup with max_age_hours parameter."""
        manager = AuthenticatedExchangeManager()
        
        # Create old exchange (older than 1 hour)
        mock_exchange1 = MagicMock()
        wrapper1 = ExchangeWrapper(mock_exchange1)
        
        # Create recent exchange (less than 1 hour old)
        mock_exchange2 = MagicMock()
        wrapper2 = ExchangeWrapper(mock_exchange2)
        
        current_time = time.time()
        old_time = current_time - 7200  # 2 hours ago
        
        with manager._request_lock:
            manager._authenticated_exchanges['old_exchange'] = wrapper1
            manager._authenticated_exchanges['recent_exchange'] = wrapper2
            manager._exchange_timestamps['old_exchange'] = old_time
            manager._exchange_timestamps['recent_exchange'] = current_time - 300  # 5 minutes ago
        
        # Cleanup with max_age_hours = 1
        manager.cleanup_unused_exchanges(max_age_hours=1.0)
        
        # Only old exchange should be removed
        assert 'old_exchange' not in manager._authenticated_exchanges
        assert 'recent_exchange' in manager._authenticated_exchanges
    
    def test_authenticated_cleanup_exchange_without_timestamp(self):
        """Test cleanup of exchange without timestamp (treat as old)."""
        manager = AuthenticatedExchangeManager()
        
        mock_exchange = MagicMock()
        wrapper = ExchangeWrapper(mock_exchange)
        
        with manager._request_lock:
            manager._authenticated_exchanges['no_timestamp'] = wrapper
            # No timestamp entry
        
        # Cleanup with max_age_hours
        manager.cleanup_unused_exchanges(max_age_hours=1.0)
        
        # Exchange without timestamp should be removed
        assert 'no_timestamp' not in manager._authenticated_exchanges
    
    def test_authenticated_close_exchange(self):
        """Test closing a specific exchange connection."""
        manager = AuthenticatedExchangeManager()
        
        # Create wrapper with refcount = 0
        mock_exchange = MagicMock()
        mock_exchange.close = MagicMock()
        wrapper = ExchangeWrapper(mock_exchange)
        
        cache_key = "binance_False_future"
        with manager._request_lock:
            manager._authenticated_exchanges[cache_key] = wrapper
            manager._exchange_timestamps[cache_key] = time.time()
        
        # Close exchange
        manager.close_exchange('binance', testnet=False, contract_type='future')
        
        # Verify exchange was removed and close was called
        assert cache_key not in manager._authenticated_exchanges
        assert cache_key not in manager._exchange_timestamps
        mock_exchange.close.assert_called_once()
    
    def test_authenticated_close_exchange_in_use(self):
        """Test that close_exchange does not close exchange in use."""
        manager = AuthenticatedExchangeManager()
        
        mock_exchange = MagicMock()
        mock_exchange.close = MagicMock()
        wrapper = ExchangeWrapper(mock_exchange)
        wrapper.increment_refcount()  # Mark as in use
        
        cache_key = "binance_False_future"
        with manager._request_lock:
            manager._authenticated_exchanges[cache_key] = wrapper
        
        # Close exchange (should not close because in use)
        manager.close_exchange('binance', testnet=False, contract_type='future')
        
        # Verify exchange was NOT removed
        assert cache_key in manager._authenticated_exchanges
        mock_exchange.close.assert_not_called()
    
    def test_authenticated_close_exchange_no_close_method(self):
        """Test closing exchange that doesn't have close method."""
        manager = AuthenticatedExchangeManager()
        
        mock_exchange = MagicMock()
        del mock_exchange.close  # Remove close method
        wrapper = ExchangeWrapper(mock_exchange)
        
        cache_key = "binance_False_future"
        with manager._request_lock:
            manager._authenticated_exchanges[cache_key] = wrapper
            manager._exchange_timestamps[cache_key] = time.time()
        
        # Close exchange (should not raise error)
        manager.close_exchange('binance', testnet=False, contract_type='future')
        
        # Verify exchange was removed
        assert cache_key not in manager._authenticated_exchanges
    
    def test_authenticated_close_exchange_not_found(self):
        """Test closing non-existent exchange."""
        manager = AuthenticatedExchangeManager()
        
        # Should not raise error
        manager.close_exchange('nonexistent', testnet=False, contract_type='future')
        
        # Verify no exchanges exist
        assert len(manager._authenticated_exchanges) == 0
    
    def test_set_exchange_credentials_clears_unused_exchanges(self, monkeypatch):
        """Test that set_exchange_credentials clears unused exchanges."""
        manager = AuthenticatedExchangeManager()
        
        mock_exchange = MagicMock()
        wrapper = ExchangeWrapper(mock_exchange)
        # refcount = 0 (unused)
        
        cache_key = "binance_False_future"
        with manager._request_lock:
            manager._authenticated_exchanges[cache_key] = wrapper
        
        # Set new credentials
        manager.set_exchange_credentials("binance", "new_key", "new_secret")
        
        # Unused exchange should be cleared
        assert cache_key not in manager._authenticated_exchanges
        assert manager._exchange_credentials['binance']['api_key'] == 'new_key'
    
    def test_set_exchange_credentials_preserves_in_use_exchanges(self):
        """Test that set_exchange_credentials does not clear exchanges in use."""
        manager = AuthenticatedExchangeManager()
        
        mock_exchange = MagicMock()
        wrapper = ExchangeWrapper(mock_exchange)
        wrapper.increment_refcount()  # Mark as in use
        
        cache_key = "binance_False_future"
        with manager._request_lock:
            manager._authenticated_exchanges[cache_key] = wrapper
        
        # Set new credentials
        manager.set_exchange_credentials("binance", "new_key", "new_secret")
        
        # Exchange in use should NOT be cleared
        assert cache_key in manager._authenticated_exchanges
    
    def test_update_default_credentials_clears_unused_exchanges(self):
        """Test that update_default_credentials clears unused exchanges."""
        manager = AuthenticatedExchangeManager()
        
        mock_exchange = MagicMock()
        wrapper = ExchangeWrapper(mock_exchange)
        # refcount = 0 (unused)
        
        cache_key = "binance_False_future"
        with manager._request_lock:
            manager._authenticated_exchanges[cache_key] = wrapper
        
        # Update credentials
        manager.update_default_credentials(api_key="new_key", api_secret="new_secret")
        
        # Unused exchange should be cleared
        assert cache_key not in manager._authenticated_exchanges
        assert manager.default_api_key == "new_key"
    
    def test_update_default_credentials_preserves_in_use_exchanges(self):
        """Test that update_default_credentials does not clear exchanges in use."""
        manager = AuthenticatedExchangeManager()
        
        mock_exchange = MagicMock()
        wrapper = ExchangeWrapper(mock_exchange)
        wrapper.increment_refcount()  # Mark as in use
        
        cache_key = "binance_False_future"
        with manager._request_lock:
            manager._authenticated_exchanges[cache_key] = wrapper
        
        # Update credentials
        manager.update_default_credentials(api_key="new_key", api_secret="new_secret")
        
        # Exchange in use should NOT be cleared
        assert cache_key in manager._authenticated_exchanges
    
    def test_public_cleanup_unused_exchanges(self):
        """Test cleanup of unused exchange connections in PublicExchangeManager."""
        manager = PublicExchangeManager()
        
        # Mock exchange connections
        mock_exchange1 = MagicMock()
        mock_exchange2 = MagicMock()
        
        with manager._request_lock:
            manager._public_exchanges['binance'] = mock_exchange1
            manager._public_exchanges['kraken'] = mock_exchange2
        
        # Verify exchanges exist
        assert len(manager._public_exchanges) == 2
        
        # Cleanup
        manager.cleanup_unused_exchanges()
        
        # Verify exchanges are cleared
        assert len(manager._public_exchanges) == 0
    
    def test_public_close_exchange(self):
        """Test closing a specific public exchange connection."""
        manager = PublicExchangeManager()
        
        # Mock exchange with close method
        mock_exchange = MagicMock()
        mock_exchange.close = MagicMock()
        
        with manager._request_lock:
            manager._public_exchanges['binance'] = mock_exchange
        
        # Close exchange
        manager.close_exchange('binance')
        
        # Verify exchange was removed and close was called
        assert 'binance' not in manager._public_exchanges
        mock_exchange.close.assert_called_once()
    
    def test_exchange_manager_cleanup_unused_exchanges(self):
        """Test cleanup in composite ExchangeManager."""
        manager = ExchangeManager()
        
        # Mock sub-managers
        mock_auth_cleanup = MagicMock()
        mock_public_cleanup = MagicMock()
        manager.authenticated.cleanup_unused_exchanges = mock_auth_cleanup
        manager.public.cleanup_unused_exchanges = mock_public_cleanup
        
        # Call cleanup
        manager.cleanup_unused_exchanges()
        
        # Verify both sub-managers were called
        mock_auth_cleanup.assert_called_once()
        mock_public_cleanup.assert_called_once()
    
    def test_exchange_manager_close_exchange(self):
        """Test closing exchange in composite ExchangeManager."""
        manager = ExchangeManager()
        
        # Mock sub-managers
        mock_auth_close = MagicMock()
        mock_public_close = MagicMock()
        manager.authenticated.close_exchange = mock_auth_close
        manager.public.close_exchange = mock_public_close
        
        # Call close
        manager.close_exchange('binance', testnet=False, contract_type='future')
        
        # Verify both sub-managers were called
        mock_auth_close.assert_called_once_with('binance', False, 'future')
        mock_public_close.assert_called_once_with('binance')


class TestThreadSafety:
    """Test thread safety of exchange manager operations."""
    
    def test_concurrent_connect_same_exchange(self, mock_binance_exchange):
        """Test concurrent connections to same exchange."""
        manager = AuthenticatedExchangeManager(api_key="test_key", api_secret="test_secret")
        
        exchanges = []
        errors = []
        
        def connect():
            try:
                exchange = manager.connect_to_exchange_with_credentials("binance")
                exchanges.append(exchange)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads connecting simultaneously
        threads = [threading.Thread(target=connect) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should succeed
        assert len(errors) == 0
        assert len(exchanges) == 10
        
        # All should get the same exchange instance
        assert all(ex is exchanges[0] for ex in exchanges)
        
        # Refcount should be 10
        cache_key = "binance_False_future"
        wrapper = manager._authenticated_exchanges[cache_key]
        assert wrapper.get_refcount() == 10
    
    def test_concurrent_cleanup_and_connect(self, mock_binance_exchange):
        """Test concurrent cleanup and connect operations."""
        manager = AuthenticatedExchangeManager(api_key="test_key", api_secret="test_secret")
        
        # Create initial exchange
        exchange = manager.connect_to_exchange_with_credentials("binance")
        manager.release_exchange("binance")  # Release to make it unused
        
        cleanup_called = []
        connect_called = []
        
        def cleanup():
            manager.cleanup_unused_exchanges()
            cleanup_called.append(1)
        
        def connect():
            try:
                manager.connect_to_exchange_with_credentials("binance")
                connect_called.append(1)
            except Exception:
                pass
        
        # Run cleanup and connect concurrently
        threads = [
            threading.Thread(target=cleanup),
            threading.Thread(target=connect),
            threading.Thread(target=connect),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Both operations should complete without errors
        assert len(cleanup_called) == 1
        assert len(connect_called) == 2
