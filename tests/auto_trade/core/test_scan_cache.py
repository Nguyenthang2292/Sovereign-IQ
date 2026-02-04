"""
Unit tests for Rust ScanCache implementation.

Tests thread-safe LRU cache with TTL expiration.
"""

import time

import atc_rust
import pytest


class TestScanCache:
    """Test suite for Rust ScanCache."""

    def test_cache_creation(self):
        """Test cache creation with default and custom parameters."""
        # Default parameters
        cache = atc_rust.ScanCache()
        assert cache.capacity() == 1000
        assert cache.len() == 0

        # Custom parameters
        cache = atc_rust.ScanCache(capacity=500, ttl_seconds=30.0)
        assert cache.capacity() == 500
        assert cache.len() == 0

    def test_cache_creation_invalid_capacity(self):
        """Test that zero capacity raises error."""
        with pytest.raises(ValueError, match="Capacity must be > 0"):
            atc_rust.ScanCache(capacity=0)

    def test_cache_set_and_get(self):
        """Test basic set and get operations."""
        cache = atc_rust.ScanCache(capacity=100, ttl_seconds=60.0)

        # Prepare data
        longs = {"BTC/USDT", "ETH/USDT"}
        shorts = {"XRP/USDT"}
        strengths = {"BTC/USDT": 0.85, "ETH/USDT": 0.72, "XRP/USDT": -0.65}

        # Set cache entry
        cache.set("test_key", longs, shorts, strengths)

        # Verify cache size
        assert cache.len() == 1

        # Get cache entry
        result = cache.get("test_key")
        assert result is not None
        assert isinstance(result, dict)
        assert "longs" in result
        assert "shorts" in result
        assert "strengths" in result

        # Verify data integrity
        assert set(result["longs"]) == longs
        assert set(result["shorts"]) == shorts
        assert result["strengths"] == strengths

    def test_cache_get_nonexistent(self):
        """Test getting non-existent key returns None."""
        cache = atc_rust.ScanCache()
        result = cache.get("nonexistent_key")
        assert result is None

    def test_cache_contains(self):
        """Test contains method."""
        cache = atc_rust.ScanCache(ttl_seconds=60.0)

        # Initially not present
        assert not cache.contains("test_key")

        # Add entry
        cache.set("test_key", {"BTC/USDT"}, set(), {"BTC/USDT": 0.8})

        # Now present
        assert cache.contains("test_key")

        # Non-existent key
        assert not cache.contains("other_key")

    def test_cache_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = atc_rust.ScanCache(ttl_seconds=1.0)  # 1 second TTL

        # Add entry
        cache.set("test_key", {"BTC/USDT"}, set(), {"BTC/USDT": 0.8})

        # Immediately available
        assert cache.contains("test_key")
        result = cache.get("test_key")
        assert result is not None

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired
        assert not cache.contains("test_key")
        result = cache.get("test_key")
        assert result is None

    def test_cache_clear(self):
        """Test clearing all cache entries."""
        cache = atc_rust.ScanCache()

        # Add multiple entries
        for i in range(5):
            cache.set(f"key_{i}", {f"SYM_{i}"}, set(), {f"SYM_{i}": 0.8})

        assert cache.len() == 5

        # Clear cache
        cache.clear()
        assert cache.len() == 0

        # Verify all entries removed
        for i in range(5):
            assert cache.get(f"key_{i}") is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when capacity is exceeded."""
        cache = atc_rust.ScanCache(capacity=3, ttl_seconds=60.0)

        # Add 3 entries (fill capacity)
        cache.set("key_1", {"SYM_1"}, set(), {"SYM_1": 0.8})
        cache.set("key_2", {"SYM_2"}, set(), {"SYM_2": 0.8})
        cache.set("key_3", {"SYM_3"}, set(), {"SYM_3": 0.8})
        assert cache.len() == 3

        # Add 4th entry (should evict oldest: key_1)
        cache.set("key_4", {"SYM_4"}, set(), {"SYM_4": 0.8})
        assert cache.len() == 3

        # key_1 should be evicted
        assert cache.get("key_1") is None

        # Others should still exist
        assert cache.get("key_2") is not None
        assert cache.get("key_3") is not None
        assert cache.get("key_4") is not None

    def test_cache_remove_expired(self):
        """Test manual removal of expired entries."""
        cache = atc_rust.ScanCache(ttl_seconds=1.0)

        # Add entries
        cache.set("key_1", {"SYM_1"}, set(), {"SYM_1": 0.8})
        cache.set("key_2", {"SYM_2"}, set(), {"SYM_2": 0.8})
        cache.set("key_3", {"SYM_3"}, set(), {"SYM_3": 0.8})
        assert cache.len() == 3

        # Wait for expiration
        time.sleep(1.5)

        # Manually remove expired entries
        removed_count = cache.remove_expired()
        assert removed_count == 3
        assert cache.len() == 0

    def test_cache_mixed_expiration(self):
        """Test with some expired and some valid entries."""
        cache = atc_rust.ScanCache(ttl_seconds=2.0)

        # Add initial entries
        cache.set("old_1", {"SYM_1"}, set(), {"SYM_1": 0.8})
        cache.set("old_2", {"SYM_2"}, set(), {"SYM_2": 0.8})

        # Wait 1.5 seconds
        time.sleep(1.5)

        # Add new entries
        cache.set("new_1", {"SYM_3"}, set(), {"SYM_3": 0.8})
        cache.set("new_2", {"SYM_4"}, set(), {"SYM_4": 0.8})

        assert cache.len() == 4

        # Wait for old entries to expire (0.5s more = 2s total)
        time.sleep(0.6)

        # Remove expired (should remove old_1 and old_2)
        removed_count = cache.remove_expired()
        assert removed_count == 2
        assert cache.len() == 2

        # Verify correct entries remain
        assert cache.get("old_1") is None
        assert cache.get("old_2") is None
        assert cache.get("new_1") is not None
        assert cache.get("new_2") is not None

    def test_cache_repr(self):
        """Test string representation."""
        cache = atc_rust.ScanCache(capacity=100, ttl_seconds=60.0)
        repr_str = repr(cache)

        assert "ScanCache" in repr_str
        assert "size=0/100" in repr_str
        assert "ttl=60" in repr_str

        # Add entry
        cache.set("key_1", {"SYM_1"}, set(), {"SYM_1": 0.8})
        repr_str = repr(cache)
        assert "size=1/100" in repr_str

    def test_cache_empty_collections(self):
        """Test caching with empty longs/shorts."""
        cache = atc_rust.ScanCache()

        # Empty longs and shorts
        cache.set("key_1", set(), set(), {})
        result = cache.get("key_1")

        assert result is not None
        assert len(result["longs"]) == 0
        assert len(result["shorts"]) == 0
        assert len(result["strengths"]) == 0

    def test_cache_large_dataset(self):
        """Test cache with larger dataset."""
        cache = atc_rust.ScanCache(capacity=1000, ttl_seconds=60.0)

        # Add 100 entries
        for i in range(100):
            longs = {f"BTC_{i}", f"ETH_{i}"}
            shorts = {f"XRP_{i}"}
            strengths = {
                f"BTC_{i}": 0.8 + (i * 0.001),
                f"ETH_{i}": 0.7 + (i * 0.001),
                f"XRP_{i}": -0.6 - (i * 0.001),
            }
            cache.set(f"key_{i}", longs, shorts, strengths)

        assert cache.len() == 100

        # Verify random entries
        result = cache.get("key_50")
        assert result is not None
        assert "BTC_50" in result["longs"]
        assert "XRP_50" in result["shorts"]
        assert abs(result["strengths"]["BTC_50"] - 0.85) < 0.001

    def test_cache_update_existing_key(self):
        """Test updating an existing cache key."""
        cache = atc_rust.ScanCache()

        # Initial entry
        cache.set("key_1", {"BTC/USDT"}, set(), {"BTC/USDT": 0.8})
        result = cache.get("key_1")
        assert result["strengths"]["BTC/USDT"] == 0.8

        # Update entry
        cache.set("key_1", {"BTC/USDT", "ETH/USDT"}, set(), {"BTC/USDT": 0.9, "ETH/USDT": 0.75})
        result = cache.get("key_1")

        # Verify updated data
        assert len(result["longs"]) == 2
        assert result["strengths"]["BTC/USDT"] == 0.9
        assert result["strengths"]["ETH/USDT"] == 0.75


@pytest.mark.slow
class TestScanCacheThreadSafety:
    """Thread-safety tests (optional, requires threading)."""

    def test_concurrent_access(self):
        """Test concurrent read/write access."""
        import threading

        cache = atc_rust.ScanCache(capacity=1000, ttl_seconds=60.0)
        errors = []

        def writer(thread_id: int):
            try:
                for i in range(50):
                    key = f"thread_{thread_id}_key_{i}"
                    cache.set(key, {f"SYM_{i}"}, set(), {f"SYM_{i}": 0.8})
            except Exception as e:
                errors.append(e)

        def reader(thread_id: int):
            try:
                for i in range(50):
                    key = f"thread_{(thread_id - 5) % 10}_key_{i}"
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        # Create threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=writer if i < 5 else reader, args=(i,))
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0
        # Cache should have entries (exact count may vary due to LRU eviction)
        assert cache.len() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
