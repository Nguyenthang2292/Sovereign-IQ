"""
Example usage of Rust ScanCache for ATC Scanner.

This demonstrates how to use the high-performance Rust cache
in place of Python's caching.py.
"""
# -*- coding: utf-8 -*-

import time

from sovereign_prime import ScanCache


def example_basic_usage():
    """Basic cache operations."""
    print("=== Basic Usage ===")

    # Create cache
    cache = ScanCache(capacity=100, ttl_seconds=60.0)
    print(f"Created cache: {cache}")

    # Store some results
    longs = {"BTC/USDT", "ETH/USDT"}
    shorts = {"XRP/USDT"}
    strengths = {"BTC/USDT": 0.85, "ETH/USDT": 0.72, "XRP/USDT": -0.65}

    cache.set("BTC:1h", longs, shorts, strengths)
    print(f"Stored entry. Cache size: {cache.len()}/{cache.capacity()}")

    # Retrieve result
    result = cache.get("BTC:1h")
    if result:
        print(f"Cache hit! Longs: {result['longs']}")
        print(f"           Shorts: {result['shorts']}")
        print(f"           Strengths: {result['strengths']}")

    # Check if key exists
    if cache.contains("BTC:1h"):
        print("Key exists in cache [OK]")

    print()


def example_ttl_expiration():
    """Demonstrate TTL expiration."""
    print("=== TTL Expiration ===")

    # Create cache with short TTL
    cache = ScanCache(capacity=100, ttl_seconds=2.0)
    print("Created cache with 2s TTL")

    # Add entry
    cache.set("test_key", {"BTC/USDT"}, set(), {"BTC/USDT": 0.8})
    print("Added entry at t=0s")

    # Immediate access
    result = cache.get("test_key")
    print(f"Access at t=0s: {'Hit' if result else 'Miss'}")

    # Wait 1 second
    time.sleep(1.0)
    result = cache.get("test_key")
    print(f"Access at t=1s: {'Hit' if result else 'Miss'}")

    # Wait another 1.5 seconds (total 2.5s)
    time.sleep(1.5)
    result = cache.get("test_key")
    print(f"Access at t=2.5s: {'Hit (expired)' if result else 'Miss (expired) [OK]'}")

    print()


def example_lru_eviction():
    """Demonstrate LRU eviction."""
    print("=== LRU Eviction ===")

    # Create small cache
    cache = ScanCache(capacity=3, ttl_seconds=60.0)
    print("Created cache with capacity=3")

    # Fill cache
    for i in range(1, 4):
        cache.set(f"key_{i}", {f"SYM_{i}"}, set(), {f"SYM_{i}": 0.8})
        print(f"Added key_{i}. Size: {cache.len()}")

    # Add 4th entry (should evict oldest)
    cache.set("key_4", {"SYM_4"}, set(), {"SYM_4": 0.8})
    print(f"Added key_4. Size: {cache.len()}")

    # Check which key was evicted
    for i in range(1, 5):
        key = f"key_{i}"
        exists = cache.contains(key)
        print(f"{key}: {'Present' if exists else 'Evicted (LRU)'}")

    print()


def example_manual_cleanup():
    """Manual cleanup of expired entries."""
    print("=== Manual Cleanup ===")

    cache = ScanCache(capacity=100, ttl_seconds=1.0)

    # Add multiple entries
    for i in range(5):
        cache.set(f"old_{i}", {f"SYM_{i}"}, set(), {f"SYM_{i}": 0.8})

    print(f"Added 5 entries. Size: {cache.len()}")

    # Wait for expiration
    time.sleep(1.5)
    print("Waited 1.5s (TTL=1.0s)")

    # Add new entries
    for i in range(3):
        cache.set(f"new_{i}", {f"SYM_{i}"}, set(), {f"SYM_{i}": 0.8})

    print(f"Added 3 new entries. Size: {cache.len()}")

    # Manual cleanup
    removed = cache.remove_expired()
    print(f"Removed {removed} expired entries. Size: {cache.len()}")

    print()


def example_cache_statistics():
    """Track cache hit/miss statistics."""
    print("=== Cache Statistics ===")

    cache = ScanCache(capacity=100, ttl_seconds=60.0)

    # Simulate cache operations
    hits = 0
    misses = 0

    # Populate cache
    for i in range(10):
        cache.set(f"sym_{i}", {f"SYM_{i}"}, set(), {f"SYM_{i}": 0.8})

    # Simulate lookups
    for i in range(20):
        key = f"sym_{i % 15}"  # Some hits, some misses
        result = cache.get(key)
        if result:
            hits += 1
        else:
            misses += 1

    hit_rate = hits / (hits + misses) * 100
    print(f"Hits: {hits}, Misses: {misses}")
    print(f"Hit rate: {hit_rate:.1f}%")
    print(f"Cache size: {cache.len()}/{cache.capacity()}")

    print()


def example_atc_scanner_integration():
    """Simulate ATCScanner integration."""
    print("=== ATCScanner Integration Simulation ===")

    # Create cache with realistic parameters
    cache = ScanCache(capacity=500, ttl_seconds=60.0)

    def scan_symbols_with_cache(symbols, timeframe):
        """Simulate scanning with cache."""
        # Create cache key
        cache_key = f"{','.join(sorted(symbols))}:{timeframe}"

        # Check cache
        cached = cache.get(cache_key)
        if cached:
            print(f"  Cache HIT for {cache_key}")
            return cached

        # Cache miss - simulate scan
        print(f"  Cache MISS for {cache_key} - performing scan...")
        time.sleep(0.01)  # Simulate scan delay

        # Store result
        longs = {sym for sym in symbols if hash(sym) % 2 == 0}
        shorts = {sym for sym in symbols if hash(sym) % 2 == 1}
        strengths = {sym: 0.8 if sym in longs else -0.6 for sym in symbols}

        cache.set(cache_key, longs, shorts, strengths)
        return {"longs": longs, "shorts": shorts, "strengths": strengths}

    # Simulate multiple scans
    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]

    for timeframe in ["1h", "15m", "5m"]:
        print(f"\nScanning {timeframe}:")
        result = scan_symbols_with_cache(symbols, timeframe)
        print(f"  Result: {len(result['longs'])} longs, {len(result['shorts'])} shorts")

    # Repeat scan (should hit cache)
    print("\nRe-scanning 1h (should hit cache):")
    result = scan_symbols_with_cache(symbols, "1h")

    print(f"\nFinal cache stats: {cache}")

    print()


def main():
    """Run all examples."""
    print("=" * 60)
    print("Rust ScanCache Examples")
    print("=" * 60)
    print()

    example_basic_usage()
    example_ttl_expiration()
    example_lru_eviction()
    example_manual_cleanup()
    example_cache_statistics()
    example_atc_scanner_integration()

    print("=" * 60)
    print("All examples completed successfully! [OK]")
    print("=" * 60)


if __name__ == "__main__":
    main()
