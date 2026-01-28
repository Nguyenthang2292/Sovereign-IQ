"""Debug cache key generation."""

import pandas as pd
import numpy as np
import hashlib
from modules.adaptive_trend_LTS.utils.cache_manager import get_cache_manager, reset_cache_manager, CacheEntry

# Test key generation
prices = pd.Series(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float))


# Test 1: Generate key twice to see if it's consistent
class TestCacheManager:
    def _generate_key(self, ma_type: str, length: int, price_data):
        data_raw = price_data.values.tobytes()
        data_hash = hashlib.md5(data_raw).hexdigest()[:16]
        key_parts = [f"ma={ma_type}", f"len={length}", f"d={data_hash}"]
        return "|".join(key_parts)

    def __init__(self):
        self._l1_cache = {}
        self._l2_cache = {}
        self._hits_l1 = 0
        self._hits_l2 = 0
        self._misses = 0

    def put(self, key, value):
        size = len(str(value))
        entry = CacheEntry(key=key, value=value, timestamp=0, hits=1, size_bytes=size)
        self._l1_cache[key] = entry
        self._l2_cache[key] = entry
        print(f"PUT: key={key}, L1 size={len(self._l1_cache)}, L2 size={len(self._l2_cache)}")

    def get(self, key):
        print(f"GET: key={key}, L1 has={key in self._l1_cache}, L2 has={key in self._l2_cache}")
        if key in self._l1_cache:
            self._hits_l1 += 1
            return self._l1_cache[key].value
        if key in self._l2_cache:
            self._hits_l2 += 1
            return self._l2_cache[key].value
        self._misses += 1
        return None


cm = TestCacheManager()
key1 = cm._generate_key("EMA", 5, prices)
key2 = cm._generate_key("EMA", 5, prices)

print(f"Key 1: {key1}")
print(f"Key 2: {key2}")
print(f"Keys match: {key1 == key2}")

# Test put then get
cm.put(key1, "test_value1")
result = cm.get(key1)
print(f"Result after get: {result}")
print(f"L1 cache: {list(cm._l1_cache.keys())}")
print(f"L2 cache: {list(cm._l2_cache.keys())}")
