# Legacy Code - Deprecated

This folder contains deprecated modules that are no longer actively used in the auto_trade system.

## Modules

### `caching.py` (Deprecated: 2026-02-02)

- **Replaced by**: Rust ScanCache in `atc_scanner.py`
- **Reason**: Rust implementation provides 10-20x performance improvement
- **Status**: ⚠️ Still imported in `signal_pipeline.py` but can be removed
- **Migration**: Replace `Cache()` with Rust ScanCache or remove entirely

### `persistence.py` (Deprecated: 2026-02-02)

- **Replaced by**: SQLite-based `persistence_sqlite.py`
- **Reason**: SQLite provides indexed queries, analytics, and better performance
- **Status**: ⚠️ Still imported in `signal_pipeline.py`
- **Migration**: Use `SignalPersistenceSQLite` instead of `SignalPersistence`

## Migration Guide

### Migrating from `Cache` to Rust ScanCache

```python
# OLD (legacy)
from modules.auto_trade.core.caching import Cache
cache = Cache()
cache.set("key", value, ttl=300)
result = cache.get("key")

# NEW (Rust)
# Use ATCScanner's built-in Rust cache
# Or remove generic caching entirely
```

### Migrating from `SignalPersistence` to `SignalPersistenceSQLite`

```python
# OLD (legacy)
from modules.auto_trade.core.persistence import SignalPersistence
persistence = SignalPersistence(storage_dir="data/signals")

# NEW (SQLite)
from modules.auto_trade.core.persistence_sqlite import SignalPersistenceSQLite
persistence = SignalPersistenceSQLite(db_path="data/signals/signals.db")
```

## Removal Timeline

- **2026-02-02**: Moved to legacy folder
- **2026-03-02** (1 month): Add deprecation warnings
- **2026-05-02** (3 months): Remove completely (breaking change)

## Need Help?

Refer to:

- [`optimization_recommendations.md`](../docs/core/optimization_recommendations.md) - Full optimization analysis
- [`persistence_sqlite_implementation_plan.md`](../docs/core/persistence_sqlite_implementation_plan.md) - SQLite migration guide
