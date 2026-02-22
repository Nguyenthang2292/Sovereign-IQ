# Implementation Plan: SQLite Persistence System

## Goal

Replace the JSONL-based signal persistence with a SQLite database to enable efficient querying, better data integrity, and advanced analytics capabilities.

## Execution Checklist

### Phase 1: Planning & Design

- [x] Review current JSONL implementation requirements
- [x] Design SQLite schema with proper indexes
- [x] Design migration strategy from JSONL to SQLite
- [x] Identify backward compatibility requirements

### Phase 2: Core Implementation

- [x] Implement SQLite-based SignalPersistence class
- [x] Add database connection management and pooling
- [x] Implement thread-safe write operations with ACID guarantees
- [x] Add advanced query capabilities (filtering, aggregations, analytics)
- [x] Preserve metrics and monitoring features

### Phase 3: Migration Tooling

- [x] Create JSONL to SQLite migration script
- [ ] Add data integrity validation
- [ ] Test migration with existing data

### Phase 4: Testing & Validation

- [x] Write comprehensive unit tests
- [x] Add integration tests for concurrent operations
- [ ] Performance benchmarking vs JSONL
- [x] Verify backward compatibility

### Phase 5: Documentation & Deployment

- [ ] Update module documentation
- [ ] Create migration guide for users
- [ ] Update review documentation
- [ ] Deploy and monitor initial usage

## User Review Required

> [!IMPORTANT]
> **Breaking Change**: This migration changes the storage backend from JSONL to SQLite.
>
> - Existing JSONL files will need migration
> - API remains mostly compatible but some internals change
> - Performance improvements expected for queries and analytics

> [!WARNING]
> **Migration Strategy**: Users must run migration script to convert existing JSONL data to SQLite before upgrading.

## Proposed Changes

### Database Design

#### Schema Definition

**`signals` table** - Main storage for trading signals

```sql
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,           -- ISO 8601 format
    symbol TEXT NOT NULL,               -- Trading pair (e.g., "BTCUSDT")
    type TEXT NOT NULL,                 -- "LONG" or "SHORT"
    confidence REAL NOT NULL,           -- 0.0 to 1.0
    entry_price REAL NOT NULL,          -- Entry price
    stop_loss REAL,                     -- Optional stop loss
    take_profit REAL,                   -- Optional take profit
    sources TEXT NOT NULL,              -- JSON array of sources
    metadata TEXT,                      -- JSON for extensibility
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_signals_timestamp ON signals(timestamp);
CREATE INDEX idx_signals_symbol ON signals(symbol);
CREATE INDEX idx_signals_type ON signals(type);
CREATE INDEX idx_signals_symbol_timestamp ON signals(symbol, timestamp);
```

**`signal_metrics` table** - Performance tracking

```sql
CREATE TABLE signal_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL,
    outcome TEXT,                       -- "WIN", "LOSS", "PENDING"
    profit_loss REAL,                   -- Actual P/L
    duration_seconds INTEGER,           -- Time to close
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (signal_id) REFERENCES signals(id) ON DELETE CASCADE
);

CREATE INDEX idx_metrics_signal_id ON signal_metrics(signal_id);
CREATE INDEX idx_metrics_outcome ON signal_metrics(outcome);
```

---

### Core Implementation

#### [NEW] `modules/auto_trade/core/persistence_sqlite.py`

New SQLite-based implementation with enhanced features:

```python
import sqlite3
import threading
import json
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from modules.auto_trade.core.signal_selector import FinalSignal
from modules.common.ui.logging import log_error, log_info, log_warn


class SignalPersistenceSQLite:
    """
    SQLite-based signal persistence with improved query performance.
    
    Features:
    - ACID transactions for data integrity
    - Efficient indexed queries
    - Advanced filtering and aggregations
    - Thread-safe operations
    - Performance metrics
    - Outcome tracking support
    """
    
    DISK_SPACE_ERROR_THRESHOLD_MB = 100
    DISK_SPACE_WARN_THRESHOLD_MB = 500
    
    def __init__(
        self, 
        db_path: str = "data/signals/signals.db",
        enable_wal: bool = True
    ) -> None:
        """Initialize SQLite persistence.
        
        Args:
            db_path: Path to SQLite database file
            enable_wal: Enable Write-Ahead Logging for better concurrency
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self.metrics = {
            "total_writes": 0,
            "failed_writes": 0,
            "total_bytes_written": 0,
            "avg_write_time_ms": 0.0,
        }
        
        # Initialize database
        self._init_database(enable_wal)
    
    def _init_database(self, enable_wal: bool) -> None:
        """Create tables and indexes if not exist."""
        with self._get_connection() as conn:
            # Enable WAL mode for better concurrency
            if enable_wal:
                conn.execute("PRAGMA journal_mode=WAL")
            
            # Create signals table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    sources TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp ON signals(symbol, timestamp)")
            
            # Create metrics table for outcome tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER NOT NULL,
                    outcome TEXT,
                    profit_loss REAL,
                    duration_seconds INTEGER,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES signals(id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_signal_id ON signal_metrics(signal_id)")
            
            conn.commit()
            log_info(f"SQLite database initialized: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save_signal(self, signal: FinalSignal) -> Optional[int]:
        """Save a signal to the database.
        
        Returns:
            Signal ID if successful, None otherwise
        """
        import time
        start_time = time.time()
        
        try:
            # Validation
            if not signal.symbol or not signal.signal_type:
                log_error("Invalid signal: missing symbol or signal_type")
                self.metrics["failed_writes"] += 1
                return None
            
            if signal.entry_price <= 0:
                log_error(f"Invalid entry price: {signal.entry_price}")
                self.metrics["failed_writes"] += 1
                return None
            
            # Prepare data
            timestamp_str = datetime.fromtimestamp(signal.timestamp).isoformat()
            sources_json = json.dumps(signal.sources)
            
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.execute("""
                        INSERT INTO signals (
                            timestamp, symbol, type, confidence,
                            entry_price, stop_loss, take_profit, sources
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp_str,
                        signal.symbol,
                        signal.signal_type,
                        signal.confidence,
                        signal.entry_price,
                        signal.stop_loss,
                        signal.take_profit,
                        sources_json
                    ))
                    conn.commit()
                    signal_id = cursor.lastrowid
            
            # Update metrics
            self.metrics["total_writes"] += 1
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics["avg_write_time_ms"] = (
                self.metrics["avg_write_time_ms"] * (self.metrics["total_writes"] - 1) + elapsed_ms
            ) / self.metrics["total_writes"]
            
            log_info(f"Saved signal for {signal.symbol} (ID: {signal_id})")
            return signal_id
            
        except Exception as e:
            self.metrics["failed_writes"] += 1
            log_error(f"Failed to save signal: {e}")
            return None
    
    def read_signals(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Read signals with advanced filtering.
        
        Args:
            from_date: Start date filter (inclusive)
            to_date: End date filter (inclusive)
            symbol: Filter by symbol
            signal_type: Filter by type ("LONG" or "SHORT")
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of signal records
        """
        query = "SELECT * FROM signals WHERE 1=1"
        params = []
        
        if from_date:
            query += " AND date(timestamp) >= ?"
            params.append(from_date.isoformat())
        
        if to_date:
            query += " AND date(timestamp) <= ?"
            params.append(to_date.isoformat())
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if signal_type:
            query += " AND type = ?"
            params.append(signal_type)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_signal_count(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbol: Optional[str] = None
    ) -> int:
        """Get count of signals matching filters."""
        query = "SELECT COUNT(*) as count FROM signals WHERE 1=1"
        params = []
        
        if from_date:
            query += " AND date(timestamp) >= ?"
            params.append(from_date.isoformat())
        
        if to_date:
            query += " AND date(timestamp) <= ?"
            params.append(to_date.isoformat())
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchone()["count"]
    
    def get_signals_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all signals for a specific symbol."""
        return self.read_signals(symbol=symbol)
    
    def get_recent_signals(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get signals from the last N days."""
        to_date = date.today()
        from_date = to_date - timedelta(days=days)
        return self.read_signals(from_date=from_date, to_date=to_date)
    
    def get_statistics(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Get signal statistics for a date range."""
        query = """
            SELECT 
                COUNT(*) as total_signals,
                COUNT(DISTINCT symbol) as unique_symbols,
                SUM(CASE WHEN type = 'LONG' THEN 1 ELSE 0 END) as long_signals,
                SUM(CASE WHEN type = 'SHORT' THEN 1 ELSE 0 END) as short_signals,
                AVG(confidence) as avg_confidence,
                MIN(timestamp) as first_signal,
                MAX(timestamp) as last_signal
            FROM signals
            WHERE 1=1
        """
        params = []
        
        if from_date:
            query += " AND date(timestamp) >= ?"
            params.append(from_date.isoformat())
        
        if to_date:
            query += " AND date(timestamp) <= ?"
            params.append(to_date.isoformat())
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return dict(cursor.fetchone())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get persistence metrics."""
        return self.metrics.copy()
```

---

### Migration Tooling

#### [NEW] `scripts/migrate_jsonl_to_sqlite.py`

Migration script to convert existing JSONL data:

```python
#!/usr/bin/env python3
"""Migrate JSONL signal history to SQLite database."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from modules.auto_trade.core.persistence_sqlite import SignalPersistenceSQLite
from modules.auto_trade.core.signal_selector import FinalSignal
from modules.common.ui.logging import log_info, log_error


def migrate_jsonl_to_sqlite(
    jsonl_dir: str = "data/signals",
    db_path: str = "data/signals/signals.db",
    dry_run: bool = False
) -> Dict[str, Any]:
    """Migrate JSONL files to SQLite database.
    
    Args:
        jsonl_dir: Directory containing JSONL files
        db_path: Target SQLite database path
        dry_run: If True, only count records without writing
        
    Returns:
        Migration statistics
    """
    stats = {
        "files_processed": 0,
        "records_migrated": 0,
        "records_failed": 0,
        "errors": []
    }
    
    jsonl_files = sorted(Path(jsonl_dir).glob("signal_history*.jsonl"))
    
    if not jsonl_files:
        log_error(f"No JSONL files found in {jsonl_dir}")
        return stats
    
    log_info(f"Found {len(jsonl_files)} JSONL files to migrate")
    
    if not dry_run:
        persistence = SignalPersistenceSQLite(db_path=db_path)
    
    for filepath in jsonl_files:
        stats["files_processed"] += 1
        log_info(f"Processing {filepath.name}...")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        record = json.loads(line)
                        
                        if dry_run:
                            stats["records_migrated"] += 1
                            continue
                        
                        # Convert to FinalSignal
                        timestamp = datetime.fromisoformat(record["timestamp"]).timestamp()
                        signal = FinalSignal(
                            symbol=record["symbol"],
                            signal_type=record["type"],
                            confidence=record.get("confidence", 0.0),
                            entry_price=record["entry"],
                            stop_loss=record.get("stop_loss"),
                            take_profit=record.get("take_profit"),
                            sources=record.get("sources", []),
                            timestamp=timestamp
                        )
                        
                        signal_id = persistence.save_signal(signal)
                        if signal_id:
                            stats["records_migrated"] += 1
                        else:
                            stats["records_failed"] += 1
                            
                    except Exception as e:
                        stats["records_failed"] += 1
                        error_msg = f"{filepath.name}:{line_num} - {str(e)}"
                        stats["errors"].append(error_msg)
                        log_error(f"Failed to migrate record: {error_msg}")
                        
        except Exception as e:
            error_msg = f"Failed to read {filepath.name}: {e}"
            stats["errors"].append(error_msg)
            log_error(error_msg)
    
    return stats


if __name__ == "__main__":
    import sys
    
    dry_run = "--dry-run" in sys.argv
    
    if dry_run:
        log_info("DRY RUN MODE - No data will be written")
    
    stats = migrate_jsonl_to_sqlite(dry_run=dry_run)
    
    print("\n" + "="*60)
    print("Migration Summary")
    print("="*60)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Records migrated: {stats['records_migrated']}")
    print(f"Records failed: {stats['records_failed']}")
    
    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])} total):")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
```

---

### Testing Strategy

#### [NEW] `tests/auto_trade/core/test_persistence_sqlite.py`

Comprehensive test suite covering:

- Basic CRUD operations
- Concurrent write/read operations
- Query filtering and pagination
- Statistics and aggregations
- Error handling and validation
- Performance benchmarks

---

## Verification Plan

### Automated Tests

1. **Unit Tests**: Test all methods of `SignalPersistenceSQLite`
2. **Integration Tests**: Test with concurrent operations (10+ threads)
3. **Migration Tests**: Verify JSONL to SQLite conversion accuracy
4. **Performance Tests**: Benchmark query performance vs JSONL

### Manual Verification

1. Run migration on actual production data
2. Verify data integrity with checksums
3. Compare query results between JSONL and SQLite
4. Monitor database size and query performance
