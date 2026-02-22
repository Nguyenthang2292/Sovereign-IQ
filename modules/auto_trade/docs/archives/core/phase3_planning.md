# Phase 3 Planning: Rust ScanCache Optional Optimizations

**Document Version**: 1.0
**Date**: 2026-02-01
**Status**: Planning (Not Started)
**Prerequisites**: Phase 2 Complete ✅

---

## Overview

Phase 3 consists of **optional optimizations** that should only be pursued based on production metrics and demonstrated needs. This document provides a roadmap for evaluating and implementing each optimization.

## Executive Summary

| Task | Priority | Effort | Risk | ROI | Recommendation |
|------|----------|--------|------|-----|----------------|
| 3.1 Remove Python Cache | Medium | Small (2-4h) | Low | High | Proceed after 1-2 weeks monitoring |
| 3.2 Monitoring & Metrics | High | Medium (6-8h) | Low | High | **Start immediately** |
| 3.3 Async/Await (Tokio) | Low | Large (16-24h) | High | Unknown | Only if metrics show GIL contention >20% |
| 3.4 Disk Persistence | Low | Medium (10-14h) | Medium | Low | Only if warm-up delay >10% impact |

**Key Insight**: Not all Phase 3 tasks provide value. Use production metrics to guide decisions.

---

## 3.1 Remove Python Cache Dependency

### Motivation
- Simplify codebase by removing dual cache implementation
- Reduce maintenance burden
- Eliminate conditional logic in cache operations

### Prerequisites
- ✅ Phase 2 complete
- 🕐 1-2 weeks of production monitoring showing:
  - 99%+ Rust cache usage
  - <0.1% fallback to Python cache
  - No Rust cache initialization failures

### Implementation Plan

#### Step 1: Production Monitoring (Week 1-2)
```python
# Add to ATCScanner initialization
log_info(f"ATCScanner: Cache type={scanner._use_rust_cache ? 'rust' : 'python'}")

# Add to _get_cached_result
if not self._use_rust_cache:
    log_warn("ATCScanner: Using Python cache fallback (investigate why)")
```

**Metrics to Track**:
- Cache type on initialization (Rust vs Python)
- Fallback occurrences per day
- Rust cache initialization errors

**Decision Gate**: If Rust cache usage <99% or fallback rate >0.1%, investigate root cause before proceeding.

#### Step 2: Deprecation Warning (Week 3)
```python
# In ATCScanner.__init__
if not use_rust_cache_config:
    log_warn(
        "Python cache is DEPRECATED and will be removed in a future version. "
        "Please ensure Rust ScanCache (sovereign_prime) is installed."
    )
```

#### Step 3: Code Removal (Week 4)
**Files to Modify**:
1. `modules/auto_trade/core/atc_scanner.py`:
   - Remove lines 309-316 (Python cache in `_set_cache`)
   - Remove lines 274-284 (Python cache in `_get_cached_result`)
   - Remove `self._cache` and `self._cache_lock` from `__init__`
   - Simplify `cache_stats()` to only return Rust stats
   - Remove `use_rust_cache` config handling

2. `config/auto_trade.py`:
   - Remove `use_rust_cache` from `ATC_SCANNER_DEFAULTS`

3. `tests/auto_trade/core/test_atc_scanner.py`:
   - Remove Python-specific cache tests (3 tests)
   - Keep Rust cache tests (9 tests)

**Expected Diff**: ~50 lines removed, ~20 lines simplified

#### Step 4: Testing
```bash
# Run all ATCScanner tests
pytest tests/auto_trade/core/test_atc_scanner.py -v

# Run cache-specific tests
pytest tests/auto_trade/core/test_atc_scanner.py::TestATCScannerCache -v

# Run integration tests
pytest tests/auto_trade/ -k "scanner" -v
```

**Acceptance Criteria**:
- ✅ All tests pass (9/12 cache tests remain)
- ✅ No Python cache code remains
- ✅ Rust cache always used (no fallback)
- ✅ Documentation updated

### Rollback Plan
If issues arise:
1. Revert commit removing Python cache
2. Set `use_rust_cache=False` in production config
3. Investigate Rust cache issue
4. Re-attempt after fix

### Estimated Timeline
- Week 1-2: Production monitoring
- Week 3: Add deprecation warnings
- Week 4: Remove code and test
- **Total**: 3-4 weeks (mostly monitoring time)

### Risk Assessment
- **Risk Level**: Low
- **Impact of Failure**: Medium (cache stops working)
- **Mitigation**: Keep Python cache in git history, easy rollback

---

## 3.2 Production Monitoring and Metrics

### Motivation
- Visibility into cache performance
- Early detection of performance degradation
- Data-driven optimization decisions
- Support for Phase 3 decision-making

### Prerequisites
None - can start immediately

### Implementation Plan

#### Step 1: Rust Metrics Struct (2 hours)

**File**: `rust_backend/src/atc_scanner_rs.rs`

```rust
/// Cache performance metrics
#[pyclass]
#[derive(Clone)]
pub struct CacheMetrics {
    #[pyo3(get)]
    pub total_hits: u64,
    #[pyo3(get)]
    pub total_misses: u64,
    #[pyo3(get)]
    pub evictions: u64,
    #[pyo3(get)]
    pub expirations: u64,
    #[pyo3(get)]
    pub avg_hit_latency_us: f64,
    #[pyo3(get)]
    pub avg_miss_latency_us: f64,
}

impl ScanCache {
    /// Get cache metrics
    pub fn get_metrics(&self) -> PyResult<CacheMetrics> {
        let cache = self.cache.read().unwrap();
        Ok(self.metrics.lock().unwrap().clone())
    }

    /// Reset metrics (for testing)
    pub fn reset_metrics(&self) {
        let mut metrics = self.metrics.lock().unwrap();
        *metrics = CacheMetrics::default();
    }
}
```

**Test**: `tests/auto_trade/core/test_scan_cache.py`
```python
def test_cache_metrics():
    cache = ScanCache(capacity=100)

    # Generate some cache operations
    cache.set("key1", {"BTC/USDT"}, set(), {"BTC/USDT": 0.8})
    cache.get("key1")  # Hit
    cache.get("nonexistent")  # Miss

    metrics = cache.get_metrics()
    assert metrics["total_hits"] == 1
    assert metrics["total_misses"] == 1
    assert metrics["avg_hit_latency_us"] > 0
```

#### Step 2: ATCScanner Integration (2 hours)

**File**: `modules/auto_trade/core/atc_scanner.py`

```python
def get_cache_metrics(self) -> Dict[str, Any]:
    """Get comprehensive cache metrics.

    Returns:
        Dict with cache performance metrics
    """
    if self._use_rust_cache and self._rust_cache:
        metrics = self._rust_cache.get_metrics()
        return {
            "type": "rust",
            "total_operations": metrics["total_hits"] + metrics["total_misses"],
            "hit_rate": metrics["total_hits"] / (metrics["total_hits"] + metrics["total_misses"] + 1e-9),
            **metrics
        }
    else:
        return {
            "type": "python",
            "total_operations": 0,
            "hit_rate": 0.0,
        }

def _log_cache_metrics_periodic(self):
    """Log cache metrics every 100 scans."""
    self._scan_count += 1
    if self._scan_count % 100 == 0:
        metrics = self.get_cache_metrics()
        log_info(
            f"ATCScanner Cache Metrics [scans={self._scan_count}]: "
            f"hit_rate={metrics['hit_rate']:.2%}, "
            f"hits={metrics['total_hits']}, "
            f"misses={metrics['total_misses']}, "
            f"evictions={metrics['evictions']}"
        )
```

#### Step 3: Prometheus Integration (Optional, 2 hours)

**File**: `modules/auto_trade/core/metrics.py`

```python
from prometheus_client import Counter, Histogram, Gauge

# Cache metrics
cache_hits = Counter('atc_cache_hits_total', 'Total cache hits')
cache_misses = Counter('atc_cache_misses_total', 'Total cache misses')
cache_hit_latency = Histogram('atc_cache_hit_latency_seconds', 'Cache hit latency')
cache_size = Gauge('atc_cache_size', 'Current cache size')

def export_cache_metrics(scanner: ATCScanner):
    """Export cache metrics to Prometheus."""
    metrics = scanner.get_cache_metrics()

    cache_hits._value.set(metrics['total_hits'])
    cache_misses._value.set(metrics['total_misses'])
    cache_size.set(metrics['size'])
```

#### Step 4: Performance Regression Tests (2 hours)

**File**: `tests/performance/test_cache_regression.py`

```python
import pytest
import json
from pathlib import Path
from sovereign_prime import ScanCache

METRICS_FILE = Path("tests/performance/cache_metrics_history.json")

def load_baseline():
    """Load baseline metrics from history."""
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            return json.load(f)
    return None

def save_metrics(metrics):
    """Save current metrics to history."""
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)

def test_cache_performance_regression():
    """Ensure cache performance doesn't regress."""
    cache = ScanCache(capacity=1000)

    # Benchmark operations
    import time
    times = []
    for i in range(10000):
        start = time.perf_counter()
        cache.set(f"key{i % 100}", {"BTC"}, set(), {"BTC": 0.8})
        cache.get(f"key{i % 100}")
        times.append(time.perf_counter() - start)

    avg_latency_us = (sum(times) / len(times)) * 1_000_000

    # Compare with baseline
    baseline = load_baseline()
    if baseline:
        baseline_latency = baseline['avg_latency_us']
        regression = (avg_latency_us - baseline_latency) / baseline_latency

        assert regression < 0.10, (
            f"Performance regression detected: "
            f"{regression:.1%} slower than baseline "
            f"({avg_latency_us:.2f} µs vs {baseline_latency:.2f} µs)"
        )

    # Save new baseline
    save_metrics({'avg_latency_us': avg_latency_us})
```

### Estimated Timeline
- Day 1: Rust metrics struct (2h)
- Day 2: ATCScanner integration (2h)
- Day 3: Prometheus integration (2h, optional)
- Day 4: Regression tests (2h)
- **Total**: 4 days (6-8 hours)

### Acceptance Criteria
- ✅ Metrics accessible via `scanner.get_cache_metrics()`
- ✅ Metrics logged every 100 scans
- ✅ Performance regression tests in CI
- ✅ (Optional) Prometheus metrics exportable

---

## 3.3 Async/Await Support (Tokio)

### Motivation
- Eliminate Python GIL blocking in cache operations
- Enable true concurrent cache access
- Improve throughput under high concurrency

### Prerequisites
- ✅ Phase 3.2 metrics implemented
- 🕐 Production metrics showing one of:
  - GIL contention >20%
  - Thread blocking time >50ms p95
  - Clear async benefit demonstrated

### Decision Gate: Should We Proceed?

**Evaluation Criteria**:
```python
# Monitor GIL contention
import sys
gil_switches = sys.getswitchinterval()  # Default 0.005s = 5ms

# If seeing:
# - Thread wait times >50ms
# - CPU usage <80% despite load
# - Lock contention in profiling
# → Consider async implementation

# Otherwise: Skip (not worth complexity)
```

### Implementation Plan (Only if Decision Gate Passed)

#### Design Considerations
1. **Tokio Runtime**:
   - Multi-threaded or current-thread?
   - Runtime lifecycle management
2. **PyO3 Async Bridge**:
   - Use `pyo3-asyncio` crate
   - Handle async/sync API coexistence
3. **API Design**:
   - Keep sync API for backward compatibility
   - Add `AsyncScanCache` as separate class

#### Architecture
```
Python asyncio
    ↓
pyo3-asyncio bridge
    ↓
Tokio async runtime
    ↓
AsyncScanCache (tokio::sync::RwLock)
```

#### Step 1: Proof of Concept (4 hours)
- Add tokio dependency to Cargo.toml
- Create minimal async cache with 2 methods
- Benchmark async vs sync with 100 concurrent tasks
- **Decision Gate**: Only proceed if >30% improvement

#### Step 2-6: Full Implementation (12-20 hours)
- See detailed subtasks in scan_cache_implementation_summary.md

### Risk Assessment
- **Risk Level**: High
- **Complexity**: Very High
- **Maintenance Burden**: High (two implementations to maintain)
- **Recommendation**: **Only pursue if clear need demonstrated**

---

## 3.4 Disk Persistence (Optional)

### Motivation
- Faster cache warm-up after restart
- Preserve cache across service restarts
- Reduce initial scan load on startup

### Prerequisites
- ✅ Phase 3.2 metrics implemented
- 🕐 Metrics showing:
  - Cache warm-up time >30 seconds
  - Hit rate <50% in first 5 minutes
  - Warm-up causing >10% performance drop

### Decision Gate: Is Persistence Needed?

**Evaluation**:
```python
# Measure warm-up time
start_time = time.time()
scanner = ATCScanner(data_fetcher)
results = scanner.scan_symbols(symbols)  # First scan (cold cache)
warm_up_time = time.time() - start_time

# Measure hit rate during warm-up
metrics_5min = scanner.get_cache_metrics()  # After 5 minutes
initial_hit_rate = metrics_5min['hit_rate']

# Decision criteria:
# - warm_up_time >30s AND
# - initial_hit_rate <50%
# → Consider persistence

# Otherwise: Skip (warm-up is acceptable)
```

### Implementation Plan (Only if Decision Gate Passed)

#### Design Choices
1. **Serialization Format**:
   - Bincode: Fastest, not human-readable
   - MessagePack: Fast, portable
   - JSON: Slow, human-readable (debugging)
   - **Recommendation**: Bincode for production

2. **Persistence Strategy**:
   - Snapshot on shutdown (simple)
   - Periodic snapshots (more resilient)
   - Write-through (complex, rarely needed)
   - **Recommendation**: Snapshot on shutdown

3. **Cache Invalidation**:
   - Check TTL on load
   - Validate entry versions
   - Discard incompatible formats

#### Implementation Steps (10-14 hours)
- See detailed subtasks in scan_cache_implementation_summary.md

### Risk Assessment
- **Risk Level**: Medium
- **Complexity**: Medium
- **File Corruption Risk**: Low (with versioning)
- **Recommendation**: Only implement if warm-up is a real problem

---

## Phase 3 Decision Framework

### Recommended Decision Process

```
┌─────────────────────────────────────┐
│  Phase 2 Complete ✅                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Start 3.2: Add Monitoring          │
│  [HIGH PRIORITY, Always Do This]    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Monitor Production for 2-4 Weeks   │
│  Collect: hit rate, latency, errors │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│  Evaluate Metrics:                                  │
│                                                     │
│  Q1: Rust cache stable (>99% usage, <0.1% errors)? │
│      YES → Proceed with 3.1 (Remove Python cache)  │
│      NO  → Investigate issues, wait longer         │
│                                                     │
│  Q2: GIL contention >20%?                           │
│      YES → Consider 3.3 (Async support)             │
│      NO  → Skip 3.3                                 │
│                                                     │
│  Q3: Warm-up delay >10% performance impact?         │
│      YES → Consider 3.4 (Persistence)               │
│      NO  → Skip 3.4                                 │
└─────────────────────────────────────────────────────┘
```

### Priority Matrix

| Scenario | Metrics | Recommended Actions |
|----------|---------|---------------------|
| **Best Case** | High hit rate, low latency, stable | 3.2 ✅, 3.1 ✅, Skip 3.3/3.4 |
| **Needs Optimization** | Low hit rate, high latency | 3.2 ✅, Investigate caching strategy, Wait on 3.1 |
| **GIL Bottleneck** | High GIL contention | 3.2 ✅, 3.3 (Async) ⚠️ |
| **Slow Warm-up** | Long initial latency | 3.2 ✅, 3.4 (Persistence) ⚠️ |
| **Unstable** | High error rate | Fix issues first, delay all Phase 3 |

---

## Success Metrics

### Phase 3 Completion Criteria

**3.1 (Remove Python Cache)**:
- ✅ Python cache code removed
- ✅ All tests pass
- ✅ No production issues for 1 week
- ✅ Rollback plan documented

**3.2 (Monitoring)**:
- ✅ Metrics dashboard operational
- ✅ Automated performance regression tests
- ✅ Metrics logged and exportable

**3.3 (Async)** - Only if pursued:
- ✅ >30% throughput improvement demonstrated
- ✅ All async tests pass
- ✅ Backward compatibility maintained

**3.4 (Persistence)** - Only if pursued:
- ✅ Warm-up time reduced >50%
- ✅ Cache survives restarts
- ✅ No file corruption in testing

---

## Timeline and Resource Allocation

### Minimum Viable Phase 3
- **3.2 Monitoring**: 6-8 hours (1 week with monitoring)
- **3.1 Remove Python Cache**: 2-4 hours (after monitoring period)
- **Total**: 8-12 hours over 3-4 weeks

### Maximum Phase 3 (All Tasks)
- **3.2 Monitoring**: 6-8 hours
- **3.1 Remove Python Cache**: 2-4 hours
- **3.3 Async Support**: 16-24 hours
- **3.4 Persistence**: 10-14 hours
- **Total**: 34-50 hours over 6-8 weeks

### Recommended Approach
1. **Week 1**: Implement 3.2 (Monitoring)
2. **Week 2-4**: Monitor production, collect metrics
3. **Week 5**: Evaluate metrics, decide on 3.1
4. **Week 6**: Implement 3.1 if metrics support it
5. **Week 7+**: Only pursue 3.3/3.4 if metrics show clear need

---

## Documentation and Communication

### Stakeholder Communication
- **Engineering Team**: Share metrics dashboard, explain decision criteria
- **Operations Team**: Provide monitoring runbook
- **Leadership**: Report on performance improvements and stability

### Documentation Updates
- ✅ Update CLAUDE.md with Phase 3 status
- ✅ Update API documentation if APIs change
- ✅ Create monitoring runbook for ops team
- ✅ Document rollback procedures

---

## Conclusion

Phase 3 is **metrics-driven and optional**. The only mandatory task is **3.2 (Monitoring)**, which provides visibility and enables data-driven decisions for other optimizations.

**Key Principles**:
1. **Measure before optimizing** - Implement 3.2 first
2. **Justify with metrics** - Only pursue optimizations that solve real problems
3. **Keep it simple** - Don't add complexity without clear benefit
4. **Be conservative** - Rust cache is already fast, don't over-engineer

**Next Steps**:
1. Review this planning document
2. Implement 3.2 (Monitoring) immediately
3. Monitor production for 2-4 weeks
4. Reconvene to evaluate metrics and decide on next steps

---

**Document Owner**: Development Team
**Last Updated**: 2026-02-01
**Next Review**: After 2-4 weeks of production monitoring
