# Auto-Trade Core Module: Optimization Recommendations

## Document Status

**Last Updated**: 2026-02-02
**Phase 2 Completion**: ✅ **COMPLETE** (2026-02-01)
**Related Documents**:
- `modules/auto_trade/docs/core/scan_cache_implementation_summary.md` - Rust ScanCache implementation (Phase 1-2 complete)
- `modules/auto_trade/docs/core/phase3_planning.md` - Phase 3 optional optimizations (planning)
- `benchmarks/atc_scanner_cache_analysis.md` - Performance analysis showing FFI overhead vs real-world speedup

**Key Achievement**: Rust ScanCache provides **33,000x effective speedup** in real-world ATCScanner usage (saves 500ms per cache hit vs 15µs FFI overhead).

**Important Note**: Some recommendations in this document were superseded by Phase 2 completion. See cross-references for current implementation status.

---

## Executive Summary

Dựa trên phân tích module `auto_trade/core`, tài liệu này đưa ra các gợi ý tối ưu hóa cho:

- **Rust Integration**: 4 cơ hội chính (High-effort, High-impact)
- **Python Code Optimization**: 6 cơ hội tối ưu (Low-Medium effort, Medium impact)
- **Structural Optimization**: 5 cơ hội cải tiến (Medium effort, High impact)
- **Conflict Assessment**: 8 xung đột tiềm năng cần lưu ý

**Status quo**: Module đã có Rust cache (10-20x speedup), Polars DataFrames, SQLite persistence

---

## 1. Rust Integration Opportunities

### 1.1 XGBoost Feature Computation (🔥 High Priority)

**Current State**:

- Python feature computation in `xgboost_filter.py`
- Uses Pandas DataFrames + Python indicator engine
- Bottleneck: `_predict_signal()` fetches OHLCV, computes indicators

**Opportunity**:

```rust
// rust_backend/src/xgboost_features.rs
pub fn compute_features_optimized(
    ohlcv_data: &[Candle],
    indicators: &IndicatorConfig
) -> PyResult<HashMap<String, f64>> {
    // Rust implementation of:
    // - RSI, MACD, BB calculations
    // - Advanced features from xgboost_LTS
    // 5-10x speedup expected
}
```

**Benefits**:

- ✅ 5-10x speedup for feature computation
- ✅ Reduces GIL contention during ML inference
- ✅ Reusable for other ML models

**Effort**: High (50-85 hours)

- Implement technical indicators in Rust (30-45h)
- PyO3 bindings for NumPy arrays (10-20h)
- Comprehensive testing against Python version (10-20h)

**Note**: Effort increased from initial estimate due to:
- Complex indicator implementations (RSI, MACD, BB, ATR, etc.)
- Need for exact numerical parity with Python versions
- Edge case handling (NaN, Inf, division by zero)

**Impact**: 🔥 **HIGH** - XGBoost filter is critical path

---

### 1.2 Signal Aggregation Pipeline (⭐ Medium Priority)

**Current State**:

- Python loops in `signal_selector.py::select_best_signal()`
- Iterates over xgboost_signals + gemini_signals

**Opportunity**:

```rust
// rust_backend/src/signal_aggregation.rs
pub fn aggregate_signals_vectorized(
    xgboost_signals: Vec<SignalData>,
    gemini_signals: HashMap<String, GeminiData>,
    weights: HashMap<String, f64>
) -> Vec<FinalSignal> {
    // Parallel aggregation using Rayon
    // 3-5x speedup for 100+ signals
}
```

**Benefits**:

- ✅ 3-5x speedup for signal selection
- ✅ Better utilization of multi-core CPUs
- ✅ Type-safe signal validation

**⚠️ FFI Overhead Warning**:
- Phase 2 benchmarks showed FFI overhead of ~15µs per Python↔Rust call
- Signal aggregation typically processes <100ms per operation
- **Recommendation**: Only pursue if profiling shows aggregation taking >500ms
- **Alternative**: If aggregation is fast, FFI overhead may negate benefits
- See `benchmarks/atc_scanner_cache_analysis.md` for detailed FFI analysis

**Effort**: Medium (20-30 hours)

- Port scoring logic to Rust
- Parallel iteration with Rayon
- Integration tests

**Impact**: ⭐ **MEDIUM** - Significant when xgboost_signals > 50

---

### 1.3 Gemini Rate Limiter (💡 Low Priority)

**Current State**:

- Python deque-based rate limiter in `gemini_integration.py`
- `_check_rate_limit()` uses time.sleep()

**Opportunity**:

```rust
// rust_backend/src/rate_limiter.rs
pub struct TokenBucketLimiter {
    capacity: u32,
    tokens: AtomicU32,
    refill_rate: Duration,
}

impl TokenBucketLimiter {
    pub async fn acquire(&self) -> PyResult<()> {
        // Async token bucket algorithm
        // Submicrosecond precision
    }
}
```

**Benefits**:

- ✅ More accurate rate limiting
- ✅ Zero GIL interaction for waiting
- ✅ async/await compatible

**Effort**: Low (8-12 hours)

- Implement token bucket in Rust
- PyO3 async bridge
- Minimal integration

**Impact**: 💡 **LOW** - Only matters at high request rates

---

### 1.4 Persistent Cache Serialization (📋 Future)

**Current State**:

- In-memory SQLite persistence
- No cross-process cache sharing

**Opportunity**:

```rust
// rust_backend/src/persistent_cache.rs
pub fn save_cache_snapshot(
    cache: &ScanCache,
    path: &str
) -> PyResult<()> {
    // Bincode serialization
    // ~10ms to save 1000 entries
}
```

**Benefits**:

- ✅ Fast startup (load cached results)
- ✅ Share cache between processes
- ✅ Crash recovery

**Effort**: Medium (15-20 hours)

- Implement serialization
- Versioning + compatibility
- Migration logic

**Impact**: 📋 **LOW-MEDIUM** - Only if warm-up time is issue

---

## 2. Python Code Optimization Opportunities

### 2.1 Consolidate Caching Implementation (📋 Phase 3 Task)

**Status**: ⚠️ **SUPERSEDED by Phase 2** - See `modules/auto_trade/docs/core/phase3_planning.md` Section 3.1

**Current State (as of Phase 2 completion)**:

- `caching.py` removed as unused
- `atc_scanner.py` has dual cache (Rust + Python fallback) - **BY DESIGN**
- Rust cache: High performance, thread-safe
- Python cache: Graceful fallback if Rust fails

**Phase 2 Decision**:
- **Kept both caches** for graceful degradation
- `use_rust_cache` config flag controls behavior (default: True)
- Python cache provides safety net for Rust initialization failures

**Phase 3 Option**:
- After 2-4 weeks of production monitoring showing >99% Rust usage
- Can remove Python cache if metrics support it
- See `phase3_planning.md` Section 3.1 for detailed removal plan

**Benefits of Current Dual-Cache Approach**:

- ✅ Graceful fallback if Rust fails
- ✅ Zero downtime during Rust issues
- ✅ Production-safe deployment

**Effort (if removing Python cache in Phase 3)**: Low (2-4 hours)
**Impact**: ⭐ **MEDIUM** - Code hygiene + clarity (but loses safety net)

---

### 2.2 Async/Await for Gemini Integration (🔥 High Priority)

**Current State**:

- `analyze_candidates_batch_async()` uses asyncio
- But `analyze_candidate()` is still sync
- Mixed sync/async API

**Optimization**:

```python
# Make ALL Gemini methods async
class GeminiIntegration:
    async def analyze_candidate(self, signal):
        await self._wait_for_rate_limit_async()
        # ... existing logic
    
    # Remove sync analyze_candidate()
```

**Benefits**:

- ✅ Consistent async API
- ✅ Better integration with signal_pipeline
- ✅ True non-blocking I/O

**Effort**: Medium (8-12 hours)

- Refactor sync → async
- Update callers in signal_pipeline
- Test async error handling

**Impact**: 🔥 **HIGH** - Improves pipeline throughput

---

### 2.3 Batch OHLCV Fetching (⭐ Medium Priority)

**Current State**:

- `xgboost_filter.py::_predict_signal()` fetches OHLCV per symbol
- N API calls for N symbols

**Optimization**:

```python
class XGBoostFilter:
    def filter_signals_batched(self, signals):
        # Fetch OHLCV for all symbols in one call
        all_symbols = [s.symbol for s in signals]
        ohlcv_batch = self.data_fetcher.fetch_ohlcv_batch(
            all_symbols, timeframe=self.prediction_timeframe
        )
        # Process in parallel
```

**Benefits**:

- ✅ Reduce API call count (N → 1)
- ✅ Lower latency (parallel fetch)
- ✅ Better cache utilization

**Effort**: Medium (10-15 hours)

- Add batch fetch to DataFetcher
- Refactor XGBoostFilter
- Handle partial failures

**Impact**: ⭐ **MEDIUM-HIGH** - Significant for 20+ symbols

---

### 2.4 Optimize Signal Selector Scoring (💡 Low Priority)

**Current State**:

- `signal_selector.py::_evaluate_candidate()` has nested conditions
- Risk/reward calculation repeated

**Optimization**:

```python
# Precompute common values
class SignalSelector:
    def select_best_signal(self, xgboost_signals, gemini_signals):
        # Vectorize scoring with numpy
        xb_confs = np.array([s.details.get("xgboost_conf", 0.0) 
                             for s in xgboost_signals])
        # Apply weights in single operation
```

**Benefits**:

- ✅ Faster scoring (2-3x for 100+ signals)
- ✅ Cleaner code
- ✅ NumPy vectorization

**Effort**: Low (4-6 hours)
**Impact**: 💡 **LOW** - Only noticeable with many signals

---

### 2.5 Migrate from JSONL to SQLite Fully (✅ Quick Win)

**Current State**:

- `persistence.py` (JSONL) still exists
- `persistence_sqlite.py` is new implementation
- Dual maintenance

**Optimization**:

```python
# 1. Mark persistence.py as deprecated
# 2. Add migration guide
# 3. Remove after 1-2 releases

# Update signal_pipeline to use SQLite by default
from modules.auto_trade.core.persistence_sqlite import SignalPersistenceSQLite

persistence = SignalPersistenceSQLite(db_path="data/signals/signals.db")
```

**Benefits**:

- ✅ Faster queries (indexed SQLite)
- ✅ Advanced analytics (get_statistics())
- ✅ Outcome tracking

**Effort**: Low (2-3 hours)
**Impact**: ⭐ **MEDIUM** - Better data management

---

### 2.6 Circuit Breaker for XGBoost (💡 Low Priority)

**Current State**:

- XGBoost filter has feature failure counter
- But no circuit breaker for model itself

**Optimization**:

```python
class XGBoostFilter:
    def __init__(self, ...):
        self._prediction_circuit = CircuitBreaker(
            name="XGBoostModel",
            failure_threshold=5,
            recovery_timeout=300
        )
    
    def filter_signals(self, signals):
        return self._prediction_circuit.call(
            lambda: self._filter_signals_internal(signals)
        )
```

**Benefits**:

- ✅ Fail-fast when model is broken
- ✅ Auto-recovery after timeout
- ✅ Consistent with GeminiAPI pattern

**Effort**: Low (3-4 hours)
**Impact**: 💡 **LOW** - Defensive programming

---

## 3. Structural Optimization Opportunities

### 3.1 Pipeline Batching for Large Symbol Lists (🔥 High Priority)

**Current State**:

- `signal_pipeline.py` limits symbols to `max_symbols_to_scan` (20)
- But no batching for intermediate steps

**Optimization**:

```python
class SignalPipeline:
    def run_pipeline(self):
        # 1. ATC scan: Already batched (atc_scanner.batch_size)
        
        # 2. XGBoost filter: Add batching
        xgboost_signals_batched = self._filter_in_batches(
            atc_signals, batch_size=10
        )
        
        # 3. Gemini analysis: Already async batched
```

**Benefits**:

- ✅ Handle 100+ symbols efficiently
- ✅ Progressive results (partial success)
- ✅ Better memory control

**Effort**: Medium (12-16 hours)

- Add batching to XGBoost filter
- Update pipeline orchestration
- Test with large symbol lists

**Impact**: 🔥 **HIGH** - Scalability bottleneck

---

### 3.2 Health Check Integration (⭐ Medium Priority)

**Current State**:

- `health.py` exists but underutilized
- Only 2 checks in signal_pipeline

**Optimization**:

```python
# Add health checks for all components
self.health_registry.register_check("XGBoostModel", self._check_xgboost)
self.health_registry.register_check("SQLitePersistence", self._check_persistence)
self.health_registry.register_check("RustCache", self._check_rust_cache)
self.health_registry.register_check("DataFetcher", self._check_data_fetcher)

# Expose /health endpoint for monitoring
```

**Benefits**:

- ✅ Comprehensive system health
- ✅ Early problem detection
- ✅ Production monitoring

**Effort**: Medium (8-12 hours)
**Impact**: ⭐ **MEDIUM** - Operational visibility

---

### 3.3 Modular ATC Configuration (💡 Low Priority)

**Current State**:

- ATC config mixed in ATCScanner config
- Hard to switch ATC strategies

**Optimization**:

```python
# Separate ATC strategy configuration
class ATCStrategyConfig:
    strategy: str  # "conservative", "aggressive", "balanced"
    params: Dict[str, Any]

class ATCScanner:
    def __init__(self, ..., strategy_config: ATCStrategyConfig):
        self.atc_config = self._load_strategy(strategy_config)
```

**Benefits**:

- ✅ Easy strategy switching
- ✅ Clearer separation of concerns
- ✅ A/B testing support

**Effort**: Medium (10-14 hours)
**Impact**: 💡 **LOW-MEDIUM** - Flexibility

---

### 3.4 Symbol Manager Caching (✅ Quick Win)

**Current State**:

- `symbol_manager.py` refreshes symbols every pipeline run
- No caching of symbol lists

**Optimization**:

```python
class SymbolManager:
    def __init__(self):
        self._symbols_cache = None
        self._cache_timestamp = 0
        self._cache_ttl = 3600  # 1 hour
    
    def get_symbols(self):
        if self._is_cache_valid():
            return self._symbols_cache
        return self.refresh_symbols()
```

**Benefits**:

- ✅ Reduce unnecessary API calls
- ✅ Faster pipeline startup
- ✅ Consistent symbol lists

**Effort**: Low (2-3 hours)
**Impact**: ⭐ **MEDIUM** - Reduces API load

---

### 3.5 Outcome Tracking Integration (📋 Future)

**Current State**:

- SQLite persistence has `update_signal_outcome()`
- But no automated outcome tracking

**Optimization**:

```python
# Create OutcomeTracker service
class OutcomeTracker:
    def __init__(self, persistence):
        self.persistence = persistence
    
    async def track_outcomes(self):
        # Periodically check signal outcomes
        # Compare entry/TP/SL with actual market data
        # Update persistence with WIN/LOSS/PENDING
```

**Benefits**:

- ✅ Automatic signal accuracy tracking
- ✅ ML model performance monitoring
- ✅ Strategy backtesting foundation

**Effort**: High (30-40 hours)

- Design outcome logic
- Integrate with data fetcher
- Build reporting dashboard

**Impact**: 📋 **HIGH (future)** - Essential for ML feedback loop

---

## 4. Potential Conflicts and Risks

### 4.1 Rust/Python FFI Overhead (❗ Critical)

**Issue**:

- Rust integration adds FFI overhead (~10-50µs per call)
- Micro-operations may be slower than pure Python

**Mitigation**:

- ✅ **Batch operations**: Pass arrays, not single values
- ✅ **Profile first**: Benchmark before Rust-ifying
- ✅ **Target hot paths**: Only optimize compute-heavy code

**Example**: XGBoost feature computation is good (500ms saved) vs token bucket rate limiter (10µs overhead)

---

### 4.2 SQLite WAL Mode Conflicts (⚠️ High)

**Issue**:

- Multiple processes accessing same SQLite DB
- WAL mode requires shared memory

**Mitigation**:

- ✅ Use connection pooling
- ✅ Set busy_timeout to avoid SQLITE_BUSY
- ✅ Document multi-process limitations

```python
# persistence_sqlite.py
conn.execute("PRAGMA busy_timeout = 5000")  # 5 seconds
```

---

### 4.3 Async/Sync Mixing (⚠️ High)

**Issue**:

- GeminiIntegration has both sync and async methods
- Easy to accidentally block event loop

**Mitigation**:

- ✅ Make all methods async
- ✅ Use `asyncio.run()` only at entry points
- ✅ Document async requirements clearly

```python
# BAD: Sync call in async context
async def pipeline():
    result = gemini.analyze_candidate(signal)  # Blocks!

# GOOD: Proper async
async def pipeline():
    result = await gemini.analyze_candidate_async(signal)
```

---

### 4.4 Cache Invalidation Strategy (⚠️ Medium)

**Issue**:

- Multiple caches: Rust ScanCache, XGBoost prediction cache, Gemini cache
- No coordinated invalidation

**Mitigation**:

- ✅ Use consistent TTL values (60s default)
- ✅ Add `clear_all_caches()` method to pipeline
- ✅ Document cache dependencies

```python
class SignalPipeline:
    def clear_all_caches(self):
        self.atc_scanner.clear_cache()
        self.xgboost_filter.clear_cache()
        self.gemini_integration.clear_cache()
```

---

### 4.5 Memory Usage with Large Batches (⚠️ Medium)

**Issue**:

- Batching 100+ symbols loads all OHLCV in memory
- Risk of OOM with aggressive batching

**Mitigation**:

- ✅ Add memory limits to batch processing
- ✅ Monitor memory usage via health checks
- ✅ Use streaming for very large datasets

```python
# atc_scanner.py
max_batch_memory = 500 * 1024 * 1024  # 500MB
if estimated_memory > max_batch_memory:
    # Split into smaller batches
```

---

### 4.6 Thread Safety in Global State (❗ Critical)

**Issue**:

- `atc_scanner.py` has class-level `_EMPTY_SCAN_RESULT`
- Mutable dict shared across threads

**Mitigation**:

- ✅ **CURRENT CODE IS SAFE**: Uses `.copy()` on access
- ✅ Consider immutable alternatives (frozendict)

```python
# Current (SAFE):
res = results_by_tf.get(tf, self._EMPTY_SCAN_RESULT.copy())

# Better (IMMUTABLE):
from types import MappingProxyType
_EMPTY_SCAN_RESULT = MappingProxyType({"longs": frozenset(), ...})
```

---

### 4.7 Dependency on Rust Build (⚠️ Medium)

**Issue**:

- Adding more Rust code increases build complexity
- Users need Rust toolchain for development

**Mitigation**:

- ✅ Provide pre-built wheels for common platforms
- ✅ Keep Python fallbacks for non-critical paths
- ✅ Document Rust setup in CONTRIBUTING.md

---

### 4.8 Backwards Compatibility (⚠️ Medium)

**Issue**:

- Removing `persistence.py` breaks existing code
- Changing async APIs affects callers

**Mitigation**:

- ✅ Deprecation warnings (2 releases minimum)
- ✅ Migration scripts for data
- ✅ Semantic versioning (major bump for breaking changes)

```python
# persistence.py
import warnings
warnings.warn(
    "SignalPersistence (JSONL) is deprecated. "
    "Use SignalPersistenceSQLite instead.",
    DeprecationWarning
)
```

---

### 4.9 Security Considerations (⚠️ High)

**Issue**:

- Optimizations may introduce security vulnerabilities
- New attack surfaces with Rust FFI, async code, and database access

**Mitigation**:

**SQL Injection (Persistence)**:
```python
# BAD: String concatenation
query = f"SELECT * FROM signals WHERE symbol = '{symbol}'"

# GOOD: Parameterized queries
query = "SELECT * FROM signals WHERE symbol = ?"
cursor.execute(query, (symbol,))
```

**API Key Exposure**:
```python
# BAD: Hardcoded or logged
log_info(f"Using Gemini API key: {api_key}")

# GOOD: Environment variables + redacted logs
api_key = os.getenv("GEMINI_API_KEY")
log_info("Using Gemini API key: [REDACTED]")
```

**Async Race Conditions**:
```python
# BAD: Shared mutable state in async
class Pipeline:
    def __init__(self):
        self.results = []  # Not thread-safe!

    async def process(self, signal):
        self.results.append(signal)  # Race condition!

# GOOD: Use asyncio.Queue or locks
class Pipeline:
    def __init__(self):
        self.results_queue = asyncio.Queue()

    async def process(self, signal):
        await self.results_queue.put(signal)  # Thread-safe
```

**Rust Memory Safety**:
- ✅ Rust compiler prevents memory vulnerabilities
- ✅ Use `#![forbid(unsafe_code)]` in Rust modules
- ✅ Audit all `unsafe` blocks if necessary

**Input Validation**:
```python
# Validate all user inputs at API boundaries
def filter_signals(self, signals: List[Signal]):
    if not signals:
        raise ValueError("signals cannot be empty")
    if len(signals) > 1000:
        raise ValueError("Too many signals (max 1000)")
    for signal in signals:
        if not signal.symbol or not signal.entry_price:
            raise ValueError(f"Invalid signal: {signal}")
```

**Recommendations**:
1. Run security scans (Snyk, Bandit) on all new code
2. Review all SQL queries for injection vulnerabilities
3. Never log sensitive data (API keys, credentials)
4. Use type hints and validation (Pydantic) for all inputs
5. Audit async code for race conditions
6. Follow OWASP Top 10 guidelines

---

## 5. Prioritized Roadmap

### Dependency Graph

Understanding task dependencies is critical for efficient execution:

```
Sequential Dependencies (must complete in order):
┌─────────────────────────────────────────────────┐
│ Phase 1 → Phase 2 → Phase 3 → Phase 4          │
└─────────────────────────────────────────────────┘

Within Phases:

Phase 1 (All can run in parallel):
├─ Consolidate caching (DONE - Phase 2)
├─ SQLite migration
├─ Symbol Manager caching
└─ Async Gemini

Phase 2 (Sequential + Parallel):
├─ XGBoost Features (START FIRST - blocks Signal Aggregation)
│   └─> Signal Aggregation (depends on XGBoost patterns)
└─ Gemini Rate Limiter (parallel, independent)

Phase 3 (Mostly parallel):
├─ Pipeline batching (depends on XGBoost batch patterns)
├─ Health check integration (independent)
└─ Batch OHLCV fetching (independent)

Phase 4 (Sequential):
├─ Outcome tracking (START FIRST - blocks others)
│   └─> Persistent cache (may use outcome data)
└─ Modular ATC config (independent)
```

**Critical Path**: Phase 1 → XGBoost Features → Signal Aggregation → Pipeline Batching

**Parallelizable**: Health checks, Gemini Rate Limiter, Symbol Manager caching

---

### Phase 1: Quick Wins (1-2 weeks)

1. ✅ Consolidate caching (remove `caching.py`) - **DONE (Phase 2)**
2. ✅ Migrate to SQLite fully (deprecate `persistence.py`)
3. ✅ Symbol Manager caching
4. ⭐ Async/Await for Gemini (high impact)

**Estimated Effort**: 20-30 hours
**Expected Impact**: 🔥 HIGH - Code clarity + performance

---

### Phase 2: Rust Integration (4-6 weeks)

1. 🔥 XGBoost Feature Computation (highest ROI, START FIRST)
2. ⭐ Signal Aggregation Pipeline (depends on XGBoost completion)
3. 💡 Gemini Rate Limiter (parallel, independent)

**Estimated Effort**: 80-125 hours (increased from 70-100h)
**Expected Impact**: 🔥 VERY HIGH - 5-10x XGBoost speedup

---

### Phase 3: Scalability (2-3 weeks)

1. 🔥 Pipeline batching for large symbol lists
2. ⭐ Health check integration
3. ⭐ Batch OHLCV fetching

**Estimated Effort**: 30-45 hours  
**Expected Impact**: 🔥 HIGH - Handle 100+ symbols

---

### Phase 4: Future Enhancements (8-12 weeks)

1. 📋 Outcome tracking integration
2. 📋 Persistent cache serialization
3. 💡 Modular ATC configuration

**Estimated Effort**: 50-70 hours  
**Expected Impact**: ⭐ MEDIUM - Long-term maintainability

---

## 6. Decision Framework

When deciding whether to implement an optimization:

### Should I use Rust?

```
IF (operation is CPU-bound)
  AND (computation time > 100ms)
  AND (FFI overhead < 5% of total time)
  AND (can batch operations)
THEN use Rust
ELSE stick with Python
```

**Examples**:

- ✅ YES: XGBoost feature computation (500ms saved)
- ✅ YES: Signal aggregation (100+ signals)
- ❌ NO: Token bucket rate limiter (10µs operation)

### Should I batch operations?

```
IF (operation is I/O-bound OR network-bound)
  AND (operates on multiple items)
  AND (order doesn't matter)
THEN use batching
ELSE process individually
```

**Examples**:

- ✅ YES: OHLCV fetching (N API calls → 1)
- ✅ YES: Gemini analysis (3 concurrent max)
- ❌ NO: Writing to SQLite (ACID transaction per signal)

### Should I use async?

```
IF (operation involves network I/O)
  OR (multiple independent I/O operations)
  AND (response time matters)
THEN use async/await
ELSE stick with sync
```

**Examples**:

- ✅ YES: Gemini API calls
- ✅ YES: Batch data fetching
- ❌ NO: XGBoost model inference (CPU-bound)

---

## 6.5 Performance Targets

Specific, measurable targets for each optimization:

| Component | Metric | Current | Target | Measurement Method |
|-----------|--------|---------|--------|-------------------|
| **XGBoost Features** | Feature computation time | 500ms | <100ms | `pytest-benchmark` on 100 OHLCV rows |
| **Signal Aggregation** | Aggregation time (100 signals) | 200ms | <50ms | Time `select_best_signal()` |
| **Gemini API** | Concurrent requests | 1 | 3 | Async batch size |
| **Pipeline Batching** | Max symbols processed | 20 | 100 | End-to-end test |
| **Cache Hit Rate** | Rust ScanCache hit rate | N/A | >70% | `cache.get_metrics()` |
| **OHLCV Fetching** | API calls (N symbols) | N calls | 1 call | Count `DataFetcher.fetch_ohlcv()` |
| **Overall Pipeline** | Total time (50 symbols) | 30s | <10s | End-to-end benchmark |

**Acceptance Criteria**:
- All targets must be met in production for 7 consecutive days
- P95 latency must not exceed 2x median
- Error rate must remain <1%
- Memory usage must not increase >20%

**Regression Detection**:
- Automated tests fail if performance drops >10% from baseline
- See `tests/performance/test_cache_regression.py` for example

---

## 7. Testing Strategy

For each optimization:

1. **Benchmark**: Measure before/after (use `pytest-benchmark`)
2. **Profile**: Identify actual bottlenecks (use `py-spy` or `cProfile`)
3. **Unit Test**: Verify correctness (100% code coverage)
4. **Integration Test**: Test end-to-end pipeline
5. **Load Test**: Simulate production workload (100+ symbols)

**Example benchmark**:

```python
# tests/performance/test_xgboost_optimization.py
def test_feature_computation_speedup(benchmark):
    result = benchmark(xgboost_filter.filter_signals, signals)
    # Assert: Rust version is 5-10x faster
    assert result.stats.median < 0.1  # 100ms target
```

---

## 8. Monitoring and Rollback Plan

### Monitoring Metrics

- **Performance**: Pipeline duration, component latencies
- **Reliability**: Error rates, cache hit rates
- **Correctness**: Signal count, score distributions

### Feature Flags

```python
# config/auto_trade.py
FEATURE_FLAGS = {
    "use_rust_xgboost_features": bool(os.getenv("USE_RUST_XGBOOST", "true")),
    "use_async_gemini": bool(os.getenv("USE_ASYNC_GEMINI", "true")),
    "use_batched_ohlcv": bool(os.getenv("USE_BATCHED_OHLCV", "false")),
}
```

### Rollback Procedure

1. Set feature flag to `false` in production
2. Monitor metrics for 24 hours
3. If stable, delete old code in next release
4. If issues, keep old code path for 2+ releases

---

## Conclusion

**Document Status**: Some recommendations superseded by Phase 2 completion. See status section at top.

**Immediate Actions** (Phase 1):

1. ~~Consolidate caching → Remove `caching.py`~~ **DONE (Phase 2)** - Kept dual cache for graceful fallback
2. SQLite migration → Deprecate `persistence.py`
3. Async Gemini → Refactor integration

**High-Impact Actions** (Phase 2):

1. Rust XGBoost features → 5-10x speedup (START FIRST)
2. Signal aggregation → Better multi-core usage (depends on XGBoost)

**Scalability Actions** (Phase 3):

1. Pipeline batching → Handle 100+ symbols
2. Health monitoring → Production readiness

**Total Estimated Effort**: 195-285 hours (~1.5-2.5 months with 1 developer)

**Effort Breakdown**:
- Phase 1: 20-30 hours (Quick wins)
- Phase 2: 80-125 hours (Rust integration, increased from 70-100h due to XGBoost complexity)
- Phase 3: 30-45 hours (Scalability)
- Phase 4: 50-70 hours (Future enhancements)
- Security audit: 15-25 hours (Added in this revision)

**Expected Performance Gain**:

- Overall pipeline: 3-5x faster
- XGBoost filtering: 5-10x faster
- Gemini integration: Better concurrency
- Symbol handling: 5x more symbols (20 → 100)

**Key Learnings from Phase 2**:
- FFI overhead (~15µs) is negligible for large operations (500ms+ saved per cache hit)
- Graceful fallback is essential for production reliability
- Comprehensive benchmarking reveals performance paradoxes (Python faster for micro-ops, Rust better for real workloads)

**See Also**:
- `modules/auto_trade/docs/core/scan_cache_implementation_summary.md` - Phase 1-2 implementation details
- `modules/auto_trade/docs/core/phase3_planning.md` - Phase 3 optional optimizations planning
- `benchmarks/atc_scanner_cache_analysis.md` - Detailed FFI overhead analysis
