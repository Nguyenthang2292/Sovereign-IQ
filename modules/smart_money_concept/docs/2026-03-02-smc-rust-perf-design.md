# SMC Rust Performance Optimization — Design Document

**Date**: 2026-03-02
**Status**: Approved
**Scope**: `modules/smart_money_concept/` — Rust acceleration + ATR deduplication

---

## 1. Goal

Tối ưu hiệu năng cho SMC module bằng Rust extension (`smc_rust`), phục vụ:

- **Batch scanner**: Chạy SMC trên hàng trăm symbol song song (Rayon)
- **Real-time**: Incremental update per-bar với O(1) complexity

Đồng thời tái sử dụng `calculate_atr_series` từ `modules.common.indicators.volatility`, loại bỏ ATR code trùng lặp.

---

## 2. Architecture Overview

```
SMCAnalyzer.run(df)
    │
    ├─ calculate_atr_series()          ← modules.common (pandas_ta, tính 1 lần)
    │
    ├─ detect_swings()                 ← smc_rust.detect_swings_rust() | scipy fallback
    │
    ├─ identify_bos_choch()            ← smc_rust.find_bos_crossovers_rust() | Python fallback
    │
    ├─ identify_order_blocks()         ← nhận atr_series param (không tự tính)
    │   └─ _filter_ob_mitigation()     ← smc_rust.filter_ob_mitigation_rust() | iterrows fallback
    │
    └─ identify_equal_hl()             ← smc_rust.detect_equal_hl_rust() | Python fallback
                                          nhận atr_series param (không tự tính)
```

**Pattern**: Python orchestrate (Pivot objects, DataFrames, charts). Rust chỉ nhận numpy arrays, trả numpy arrays. Graceful fallback nếu Rust chưa build.

---

## 3. Computation Hotspots

| # | Hotspot | File | Vấn đề hiện tại | % thời gian |
|---|---------|------|------------------|-------------|
| H1 | Swing detection | `core/swing.py` | `scipy.signal.argrelextrema` 2×2 calls, dependency nặng | ~30% |
| H2 | BOS/CHoCH crossover | `core/bos.py` | N vòng lặp Python scan M bars, O(N×M) | ~25% |
| H3 | OB creation + mitigation | `core/order_block.py` | ATR recompute per-event (M-01) + `iterrows()` | ~35% |
| H4 | Equal HL detection | `core/equal_hl.py` | Trùng lặp compute_atr + swing detect | ~10% |

---

## 4. Rust Extension: `smc_rust`

### 4.1 Directory Structure

```
smart_money_concept/
├── rust_extensions/                ← NEW
│   ├── .cargo/
│   │   └── config.toml
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       ├── lib.rs                  ← PyO3 #[pymodule] entry
│       ├── swing.rs                ← detect_swings_rust
│       ├── crossover.rs            ← find_bos_crossovers_rust
│       ├── mitigation.rs           ← filter_ob_mitigation_rust
│       ├── equal_hl.rs             ← detect_equal_hl_rust
│       ├── batch.rs                ← compute_smc_batch (Phase 2)
│       └── incremental.rs          ← update_smc_incremental (Phase 3)
├── smc_rust.pyi                    ← NEW: Type stubs cho IDE
├── core/
│   ├── swing.py                    ← SỬA: fallback scipy → smc_rust
│   ├── bos.py                      ← SỬA: fallback Python loop → smc_rust
│   ├── order_block.py              ← SỬA: xoá _calculate_atr, nhận atr param
│   ├── equal_hl.py                 ← SỬA: xoá compute_atr import, nhận atr param
│   ├── trend.py                    ← SỬA: xoá compute_atr()
│   ├── analyzer.py                 ← SỬA: tính ATR 1 lần, truyền xuống
│   └── constants.py
├── benchmarks/                     ← NEW
│   └── bench_smc_rust.py
└── tests/
    └── test_rust_parity.py         ← NEW
```

### 4.2 Cargo.toml

```toml
[package]
name = "smc_rust"
version = "0.1.0"
edition = "2021"

[lib]
name = "smc_rust"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.23", features = ["extension-module"] }
numpy = "0.23"
ndarray = "0.15"
rayon = "1.8"         # Phase 2: batch parallel

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
```

### 4.3 pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "smc-rust"
version = "0.1.0"
requires-python = ">=3.9"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
]
```

### 4.4 Function Signatures

#### `detect_swings_rust`

```rust
/// Detect local extrema (swing highs/lows) using window scan.
/// Replaces scipy.signal.argrelextrema — same algorithm, no scipy dependency.
///
/// Returns: (high_indices: Vec<usize>, low_indices: Vec<usize>)
#[pyfunction]
fn detect_swings_rust(
    highs: PyReadonlyArray1<f64>,
    lows: PyReadonlyArray1<f64>,
    order: usize,
) -> (Vec<usize>, Vec<usize>)
```

#### `find_bos_crossovers_rust`

```rust
/// Batch-find close crossover points for BOS/CHoCH detection.
/// For each pivot, scan forward to find first bar where close crosses pivot level.
///
/// Returns: Vec<i64> — crossing bar index for each pivot (-1 if not found)
#[pyfunction]
fn find_bos_crossovers_rust(
    closes: PyReadonlyArray1<f64>,
    timestamps: PyReadonlyArray1<i64>,     // epoch ms
    pivot_levels: PyReadonlyArray1<f64>,
    pivot_start_idx: PyReadonlyArray1<i64>,
    pivot_end_idx: PyReadonlyArray1<i64>,
    is_bullish: bool,
) -> Vec<i64>
```

#### `filter_ob_mitigation_rust`

```rust
/// Check which OrderBlocks have been mitigated by subsequent price action.
///
/// Returns: Vec<bool> — true = keep, false = mitigated (remove)
#[pyfunction]
fn filter_ob_mitigation_rust(
    highs: PyReadonlyArray1<f64>,
    lows: PyReadonlyArray1<f64>,
    ob_end_indices: PyReadonlyArray1<i64>,
    ob_bar_lows: PyReadonlyArray1<f64>,
    ob_bar_highs: PyReadonlyArray1<f64>,
    ob_biases: PyReadonlyArray1<i32>,      // BULLISH=1, BEARISH=-1
) -> Vec<bool>
```

#### `detect_equal_hl_rust`

```rust
/// Find Equal High / Equal Low pairs where |level[i] - level[i+size]| < threshold.
///
/// Returns: Vec<(usize, usize)> — pairs of indices
#[pyfunction]
fn detect_equal_hl_rust(
    levels: PyReadonlyArray1<f64>,
    threshold: f64,
    size: usize,
) -> Vec<(usize, usize)>
```

---

## 5. Python-side Refactoring

### 5.1 ATR Deduplication

**Xoá**:

- `core/trend.py:compute_atr()` — 28 dòng, Python loop thủ công
- `core/order_block.py:_calculate_atr()` — 13 dòng, pandas rolling tự viết
- `core/equal_hl.py` — xoá `from .trend import compute_atr`

**Thay bằng**: `calculate_atr_series` từ `modules.common.indicators.volatility`

```python
# core/analyzer.py
from modules.common.indicators.volatility import calculate_atr_series

class SMCAnalyzer:
    def run(self, df):
        ...
        atr_series = calculate_atr_series(
            df["High"], df["Low"], df["Close"], length=14
        )
        # Truyền xuống order_block
        ob_internal = identify_order_blocks_from_structure(
            df, bullish_events, bearish_events, atr_series=atr_series
        )
        # Truyền xuống equal_hl
        equal_hl = identify_equal_hl(
            df, internal_highs, internal_lows, atr_series=atr_series
        )
```

### 5.2 Graceful Fallback Pattern

Mỗi file core áp dụng pattern thống nhất:

```python
# core/swing.py
try:
    from smc_rust import detect_swings_rust
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

def _detect_swing_pivots(df, order, is_internal):
    if _HAS_RUST:
        high_idx, low_idx = detect_swings_rust(
            df["High"].values, df["Low"].values, order
        )
        # convert indices → Pivot objects
        ...
    else:
        # existing scipy fallback (unchanged)
        from scipy.signal import argrelextrema
        ...
```

Tương tự cho `bos.py`, `order_block.py`, `equal_hl.py`.

**Zero breaking change**: Nếu `smc_rust` chưa build → Python fallback chạy bình thường. 53 tests hiện có phải pass 100%.

---

## 6. Batch Processing (Phase 2)

```rust
/// Process SMC analysis for multiple symbols in parallel (Rayon).
/// Each symbol runs full pipeline: swing → crossover → OB → mitigation.
///
/// Input: HashMap<symbol, SmcInput{highs, lows, closes, atr}>
/// Output: HashMap<symbol, SmcOutput{swing_idx, bos_idx, ob_list, equal_hl}>
#[pyfunction]
fn compute_smc_batch(
    symbols_data: HashMap<String, SmcInput>,
    internal_order: usize,
    external_order: usize,
) -> HashMap<String, SmcOutput>
```

**Rayon parallelism** theo symbol — giống `compute_atc_signals_batch_cpu` trong `atc_rust`.

**Kỳ vọng**: 100 symbols × 1500 bars, từ ~30s (Python) → ~1s (Rust+Rayon).

---

## 7. Incremental Update (Phase 3)

```rust
/// Stateful per-bar update for real-time trading.
/// Maintains swing history, active OBs, current trend.
///
/// O(1) per bar instead of O(N) full recompute.
fn update_smc_incremental(
    state: &mut SmcState,
    new_high: f64,
    new_low: f64,
    new_close: f64,
    new_atr: f64,
) -> SmcUpdate {
    // 1. Check if new bar creates swing (window buffer)
    // 2. Check crossover against active pivots
    // 3. Check OB mitigation against new bar
    // 4. Update trend state
}
```

`SmcState` lưu:

- Ring buffer các bar gần nhất (size = max(internal_order, external_order))
- Danh sách swing pivots hiện tại
- Active OBs chưa bị mitigate
- Current trend (BULLISH/BEARISH/NEUTRAL)
- Last structure break direction

Pattern giống `update_incremental_atc_rust` trong `atc_rust`.

---

## 8. Phased Implementation

| Phase | Scope | Files mới/sửa | Priority |
|-------|-------|----------------|----------|
| **Phase 1** | 4 stateless Rust functions + Python fallback + ATR dedup | `rust_extensions/src/*.rs`, `smc_rust.pyi`, `core/swing.py`, `core/bos.py`, `core/order_block.py`, `core/equal_hl.py`, `core/trend.py`, `core/analyzer.py` | 🔴 Cao |
| **Phase 2** | `compute_smc_batch` với Rayon | `rust_extensions/src/batch.rs`, `smc_rust.pyi` | 🔴 Cao |
| **Phase 3** | `update_smc_incremental` | `rust_extensions/src/incremental.rs`, wrapper Python | 🟡 Trung bình |

---

## 9. Testing Strategy

| Layer | Tool | Nội dung |
|-------|------|---------|
| **Rust unit tests** | `cargo test` | Từng function so sánh output với expected hardcoded values |
| **Python parity tests** | `pytest` | Chạy cả Rust path và Python fallback, `assert_array_equal` — kết quả phải giống 100% |
| **Benchmark** | `pytest-benchmark` | So sánh thời gian Rust vs Python trên 500/1000/1500 bars, ghi vào `benchmarks/` |
| **Regression** | 53 tests hiện có | Phải pass 100% — zero breaking change |

### Parity test example

```python
# tests/test_rust_parity.py
import pytest
import numpy as np

def test_swing_detection_parity(sample_ohlcv_df):
    """Rust and Python paths must produce identical results."""
    from modules.smart_money_concept.core.swing import (
        _detect_swing_pivots,
    )
    
    # Force Python path
    import modules.smart_money_concept.core.swing as swing_mod
    orig = swing_mod._HAS_RUST
    
    swing_mod._HAS_RUST = False
    result_py = _detect_swing_pivots(sample_ohlcv_df, order=5, is_internal=True)
    
    swing_mod._HAS_RUST = True
    result_rs = _detect_swing_pivots(sample_ohlcv_df, order=5, is_internal=True)
    
    swing_mod._HAS_RUST = orig
    
    assert len(result_py[0]) == len(result_rs[0])
    assert len(result_py[1]) == len(result_rs[1])
    for py_pivot, rs_pivot in zip(result_py[0], result_rs[0]):
        assert py_pivot.level == rs_pivot.level
        assert py_pivot.bar_time == rs_pivot.bar_time


def test_bos_crossover_parity(sample_ohlcv_df):
    """BOS crossover detection: Rust == Python."""
    ...


def test_ob_mitigation_parity(sample_ohlcv_df):
    """OB mitigation filter: Rust == Python."""
    ...
```

---

## 10. Build & Development

```bash
# Build Rust extension (development)
cd modules/smart_money_concept/rust_extensions
maturin develop --release

# Run Rust unit tests
cargo test

# Run Python tests (auto-detects Rust availability)
pytest modules/smart_money_concept/tests/ -v

# Run parity tests specifically
pytest modules/smart_money_concept/tests/test_rust_parity.py -v

# Run benchmarks
pytest modules/smart_money_concept/benchmarks/bench_smc_rust.py --benchmark-only
```

---

## 11. Expected Performance Gains

| Component | Python (500 bars) | Rust (500 bars) | Speedup |
|-----------|-------------------|-----------------|---------|
| Swing detection | ~5ms (scipy) | ~0.05ms | ~100x |
| BOS crossover (20 pivots) | ~8ms (Python loop) | ~0.1ms | ~80x |
| OB mitigation (10 OBs) | ~12ms (iterrows) | ~0.1ms | ~120x |
| Equal HL | ~1ms | ~0.01ms | ~100x |
| **Full pipeline (1 symbol)** | **~26ms** | **~0.3ms** | **~85x** |
| **Scanner (100 symbols, Rayon)** | **~2.6s** | **~30ms** | **~85x** |

> Lưu ý: Số liệu là ước tính dựa trên benchmark patterns từ `atc_rust`. Benchmark thực tế sẽ được ghi nhận tại `benchmarks/`.

---

## 12. Summary

- **Tạo `smc_rust` crate riêng biệt** (PyO3 + maturin), 4 stateless functions + batch + incremental
- **Xoá ATR trùng lặp** trong `core/trend.py` và `core/order_block.py`, tái sử dụng `calculate_atr_series` từ `modules.common`
- **Graceful fallback** — module chạy bình thường nếu chưa build Rust
- **Parity tests** đảm bảo Rust output == Python output
- **3 phase** triển khai: stateless → batch → incremental
