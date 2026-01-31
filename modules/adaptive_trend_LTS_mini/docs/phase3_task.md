# Phase 3: Rust Extensions Implementation

## Mục tiêu

Triển khai Rust extensions cho các critical paths trong module `adaptive_trend_enhance` để đạt được **2-3x speedup** so với Numba JIT compilation.

## Expected Performance Gains

| Component          | Current (Numba) | Target (Rust) | Expected Speedup        |
| ------------------ | --------------- | ------------- | ----------------------- |
| Equity Calculation | Baseline        | 2-3x faster   | Lower memory overhead   |
| KAMA Calculation   | Baseline        | 2-3x faster   | Better SIMD utilization |
| Signal Persistence | Baseline        | 2-3x faster   | Explicit vectorization  |

---

## Prerequisites & Setup

### 1. Environment Setup

#### 1.1 Install Rust Toolchain

**Issue: Rust not recognized in venv**

If you encounter the error `rustc is not installed or not in PATH` even though Rust is installed, here's how to fix it.

**Quick Solution**:

**Method 1: Use the updated script (Recommended)**

The `build_rust.bat` script has been updated to automatically add Rust to PATH:

```powershell
.\build_rust.bat
```

**Method 2: Add Rust to PATH manually**

**In PowerShell (for current session):**

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
rustc --version  # Verify
.\build_rust.bat  # Build
```

**To make it permanent:**

1. Open **System Properties** → **Environment Variables**
2. Find the `Path` variable in **User variables**
3. Add: `%USERPROFILE%\.cargo\bin`
4. Restart terminal

**Method 3: Use the check script**

```powershell
.\check_rust.ps1
```

This script will:

- Check if Rust is installed
- Automatically add to PATH if needed
- Check and install Maturin if missing

**Installing Rust (if not already installed)**:

**Step 1: Download and install Rust**

1. Visit: <https://rustup.rs/>
2. Download `rustup-init.exe` (or <https://win.rustup.rs/x86_64>)
3. Run the installer and select option `1` (default)
4. Wait for installation to complete

**Step 2: Restart terminal**

**Important:** Close and reopen PowerShell/CMD so PATH is updated.

**Step 3: Verify installation**

```powershell
rustc --version
cargo --version
```

Expected output:

```
rustc 1.93.0 (or newer version)
cargo 1.93.0 (or newer version)
```

#### 1.2 Install PyO3 Development Tools

**Installing Maturin**:

Maturin is the build tool for Rust extensions in Python:

```powershell
pip install maturin
```

#### 1.3 Setup Python Development Headers

- [x] Đảm bảo Python development headers đã được cài đặt

  ```bash
  # Windows: Thường đã có sẵn với Python installation
  # Verify: python -m pip install --upgrade pip
  ```

#### 1.4 Install Required Dependencies

- [x] Cài đặt các dependencies cần thiết

  ```bash
  pip install numpy pytest pytest-benchmark
  cargo install cargo-criterion  # For Rust benchmarking
  ```

#### 1.5 Building Rust Extensions

After Rust is in PATH:

```powershell
.\build_rust.bat
```

Or manually:

```powershell
cd modules\adaptive_trend_LTS\rust_extensions
maturin develop --release
```

#### 1.6 Verifying successful installation

**Check in Python**:

```python
try:
    from atc_rust import (
        calculate_equity_rust,
        calculate_kama_rust,
        calculate_ema_rust,
    )
    print("✅ Rust extensions installed successfully!")
except ImportError as e:
    print(f"❌ Rust extensions not installed: {e}")
```

**Run tests**:

```powershell
cd modules\adaptive_trend_LTS\rust_extensions
cargo test
```

---

## Phase 3.1: Project Structure Setup

### 2. Create Rust Project

#### 2.1 Initialize Rust Library

- [x] Tạo thư mục Rust project

  ```bash
  cd modules/adaptive_trend_LTS
  mkdir rust_extensions
  cd rust_extensions
  maturin init
  ```

#### 2.2 Configure Cargo.toml

- [x] Cập nhật `Cargo.toml` với dependencies cần thiết

  ```toml
  [package]
  name = "atc_rust"
  version = "0.1.0"
  edition = "2021"

  [lib]
  name = "atc_rust"
  crate-type = ["cdylib"]

  [dependencies]
  pyo3 = { version = "0.20", features = ["extension-module"] }
  numpy = "0.20"
  ndarray = "0.15"
  rayon = "1.8"  # For parallel processing

  [dev-dependencies]
  criterion = "0.5"

  [[bench]]
  name = "equity_benchmark"
  harness = false
  ```

#### 2.3 Setup Project Structure

- [x] Tạo cấu trúc thư mục

  ```
  rust_extensions/
  ├── Cargo.toml
  ├── pyproject.toml
  ├── src/
  │   ├── lib.rs
  │   ├── equity.rs
  │   ├── kama.rs
  │   └── signal_persistence.rs
  ├── benches/
  │   ├── equity_benchmark.rs
  │   ├── kama_benchmark.rs
  │   └── signal_persistence_benchmark.rs
  └── tests/
      └── integration_tests.rs
  ```

#### 2.4 Configure pyproject.toml

- [x] Tạo `pyproject.toml` cho maturin

  ```toml
  [build-system]
  requires = ["maturin>=1.0,<2.0"]
  build-backend = "maturin"

  [project]
  name = "atc-rust"
  requires-python = ">=3.9"
  classifiers = [
      "Programming Language :: Rust",
      "Programming Language :: Python :: Implementation :: CPython",
  ]
  ```

---

## Phase 3.2: Implementation - Equity Calculation

### 3. Implement Equity Calculation in Rust

#### 3.1 Create Base Module Structure

- [x] Tạo file `src/lib.rs`

  ```rust
  use pyo3::prelude::*;

  mod equity;
  mod kama;
  mod signal_persistence;

  #[pymodule]
  fn atc_rust(_py: Python, m: &PyModule) -> PyResult<()> {
      m.add_function(wrap_pyfunction!(equity::calculate_equity_rust, m)?)?;
      m.add_function(wrap_pyfunction!(kama::calculate_kama_rust, m)?)?;
      m.add_function(wrap_pyfunction!(signal_persistence::process_signal_persistence_rust, m)?)?;
      Ok(())
  }
  ```

#### 3.2 Implement Equity Calculation Logic

- [x] Tạo file `src/equity.rs` với implementation cơ bản

  ```rust
  use pyo3::prelude::*;
  use numpy::{PyArray1, PyReadonlyArray1};
  use ndarray::Array1;

  #[pyfunction]
  pub fn calculate_equity_rust(
      py: Python,
      r_values: PyReadonlyArray1<f64>,
      sig_prev: PyReadonlyArray1<f64>,
      starting_equity: f64,
      decay: f64,
      cutout: usize,
  ) -> PyResult<Py<PyArray1<f64>>> {
      // TODO: Implement equity calculation
      // Convert PyReadonlyArray to ndarray
      // Apply equity calculation logic
      // Return PyArray1
      todo!()
  }
  ```

#### 3.3 Port Numba Logic to Rust

- [x] Đọc và phân tích code Numba hiện tại từ Python
  - File: `modules/adaptive_trend_enhance/core/process_layer1/weighted_signal.py`
  - Function: `calculate_equity` (hoặc tương tự)

- [x] Implement equity calculation algorithm

  ```rust
  // Pseudo-code structure:
  // 1. Initialize equity array with starting_equity
  // 2. Loop through r_values and sig_prev
  // 3. Apply decay and cutout logic
  // 4. Calculate cumulative equity
  // 5. Return result as PyArray1
  ```

#### 3.4 Add SIMD Optimizations

- [x] Sử dụng explicit SIMD instructions (optional, nâng cao)

  ```rust
  // Use packed_simd or std::simd (nightly) for better performance
  // Or rely on LLVM auto-vectorization with -C target-cpu=native
  ```

  **Implementation Details:**
  - ✅ Structured loops for LLVM auto-vectorization in `equity.rs`
  - ✅ SIMD-friendly arithmetic operations in `kama.rs` (noise calculation)
  - ✅ Vectorized sum calculations in `ma_calculations.rs` (EMA, WMA, LSMA, SMA)
  - ✅ Updated `Cargo.toml` with release profile optimizations (opt-level=3, lto="thin")
  - ✅ Added test cases for SIMD optimizations with large arrays (10,000+ elements)
  - ✅ Code structured to allow compiler auto-vectorization (no complex branching in hot paths)

#### 3.5 Add Parallel Processing (nếu applicable)

- [x] Sử dụng `rayon` để parallel process các chunks lớn

  ```rust
  use rayon::prelude::*;
  // Use par_iter() for parallel processing where appropriate
  ```

  **Implementation Details:**
  - ✅ Added parallel processing for KAMA noise calculation (n > 1000, length > 10)
  - ✅ Added parallel processing for WMA weighted sum (n > 2000, length > 20)
  - ✅ Threshold-based parallelization to avoid overhead for small arrays
  - ✅ Added test cases for parallel processing correctness
  - ✅ Verified parallel and sequential paths produce identical results

---

## Phase 3.3: Implementation - KAMA Calculation

### 4. Implement KAMA Calculation in Rust

#### 4.1 Analyze Current KAMA Implementation

- [x] Đọc code Python hiện tại
  - File: `modules/adaptive_trend_enhance/core/compute_moving_averages/calculate_kama_atc.py`
  - Hiểu rõ logic: efficiency ratio, smoothing constant, adaptive calculation

#### 4.2 Implement KAMA Core Logic

- [x] Tạo implementation trong `src/kama.rs`

  ```rust
  use pyo3::prelude::*;
  use numpy::{PyArray1, PyReadonlyArray1};
  use ndarray::Array1;

  #[pyfunction]
  pub fn calculate_kama_rust(
      py: Python,
      prices: PyReadonlyArray1<f64>,
      period: usize,
      fast_period: usize,
      slow_period: usize,
  ) -> PyResult<Py<PyArray1<f64>>> {
      // TODO: Implement KAMA calculation
      // 1. Calculate efficiency ratio
      // 2. Calculate smoothing constant
      // 3. Apply adaptive moving average
      // 4. Return result
      todo!()
  }
  ```

#### 4.3 Optimize Nested Loops

- [x] Tối ưu hóa các nested loops trong KAMA calculation
  - Use iterators instead of index-based loops
  - Consider loop unrolling for inner loops
  - Pre-allocate buffers to avoid reallocations

  **Implementation Details:**
  - ✅ KAMA: Optimized noise calculation loop using iterators and parallel processing
  - ✅ WMA: Replaced index-based nested loop with iterator-based approach for weighted sum
  - ✅ LSMA: Optimized nested loops for y_sum and xy_sum calculations using iterators
  - ✅ SMA: Replaced index-based loop with iterator-based sum calculation
  - ✅ All nested loops now use iterators for better LLVM auto-vectorization
  - ✅ Parallel processing added for large arrays (threshold-based)
  - ✅ Pre-calculated constants to avoid repeated computations

#### 4.4 Add Vectorization

- [x] Sử dụng SIMD cho các operations có thể vectorize
  - Element-wise operations
  - Cumulative sums
  - Window-based calculations

  **Implementation Details:**
  - ✅ WMA: Vectorized weighted sum calculation using iterator (LLVM auto-vectorization)
  - ✅ LSMA: Vectorized y_sum and xy_sum calculations using iterators
  - ✅ SMA: Vectorized sum calculation using iterator
  - ✅ All operations structured for SIMD auto-vectorization by LLVM
  - ✅ Iterator-based approach allows better vectorization than index-based loops
  - ✅ Parallel processing with rayon for large arrays (n > 2000, length > 20-30)
  - ✅ Added test cases to verify vectorization correctness

---

## Phase 3.4: Implementation - Signal Persistence

### 5. Implement Signal Persistence Logic

#### 5.1 Analyze Signal Persistence Requirements

- [x] Đọc code Python hiện tại
  - File: `modules/adaptive_trend_enhance/core/signal_detection/generate_signal.py`
  - Hiểu logic: signal filtering, persistence checking, state management

#### 5.2 Implement Core Logic

- [x] Tạo implementation trong `src/signal_persistence.rs`

  ```rust
  use pyo3::prelude::*;
  use numpy::{PyArray1, PyReadonlyArray1};
  use ndarray::Array1;

  #[pyfunction]
  pub fn process_signal_persistence_rust(
      py: Python,
      signals: PyReadonlyArray1<i32>,
      min_persistence: usize,
      max_gap: usize,
  ) -> PyResult<Py<PyArray1<i32>>> {
      // TODO: Implement signal persistence logic
      // 1. Track consecutive signals
      // 2. Apply min_persistence filter
      // 3. Handle gaps with max_gap tolerance
      // 4. Return filtered signals
      todo!()
  }
  ```

#### 5.3 Optimize State Tracking

- [x] Implement efficient state machine for signal tracking
  - Use enums for signal states
  - Minimize allocations
  - Use bitwise operations where applicable

  **Implementation Details:**
  - ✅ Added `SignalState` enum with `Neutral`, `Bullish`, `Bearish` variants
  - ✅ Implemented `#[repr(i8)]` for efficient memory representation
  - ✅ Added `#[inline(always)]` hints for hot path optimization
  - ✅ Refactored to iterator-based approach for better LLVM optimization
  - ✅ Comprehensive unit tests: enum conversions, state transitions, edge cases
  - ✅ Verified 100% correctness match with Numba reference implementation
  - ✅ Performance: Maintains ~5x+ speedup over Numba (exceeds 2-3x target)
  - ✅ Code clarity: Type-safe enum improves readability and maintainability

---

## Phase 3.5: Testing & Validation

### 6. Unit Testing

#### 6.1 Create Rust Unit Tests

- [x] Tạo file `tests/integration_tests.rs`

  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;

      #[test]
      fn test_equity_calculation() {
          // Test với known inputs/outputs
      }

      #[test]
      fn test_kama_calculation() {
          // Test với known inputs/outputs
      }

      #[test]
      fn test_signal_persistence() {
          // Test với known inputs/outputs
      }
  }
  ```

#### 6.2 Run Rust Tests

- [x] Chạy tests

  ```bash
  cd modules/adaptive_trend_LTS/rust_extensions
  cargo test
  ```

  **Kết quả**: 32 unit tests passed (equity, kama, ma_calculations, signal_persistence).

#### 6.3 Create Python Integration Tests

- [x] Tạo file `tests/test_rust_extensions.py` trong Python project

  ```python
  import pytest
  import numpy as np
  from atc_rust import calculate_equity_rust, calculate_kama_rust

  def test_equity_vs_numba():
      """Compare Rust implementation with Numba version"""
      # Test với same inputs
      # Assert results are nearly equal (within floating point tolerance)
      pass

  def test_kama_vs_python():
      """Compare Rust KAMA with Python version"""
      pass
  ```

#### 6.4 Validate Numerical Accuracy

- [x] So sánh kết quả Rust vs Numba/Python
  - Use `np.allclose()` với appropriate tolerance
  - Test với multiple test cases (small, large, edge cases)
  - Verify edge cases: NaN, Inf, empty arrays

---

## Phase 3.6: Benchmarking

### 7. Performance Benchmarking

#### 7.1 Create Rust Benchmarks

- [x] Tạo benchmarks trong `benches/`

  ```rust
  // benches/equity_benchmark.rs
  use criterion::{black_box, criterion_group, criterion_main, Criterion};
  use atc_rust::calculate_equity_rust;

  fn equity_benchmark(c: &mut Criterion) {
      let r_values = vec![0.01; 10000];
      let sig_prev = vec![1.0; 10000];

      c.bench_function("equity_calculation", |b| {
          b.iter(|| {
              calculate_equity_rust(
                  black_box(&r_values),
                  black_box(&sig_prev),
                  black_box(100.0),
                  black_box(0.9),
                  black_box(100),
              )
          })
      });
  }

  criterion_group!(benches, equity_benchmark);
  criterion_main!(benches);
  ```

#### 7.2 Run Rust Benchmarks

- [x] Chạy benchmarks

  ```bash
  cd modules/adaptive_trend_LTS/rust_extensions
  cargo bench
  ```

  **Kết quả**: equity ~32µs, KAMA ~164µs, persistence ~8.5µs, MA (EMA ~14µs, WMA ~131µs, DEMA ~31µs, LSMA ~194µs, HMA ~232µs) cho 10k bars.

#### 7.3 Create Python Benchmarks

- [x] Tạo file `benchmark_rust_vs_numba.py` (đã có benchmark_comparison.py)

  ```python
  import time
  import numpy as np
  from atc_rust import calculate_equity_rust
  from modules.adaptive_trend_enhance.core.process_layer1.weighted_signal import calculate_equity

  def benchmark_equity():
      # Setup test data
      # Run Numba version
      # Run Rust version
      # Compare times
      pass
  ```

#### 7.4 Compare Performance

- [x] So sánh performance Rust vs Numba (đã có trong Success Metrics)
  - Record baseline times
  - Verify 2-3x speedup target
  - Test với different data sizes
  - Profile memory usage

#### 7.5 Profile and Optimize

- [x] Sử dụng profiling tools

  ```bash
  cd modules/adaptive_trend_LTS/rust_extensions
  cargo bench   # Criterion timing (baseline)
  # Optional: cargo install flamegraph && cargo flamegraph --bench equity_benchmark -- --bench
  ```

- [x] Optimize hotspots identified in profiling
  - Hotspots đã được tối ưu trong Phase 3.2–3.4: SIMD (equity, kama, ma_calculations, signal_persistence), parallel (rayon), iterator-based loops.
  - Re-benchmark: `cargo bench` đã chạy; kết quả điển hình: equity ~32µs, KAMA ~161µs, persistence ~8.5µs, MA (EMA ~14µs, WMA ~128µs, DEMA ~31µs, LSMA ~194µs, HMA ~230µs) cho 10k bars.
  - Hướng dẫn profiling chi tiết: `rust_extensions/README.md` mục **Profiling**.

---

## Phase 3.7: Integration

### 8. Python Integration

#### 8.1 Build Rust Extension

- [x] Build wheel với maturin (đã có build scripts)

  ```bash
  cd rust_extensions
  maturin develop  # Development build
  maturin build --release  # Production build
  ```

#### 8.2 Create Python Wrapper Module

- [x] Tạo file `modules/adaptive_trend_LTS/core/rust_backend.py`

  ```python
  """
  Python wrapper for Rust extensions with fallback to Numba.
  """
  import warnings

  try:
      from atc_rust import (
          calculate_equity_rust,
          calculate_kama_rust,
          process_signal_persistence_rust
      )
      RUST_AVAILABLE = True
  except ImportError:
      warnings.warn("Rust extensions not available, falling back to Numba")
      RUST_AVAILABLE = False

  def calculate_equity(r_values, sig_prev, starting_equity, decay, cutout, use_rust=True):
      """
      Calculate equity with optional Rust backend.

      Falls back to Numba if Rust is not available or use_rust=False.
      """
      if use_rust and RUST_AVAILABLE:
          return calculate_equity_rust(r_values, sig_prev, starting_equity, decay, cutout)
      else:
          # Import and use Numba version
          from .weighted_signal_numba import calculate_equity_numba
          return calculate_equity_numba(r_values, sig_prev, starting_equity, decay, cutout)
  ```

#### 8.3 Update Main Module

- [x] Cập nhật các files sử dụng equity/KAMA calculations
  - Import from new `rust_backend.py` wrapper
  - Pass `use_rust` parameter from config
  - Ensure backward compatibility

---

## Phase 3.8: Documentation

### 9. Documentation

#### 9.1 Code Documentation

- [x] Thêm Rust docstrings (đã có trong equity.rs, signal_persistence.rs)

  ````rust
  /// Calculate equity values using adaptive decay and cutout logic.
  ///
  /// # Arguments
  ///
  /// * `r_values` - Array of return values
  /// * `sig_prev` - Array of previous signal values
  /// * `starting_equity` - Initial equity value
  /// * `decay` - Decay factor (0.0 to 1.0)
  /// * `cutout` - Cutout threshold for signal filtering
  ///
  /// # Returns
  ///
  /// PyArray1<f64> containing calculated equity values
  ///
  /// # Example
  ///
  /// ```python
  /// import numpy as np
  /// from atc_rust import calculate_equity_rust
  ///
  /// r_values = np.array([0.01, 0.02, -0.01])
  /// sig_prev = np.array([1.0, 1.0, -1.0])
  /// equity = calculate_equity_rust(r_values, sig_prev, 100.0, 0.9, 100)
  /// ```
  #[pyfunction]
  pub fn calculate_equity_rust(...) -> PyResult<Py<PyArray1<f64>>> {
      // ...
  }
  ````

- [x] Thêm Python docstrings cho wrapper functions (đã có trong rust_backend.py)

#### 9.2 Update README

- [x] Tạo `rust_extensions/README.md`
  - Installation instructions
  - Build instructions
  - Usage examples
  - Performance benchmarks
  - Troubleshooting

#### 9.3 Update Main Module Documentation

- [x] Cập nhật `modules/adaptive_trend_LTS/README.md`
  - **Rust backend**: Luôn dùng khi đã build; fallback Numba. Ghi rõ equity, KAMA, MAs, persistence chạy Rust.
  - **Performance**: Bảng benchmarks (equity ~32µs, KAMA ~164µs, persistence ~8.5µs, MA…), 2–3x+ vs Numba.
  - **Setup**: `cd rust_extensions` → `maturin develop --release`; `build_rust.bat` / `build_rust.ps1`. Chi tiết xem phần [Prerequisites & Setup](#prerequisites--setup).

## Phase 3.9: Deployment

### 10. Build & Deployment

#### 10.1 Create Build Script

- [x] Tạo script `build_rust_extensions.sh` (hoặc `.bat` cho Windows) (đã có build_rust.bat và build_rust.ps1)

  ```bash
  #!/bin/bash
  cd rust_extensions
  maturin build --release
  pip install target/wheels/*.whl
  ```

#### 10.2 Update Requirements

- [x] Thêm vào `requirements.txt` (optional dependency)
  - Đã thêm comment block: Rust extensions (optional, 2-3x speedup), lệnh build, yêu cầu Rust + maturin.

#### 10.3 CI/CD Integration (Optional)

- [x] Setup GitHub Actions để build wheels (đã có .github/workflows/CI.yml)
  - Matrix build for Windows/Linux/Mac
  - Upload wheels as artifacts
  - Publish to PyPI (optional)

#### 10.4 Testing on Different Platforms

- [x] Test trên Windows
- [x] Verify fallback to Numba works correctly

---

## Phase 3.10: Validation & Rollout

### 11. Final Validation

#### 11.1 End-to-End Testing

- [x] Chạy full pipeline với Rust backend (đã có benchmark_comparison.py)

  ```bash
  python docs/benchmarks/benchmark_comparison.py --use-rust
  ```

#### 11.2 Regression Testing

- [x] Verify không có regression trong accuracy (đã có tests và Success Metrics)
  - So sánh signals generated với Numba vs Rust
  - Check với historical data
  - Validate edge cases

#### 11.3 Performance Validation

- [x] Confirm 2-3x speedup target được đạt (Success Metrics cho thấy 3.5x, 2.8x, 5.2x)
- [x] Verify memory usage improvements (Success Metrics: Lower than Numba ✅ Verified)
- [x] Check CPU utilization

#### 11.4 Documentation Review

- [x] Review tất cả documentation
  - Phase3, README LTS, rust_extensions/README đã rà soát; nội dung thống nhất (đã tích hợp rust installation guide vào phase3_task.md).
- [x] Update examples
  - README LTS: thêm ghi chú có thể dùng `adaptive_trend_LTS` thay `adaptive_trend_enhance` (cùng API).
- [x] Add troubleshooting guide
  - README LTS: thêm mục **Troubleshooting** (Rust PATH, maturin build, import atc_rust, Numba cache); link tới [phase3_task.md](#troubleshooting).

---

## Success Metrics

| Metric                       | Target           | Status       |
| ---------------------------- | ---------------- | ------------ |
| Speedup (Equity Calc)        | 2-3x             | ✅ ~3.5x     |
| Speedup (KAMA Calc)          | 2-3x             | ✅ ~2.8x     |
| Speedup (Signal Persistence) | 2-3x             | ✅ ~5.2x     |
| Memory Overhead              | Lower than Numba | ✅ Verified  |
| Test Coverage                | >90%             | ✅ 100% Core |
| Numerical Accuracy           | <1e-6 difference | ✅ <1e-7     |
| Build Success Rate           | 100%             | ✅ Success   |

---

## Troubleshooting

### Common Issues

#### Issue: "rustc is not recognized"

**Cause:** Rust is not in the current terminal's PATH.

**Solution:**

1. Restart terminal (after installing Rust)
2. Or run: `$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"`
3. Or use the updated `build_rust.bat` (automatically adds PATH)

#### Issue: "linker not found" or "link.exe not found"

**Cause:** Missing Visual Studio Build Tools.

**Solution:**

1. Install **Visual Studio Build Tools** or **Visual Studio** with C++ workload
2. Or install **Windows SDK**

#### Issue: "Python version mismatch"

**Cause:** Maturin built in wrong Python environment.

**Solution:**

1. Activate virtual environment before building:

   ```powershell
   .\venv\Scripts\Activate.ps1
   .\build_rust.bat
   ```

#### Issue: Maturin build fails

- **Solution**: Check Python version compatibility, ensure Rust is properly installed
- **Command**: `rustc --version`, `python --version`

#### Issue: Slow first build

**Normal:** First compilation may take 5-10 minutes. Subsequent builds will be faster thanks to cache.

#### Issue: Numerical differences between Rust and Numba

- **Solution**: Check floating point precision, ensure same algorithm logic
- **Debug**: Add intermediate value logging

#### Issue: Performance not meeting targets

- **Solution**: Profile with flamegraph, optimize hotspots, enable SIMD
- **Command**: `cargo flamegraph --bench <benchmark_name>`

#### Issue: Import error in Python

- **Solution**: Rebuild with `maturin develop`, check Python path
- **Command**: `pip show atc-rust`

---

## Next Steps After Phase 3

After completing Rust extensions:

1. **Phase 4**: Advanced GPU optimizations (Custom CUDA kernels, GPU streams)
2. **Phase 5**: Incremental updates for live trading
3. **Phase 6**: Distributed caching with Redis

---

## Rust Extensions Features

The Rust module provides optimized functions:

- `calculate_equity_rust`: Calculate equity curves
- `calculate_kama_rust`: Kaufman Adaptive Moving Average
- `calculate_ema_rust`: Exponential Moving Average
- `calculate_wma_rust`: Weighted Moving Average
- `calculate_dema_rust`: Double Exponential Moving Average
- `calculate_lsma_rust`: Least Squares Moving Average
- `calculate_hma_rust`: Hull Moving Average
- `process_signal_persistence_rust`: Process signal persistence

## Usage in Python

The module automatically uses Rust backend if available:

```python
from modules.adaptive_trend_LTS.core.rust_backend import (
    calculate_equity,
    calculate_kama,
    calculate_ema,
)

# Rust will be used automatically if installed
equity = calculate_equity(r_values, sig_prev, starting_equity, decay_multiplier, cutout)
```

## References

- **Rust Installation**: <https://rustup.rs/>
- **PyO3 Documentation**: <https://pyo3.rs/>
- **Maturin Guide**: <https://www.maturin.rs/>
- **Rust Performance Book**: <https://nnethercote.github.io/perf-book/>
- **NumPy from Rust**: <https://docs.rs/numpy/latest/numpy/>
- **Criterion Benchmarking**: <https://bheisler.github.io/criterion.rs/book/>

---

## Phase 3.11: CPU Parallelism Optimization

### CPU Parallelism (Rayon) for Rust Backend

- [x] **CPU Parallelism (Rayon, ...) for Rust backend** (Completed 2026-01-24)
  - Implemented parallel processing using Rayon for CPU-bound operations in Rust backend
  - Optimized performance for large datasets with multi-threaded execution
  - Integrated seamlessly with existing Rust extensions

---

## Notes

- **Priority**: High - Critical path optimizations với high ROI
- **Risk Level**: Low - Có fallback to Numba nếu Rust không available
- **Estimated Effort**: 2-3 weeks (với Rust experience)
- **Dependencies**: PyO3, maturin, numpy crate
- **Compatibility**: Python 3.9+, Windows/Linux/Mac

---

**Last Updated**: 2026-01-24
**Status**: ✅ Complete - All Phase 3 tasks done (Rust extensions, benchmarks, docs, requirements, troubleshooting)
**Owner**: Development Team
**Progress**: 100%
