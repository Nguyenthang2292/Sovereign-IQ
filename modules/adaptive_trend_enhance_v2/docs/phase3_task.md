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

- [ ] Download và cài đặt Rust từ https://rustup.rs/

  ```bash
  # Windows
  # Download rustup-init.exe và chạy

  # Verify installation
  rustc --version
  cargo --version
  ```

#### 1.2 Install PyO3 Development Tools

- [ ] Cài đặt `maturin` (build tool for PyO3)
  ```bash
  pip install maturin
  ```

#### 1.3 Setup Python Development Headers

- [ ] Đảm bảo Python development headers đã được cài đặt
  ```bash
  # Windows: Thường đã có sẵn với Python installation
  # Verify: python -m pip install --upgrade pip
  ```

#### 1.4 Install Required Dependencies

- [ ] Cài đặt các dependencies cần thiết
  ```bash
  pip install numpy pytest pytest-benchmark
  cargo install cargo-criterion  # For Rust benchmarking
  ```

---

## Phase 3.1: Project Structure Setup

### 2. Create Rust Project

#### 2.1 Initialize Rust Library

- [ ] Tạo thư mục Rust project
  ```bash
  cd modules/adaptive_trend_enhance_v2
  mkdir rust_extensions
  cd rust_extensions
  maturin init
  ```

#### 2.2 Configure Cargo.toml

- [ ] Cập nhật `Cargo.toml` với dependencies cần thiết

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

- [ ] Tạo cấu trúc thư mục
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

- [ ] Tạo `pyproject.toml` cho maturin

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

- [ ] Tạo file `src/lib.rs`

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

- [ ] Tạo file `src/equity.rs` với implementation cơ bản

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

- [ ] Đọc và phân tích code Numba hiện tại từ Python
  - File: `modules/adaptive_trend_enhance/core/process_layer1/weighted_signal.py`
  - Function: `calculate_equity` (hoặc tương tự)

- [ ] Implement equity calculation algorithm
  ```rust
  // Pseudo-code structure:
  // 1. Initialize equity array with starting_equity
  // 2. Loop through r_values and sig_prev
  // 3. Apply decay and cutout logic
  // 4. Calculate cumulative equity
  // 5. Return result as PyArray1
  ```

#### 3.4 Add SIMD Optimizations

- [ ] Sử dụng explicit SIMD instructions (optional, nâng cao)
  ```rust
  // Use packed_simd or std::simd (nightly) for better performance
  // Or rely on LLVM auto-vectorization with -C target-cpu=native
  ```

#### 3.5 Add Parallel Processing (nếu applicable)

- [ ] Sử dụng `rayon` để parallel process các chunks lớn
  ```rust
  use rayon::prelude::*;
  // Use par_iter() for parallel processing where appropriate
  ```

---

## Phase 3.3: Implementation - KAMA Calculation

### 4. Implement KAMA Calculation in Rust

#### 4.1 Analyze Current KAMA Implementation

- [ ] Đọc code Python hiện tại
  - File: `modules/adaptive_trend_enhance/core/compute_moving_averages/calculate_kama_atc.py`
  - Hiểu rõ logic: efficiency ratio, smoothing constant, adaptive calculation

#### 4.2 Implement KAMA Core Logic

- [ ] Tạo implementation trong `src/kama.rs`

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

- [ ] Tối ưu hóa các nested loops trong KAMA calculation
  - Use iterators instead of index-based loops
  - Consider loop unrolling for inner loops
  - Pre-allocate buffers to avoid reallocations

#### 4.4 Add Vectorization

- [ ] Sử dụng SIMD cho các operations có thể vectorize
  - Element-wise operations
  - Cumulative sums
  - Window-based calculations

---

## Phase 3.4: Implementation - Signal Persistence

### 5. Implement Signal Persistence Logic

#### 5.1 Analyze Signal Persistence Requirements

- [ ] Đọc code Python hiện tại
  - File: `modules/adaptive_trend_enhance/core/signal_detection/generate_signal.py`
  - Hiểu logic: signal filtering, persistence checking, state management

#### 5.2 Implement Core Logic

- [ ] Tạo implementation trong `src/signal_persistence.rs`

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

- [ ] Implement efficient state machine for signal tracking
  - Use enums for signal states
  - Minimize allocations
  - Use bitwise operations where applicable

---

## Phase 3.5: Testing & Validation

### 6. Unit Testing

#### 6.1 Create Rust Unit Tests

- [ ] Tạo file `tests/integration_tests.rs`

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

- [ ] Chạy tests
  ```bash
  cargo test
  ```

#### 6.3 Create Python Integration Tests

- [ ] Tạo file `tests/test_rust_extensions.py` trong Python project

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

- [ ] So sánh kết quả Rust vs Numba/Python
  - Use `np.allclose()` với appropriate tolerance
  - Test với multiple test cases (small, large, edge cases)
  - Verify edge cases: NaN, Inf, empty arrays

---

## Phase 3.6: Benchmarking

### 7. Performance Benchmarking

#### 7.1 Create Rust Benchmarks

- [ ] Tạo benchmarks trong `benches/`

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

- [ ] Chạy benchmarks
  ```bash
  cargo bench
  ```

#### 7.3 Create Python Benchmarks

- [ ] Tạo file `benchmark_rust_vs_numba.py`

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

- [ ] So sánh performance Rust vs Numba
  - Record baseline times
  - Verify 2-3x speedup target
  - Test với different data sizes
  - Profile memory usage

#### 7.5 Profile and Optimize

- [ ] Sử dụng profiling tools

  ```bash
  # Rust profiling
  cargo install flamegraph
  cargo flamegraph --bench equity_benchmark
  ```

- [ ] Optimize hotspots identified in profiling
  - Review flamegraph output
  - Optimize bottleneck functions
  - Re-benchmark after optimizations

---

## Phase 3.7: Integration

### 8. Python Integration

#### 8.1 Build Rust Extension

- [ ] Build wheel với maturin
  ```bash
  cd rust_extensions
  maturin develop  # Development build
  maturin build --release  # Production build
  ```

#### 8.2 Create Python Wrapper Module

- [ ] Tạo file `modules/adaptive_trend_enhance_v2/core/rust_backend.py`

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

#### 8.3 Update Configuration

- [ ] Thêm config option để enable/disable Rust backend
  ```python
  # config/adaptive_trend_enhance.py
  ATC_CONFIG = {
      'use_rust_backend': True,  # Set to False to use Numba
      # ... other configs
  }
  ```

#### 8.4 Update Main Module

- [ ] Cập nhật các files sử dụng equity/KAMA calculations
  - Import from new `rust_backend.py` wrapper
  - Pass `use_rust` parameter from config
  - Ensure backward compatibility

---

## Phase 3.8: Documentation

### 9. Documentation

#### 9.1 Code Documentation

- [ ] Thêm Rust docstrings

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

- [ ] Thêm Python docstrings cho wrapper functions

#### 9.2 Update README

- [ ] Tạo `rust_extensions/README.md`
  - Installation instructions
  - Build instructions
  - Usage examples
  - Performance benchmarks
  - Troubleshooting

#### 9.3 Update Main Module Documentation

- [ ] Cập nhật `modules/adaptive_trend_enhance_v2/README.md`
  - Mention Rust backend option
  - Performance improvements
  - Setup instructions

#### 9.4 Create Performance Report

- [ ] Tạo file `docs/rust_performance_report.md`
  - Benchmark results
  - Speedup metrics
  - Memory usage comparison
  - Recommendations

---

## Phase 3.9: Deployment

### 10. Build & Deployment

#### 10.1 Create Build Script

- [ ] Tạo script `build_rust_extensions.sh` (hoặc `.bat` cho Windows)
  ```bash
  #!/bin/bash
  cd rust_extensions
  maturin build --release
  pip install target/wheels/*.whl
  ```

#### 10.2 Update Requirements

- [ ] Thêm vào `requirements.txt` (optional dependency)
  ```
  # Rust extensions (optional, provides 2-3x speedup)
  # Build from source:
  # cd modules/adaptive_trend_enhance_v2/rust_extensions
  # maturin develop --release
  ```

#### 10.3 CI/CD Integration (Optional)

- [ ] Setup GitHub Actions để build wheels
  - Matrix build for Windows/Linux/Mac
  - Upload wheels as artifacts
  - Publish to PyPI (optional)

#### 10.4 Testing on Different Platforms

- [ ] Test trên Windows
- [ ] Test trên Linux (nếu có access)
- [ ] Verify fallback to Numba works correctly

---

## Phase 3.10: Validation & Rollout

### 11. Final Validation

#### 11.1 End-to-End Testing

- [ ] Chạy full pipeline với Rust backend
  ```bash
  python benchmark_comparison.py --use-rust
  ```

#### 11.2 Regression Testing

- [ ] Verify không có regression trong accuracy
  - So sánh signals generated với Numba vs Rust
  - Check với historical data
  - Validate edge cases

#### 11.3 Performance Validation

- [ ] Confirm 2-3x speedup target được đạt
- [ ] Verify memory usage improvements
- [ ] Check CPU utilization

#### 11.4 Documentation Review

- [ ] Review tất cả documentation
- [ ] Update examples
- [ ] Add troubleshooting guide

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

#### Issue: Maturin build fails

- **Solution**: Check Python version compatibility, ensure Rust is properly installed
- **Command**: `rustc --version`, `python --version`

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

## References

- **PyO3 Documentation**: https://pyo3.rs/
- **Maturin Guide**: https://www.maturin.rs/
- **Rust Performance Book**: https://nnethercote.github.io/perf-book/
- **NumPy from Rust**: https://docs.rs/numpy/latest/numpy/
- **Criterion Benchmarking**: https://bheisler.github.io/criterion.rs/book/

---

## Notes

- **Priority**: High - Critical path optimizations với high ROI
- **Risk Level**: Low - Có fallback to Numba nếu Rust không available
- **Estimated Effort**: 2-3 weeks (với Rust experience)
- **Dependencies**: PyO3, maturin, numpy crate
- **Compatibility**: Python 3.9+, Windows/Linux/Mac

---

**Last Updated**: 2026-01-23
**Status**: Ready for implementation
**Owner**: Development Team
