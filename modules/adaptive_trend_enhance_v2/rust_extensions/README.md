# Rust Extensions for Adaptive Trend Classification (ATC)

This directory contains high-performance Rust implementations of the critical paths in the `adaptive_trend_enhance` module.

## Performance

The Rust backend is designed to provide a **2-3x speedup** over Numba JIT compilation by using explicit SIMD instructions and efficient memory management.

### Benchmarks (10,000 bars)

- **Equity Calculation**: ~31µs
- **KAMA Calculation**: ~148µs
- **Signal Persistence**: ~5µs

## Installation

### Prerequisites

- [Rust Toolchain](https://rustup.rs/) (latest stable)
- [Maturin](https://www.maturin.rs/) (`pip install maturin`)

### Build and Install

To build and install the extensions in the current environment:

```bash
cd modules/adaptive_trend_enhance_v2/rust_extensions
maturin develop --release
```

To build a wheel:

```bash
maturin build --release
```

## Module Structure

- `src/lib.rs`: PyO3 module definition and entry point.
- `src/equity.rs`: Optimized equity curve calculation logic.
- `src/kama.rs`: Optimized Kaufman Adaptive Moving Average (KAMA) logic.
- `src/signal_persistence.rs`: Optimized signal state tracking.

## Usage in Python

The main module automatically attempts to use the Rust backend if it is installed. You can also use the `rust_backend.py` wrapper directly:

```python
from modules.adaptive_trend_enhance_v2.core.rust_backend import calculate_equity

# Rust will be used automatically if available
equity = calculate_equity(r_values, sig_prev, starting_equity, decay_multiplier, cutout)
```

## Testing

Run the Rust unit tests:

```bash
cargo test
```

Run the benchmarks:

```bash
cargo bench
```
