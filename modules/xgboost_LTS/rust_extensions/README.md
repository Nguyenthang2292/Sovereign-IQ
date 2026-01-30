# XGBoost Rust Extensions

High-performance Rust implementations for XGBoost labeling and feature engineering.

## Building

```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release

# With native CPU optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Running Benchmarks

```bash
cargo bench
```

## Testing

```bash
cargo test
```

## Integration with Python

After building, the compiled library will be available as `xgboost_rust.pyd` (Windows) or `xgboost_rust.so` (Linux/Mac) in the `target/release` directory.

Copy it to the `modules/xgboost_LTS/rust_extensions/` directory to use from Python:

```python
from modules.xgboost_LTS.rust_extensions import rolling_quantile_rust

result = rolling_quantile_rust(data, window=50, q=0.5)
```

## Performance

Expected speedups over pure Python/Numba:
- Rolling calculations: 2-3x faster
- Volatility multiplier: 3-5x faster
- Full labeling pipeline: 2-5x faster
