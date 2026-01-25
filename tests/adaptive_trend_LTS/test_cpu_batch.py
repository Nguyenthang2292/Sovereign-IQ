import time

import numpy as np
import pandas as pd
import pytest

try:
    import atc_rust

    from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals
except ImportError:
    pytest.skip("ATC modules or rust extension not available", allow_module_level=True)


@pytest.fixture
def sample_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    n = 1000
    # Random walk
    prices = 100 + np.cumsum(np.random.normal(0, 1, n))
    return pd.Series(prices)


def test_cpu_batch_vs_python(sample_data):
    """Verify that CPU Batch Rust implementation matches Python implementation."""

    # Prepare input data
    symbols_data = {
        "SYM1": sample_data,
        "SYM2": sample_data * 1.1,  # Different scale
        # Add more similar data
        "SYM3": sample_data + 10,
    }

    config = {
        "ema_len": 20,
        "robustness": "Medium",
        "La": 0.02,
        "De": 0.03,
        "cutout": 0,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
    }

    # 1. Run Python Implementation (Sequential)
    # We run it per symbol manually as compute_atc_signals processes one
    python_results = {}
    start_py = time.time()
    for sym, prices in symbols_data.items():
        res = compute_atc_signals(
            prices,
            prefer_gpu=False,  # Force Python/Rust sequential
            use_cuda=False,
            **config,
        )
        python_results[sym] = res["Average_Signal"]
    end_py = time.time()

    # 2. Run Rust Batch CPU Implementation
    start_rust = time.time()
    # Ensure inputs match signature
    # compute_atc_signals_batch_cpu(symbols_data, ...)
    # Note: args need to match exact signature in lib.rs
    rust_results = atc_rust.compute_atc_signals_batch_cpu(
        symbols_data,
        ema_len=config["ema_len"],
        hull_len=28,  # Defaults
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        # weights defaults 1.0
        ema_w=1.0,
        hma_w=1.0,
        wma_w=1.0,
        dema_w=1.0,
        lsma_w=1.0,
        kama_w=1.0,
        robustness=config["robustness"],
        La=config["La"],
        De=config["De"],
        cutout=config["cutout"],
        long_threshold=config["long_threshold"],
        short_threshold=config["short_threshold"],
        _strategy_mode=False,
    )
    end_rust = time.time()

    print(f"\nTime Python (Seq): {end_py - start_py:.4f}s")
    print(f"Time Rust (Batch): {end_rust - start_rust:.4f}s")

    # 3. Compare Results
    for sym, py_res in python_results.items():
        rust_res = rust_results[sym]

        # Convert Rust numpy array to Series for comparison if needed, or just compare values
        rust_series = pd.Series(rust_res, index=py_res.index)

        # Check for NaN match
        # NaNs usually at start
        valid_idx = ~py_res.isna()

        # Compare valid values
        # Allow small tolerance
        diff = (py_res[valid_idx] - rust_series[valid_idx]).abs()
        max_diff = diff.max()
        print(f"Symbol {sym} Max Diff: {max_diff}")

        # Note: Precision differences might exist between vectorized Python and Rust
        # But should be very small (< 1e-6 typically, or maybe slightly higher due to ops order)
        if max_diff > 1e-6:
            # Debug where it fails
            idx_fail = diff.idxmax()
            print(f"Failed at {idx_fail}: Py={py_res[idx_fail]}, Rust={rust_series[idx_fail]}")

        # We assert a reasonable tolerance.
        # Python impl uses float64, Rust uses f64.
        assert max_diff < 1e-5, f"Mismatch for {sym}"


def test_cpu_batch_performance():
    """Performance benchmark for CPU Batch."""
    # Generate larger workload
    n_bars = 1500
    n_symbols = 50  # Enough to see parallelism benefit

    prices = np.random.normal(100, 1, n_bars).cumsum()
    sample_series = pd.Series(prices)

    symbols_data = {f"SYM_{i}": sample_series for i in range(n_symbols)}

    start = time.time()
    _ = atc_rust.compute_atc_signals_batch_cpu(symbols_data, ema_len=28, robustness="Medium", La=0.02, De=0.03)
    duration = time.time() - start
    print(f"\nProcessed {n_symbols} symbols x {n_bars} bars in {duration:.4f}s")
    print(f"Throughput: {n_symbols / duration:.2f} symbols/s")

    # Assert it's reasonably fast (e.g. > 10 symbols/s on typical CPU)
    # This is soft assertion
    if duration > 0:
        assert n_symbols / duration > 5.0
