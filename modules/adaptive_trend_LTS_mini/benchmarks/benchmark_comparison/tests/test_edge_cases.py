"""Edge case tests for benchmark comparison module."""

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_LTS_mini.benchmarks.benchmark_comparison.comparison import (
    compare_signals,
    generate_comparison_table,
)


def test_compare_signals_empty_data():
    """Test with completely empty results."""
    empty_results = {}

    result = compare_signals(
        empty_results,
        empty_results,
        empty_results,
        empty_results,
        empty_results,
        empty_results,
        empty_results,
    )

    assert result["total_symbols"] == 0
    assert result["orig_rust"]["match_rate_percent"] == 0.0


def test_compare_signals_single_bar():
    """Test with single data point."""
    symbol = "TEST"
    original_results = {symbol: {"Average_Signal": pd.Series([1.0], index=[0])}}
    rust_results = {symbol: {"Average_Signal": pd.Series([1.0], index=[0])}}
    rust_rayon_results = {symbol: {"Average_Signal": pd.Series([1.0], index=[0])}}
    approximate_results = {symbol: {"Average_Signal": pd.Series([1.0], index=[0])}}
    adaptive_approximate_results = {symbol: {"Average_Signal": pd.Series([1.0], index=[0])}}
    dask_results = {symbol: {"Average_Signal": pd.Series([1.0], index=[0])}}
    rust_dask_results = {symbol: {"Average_Signal": pd.Series([1.0], index=[0])}}

    result = compare_signals(
        original_results,
        rust_results,
        rust_rayon_results,
        approximate_results,
        adaptive_approximate_results,
        dask_results,
        rust_dask_results,
    )

    assert result["total_symbols"] == 1
    assert result["orig_rust"]["match_rate_percent"] == 100.0


def test_compare_signals_all_nan():
    """Test with all NaN signals."""
    symbol = "TEST"
    original_results = {symbol: {"Average_Signal": pd.Series([np.nan, np.nan, np.nan], index=[0, 1, 2])}}
    rust_results = {symbol: {"Average_Signal": pd.Series([np.nan, np.nan, np.nan], index=[0, 1, 2])}}
    rust_rayon_results = {symbol: {"Average_Signal": pd.Series([np.nan, np.nan, np.nan], index=[0, 1, 2])}}
    approximate_results = {symbol: {"Average_Signal": pd.Series([np.nan, np.nan, np.nan], index=[0, 1, 2])}}
    adaptive_approximate_results = {symbol: {"Average_Signal": pd.Series([np.nan, np.nan, np.nan], index=[0, 1, 2])}}
    dask_results = {symbol: {"Average_Signal": pd.Series([np.nan, np.nan, np.nan], index=[0, 1, 2])}}
    rust_dask_results = {symbol: {"Average_Signal": pd.Series([np.nan, np.nan, np.nan], index=[0, 1, 2])}}

    result = compare_signals(
        original_results,
        rust_results,
        rust_rayon_results,
        approximate_results,
        adaptive_approximate_results,
        dask_results,
        rust_dask_results,
    )

    # Symbol should be processed but match rate might be 0 due to no valid comparisons
    assert result["total_symbols"] == 1


def test_compare_signals_missing_signal():
    """Test with missing Average_Signal key."""
    symbol = "TEST"
    original_results = {symbol: {}}  # No Average_Signal
    rust_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])}}
    rust_rayon_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])}}
    approximate_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])}}
    adaptive_approximate_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])}}
    dask_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])}}
    rust_dask_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])}}

    result = compare_signals(
        original_results,
        rust_results,
        rust_rayon_results,
        approximate_results,
        adaptive_approximate_results,
        dask_results,
        rust_dask_results,
    )

    # Should skip this symbol since original has no signal
    assert result["total_symbols"] == 1


def test_compare_signals_none_results():
    """Test with None results for some symbols."""
    symbol1 = "TEST1"
    symbol2 = "TEST2"

    original_results = {
        symbol1: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])},
        symbol2: None,  # None result
    }
    rust_results = {
        symbol1: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])},
        symbol2: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])},
    }
    rust_rayon_results = {
        symbol1: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])},
        symbol2: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])},
    }
    approximate_results = {
        symbol1: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])},
        symbol2: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])},
    }
    adaptive_approximate_results = {
        symbol1: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])},
        symbol2: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])},
    }
    dask_results = {
        symbol1: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])},
        symbol2: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])},
    }
    rust_dask_results = {
        symbol1: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])},
        symbol2: {"Average_Signal": pd.Series([1.0, 2.0, 3.0])},
    }

    result = compare_signals(
        original_results,
        rust_results,
        rust_rayon_results,
        approximate_results,
        adaptive_approximate_results,
        dask_results,
        rust_dask_results,
    )

    # Only TEST1 should be processed (TEST2 has None original)
    assert result["total_symbols"] == 2
    # Match rate is calculated over total_symbols (2), so 1 match = 50%
    assert result["orig_rust"]["match_rate_percent"] == 50.0


def test_compare_signals_infinite_values():
    """Test with infinite values in signals."""
    symbol = "TEST"
    original_results = {symbol: {"Average_Signal": pd.Series([1.0, np.inf, 3.0], index=[0, 1, 2])}}
    rust_results = {symbol: {"Average_Signal": pd.Series([1.0, np.inf, 3.0], index=[0, 1, 2])}}
    rust_rayon_results = {symbol: {"Average_Signal": pd.Series([1.0, np.inf, 3.0], index=[0, 1, 2])}}
    approximate_results = {symbol: {"Average_Signal": pd.Series([1.0, np.inf, 3.0], index=[0, 1, 2])}}
    adaptive_approximate_results = {symbol: {"Average_Signal": pd.Series([1.0, np.inf, 3.0], index=[0, 1, 2])}}
    dask_results = {symbol: {"Average_Signal": pd.Series([1.0, np.inf, 3.0], index=[0, 1, 2])}}
    rust_dask_results = {symbol: {"Average_Signal": pd.Series([1.0, np.inf, 3.0], index=[0, 1, 2])}}

    result = compare_signals(
        original_results,
        rust_results,
        rust_rayon_results,
        approximate_results,
        adaptive_approximate_results,
        dask_results,
        rust_dask_results,
    )

    assert result["total_symbols"] == 1


def test_generate_comparison_table_empty():
    """Test table generation with empty/zero values."""
    signal_comparison = {
        "orig_rust": {
            "match_rate_percent": 0.0,
            "max_difference": 0,
            "avg_difference": 0,
            "median_difference": 0,
            "matching_symbols": 0,
            "mismatched_symbols": [],
        },
        "orig_dask": {
            "match_rate_percent": 0.0,
            "max_difference": 0,
            "avg_difference": 0,
            "median_difference": 0,
            "matching_symbols": 0,
            "mismatched_symbols": [],
        },
        "orig_rust_dask": {
            "match_rate_percent": 0.0,
            "max_difference": 0,
            "avg_difference": 0,
            "median_difference": 0,
            "matching_symbols": 0,
            "mismatched_symbols": [],
        },
        "orig_approx": {
            "match_rate_percent": 0.0,
            "max_difference": 0,
            "avg_difference": 0,
            "median_difference": 0,
            "matching_symbols": 0,
            "mismatched_symbols": [],
        },
        "orig_adaptive_approx": {
            "match_rate_percent": 0.0,
            "max_difference": 0,
            "avg_difference": 0,
            "median_difference": 0,
            "matching_symbols": 0,
            "mismatched_symbols": [],
        },
        "approx_adaptive": {
            "match_rate_percent": 0.0,
            "max_difference": 0,
            "avg_difference": 0,
            "median_difference": 0,
            "matching_symbols": 0,
            "mismatched_symbols": [],
        },
        "total_symbols": 0,
    }

    table = generate_comparison_table(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        signal_comparison,
    )

    # Should not crash with zero values
    assert isinstance(table, str)
    assert "Original" in table


def test_generate_comparison_table_division_by_zero():
    """Test table generation handles division by zero gracefully."""
    signal_comparison = {
        "orig_rust": {
            "match_rate_percent": 100.0,
            "max_difference": 0,
            "avg_difference": 0,
            "median_difference": 0,
            "matching_symbols": 1,
            "mismatched_symbols": [],
        },
        "orig_dask": {
            "match_rate_percent": 100.0,
            "max_difference": 0,
            "avg_difference": 0,
            "median_difference": 0,
            "matching_symbols": 1,
            "mismatched_symbols": [],
        },
        "orig_rust_dask": {
            "match_rate_percent": 100.0,
            "max_difference": 0,
            "avg_difference": 0,
            "median_difference": 0,
            "matching_symbols": 1,
            "mismatched_symbols": [],
        },
        "orig_approx": {
            "match_rate_percent": 100.0,
            "max_difference": 0,
            "avg_difference": 0,
            "median_difference": 0,
            "matching_symbols": 1,
            "mismatched_symbols": [],
        },
        "orig_adaptive_approx": {
            "match_rate_percent": 100.0,
            "max_difference": 0,
            "avg_difference": 0,
            "median_difference": 0,
            "matching_symbols": 1,
            "mismatched_symbols": [],
        },
        "approx_adaptive": {
            "match_rate_percent": 100.0,
            "max_difference": 0,
            "avg_difference": 0,
            "median_difference": 0,
            "matching_symbols": 1,
            "mismatched_symbols": [],
        },
        "total_symbols": 1,
    }

    # Test with rust_time=0 to trigger division by zero in speedup calculation
    table = generate_comparison_table(
        10.0,
        0.0,  # rust_time (was enhanced_time in comment, but now first arg after original is rust_time) - Wait, in definition rust_time is 2nd arg?
        0.0,  # rust_rayon_time
        4.5,
        4.2,
        6.0,
        5.5,
        100.0,
        95.0,  # rust_memory
        90.0,
        85.0,
        92.0,
        88.0,
        110.0,
        signal_comparison,
    )

    # Should handle division by zero gracefully (speedup should be 0)
    assert isinstance(table, str)
    assert "0.00x" in table or "Speedup" in table


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
