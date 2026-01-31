"""Unit tests for benchmark comparison module."""

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_LTS_mini.benchmarks.benchmark_comparison.comparison import (
    SIGNAL_MATCH_TOLERANCE,
    compare_signals,
    generate_comparison_table,
)


def test_compare_signals_basic():
    """Test basic signal comparison functionality."""
    # Create sample results
    symbol = "TEST"
    original_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])}}
    rust_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])}}
    rust_rayon_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])}}
    approximate_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])}}
    adaptive_approximate_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])}}
    dask_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])}}
    rust_dask_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])}}

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
    assert result["orig_approx"]["match_rate_percent"] == 100.0


def test_compare_signals_with_mismatch():
    """Test signal comparison with mismatched signals."""
    symbol = "TEST"
    original_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])}}
    # Rust has a significant difference
    rust_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.5], index=[0, 1, 2])}}
    rust_rayon_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])}}
    approximate_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])}}
    adaptive_approximate_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])}}
    dask_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])}}
    rust_dask_results = {symbol: {"Average_Signal": pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])}}

    result = compare_signals(
        original_results,
        rust_results,
        rust_rayon_results,
        approximate_results,
        adaptive_approximate_results,
        dask_results,
        rust_dask_results,
    )

    assert result["orig_rust"]["match_rate_percent"] == 0.0
    assert result["orig_rust"]["max_difference"] > SIGNAL_MATCH_TOLERANCE


def test_compare_signals_with_nan():
    """Test signal comparison with NaN values."""
    symbol = "TEST"
    original_results = {symbol: {"Average_Signal": pd.Series([1.0, np.nan, 3.0], index=[0, 1, 2])}}
    rust_results = {symbol: {"Average_Signal": pd.Series([1.0, np.nan, 3.0], index=[0, 1, 2])}}
    rust_rayon_results = {symbol: {"Average_Signal": pd.Series([1.0, np.nan, 3.0], index=[0, 1, 2])}}
    approximate_results = {symbol: {"Average_Signal": pd.Series([1.0, np.nan, 3.0], index=[0, 1, 2])}}
    adaptive_approximate_results = {symbol: {"Average_Signal": pd.Series([1.0, np.nan, 3.0], index=[0, 1, 2])}}
    dask_results = {symbol: {"Average_Signal": pd.Series([1.0, np.nan, 3.0], index=[0, 1, 2])}}
    rust_dask_results = {symbol: {"Average_Signal": pd.Series([1.0, np.nan, 3.0], index=[0, 1, 2])}}

    result = compare_signals(
        original_results,
        rust_results,
        rust_rayon_results,
        approximate_results,
        adaptive_approximate_results,
        dask_results,
        rust_dask_results,
    )

    # Should handle NaN values gracefully
    assert result["total_symbols"] == 1
    assert result["orig_rust"]["match_rate_percent"] == 100.0


def test_compare_signals_empty_data():
    """Test signal comparison with empty data."""
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


def test_generate_comparison_table():
    """Test table generation."""
    signal_comparison = {
        "orig_rust": {
            "match_rate_percent": 100.0,
            "max_difference": 0.0,
            "avg_difference": 0.0,
            "median_difference": 0.0,
            "matching_symbols": 20,
            "mismatched_symbols": [],
        },
        "orig_dask": {
            "match_rate_percent": 100.0,
            "max_difference": 0.0,
            "avg_difference": 0.0,
            "median_difference": 0.0,
            "matching_symbols": 20,
            "mismatched_symbols": [],
        },
        "orig_rust_dask": {
            "match_rate_percent": 100.0,
            "max_difference": 0.0,
            "avg_difference": 0.0,
            "median_difference": 0.0,
            "matching_symbols": 20,
            "mismatched_symbols": [],
        },
        "orig_approx": {
            "match_rate_percent": 100.0,
            "max_difference": 0.0,
            "avg_difference": 0.0,
            "median_difference": 0.0,
            "matching_symbols": 20,
            "mismatched_symbols": [],
        },
        "orig_adaptive_approx": {
            "match_rate_percent": 100.0,
            "max_difference": 0.0,
            "avg_difference": 0.0,
            "median_difference": 0.0,
            "matching_symbols": 20,
            "mismatched_symbols": [],
        },
        "approx_adaptive": {
            "match_rate_percent": 95.0,
            "max_difference": 0.3,
            "avg_difference": 0.08,
            "median_difference": 0.04,
            "matching_symbols": 19,
            "mismatched_symbols": ["SYM2"],
        },
        "total_symbols": 20,
    }

    table = generate_comparison_table(
        10.0,
        4.0,
        3.0,
        4.5,
        4.2,
        6.0,
        5.5,
        100.0,
        90.0,
        85.0,
        92.0,
        88.0,
        110.0,
        105.0,
        signal_comparison,
    )

    # Check that table contains expected data
    assert "Original" in table
    assert "Rust" in table
    assert "Dask" in table
    assert "100.00%" in table or "100.0%" in table


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
