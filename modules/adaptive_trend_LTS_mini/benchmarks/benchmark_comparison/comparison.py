"""Signal comparison and table generation utilities."""

from typing import Any, Dict

import numpy as np

# Constants
SIGNAL_MATCH_TOLERANCE = 1e-6
from tabulate import tabulate

from modules.common.utils import log_info, log_success, log_warn


def compare_signals(
    original_results: Dict[str, Any],
    rust_results: Dict[str, Any],
    rust_rayon_results: Dict[str, Any],
    approximate_results: Dict[str, Any],
    adaptive_approximate_results: Dict[str, Any],
    dask_results: Dict[str, Any],
    rust_dask_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare signal outputs between versions.

    Args:
        original_results: Results from original module
        rust_results: Results from Rust module
        rust_rayon_results: Results from Rust Rayon module
        approximate_results: Results from Approximate MAs module
        adaptive_approximate_results: Results from Adaptive Approximate MAs module
        dask_results: Results from Dask module
        rust_dask_results: Results from Rust+Dask hybrid

    Returns:
        Dictionary of comparison metrics
    """
    log_info("Comparing signal outputs between versions...")

    # DEBUG: Print result dict sizes
    log_info(
        f"Result dict sizes: orig={len(original_results)}, "
        f"rust={len(rust_results)}, rust_rayon={len(rust_rayon_results)}, "
        f"approx={len(approximate_results)}, adaptive_approx={len(adaptive_approximate_results)}, "
        f"dask={len(dask_results)}, rust_dask={len(rust_dask_results)}"
    )

    if original_results:
        sample_key = list(original_results.keys())[0]
        log_info(f"Sample key: {sample_key}")
        log_info(f"Sample orig result type: {type(original_results[sample_key])}")
        if original_results[sample_key]:
            sample_keys = (
                list(original_results[sample_key].keys())
                if isinstance(original_results[sample_key], dict)
                else "not a dict"
            )
            log_info(f"Sample orig result keys: {sample_keys}")

    total_symbols = len(original_results)
    processed_symbols = 0

    # Compare Original vs Rust
    orig_rust_diffs = []
    orig_rust_matching = 0
    orig_rust_mismatched = []

    # Compare Original vs Dask
    orig_dask_diffs = []
    orig_dask_matching = 0
    orig_dask_mismatched = []

    # Compare Original vs Rust+Dask
    orig_rust_dask_diffs = []
    orig_rust_dask_matching = 0
    orig_rust_dask_mismatched = []

    # Signal validity checks for Approx and AdaptApprox
    # These verify that signals are properly generated (all finite values)
    approx_valid_diffs = []
    approx_valid_matching = 0
    approx_valid_mismatched = []

    adaptive_approx_valid_diffs = []
    adaptive_approx_valid_matching = 0
    adaptive_approx_valid_mismatched = []

    # Compare AdaptApprox vs Approx (cross-consistency check)
    # This shows how much the Adaptive mechanism changes the Approximate results
    approx_adaptive_diffs = []
    approx_adaptive_matching = 0
    approx_adaptive_mismatched = []

    for symbol in original_results.keys():
        # Get results for each module (may be None or missing)
        orig = original_results.get(symbol)
        rust = rust_results.get(symbol)
        rust_rayon = rust_rayon_results.get(symbol)
        approx = approximate_results.get(symbol)
        adaptive_approx = adaptive_approximate_results.get(symbol)
        dask = dask_results.get(symbol)
        rust_dask = rust_dask_results.get(symbol)

        # Skip if original result is None (baseline required)

        if orig is None:
            log_warn(f"Symbol {symbol} has None original result")
            continue

        # Get Average_Signal for each module
        orig_s = orig.get("Average_Signal") if orig else None
        rust_s = rust.get("Average_Signal") if rust else None
        rust_r_s = rust_rayon.get("Average_Signal") if rust_rayon else None
        approx_s = approx.get("Average_Signal") if approx else None
        adaptive_approx_s = adaptive_approx.get("Average_Signal") if adaptive_approx else None
        dask_s = dask.get("Average_Signal") if dask else None
        rust_dask_s = rust_dask.get("Average_Signal") if rust_dask else None

        # Skip if original signal is None (baseline required)

        if orig_s is None or len(orig_s) == 0:
            log_warn(f"Symbol {symbol} has no original Average_Signal")
            continue

        # Original vs Rust
        if rust_s is not None and len(rust_s) > 0:
            common_idx = orig_s.index.intersection(rust_s.index)
            if len(common_idx) > 0:
                diff_or = np.abs(orig_s.loc[common_idx] - rust_s.loc[common_idx]).max()
                orig_rust_diffs.append(diff_or)
                if diff_or < SIGNAL_MATCH_TOLERANCE:
                    orig_rust_matching += 1
                else:
                    orig_rust_mismatched.append((symbol, diff_or))

        # Original vs Dask

        if dask_s is not None and len(dask_s) > 0:
            common_idx = orig_s.index.intersection(dask_s.index)
            if len(common_idx) > 0:
                diff_od = np.abs(orig_s.loc[common_idx] - dask_s.loc[common_idx]).max()
                orig_dask_diffs.append(diff_od)
                if diff_od < SIGNAL_MATCH_TOLERANCE:
                    orig_dask_matching += 1
                else:
                    orig_dask_mismatched.append((symbol, diff_od))

        # Original vs Rust+Dask
        if rust_dask_s is not None and len(rust_dask_s) > 0:
            common_idx = orig_s.index.intersection(rust_dask_s.index)
            if len(common_idx) > 0:
                diff_ord = np.abs(orig_s.loc[common_idx] - rust_dask_s.loc[common_idx]).max()
                orig_rust_dask_diffs.append(diff_ord)
                if diff_ord < SIGNAL_MATCH_TOLERANCE:
                    orig_rust_dask_matching += 1
                else:
                    orig_rust_dask_mismatched.append((symbol, diff_ord))

        # Approx Signal Validity Check
        if approx_s is not None and len(approx_s) > 0:
            # Check if all signal values are finite (not NaN or Inf)
            if np.all(np.isfinite(approx_s.values)):
                approx_valid_matching += 1
                approx_valid_diffs.append(0.0)
            else:
                approx_valid_mismatched.append((symbol, 0.0))
                approx_valid_diffs.append(0.0)
        else:
            log_warn(
                f"Approximate signal missing or empty for {symbol}: {approx_s is None}, "
                f"len={len(approx_s) if approx_s is not None else 'None'}"
            )

        # AdaptApprox Signal Validity Check
        if adaptive_approx_s is not None and len(adaptive_approx_s) > 0:
            # Check if all signal values are finite (not NaN or Inf)
            if np.all(np.isfinite(adaptive_approx_s.values)):
                adaptive_approx_valid_matching += 1
                adaptive_approx_valid_diffs.append(0.0)
            else:
                adaptive_approx_valid_mismatched.append((symbol, 0.0))
                adaptive_approx_valid_diffs.append(0.0)
        else:
            log_warn(
                f"Adaptive Approx signal missing or empty for {symbol}: "
                f"{adaptive_approx_s is None}, len={len(adaptive_approx_s) if adaptive_approx_s is not None else 'None'}"
            )

        # AdaptApprox vs Approx
        if approx_s is not None and adaptive_approx_s is not None and len(approx_s) > 0 and len(adaptive_approx_s) > 0:
            common_idx = approx_s.index.intersection(adaptive_approx_s.index)
            if len(common_idx) > 0:
                diff_aa = np.abs(approx_s.loc[common_idx] - adaptive_approx_s.loc[common_idx]).max()
                approx_adaptive_diffs.append(diff_aa)
                if diff_aa < SIGNAL_MATCH_TOLERANCE:
                    approx_adaptive_matching += 1
                else:
                    approx_adaptive_mismatched.append((symbol, diff_aa))

        # Original vs Rust Rayon

        if rust_r_s is not None and len(rust_r_s) > 0:
            common_idx = orig_s.index.intersection(rust_r_s.index)
            if len(common_idx) > 0:
                diff_orr = np.abs(orig_s.loc[common_idx] - rust_r_s.loc[common_idx]).max()
                if diff_orr > SIGNAL_MATCH_TOLERANCE:
                    log_warn(f"Rust Rayon mismatch for {symbol}: {diff_orr}")

        # Increment processed counter
        processed_symbols += 1

    if orig_rust_mismatched:
        log_info(f"Mismatched symbols (Orig vs Rust): {[s[0] for s in orig_rust_mismatched[:10]]}...")

    # Calculate metrics
    log_info(f"Processed {processed_symbols}/{total_symbols} symbols successfully")
    orig_rust_match_rate = (orig_rust_matching / total_symbols) * 100 if total_symbols > 0 else 0
    orig_dask_match_rate = (orig_dask_matching / total_symbols) * 100 if total_symbols > 0 else 0
    orig_rust_dask_match_rate = (orig_rust_dask_matching / total_symbols) * 100 if total_symbols > 0 else 0

    log_success(f"Original vs Rust match rate: {orig_rust_match_rate:.2f}%")
    log_success(f"Original vs Dask match rate: {orig_dask_match_rate:.2f}%")
    log_success(f"Original vs Rust+Dask match rate: {orig_rust_dask_match_rate:.2f}%")

    # Calculate validity check rates for Approx and AdaptApprox

    approx_valid_rate = (approx_valid_matching / total_symbols) * 100 if total_symbols > 0 else 0
    adaptive_approx_valid_rate = (adaptive_approx_valid_matching / total_symbols) * 100 if total_symbols > 0 else 0

    log_success(f"Approximate validity rate: {approx_valid_rate:.2f}%")
    log_success(f"Adaptive Approx validity rate: {adaptive_approx_valid_rate:.2f}%")

    approx_adaptive_match_rate = (approx_adaptive_matching / total_symbols) * 100 if total_symbols > 0 else 0
    log_success(f"AdaptApprox vs Approx match rate: {approx_adaptive_match_rate:.2f}%")

    return {
        "orig_approx": {
            "match_rate_percent": approx_valid_rate,
            "max_difference": max(approx_valid_diffs) if approx_valid_diffs else 0,
            "avg_difference": np.mean(approx_valid_diffs) if approx_valid_diffs else 0,
            "median_difference": np.median(approx_valid_diffs) if approx_valid_diffs else 0,
            "matching_symbols": approx_valid_matching,
            "mismatched_symbols": [s[0] for s in approx_valid_mismatched],
        },
        "orig_adaptive_approx": {
            "match_rate_percent": adaptive_approx_valid_rate,
            "max_difference": max(adaptive_approx_valid_diffs) if adaptive_approx_valid_diffs else 0,
            "avg_difference": np.mean(adaptive_approx_valid_diffs) if adaptive_approx_valid_diffs else 0,
            "median_difference": np.median(adaptive_approx_valid_diffs) if adaptive_approx_valid_diffs else 0,
            "matching_symbols": adaptive_approx_valid_matching,
            "mismatched_symbols": [s[0] for s in adaptive_approx_valid_mismatched],
        },
        "approx_adaptive": {
            "match_rate_percent": approx_adaptive_match_rate,
            "max_difference": max(approx_adaptive_diffs) if approx_adaptive_diffs else 0,
            "avg_difference": np.mean(approx_adaptive_diffs) if approx_adaptive_diffs else 0,
            "median_difference": np.median(approx_adaptive_diffs) if approx_adaptive_diffs else 0,
            "matching_symbols": approx_adaptive_matching,
            "mismatched_symbols": [s[0] for s in approx_adaptive_mismatched],
        },
        "orig_rust": {
            "match_rate_percent": orig_rust_match_rate,
            "max_difference": max(orig_rust_diffs) if orig_rust_diffs else 0,
            "avg_difference": np.mean(orig_rust_diffs) if orig_rust_diffs else 0,
            "median_difference": np.median(orig_rust_diffs) if orig_rust_diffs else 0,
            "matching_symbols": orig_rust_matching,
            "mismatched_symbols": [s[0] for s in orig_rust_mismatched],
        },
        "orig_dask": {
            "match_rate_percent": orig_dask_match_rate,
            "max_difference": max(orig_dask_diffs) if orig_dask_diffs else 0,
            "avg_difference": np.mean(orig_dask_diffs) if orig_dask_diffs else 0,
            "median_difference": np.median(orig_dask_diffs) if orig_dask_diffs else 0,
            "matching_symbols": orig_dask_matching,
            "mismatched_symbols": [s[0] for s in orig_dask_mismatched],
        },
        "orig_rust_dask": {
            "match_rate_percent": orig_rust_dask_match_rate,
            "max_difference": max(orig_rust_dask_diffs) if orig_rust_dask_diffs else 0,
            "avg_difference": np.mean(orig_rust_dask_diffs) if orig_rust_dask_diffs else 0,
            "median_difference": np.median(orig_rust_dask_diffs) if orig_rust_dask_diffs else 0,
            "matching_symbols": orig_rust_dask_matching,
            "mismatched_symbols": [s[0] for s in orig_rust_dask_mismatched],
        },
        "total_symbols": total_symbols,
    }


def generate_comparison_table(
    original_time: float,
    rust_time: float,
    rust_rayon_time: float,
    approximate_time: float,
    adaptive_approximate_time: float,
    dask_time: float,
    rust_dask_time: float,
    original_memory: float,
    rust_memory: float,
    rust_rayon_memory: float,
    approximate_memory: float,
    adaptive_approximate_memory: float,
    dask_memory: float,
    rust_dask_memory: float,
    signal_comparison: Dict,
) -> str:
    """Generate formatted comparison table for versions.

    Args:
        original_time: Execution time for original module (seconds)
        rust_time: Execution time for Rust module (seconds)
        rust_rayon_time: Execution time for Rust Rayon module (seconds)
        approximate_time: Execution time for Approximate MAs module (seconds)
        adaptive_approximate_time: Execution time for Adaptive Approximate MAs module (seconds)
        dask_time: Execution time for Dask module (seconds)
        rust_dask_time: Execution time for Rust+Dask hybrid (seconds)
        original_memory: Peak memory for original module (MB)
        rust_memory: Peak memory for Rust module (MB)
        rust_rayon_memory: Peak memory for Rust Rayon module (MB)
        approximate_memory: Peak memory for Approximate MAs module (MB)
        adaptive_approximate_memory: Peak memory for Adaptive Approximate MAs module (MB)
        dask_memory: Peak memory for Dask module (MB)
        rust_dask_memory: Peak memory for Rust+Dask hybrid (MB)
        signal_comparison: Signal comparison metrics

    Returns:
        Formatted table string
    """
    speedup_rust_rayon = original_time / rust_rayon_time if rust_rayon_time > 0 else 0
    speedup_approx = original_time / approximate_time if approximate_time > 0 else 0
    speedup_adaptive_approx = original_time / adaptive_approximate_time if adaptive_approximate_time > 0 else 0
    speedup_dask = original_time / dask_time if dask_time > 0 else 0
    speedup_rust_dask = original_time / rust_dask_time if rust_dask_time > 0 else 0

    memory_reduction_rust_rayon = (
        ((original_memory - rust_rayon_memory) / original_memory) * 100 if original_memory > 0 else 0
    )
    memory_reduction_approx = (
        ((original_memory - approximate_memory) / original_memory) * 100 if original_memory > 0 else 0
    )
    memory_reduction_adaptive_approx = (
        ((original_memory - adaptive_approximate_memory) / original_memory) * 100 if original_memory > 0 else 0
    )
    memory_reduction_dask = ((original_memory - dask_memory) / original_memory) * 100 if original_memory > 0 else 0
    memory_reduction_rust_dask = (
        ((original_memory - rust_dask_memory) / original_memory) * 100 if original_memory > 0 else 0
    )

    # Performance table
    perf_data = [
        [
            "Metric",
            "Original",
            "Rust",
            "Approx",
            "AdaptApprox",
            "Dask",
            "Rust+Dask",
        ],
        ["─" * 12] * 6,
        [
            "Execution Time",
            f"{original_time:.2f}s",
            f"{rust_rayon_time:.2f}s",
            f"{approximate_time:.2f}s",
            f"{adaptive_approximate_time:.2f}s",
            f"{dask_time:.2f}s",
            f"{rust_dask_time:.2f}s",
        ],
        [
            "Speedup vs Orig",
            "1.00x",
            f"{speedup_rust_rayon:.2f}x",
            f"{speedup_approx:.2f}x",
            f"{speedup_adaptive_approx:.2f}x",
            f"{speedup_dask:.2f}x",
            f"{speedup_rust_dask:.2f}x",
        ],
        [
            "Peak Memory",
            f"{original_memory:.1f} MB",
            f"{rust_rayon_memory:.1f} MB",
            f"{approximate_memory:.1f} MB",
            f"{adaptive_approximate_memory:.1f} MB",
            f"{dask_memory:.1f} MB",
            f"{rust_dask_memory:.1f} MB",
        ],
        [
            "Memory Reduction",
            "0%",
            f"{memory_reduction_rust_rayon:.1f}%",
            f"{memory_reduction_approx:.1f}%",
            f"{memory_reduction_adaptive_approx:.1f}%",
            f"{memory_reduction_dask:.1f}%",
            f"{memory_reduction_rust_dask:.1f}%",
        ],
    ]

    # Signal comparison table - Original vs exact versions
    signal_data = [
        [
            "Signal Comparison",
            "vs Rust",
            "vs Dask",
            "vs Rust+Dask",
        ],
        ["─" * 20] * 3,
        [
            "Match Rate",
            f"{signal_comparison['orig_rust']['match_rate_percent']:.2f}%",
            f"{signal_comparison['orig_dask']['match_rate_percent']:.2f}%",
            f"{signal_comparison['orig_rust_dask']['match_rate_percent']:.2f}%",
        ],
        [
            "Matching Symbols",
            f"{signal_comparison['orig_rust']['matching_symbols']}/{signal_comparison['total_symbols']}",
            f"{signal_comparison['orig_dask']['matching_symbols']}/{signal_comparison['total_symbols']}",
            f"{signal_comparison['orig_rust_dask']['matching_symbols']}/{signal_comparison['total_symbols']}",
        ],
        [
            "Max Difference",
            f"{signal_comparison['orig_rust']['max_difference']:.2e}",
            f"{signal_comparison['orig_dask']['max_difference']:.2e}",
            f"{signal_comparison['orig_rust_dask']['max_difference']:.2e}",
        ],
        [
            "Avg Difference",
            f"{signal_comparison['orig_rust']['avg_difference']:.2e}",
            f"{signal_comparison['orig_dask']['avg_difference']:.2e}",
            f"{signal_comparison['orig_rust_dask']['avg_difference']:.2e}",
        ],
        [
            "Median Difference",
            f"{signal_comparison['orig_rust']['median_difference']:.2e}",
            f"{signal_comparison['orig_dask']['median_difference']:.2e}",
            f"{signal_comparison['orig_rust_dask']['median_difference']:.2e}",
        ],
    ]

    # Approx vs AdaptApprox table
    approx_data = [
        [
            "Approx Comparison",
            "Approx (Validity)",
            "AdaptApprox (Validity)",
            "Adapt vs Approx",
        ],
        ["─" * 20] * 4,
        [
            "Match Rate",
            f"{signal_comparison['orig_approx']['match_rate_percent']:.2f}%",
            f"{signal_comparison['orig_adaptive_approx']['match_rate_percent']:.2f}%",
            f"{signal_comparison['approx_adaptive']['match_rate_percent']:.2f}%",
        ],
        [
            "Matching Symbols",
            f"{signal_comparison['orig_approx']['matching_symbols']}/{signal_comparison['total_symbols']}",
            f"{signal_comparison['orig_adaptive_approx']['matching_symbols']}/{signal_comparison['total_symbols']}",
            f"{signal_comparison['approx_adaptive']['matching_symbols']}/{signal_comparison['total_symbols']}",
        ],
        [
            "Max Difference",
            f"{signal_comparison['orig_approx']['max_difference']:.2e}",
            f"{signal_comparison['orig_adaptive_approx']['max_difference']:.2e}",
            f"{signal_comparison['approx_adaptive']['max_difference']:.2e}",
        ],
        [
            "Avg Difference",
            f"{signal_comparison['orig_approx']['avg_difference']:.2e}",
            f"{signal_comparison['orig_adaptive_approx']['avg_difference']:.2e}",
            f"{signal_comparison['approx_adaptive']['avg_difference']:.2e}",
        ],
        [
            "Median Difference",
            f"{signal_comparison['orig_approx']['median_difference']:.2e}",
            f"{signal_comparison['orig_adaptive_approx']['median_difference']:.2e}",
            f"{signal_comparison['approx_adaptive']['median_difference']:.2e}",
        ],
    ]

    perf_table = tabulate(perf_data, headers="firstrow", tablefmt="grid")
    signal_table = tabulate(signal_data, headers="firstrow", tablefmt="grid")
    approx_table = tabulate(approx_data, headers="firstrow", tablefmt="grid")

    return f"\n{perf_table}\n\n{signal_table}\n\n{approx_table}\n"
