"""Signal comparison and table generation utilities."""

from typing import Dict

import numpy as np
from tabulate import tabulate

from modules.common.utils import log_info, log_success, log_warn

def compare_signals(
    original_results: Dict[str, Dict],
    enhanced_results: Dict[str, Dict],
    rust_results: Dict[str, Dict],
    rust_rayon_results: Dict[str, Dict],
    cuda_results: Dict[str, Dict],
    dask_results: Dict[str, Dict],
    rust_dask_results: Dict[str, Dict],
    cuda_dask_results: Dict[str, Dict],
    rust_cuda_dask_results: Dict[str, Dict],
) -> Dict[str, any]:
    """Compare signal outputs between all 8 versions: original, enhanced, rust, cuda, dask, and their combinations.

    Args:
        original_results: Results from original module
        enhanced_results: Results from enhanced module
        rust_results: Results from Rust module
        rust_rayon_results: Results from Rust Rayon module
        cuda_results: Results from CUDA module
        dask_results: Results from Dask module
        rust_dask_results: Results from Rust+Dask hybrid
        cuda_dask_results: Results from CUDA+Dask hybrid
        rust_cuda_dask_results: Results from Rust+CUDA+Dask hybrid

    Returns:
        Dictionary of comparison metrics
    """
    log_info("Comparing signal outputs between all 8 versions...")

    # DEBUG: Print result dict sizes and sample keys
    log_info(
        f"Result dict sizes: orig={len(original_results)}, enh={len(enhanced_results)}, "
        f"rust={len(rust_results)}, rust_rayon={len(rust_rayon_results)}, cuda={len(cuda_results)}, "
        f"dask={len(dask_results)}, rust_dask={len(rust_dask_results)}, "
        f"cuda_dask={len(cuda_dask_results)}, rust_cuda_dask={len(rust_cuda_dask_results)}"
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

    # Compare Original vs Enhanced
    orig_enh_diffs = []
    orig_enh_matching = 0
    orig_enh_mismatched = []

    # Compare Original vs Rust
    orig_rust_diffs = []
    orig_rust_matching = 0
    orig_rust_mismatched = []

    # Compare Original vs CUDA
    orig_cuda_diffs = []
    orig_cuda_matching = 0
    orig_cuda_mismatched = []

    # Compare Enhanced vs Rust
    enh_rust_diffs = []
    enh_rust_matching = 0
    enh_rust_mismatched = []

    # Compare Rust vs CUDA
    rust_cuda_diffs = []
    rust_cuda_matching = 0
    rust_cuda_mismatched = []

    # Compare Original vs Dask
    orig_dask_diffs = []
    orig_dask_matching = 0
    orig_dask_mismatched = []

    # Compare Original vs Rust+Dask
    orig_rust_dask_diffs = []
    orig_rust_dask_matching = 0
    orig_rust_dask_mismatched = []

    # Compare Original vs CUDA+Dask
    orig_cuda_dask_diffs = []
    orig_cuda_dask_matching = 0
    orig_cuda_dask_mismatched = []

    # Compare Original vs Rust+CUDA+Dask
    orig_all_three_diffs = []
    orig_all_three_matching = 0
    orig_all_three_mismatched = []

    for symbol in original_results.keys():
        if (
            symbol not in enhanced_results
            or symbol not in rust_results
            or symbol not in rust_rayon_results
            or symbol not in cuda_results
            or symbol not in dask_results
            or symbol not in rust_dask_results
            or symbol not in cuda_dask_results
            or symbol not in rust_cuda_dask_results
        ):
            log_warn(f"Symbol {symbol} missing in results")
            continue

        orig = original_results[symbol]
        enh = enhanced_results[symbol]
        rust = rust_results[symbol]
        rust_rayon = rust_rayon_results[symbol]
        cuda = cuda_results[symbol]
        dask = dask_results[symbol]
        rust_dask = rust_dask_results[symbol]
        cuda_dask = cuda_dask_results[symbol]
        all_three = rust_cuda_dask_results[symbol]

        if (
            orig is None
            or enh is None
            or rust is None
            or rust_rayon is None
            or cuda is None
            or dask is None
            or rust_dask is None
            or cuda_dask is None
            or all_three is None
        ):
            log_warn(f"Symbol {symbol} has None result")
            continue

        # Compare Average_Signal
        orig_s = orig.get("Average_Signal")
        enh_s = enh.get("Average_Signal")
        rust_s = rust.get("Average_Signal")
        rust_r_s = rust_rayon.get("Average_Signal")
        cuda_s = cuda.get("Average_Signal")
        dask_s = dask.get("Average_Signal")
        rust_dask_s = rust_dask.get("Average_Signal")
        cuda_dask_s = cuda_dask.get("Average_Signal")
        all_three_s = all_three.get("Average_Signal")

        if (
            orig_s is None
            or enh_s is None
            or rust_s is None
            or rust_r_s is None
            or cuda_s is None
            or dask_s is None
            or rust_dask_s is None
            or cuda_dask_s is None
            or all_three_s is None
        ):
            log_warn(f"Symbol {symbol} missing Average_Signal")
            continue

        # Find common index across all versions
        common_index = (
            orig_s.index.intersection(enh_s.index)
            .intersection(rust_s.index)
            .intersection(rust_r_s.index)
            .intersection(cuda_s.index)
            .intersection(dask_s.index)
            .intersection(rust_dask_s.index)
            .intersection(cuda_dask_s.index)
            .intersection(all_three_s.index)
        )

        if len(common_index) == 0:
            log_warn(f"No common index after alignment for {symbol}")
            continue

        # Reindex all series to common index
        orig_s = orig_s.loc[common_index]
        enh_s = enh_s.loc[common_index]
        rust_s = rust_s.loc[common_index]
        rust_r_s = rust_r_s.loc[common_index]
        cuda_s = cuda_s.loc[common_index]
        dask_s = dask_s.loc[common_index]
        rust_dask_s = rust_dask_s.loc[common_index]
        cuda_dask_s = cuda_dask_s.loc[common_index]
        all_three_s = all_three_s.loc[common_index]

        # Original vs Enhanced
        diff_oe = np.abs(orig_s - enh_s).max()
        orig_enh_diffs.append(diff_oe)
        if diff_oe < 1e-6:
            orig_enh_matching += 1
        else:
            orig_enh_mismatched.append((symbol, diff_oe))

        # Original vs Rust
        diff_or = np.abs(orig_s - rust_s).max()
        orig_rust_diffs.append(diff_or)
        if diff_or < 1e-6:
            orig_rust_matching += 1
        else:
            orig_rust_mismatched.append((symbol, diff_or))

        # Original vs CUDA
        diff_oc = np.abs(orig_s - cuda_s).max()
        orig_cuda_diffs.append(diff_oc)
        if diff_oc < 1e-6:
            orig_cuda_matching += 1
        else:
            orig_cuda_mismatched.append((symbol, diff_oc))

        # Enhanced vs Rust
        diff_er = np.abs(enh_s - rust_s).max()
        enh_rust_diffs.append(diff_er)
        if diff_er < 1e-6:
            enh_rust_matching += 1
        else:
            enh_rust_mismatched.append((symbol, diff_er))

        # Rust vs CUDA
        diff_rc = np.abs(rust_s - cuda_s).max()
        rust_cuda_diffs.append(diff_rc)
        if diff_rc < 1e-6:
            rust_cuda_matching += 1
        else:
            rust_cuda_mismatched.append((symbol, diff_rc))

        # Original vs Dask
        diff_od = np.abs(orig_s - dask_s).max()
        orig_dask_diffs.append(diff_od)
        if diff_od < 1e-6:
            orig_dask_matching += 1
        else:
            orig_dask_mismatched.append((symbol, diff_od))

        # Original vs Rust+Dask
        diff_ord = np.abs(orig_s - rust_dask_s).max()
        orig_rust_dask_diffs.append(diff_ord)
        if diff_ord < 1e-6:
            orig_rust_dask_matching += 1
        else:
            orig_rust_dask_mismatched.append((symbol, diff_ord))

        # Original vs CUDA+Dask
        diff_ocd = np.abs(orig_s - cuda_dask_s).max()
        orig_cuda_dask_diffs.append(diff_ocd)
        if diff_ocd < 1e-6:
            orig_cuda_dask_matching += 1
        else:
            orig_cuda_dask_mismatched.append((symbol, diff_ocd))

        # Original vs Rust+CUDA+Dask
        diff_oall = np.abs(orig_s - all_three_s).max()
        orig_all_three_diffs.append(diff_oall)
        if diff_oall < 1e-6:
            orig_all_three_matching += 1
        else:
            orig_all_three_mismatched.append((symbol, diff_oall))

        # Original vs Rust Rayon
        diff_orr = np.abs(orig_s - rust_r_s).max()
        if diff_orr > 1e-6:
            log_warn(f"Rust Rayon mismatch for {symbol}: {diff_orr}")

        # Increment processed counter
        processed_symbols += 1

    if orig_rust_mismatched:
        log_info(f"Mismatched symbols (Orig vs Rust): {[s[0] for s in orig_rust_mismatched[:10]]}...")
    if orig_cuda_mismatched:
        log_info(f"Mismatched symbols (Orig vs CUDA): {[s[0] for s in orig_cuda_mismatched[:10]]}...")

    # Calculate metrics
    log_info(f"Processed {processed_symbols}/{total_symbols} symbols successfully")
    orig_enh_match_rate = (orig_enh_matching / total_symbols) * 100 if total_symbols > 0 else 0
    orig_rust_match_rate = (orig_rust_matching / total_symbols) * 100 if total_symbols > 0 else 0
    orig_cuda_match_rate = (orig_cuda_matching / total_symbols) * 100 if total_symbols > 0 else 0
    enh_rust_match_rate = (enh_rust_matching / total_symbols) * 100 if total_symbols > 0 else 0
    rust_cuda_match_rate = (rust_cuda_matching / total_symbols) * 100 if total_symbols > 0 else 0
    orig_dask_match_rate = (orig_dask_matching / total_symbols) * 100 if total_symbols > 0 else 0
    orig_rust_dask_match_rate = (orig_rust_dask_matching / total_symbols) * 100 if total_symbols > 0 else 0
    orig_cuda_dask_match_rate = (orig_cuda_dask_matching / total_symbols) * 100 if total_symbols > 0 else 0
    orig_all_three_match_rate = (orig_all_three_matching / total_symbols) * 100 if total_symbols > 0 else 0

    log_success(f"Original vs Enhanced match rate: {orig_enh_match_rate:.2f}%")
    log_success(f"Original vs Rust match rate: {orig_rust_match_rate:.2f}%")
    log_success(f"Original vs CUDA match rate: {orig_cuda_match_rate:.2f}%")
    log_success(f"Original vs Dask match rate: {orig_dask_match_rate:.2f}%")
    log_success(f"Original vs Rust+Dask match rate: {orig_rust_dask_match_rate:.2f}%")
    log_success(f"Original vs CUDA+Dask match rate: {orig_cuda_dask_match_rate:.2f}%")
    log_success(f"Original vs Rust+CUDA+Dask match rate: {orig_all_three_match_rate:.2f}%")
    log_success(f"Enhanced vs Rust match rate: {enh_rust_match_rate:.2f}%")
    log_success(f"Rust vs CUDA match rate: {rust_cuda_match_rate:.2f}%")

    return {
        "orig_enh": {
            "match_rate_percent": orig_enh_match_rate,
            "max_difference": max(orig_enh_diffs) if orig_enh_diffs else 0,
            "avg_difference": np.mean(orig_enh_diffs) if orig_enh_diffs else 0,
            "median_difference": np.median(orig_enh_diffs) if orig_enh_diffs else 0,
            "matching_symbols": orig_enh_matching,
            "mismatched_symbols": [s[0] for s in orig_enh_mismatched],
        },
        "orig_rust": {
            "match_rate_percent": orig_rust_match_rate,
            "max_difference": max(orig_rust_diffs) if orig_rust_diffs else 0,
            "avg_difference": np.mean(orig_rust_diffs) if orig_rust_diffs else 0,
            "median_difference": np.median(orig_rust_diffs) if orig_rust_diffs else 0,
            "matching_symbols": orig_rust_matching,
            "mismatched_symbols": [s[0] for s in orig_rust_mismatched],
        },
        "orig_cuda": {
            "match_rate_percent": orig_cuda_match_rate,
            "max_difference": max(orig_cuda_diffs) if orig_cuda_diffs else 0,
            "avg_difference": np.mean(orig_cuda_diffs) if orig_cuda_diffs else 0,
            "median_difference": np.median(orig_cuda_diffs) if orig_cuda_diffs else 0,
            "matching_symbols": orig_cuda_matching,
            "mismatched_symbols": [s[0] for s in orig_cuda_mismatched],
        },
        "enh_rust": {
            "match_rate_percent": enh_rust_match_rate,
            "max_difference": max(enh_rust_diffs) if enh_rust_diffs else 0,
            "avg_difference": np.mean(enh_rust_diffs) if enh_rust_diffs else 0,
            "median_difference": np.median(enh_rust_diffs) if enh_rust_diffs else 0,
            "matching_symbols": enh_rust_matching,
            "mismatched_symbols": [s[0] for s in enh_rust_mismatched],
        },
        "rust_cuda": {
            "match_rate_percent": rust_cuda_match_rate,
            "max_difference": max(rust_cuda_diffs) if rust_cuda_diffs else 0,
            "avg_difference": np.mean(rust_cuda_diffs) if rust_cuda_diffs else 0,
            "median_difference": np.median(rust_cuda_diffs) if rust_cuda_diffs else 0,
            "matching_symbols": rust_cuda_matching,
            "mismatched_symbols": [s[0] for s in rust_cuda_mismatched],
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
        "orig_cuda_dask": {
            "match_rate_percent": orig_cuda_dask_match_rate,
            "max_difference": max(orig_cuda_dask_diffs) if orig_cuda_dask_diffs else 0,
            "avg_difference": np.mean(orig_cuda_dask_diffs) if orig_cuda_dask_diffs else 0,
            "median_difference": np.median(orig_cuda_dask_diffs) if orig_cuda_dask_diffs else 0,
            "matching_symbols": orig_cuda_dask_matching,
            "mismatched_symbols": [s[0] for s in orig_cuda_dask_mismatched],
        },
        "orig_all_three": {
            "match_rate_percent": orig_all_three_match_rate,
            "max_difference": max(orig_all_three_diffs) if orig_all_three_diffs else 0,
            "avg_difference": np.mean(orig_all_three_diffs) if orig_all_three_diffs else 0,
            "median_difference": np.median(orig_all_three_diffs) if orig_all_three_diffs else 0,
            "matching_symbols": orig_all_three_matching,
            "mismatched_symbols": [s[0] for s in orig_all_three_mismatched],
        },
        "total_symbols": total_symbols,
    }


def generate_comparison_table(
    original_time: float,
    enhanced_time: float,
    rust_time: float,
    rust_rayon_time: float,
    cuda_time: float,
    dask_time: float,
    rust_dask_time: float,
    cuda_dask_time: float,
    all_three_time: float,
    original_memory: float,
    enhanced_memory: float,
    rust_memory: float,
    rust_rayon_memory: float,
    cuda_memory: float,
    dask_memory: float,
    rust_dask_memory: float,
    cuda_dask_memory: float,
    all_three_memory: float,
    signal_comparison: Dict,
) -> str:
    """Generate formatted comparison table for all 8 versions.

    Args:
        original_time: Execution time for original module (seconds)
        enhanced_time: Execution time for enhanced module (seconds)
        rust_time: Execution time for Rust module (seconds)
        rust_rayon_time: Execution time for Rust Rayon module (seconds)
        cuda_time: Execution time for CUDA module (seconds)
        dask_time: Execution time for Dask module (seconds)
        rust_dask_time: Execution time for Rust+Dask hybrid (seconds)
        cuda_dask_time: Execution time for CUDA+Dask hybrid (seconds)
        all_three_time: Execution time for Rust+CUDA+Dask hybrid (seconds)
        original_memory: Peak memory for original module (MB)
        enhanced_memory: Peak memory for enhanced module (MB)
        rust_memory: Peak memory for Rust module (MB)
        rust_rayon_memory: Peak memory for Rust Rayon module (MB)
        cuda_memory: Peak memory for CUDA module (MB)
        dask_memory: Peak memory for Dask module (MB)
        rust_dask_memory: Peak memory for Rust+Dask hybrid (MB)
        cuda_dask_memory: Peak memory for CUDA+Dask hybrid (MB)
        all_three_memory: Peak memory for Rust+CUDA+Dask hybrid (MB)
        signal_comparison: Signal comparison metrics

    Returns:
        Formatted table string
    """
    speedup_enh = original_time / enhanced_time if enhanced_time > 0 else 0
    speedup_rust_rayon = original_time / rust_rayon_time if rust_rayon_time > 0 else 0
    speedup_cuda = original_time / cuda_time if cuda_time > 0 else 0
    speedup_dask = original_time / dask_time if dask_time > 0 else 0
    speedup_rust_dask = original_time / rust_dask_time if rust_dask_time > 0 else 0
    speedup_cuda_dask = original_time / cuda_dask_time if cuda_dask_time > 0 else 0
    speedup_all_three = original_time / all_three_time if all_three_time > 0 else 0

    memory_reduction_enh = (
        ((original_memory - enhanced_memory) / original_memory) * 100 if original_memory > 0 else 0
    )
    memory_reduction_rust_rayon = (
        ((original_memory - rust_rayon_memory) / original_memory) * 100 if original_memory > 0 else 0
    )
    memory_reduction_cuda = (
        ((original_memory - cuda_memory) / original_memory) * 100 if original_memory > 0 else 0
    )
    memory_reduction_dask = (
        ((original_memory - dask_memory) / original_memory) * 100 if original_memory > 0 else 0
    )
    memory_reduction_rust_dask = (
        ((original_memory - rust_dask_memory) / original_memory) * 100 if original_memory > 0 else 0
    )
    memory_reduction_cuda_dask = (
        ((original_memory - cuda_dask_memory) / original_memory) * 100 if original_memory > 0 else 0
    )
    memory_reduction_all_three = (
        ((original_memory - all_three_memory) / original_memory) * 100 if original_memory > 0 else 0
    )

    # Performance table - 8 versions
    perf_data = [
        [
            "Metric",
            "Original",
            "Enhanced",
            "Rust",
            "CUDA",
            "Dask",
            "Rust+Dask",
            "CUDA+Dask",
            "All Three",
        ],
        ["─" * 12] * 8,
        [
            "Execution Time",
            f"{original_time:.2f}s",
            f"{enhanced_time:.2f}s",
            f"{rust_rayon_time:.2f}s",
            f"{cuda_time:.2f}s",
            f"{dask_time:.2f}s",
            f"{rust_dask_time:.2f}s",
            f"{cuda_dask_time:.2f}s",
            f"{all_three_time:.2f}s",
        ],
        [
            "Speedup vs Orig",
            "1.00x",
            f"{speedup_enh:.2f}x",
            f"{speedup_rust_rayon:.2f}x",
            f"{speedup_cuda:.2f}x",
            f"{speedup_dask:.2f}x",
            f"{speedup_rust_dask:.2f}x",
            f"{speedup_cuda_dask:.2f}x",
            f"{speedup_all_three:.2f}x",
        ],
        [
            "Peak Memory",
            f"{original_memory:.1f} MB",
            f"{enhanced_memory:.1f} MB",
            f"{rust_rayon_memory:.1f} MB",
            f"{cuda_memory:.1f} MB",
            f"{dask_memory:.1f} MB",
            f"{rust_dask_memory:.1f} MB",
            f"{cuda_dask_memory:.1f} MB",
            f"{all_three_memory:.1f} MB",
        ],
        [
            "Memory Reduction",
            "0%",
            f"{memory_reduction_enh:.1f}%",
            f"{memory_reduction_rust_rayon:.1f}%",
            f"{memory_reduction_cuda:.1f}%",
            f"{memory_reduction_dask:.1f}%",
            f"{memory_reduction_rust_dask:.1f}%",
            f"{memory_reduction_cuda_dask:.1f}%",
            f"{memory_reduction_all_three:.1f}%",
        ],
    ]

    # Signal comparison table - Original vs all versions
    signal_data = [
        [
            "Signal Comparison",
            "vs Enhanced",
            "vs Rust",
            "vs CUDA",
            "vs Dask",
            "vs Rust+Dask",
            "vs CUDA+Dask",
            "vs All Three",
        ],
        ["─" * 20] * 8,
        [
            "Match Rate",
            f"{signal_comparison['orig_enh']['match_rate_percent']:.2f}%",
            f"{signal_comparison['orig_rust']['match_rate_percent']:.2f}%",
            f"{signal_comparison['orig_cuda']['match_rate_percent']:.2f}%",
            f"{signal_comparison['orig_dask']['match_rate_percent']:.2f}%",
            f"{signal_comparison['orig_rust_dask']['match_rate_percent']:.2f}%",
            f"{signal_comparison['orig_cuda_dask']['match_rate_percent']:.2f}%",
            f"{signal_comparison['orig_all_three']['match_rate_percent']:.2f}%",
        ],
        [
            "Matching Symbols",
            f"{signal_comparison['orig_enh']['matching_symbols']}/{signal_comparison['total_symbols']}",
            f"{signal_comparison['orig_rust']['matching_symbols']}/{signal_comparison['total_symbols']}",
            f"{signal_comparison['orig_cuda']['matching_symbols']}/{signal_comparison['total_symbols']}",
            f"{signal_comparison['orig_dask']['matching_symbols']}/{signal_comparison['total_symbols']}",
            f"{signal_comparison['orig_rust_dask']['matching_symbols']}/{signal_comparison['total_symbols']}",
            f"{signal_comparison['orig_cuda_dask']['matching_symbols']}/{signal_comparison['total_symbols']}",
            f"{signal_comparison['orig_all_three']['matching_symbols']}/{signal_comparison['total_symbols']}",
        ],
        [
            "Max Difference",
            f"{signal_comparison['orig_enh']['max_difference']:.2e}",
            f"{signal_comparison['orig_rust']['max_difference']:.2e}",
            f"{signal_comparison['orig_cuda']['max_difference']:.2e}",
            f"{signal_comparison['orig_dask']['max_difference']:.2e}",
            f"{signal_comparison['orig_rust_dask']['max_difference']:.2e}",
            f"{signal_comparison['orig_cuda_dask']['max_difference']:.2e}",
            f"{signal_comparison['orig_all_three']['max_difference']:.2e}",
        ],
        [
            "Avg Difference",
            f"{signal_comparison['orig_enh']['avg_difference']:.2e}",
            f"{signal_comparison['orig_rust']['avg_difference']:.2e}",
            f"{signal_comparison['orig_cuda']['avg_difference']:.2e}",
            f"{signal_comparison['orig_dask']['avg_difference']:.2e}",
            f"{signal_comparison['orig_rust_dask']['avg_difference']:.2e}",
            f"{signal_comparison['orig_cuda_dask']['avg_difference']:.2e}",
            f"{signal_comparison['orig_all_three']['avg_difference']:.2e}",
        ],
        [
            "Median Difference",
            f"{signal_comparison['orig_enh']['median_difference']:.2e}",
            f"{signal_comparison['orig_rust']['median_difference']:.2e}",
            f"{signal_comparison['orig_cuda']['median_difference']:.2e}",
            f"{signal_comparison['orig_dask']['median_difference']:.2e}",
            f"{signal_comparison['orig_rust_dask']['median_difference']:.2e}",
            f"{signal_comparison['orig_cuda_dask']['median_difference']:.2e}",
            f"{signal_comparison['orig_all_three']['median_difference']:.2e}",
        ],
    ]

    perf_table = tabulate(perf_data, headers="firstrow", tablefmt="grid")
    signal_table = tabulate(signal_data, headers="firstrow", tablefmt="grid")

    return f"\n{perf_table}\n\n{signal_table}\n"


