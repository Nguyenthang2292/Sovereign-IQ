"""Signal comparison and table generation utilities for XGBoost benchmark."""

from typing import Any, Dict

import numpy as np
from tabulate import tabulate

from modules.common.utils import log_info, log_warn


def compare_results(
    original_results: Dict[str, Any],
    rust_results: Dict[str, Any],
    gpu_results: Dict[str, Any],
    cached_results: Dict[str, Any],
    batch_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare outputs between versions."""
    log_info("Comparing results between versions...")

    comparisons = {}

    # Extract accuracies
    def extract_metrics(results):
        metrics = {}
        valid_count = 0
        total_cv_acc = 0.0
        total_test_acc = 0.0

        for symbol, res in results.items():
            if isinstance(res, dict) and res.get("Success", False) is False:
                # Check if it is a result from batch runner which might return result dict directly or wrapper
                pass

            # Handle wrapper from runners.py
            if "Model" in res:
                res = res["Model"]  # This is the dict returned by train_and_predict
            elif "result" in res:  # Batch runner wrapper
                res = res["result"]

            if isinstance(res, dict) and "mean_cv_accuracy" in res:
                total_cv_acc += res["mean_cv_accuracy"]
                total_test_acc += res["test_accuracy"]
                valid_count += 1

        return {
            "avg_cv_accuracy": total_cv_acc / valid_count if valid_count > 0 else 0.0,
            "avg_test_accuracy": total_test_acc / valid_count if valid_count > 0 else 0.0,
            "valid_symbols": valid_count,
        }

    comparisons["original"] = extract_metrics(original_results)
    comparisons["rust"] = extract_metrics(rust_results)
    comparisons["gpu"] = extract_metrics(gpu_results)
    comparisons["cached"] = extract_metrics(cached_results)
    comparisons["batch"] = extract_metrics(batch_results)

    return comparisons


def generate_comparison_table(
    original_time: float,
    rust_time: float,
    gpu_time: float,
    cached_time: float,
    batch_time: float,
    original_memory: float,
    rust_memory: float,
    gpu_memory: float,
    cached_memory: float,
    batch_memory: float,
    metrics: Dict[str, Any],
) -> str:
    """Generate formatted comparison table."""

    # Calculate speedups
    speedup_rust = original_time / rust_time if rust_time > 0 else 0
    speedup_gpu = original_time / gpu_time if gpu_time > 0 else 0
    speedup_cached = original_time / cached_time if cached_time > 0 else 0
    speedup_batch = original_time / batch_time if batch_time > 0 else 0

    # Memory reduction
    mem_red_rust = ((original_memory - rust_memory) / original_memory * 100) if original_memory > 0 else 0
    mem_red_gpu = ((original_memory - gpu_memory) / original_memory * 100) if original_memory > 0 else 0
    mem_red_cached = ((original_memory - cached_memory) / original_memory * 100) if original_memory > 0 else 0
    mem_red_batch = ((original_memory - batch_memory) / original_memory * 100) if original_memory > 0 else 0

    table_data = [
        ["Metric", "Original (Simulated)", "Rust (CPU)", "GPU Accelerated", "Cached", "Batch Parallel"],
        ["─" * 12] * 6,
        [
            "Time (s)",
            f"{original_time:.2f}",
            f"{rust_time:.2f}",
            f"{gpu_time:.2f}",
            f"{cached_time:.2f}",
            f"{batch_time:.2f}",
        ],
        [
            "Speedup",
            "1.00x",
            f"{speedup_rust:.2f}x",
            f"{speedup_gpu:.2f}x",
            f"{speedup_cached:.2f}x",
            f"{speedup_batch:.2f}x",
        ],
        [
            "Memory (MB)",
            f"{original_memory:.1f}",
            f"{rust_memory:.1f}",
            f"{gpu_memory:.1f}",
            f"{cached_memory:.1f}",
            f"{batch_memory:.1f}",
        ],
        [
            "CV Accuracy",
            f"{metrics['original']['avg_cv_accuracy']:.2%}",
            f"{metrics['rust']['avg_cv_accuracy']:.2%}",
            f"{metrics['gpu']['avg_cv_accuracy']:.2%}",
            f"{metrics['cached']['avg_cv_accuracy']:.2%}",
            f"{metrics['batch']['avg_cv_accuracy']:.2%}",
        ],
        [
            "Valid Symbols",
            f"{metrics['original']['valid_symbols']}",
            f"{metrics['rust']['valid_symbols']}",
            f"{metrics['gpu']['valid_symbols']}",
            f"{metrics['cached']['valid_symbols']}",
            f"{metrics['batch']['valid_symbols']}",
        ],
    ]

    return tabulate(table_data, headers="firstrow", tablefmt="grid")
