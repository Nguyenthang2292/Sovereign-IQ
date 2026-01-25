"""
Display utilities for batch scanner CLI.
"""

from typing import Any, Dict

from colorama import Fore

from modules.common.utils import color_text


def display_configuration_summary(config: Dict[str, Any]) -> None:
    """Display configuration summary before scan."""
    print("\n" + color_text("=" * 50, Fore.CYAN))
    print(color_text("CONFIGURATION SUMMARY", Fore.CYAN))
    print(color_text("=" * 50, Fore.CYAN))

    print(f"Analysis Mode: {config.get('analysis_mode', 'N/A')}")
    print(f"Timeframe: {config.get('timeframe', 'N/A')}")
    if config.get("timeframes"):
        print(f"Timeframes: {', '.join(config['timeframes'])}")
    print(f"Max Symbols: {config.get('max_symbols', 'All')}")
    print(f"Cooldown: {config.get('cooldown', 'N/A')}s")
    print(f"Limit: {config.get('limit', 'N/A')}")

    if config.get("enable_pre_filter"):
        stage0_sample = config.get("stage0_random_sample")
        if stage0_sample:
            print(f"Stage 0 Random Sampling: {stage0_sample} symbols")
        else:
            print("Stage 0 Random Sampling: Disabled")
        print(f"Pre-filter Mode: {config.get('pre_filter_mode', 'N/A')}")
        pre_filter_percentage = config.get("pre_filter_percentage")
        if pre_filter_percentage is not None:
            print(f"Pre-filter Percentage: {pre_filter_percentage}%")
        else:
            print("Pre-filter Percentage: 10% (default)")
        print(f"Fast Mode: {config.get('fast_mode', 'N/A')}")
        spc_config = config.get("spc_config", {})
        if spc_config.get("preset"):
            print(f"SPC Preset: {spc_config['preset']}")
        else:
            print("SPC Config: Custom")
    else:
        print("Pre-filter: Disabled")

    rf_model = config.get("random_forest_model", {})
    if rf_model.get("status"):
        print(f"RF Model: Available ({rf_model['status'].get('model_path', 'Unknown')})")
    else:
        print("RF Model: None")

    # ATC Performance Summary
    atc_perf = config.get("atc_performance", {})
    if atc_perf:
        print("\n" + color_text("ATC Performance:", Fore.GREEN))
        backend = "CUDA (GPU)" if atc_perf.get("use_cuda") else "Rust Rayon (CPU)"
        print(f"  Backend: {backend}")
        print(
            f"  Parallel L1/L2: {'Yes' if atc_perf.get('parallel_l1') and atc_perf.get('parallel_l2') else 'Limited'}"
        )

        # Dask status
        use_dask = atc_perf.get("use_dask", False)
        if use_dask:
            nparts = atc_perf.get("npartitions") or "Auto"
            print(f"  Dask (Out-of-Core): Enabled ({nparts} partitions)")
        else:
            print("  Dask (Out-of-Core): Disabled")

    print(color_text("=" * 50, Fore.CYAN))
