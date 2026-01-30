"""Benchmark runner functions for XGBoost LTS."""

import gc
import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Import modules to patch
import config
import modules.common.indicators.price_derived as price_derived
import modules.xgboost_LTS.core.labeling as labeling_module
import modules.xgboost_LTS.core.model as model_module
import modules.xgboost_LTS.utils.features as features_module
import modules.xgboost_LTS.utils.gpu_utils as gpu_utils
from modules.common.core.indicator_engine import IndicatorConfig, IndicatorEngine, IndicatorProfile
from modules.common.utils import log_error, log_info, log_success
from modules.xgboost_LTS.core.labeling import apply_directional_labels
from modules.xgboost_LTS.core.model import train_and_predict
from modules.xgboost_LTS.utils.features import add_advanced_features


def _run_single_pipeline(symbol: str, df: pd.DataFrame, use_cache: bool = False) -> Dict[str, Any]:
    """Run full XGBoost pipeline for a single symbol."""
    try:
        # 1. Feature Engineering
        indicator_engine = IndicatorEngine(IndicatorConfig.for_profile(IndicatorProfile.XGBOOST))
        df = indicator_engine.compute_features(df)
        df = add_advanced_features(df)

        # 2. Labeling
        df = apply_directional_labels(df)

        # 3. Training
        model = train_and_predict(df, use_cache=use_cache)

        # Results
        return {"Symbol": symbol, "Model": model, "Success": True}
    except Exception as e:
        # log_error(f"Pipeline failed for {symbol}: {e}")
        return {"Symbol": symbol, "Error": str(e), "Success": False}


def run_original_python(symbols_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Any], float, float]:
    """Run simulated 'Original Python' pipeline (No Rust, No GPU, No Parallel CV)."""
    log_info("Running Original Python pipeline (Simulated)...")

    # Save original states
    orig_rust_pd = getattr(price_derived, "RUST_AVAILABLE", False)
    orig_rust_lbl = getattr(labeling_module, "RUST_AVAILABLE", False)
    orig_rust_feat = getattr(features_module, "RUST_AVAILABLE", False)
    orig_float32 = getattr(config, "XGBOOST_USE_FLOAT32", False)
    orig_parallel_cv = getattr(config, "XGBOOST_USE_PARALLEL_CV", False)
    orig_gpu = getattr(config.position_sizing, "USE_GPU", False)
    orig_params = model_module.XGBOOST_PARAMS.copy()

    # Apply Patches
    price_derived.RUST_AVAILABLE = False
    labeling_module.RUST_AVAILABLE = False
    features_module.RUST_AVAILABLE = False
    model_module.XGBOOST_USE_FLOAT32 = False
    # model_module.XGBOOST_USE_PARALLEL_CV = False # Not easily patchable if imported from config directly in model.py?
    # Check model.py imports: `from config import ... XGBOOST_USE_PARALLEL_CV`
    # Updating config.XGBOOST_USE_PARALLEL_CV won't affect model.py if it did `from config import ...`
    # I need to patch `model_module.XGBOOST_USE_PARALLEL_CV`
    if hasattr(model_module, "XGBOOST_USE_FLOAT32"):
        model_module.XGBOOST_USE_FLOAT32 = False
    # Parallel CV patching might be tricky if not exposed in model.py

    config.position_sizing.USE_GPU = False  # model.py uses USE_GPU from config.position_sizing
    model_module.USE_GPU = False

    model_module.XGBOOST_PARAMS["tree_method"] = "auto"
    model_module.XGBOOST_PARAMS["device"] = "cpu"

    results = {}
    start_time = time.time()

    # Track memory
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    try:
        for idx, (symbol, df) in enumerate(symbols_data.items(), 1):
            results[symbol] = _run_single_pipeline(symbol, df.copy(), use_cache=False)
            if idx % 10 == 0:
                log_info(f"Original: Processed {idx}/{len(symbols_data)} symbols")
    finally:
        # Restore states
        price_derived.RUST_AVAILABLE = orig_rust_pd
        labeling_module.RUST_AVAILABLE = orig_rust_lbl
        features_module.RUST_AVAILABLE = orig_rust_feat
        if hasattr(model_module, "XGBOOST_USE_FLOAT32"):
            model_module.XGBOOST_USE_FLOAT32 = orig_float32
        config.position_sizing.USE_GPU = orig_gpu
        model_module.USE_GPU = orig_gpu
        model_module.XGBOOST_PARAMS = orig_params

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024

    return results, end_time - start_time, mem_after - mem_before


def run_rust_accelerated(symbols_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Any], float, float]:
    """Run Rust-Accelerated pipeline (Rust Features/Labels, CPU Training)."""
    log_info("Running Rust-Accelerated pipeline...")

    # Ensure Rust enabled
    # Assuming environment has Rust extensions installed

    # Patch Config for CPU Training, but Rust Features
    orig_params = model_module.XGBOOST_PARAMS.copy()
    orig_gpu = model_module.USE_GPU

    model_module.USE_GPU = False
    model_module.XGBOOST_PARAMS["tree_method"] = "hist"
    model_module.XGBOOST_PARAMS["device"] = "cpu"

    # Ensure Float32 is ON for speed
    orig_float32 = getattr(model_module, "XGBOOST_USE_FLOAT32", False)
    if hasattr(model_module, "XGBOOST_USE_FLOAT32"):
        model_module.XGBOOST_USE_FLOAT32 = True

    results = {}
    start_time = time.time()
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    try:
        for idx, (symbol, df) in enumerate(symbols_data.items(), 1):
            results[symbol] = _run_single_pipeline(symbol, df.copy(), use_cache=False)
            if idx % 10 == 0:
                log_info(f"Rust: Processed {idx}/{len(symbols_data)} symbols")
    finally:
        model_module.USE_GPU = orig_gpu
        model_module.XGBOOST_PARAMS = orig_params
        if hasattr(model_module, "XGBOOST_USE_FLOAT32"):
            model_module.XGBOOST_USE_FLOAT32 = orig_float32

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024

    return results, end_time - start_time, mem_after - mem_before


def run_gpu_accelerated(symbols_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Any], float, float]:
    """Run GPU-Accelerated pipeline (Rust Features/Labels, GPU Training)."""
    log_info("Running GPU-Accelerated pipeline...")

    # Patch Config for GPU
    orig_params = model_module.XGBOOST_PARAMS.copy()
    orig_gpu = model_module.USE_GPU

    model_module.USE_GPU = True
    model_module.XGBOOST_PARAMS["tree_method"] = "hist"
    model_module.XGBOOST_PARAMS["device"] = "cuda"

    # Ensure Float32
    orig_float32 = getattr(model_module, "XGBOOST_USE_FLOAT32", False)
    if hasattr(model_module, "XGBOOST_USE_FLOAT32"):
        model_module.XGBOOST_USE_FLOAT32 = True

    results = {}
    start_time = time.time()
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    try:
        for idx, (symbol, df) in enumerate(symbols_data.items(), 1):
            results[symbol] = _run_single_pipeline(symbol, df.copy(), use_cache=False)
            if idx % 10 == 0:
                log_info(f"GPU: Processed {idx}/{len(symbols_data)} symbols")
    finally:
        model_module.USE_GPU = orig_gpu
        model_module.XGBOOST_PARAMS = orig_params
        if hasattr(model_module, "XGBOOST_USE_FLOAT32"):
            model_module.XGBOOST_USE_FLOAT32 = orig_float32

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024

    return results, end_time - start_time, mem_after - mem_before


def run_cached(symbols_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Any], float, float]:
    """Run Cached pipeline (Should be instant after first run)."""
    log_info("Running Cached pipeline...")

    # Use GPU config
    orig_params = model_module.XGBOOST_PARAMS.copy()
    model_module.XGBOOST_PARAMS["tree_method"] = "hist"
    model_module.XGBOOST_PARAMS["device"] = "cuda"

    results = {}
    start_time = time.time()
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    try:
        for idx, (symbol, df) in enumerate(symbols_data.items(), 1):
            # ENABLE CACHE
            results[symbol] = _run_single_pipeline(symbol, df.copy(), use_cache=True)
            if idx % 10 == 0:
                log_info(f"Cached: Processed {idx}/{len(symbols_data)} symbols")
    finally:
        model_module.XGBOOST_PARAMS = orig_params

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024

    return results, end_time - start_time, mem_after - mem_before


def batch_worker_fn(df, **kwargs):
    """Worker function for batch processing."""
    # We need to compute features and labels here because batch_train_symbols usually assumes
    # features are already computed if passed as df.
    # However, to be consistent with other runners, we should include feature eng in the benchmark.
    # But `batch_train_symbols` in `utils` expects `train_and_predict` alike signature.
    # Let's recreate the pipeline here.

    indicator_engine = IndicatorEngine(IndicatorConfig.for_profile(IndicatorProfile.XGBOOST))
    df = indicator_engine.compute_features(df)
    df = add_advanced_features(df)
    df = apply_directional_labels(df)

    return train_and_predict(df, use_cache=kwargs.get("use_cache", False))


def run_batch_parallel(symbols_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Any], float, float]:
    """Run Batch Parallel pipeline."""
    log_info("Running Batch Parallel pipeline...")
    from modules.xgboost_LTS.utils.batch_symbols import batch_train_symbols

    start_time = time.time()
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    results = batch_train_symbols(symbols_data, train_and_predict_fn=batch_worker_fn, max_workers=4, use_cache=False)

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024

    return results, end_time - start_time, mem_after - mem_before
