"""Stub for atc_rust (Rust extension). Enables type hints and IDE completion."""

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

# --- Equity ---
def calculate_equity_rust(
    r_values: np.ndarray,
    sig_prev_values: np.ndarray,
    starting_equity: float = ...,
    decay_multiplier: float = ...,
    cutout: int = ...,
) -> np.ndarray: ...

# --- KAMA ---
def calculate_kama_rust(
    prices: np.ndarray,
    length: int = ...,
) -> np.ndarray: ...

# --- Signal persistence ---
def process_signal_persistence_rust(
    up: np.ndarray,
    down: np.ndarray,
) -> np.ndarray: ...

# --- MA calculations (CPU) ---
def calculate_ema_rust(
    prices: np.ndarray,
    length: int = ...,
) -> np.ndarray: ...
def calculate_wma_rust(
    prices: np.ndarray,
    length: int = ...,
) -> np.ndarray: ...
def calculate_dema_rust(
    prices: np.ndarray,
    length: int = ...,
) -> np.ndarray: ...
def calculate_lsma_rust(
    prices: np.ndarray,
    length: int = ...,
) -> np.ndarray: ...
def calculate_hma_rust(
    prices: np.ndarray,
    length: int = ...,
) -> np.ndarray: ...

# --- CUDA: Equity ---
def calculate_equity_cuda(
    r_values: np.ndarray,
    sig_prev_values: np.ndarray,
    starting_equity: float = ...,
    decay_multiplier: float = ...,
    cutout: int = ...,
) -> np.ndarray: ...

# --- CUDA: MA ---
def calculate_ema_cuda(
    prices: np.ndarray,
    length: int = ...,
) -> np.ndarray: ...
def calculate_kama_cuda(
    prices: np.ndarray,
    length: int = ...,
) -> np.ndarray: ...
def calculate_wma_cuda(
    prices: np.ndarray,
    length: int = ...,
) -> np.ndarray: ...
def calculate_hma_cuda(
    prices: np.ndarray,
    length: int = ...,
) -> np.ndarray: ...

# --- CUDA: Signal ---
def calculate_average_signal_cuda(
    signals: np.ndarray,
    equities: np.ndarray,
    long_threshold: float = ...,
    short_threshold: float = ...,
    cutout: int = ...,
) -> np.ndarray: ...
def classify_trend_cuda(
    signals: np.ndarray,
    long_threshold: float = ...,
    short_threshold: float = ...,
) -> np.ndarray: ...
def calculate_and_classify_cuda(
    signals: np.ndarray,
    equities: np.ndarray,
    long_threshold: float = ...,
    short_threshold: float = ...,
    cutout: int = ...,
) -> Tuple[np.ndarray, np.ndarray]: ...

# --- Batch: CUDA (all symbols in one kernel) ---
def compute_atc_signals_batch(
    symbols_data: Dict[str, np.ndarray],
    ema_len: int = ...,
    hull_len: int = ...,
    wma_len: int = ...,
    dema_len: int = ...,
    lsma_len: int = ...,
    kama_len: int = ...,
    ema_w: float = ...,
    hma_w: float = ...,
    wma_w: float = ...,
    dema_w: float = ...,
    lsma_w: float = ...,
    kama_w: float = ...,
    robustness: str = ...,
    la: float = ...,
    de: float = ...,
    cutout: int = ...,
    long_threshold: float = ...,
    short_threshold: float = ...,
    _strategy_mode: bool = ...,
) -> Dict[str, np.ndarray]: ...

# --- Batch: CPU (Rayon) ---
def compute_atc_signals_batch_cpu(
    symbols_data: Mapping[str, np.ndarray],
    ema_len: int = ...,
    hull_len: int = ...,
    wma_len: int = ...,
    dema_len: int = ...,
    lsma_len: int = ...,
    kama_len: int = ...,
    ema_w: float = ...,
    hma_w: float = ...,
    wma_w: float = ...,
    dema_w: float = ...,
    lsma_w: float = ...,
    kama_w: float = ...,
    robustness: str = ...,
    la: float = ...,
    de: float = ...,
    cutout: int = ...,
    long_threshold: float = ...,
    short_threshold: float = ...,
    _strategy_mode: bool = ...,
) -> Dict[str, np.ndarray]: ...

# --- Liquidity metrics (Stage 0 sampling, Rayon) ---
def compute_liquidity_metrics_batch(
    ohlcv_data: Dict[str, Dict[str, List[float]]],
    lookback: int = ...,
) -> Dict[str, Any]: ...

# --- Incremental ATC (single-bar update) ---
def update_incremental_atc_rust(
    state: Dict[str, Any],
    new_price: float,
    config: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]: ...
