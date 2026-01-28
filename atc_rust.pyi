from typing import Dict, Mapping

import numpy as np

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
