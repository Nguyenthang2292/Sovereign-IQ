from __future__ import annotations

import logging
from typing import Optional

import numpy as np

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:  # pragma: no cover
    _HAS_CUPY = False


logger = logging.getLogger(__name__)


def _calculate_ma_gpu(prices: np.ndarray, length: int, ma_type: str) -> Optional[np.ndarray]:
    if not _HAS_CUPY:
        return None

    try:
        prices_gpu = cp.asarray(prices)

        if ma_type == "EMA":
            result_gpu = _calculate_ema_gpu(prices_gpu, length)
        elif ma_type == "WMA":
            result_gpu = _calculate_wma_gpu(prices_gpu, length)
        elif ma_type == "DEMA":
            result_gpu = _calculate_dema_gpu(prices_gpu, length)
        elif ma_type == "LSMA":
            result_gpu = _calculate_lsma_gpu(prices_gpu, length)
        else:
            return None

        if result_gpu is None:
            return None

        return cp.asnumpy(result_gpu)
    except Exception as e:  # pragma: no cover
        logger.debug(f"GPU calculation failed for {ma_type}: {e}")
        return None


def _calculate_ema_gpu(prices_gpu, length: int):
    n = len(prices_gpu)
    alpha = 2.0 / (length + 1.0)

    ema_gpu = cp.full(n, cp.nan, dtype=cp.float64)
    ema_gpu[0] = prices_gpu[0]

    for i in range(1, n):
        ema_gpu[i] = alpha * prices_gpu[i] + (1.0 - alpha) * ema_gpu[i - 1]

    return ema_gpu


def _calculate_wma_gpu(prices_gpu, length: int):
    n = len(prices_gpu)
    wma_gpu = cp.full(n, cp.nan, dtype=cp.float64)

    if n < length:
        return wma_gpu

    denominator = length * (length + 1) / 2.0

    for i in range(length - 1, n):
        weights = cp.arange(1, length + 1, dtype=cp.float64)
        window = prices_gpu[i - length + 1 : i + 1]
        wma_gpu[i] = cp.sum(window * weights[::-1]) / denominator

    return wma_gpu


def _calculate_dema_gpu(prices_gpu, length: int):
    ema1_gpu = _calculate_ema_gpu(prices_gpu, length)
    ema2_gpu = _calculate_ema_gpu(ema1_gpu, length)
    return 2.0 * ema1_gpu - ema2_gpu


def _calculate_lsma_gpu(prices_gpu, length: int):
    # For now, fallback to CPU (GPU implementation more complex)
    return None


__all__ = [
    "_HAS_CUPY",
    "_calculate_ma_gpu",
    "_calculate_ema_gpu",
    "_calculate_wma_gpu",
    "_calculate_dema_gpu",
    "_calculate_lsma_gpu",
]
