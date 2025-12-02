"""
Adaptive Trend Classification (ATC) - Main computation module.

Module này cung cấp hàm chính `compute_atc_signals` để tính toán ATC signals
từ price data. ATC sử dụng nhiều loại Moving Averages (EMA, HMA, WMA, DEMA, LSMA, KAMA)
với adaptive weighting dựa trên simulated equity curves.

Cấu trúc tính toán:
1. Layer 1: Tính signals cho từng loại MA dựa trên equity curves
2. Layer 2: Tính trọng số từ Layer 1 signals
3. Final: Kết hợp tất cả để tạo Average_Signal

Các module hỗ trợ:
- utils.py: Core utilities (rate_of_change, diflen, exp_growth)
- moving_averages.py: MA calculations
- signals.py: Signal generation
- equity.py: Equity curve calculations
- layer1.py: Layer 1 processing
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .equity import equity_series
from .layer1 import cut_signal, _layer1_signal_for_ma
from .moving_averages import set_of_moving_averages
from .utils import rate_of_change


def compute_atc_signals(
    prices: pd.Series,
    src: Optional[pd.Series] = None,
    *,
    ema_len: int = 28,
    hull_len: int = 28,
    wma_len: int = 28,
    dema_len: int = 28,
    lsma_len: int = 28,
    kama_len: int = 28,
    ema_w: float = 1.0,
    hma_w: float = 1.0,
    wma_w: float = 1.0,
    dema_w: float = 1.0,
    lsma_w: float = 1.0,
    kama_w: float = 1.0,
    robustness: str = "Medium",
    La: float = 0.02,
    De: float = 0.03,
    cutout: int = 0,
) -> dict[str, pd.Series]:
    """
    Tính toán Adaptive Trend Classification (ATC) signals.

    Hàm này orchestrates toàn bộ quá trình tính toán ATC:
    1. Tính các Moving Averages với nhiều lengths
    2. Tính Layer 1 signals cho từng loại MA
    3. Tính Layer 2 weights từ Layer 1 signals
    4. Kết hợp tất cả để tạo Average_Signal

    Args:
        prices: Series giá (thường là close) để tính rate of change và signals.
        src: Series nguồn để tính MA. Nếu None, dùng prices.
        ema_len, hull_len, wma_len, dema_len, lsma_len, kama_len:
            Độ dài window cho từng loại MA.
        ema_w, hma_w, wma_w, dema_w, lsma_w, kama_w:
            Trọng số khởi tạo cho từng họ MA ở Layer 2.
        robustness: "Narrow" / "Medium" / "Wide" - độ rộng của length offsets.
        La: Lambda (growth rate) cho exponential growth factor.
        De: Decay rate cho equity calculations.
        cutout: Số bar bỏ qua đầu chuỗi.

    Returns:
        Dictionary chứa:
            - EMA_Signal, HMA_Signal, WMA_Signal, DEMA_Signal, LSMA_Signal, KAMA_Signal:
              Layer 1 signals cho từng loại MA
            - EMA_S, HMA_S, WMA_S, DEMA_S, LSMA_S, KAMA_S:
              Layer 2 equity weights
            - Average_Signal: Final combined signal
    """
    if src is None:
        src = prices

    # DECLARE MOVING AVERAGES (SetOfMovingAverages)
    EMA = set_of_moving_averages(ema_len, src, "EMA", robustness=robustness)
    HMA = set_of_moving_averages(hull_len, src, "HMA", robustness=robustness)
    WMA = set_of_moving_averages(wma_len, src, "WMA", robustness=robustness)
    DEMA = set_of_moving_averages(dema_len, src, "DEMA", robustness=robustness)
    LSMA = set_of_moving_averages(lsma_len, src, "LSMA", robustness=robustness)
    KAMA = set_of_moving_averages(kama_len, src, "KAMA", robustness=robustness)

    # MAIN CALCULATIONS - Adaptability Layer 1
    EMA_Signal, _, _ = _layer1_signal_for_ma(prices, EMA, L=La, De=De, cutout=cutout)
    HMA_Signal, _, _ = _layer1_signal_for_ma(prices, HMA, L=La, De=De, cutout=cutout)
    WMA_Signal, _, _ = _layer1_signal_for_ma(prices, WMA, L=La, De=De, cutout=cutout)
    DEMA_Signal, _, _ = _layer1_signal_for_ma(prices, DEMA, L=La, De=De, cutout=cutout)
    LSMA_Signal, _, _ = _layer1_signal_for_ma(prices, LSMA, L=La, De=De, cutout=cutout)
    KAMA_Signal, _, _ = _layer1_signal_for_ma(prices, KAMA, L=La, De=De, cutout=cutout)

    # Adaptability Layer 2
    R = rate_of_change(prices)
    EMA_S = equity_series(ema_w, EMA_Signal, R, L=La, De=De, cutout=cutout)
    HMA_S = equity_series(hma_w, HMA_Signal, R, L=La, De=De, cutout=cutout)
    WMA_S = equity_series(wma_w, WMA_Signal, R, L=La, De=De, cutout=cutout)
    DEMA_S = equity_series(dema_w, DEMA_Signal, R, L=La, De=De, cutout=cutout)
    LSMA_S = equity_series(lsma_w, LSMA_Signal, R, L=La, De=De, cutout=cutout)
    KAMA_S = equity_series(kama_w, KAMA_Signal, R, L=La, De=De, cutout=cutout)

    # FINAL CALCULATIONS
    nom = (
        cut_signal(EMA_Signal) * EMA_S
        + cut_signal(HMA_Signal) * HMA_S
        + cut_signal(WMA_Signal) * WMA_S
        + cut_signal(DEMA_Signal) * DEMA_S
        + cut_signal(LSMA_Signal) * LSMA_S
        + cut_signal(KAMA_Signal) * KAMA_S
    )
    den = EMA_S + HMA_S + WMA_S + DEMA_S + LSMA_S + KAMA_S
    Average_Signal = nom / den

    return {
        "EMA_Signal": EMA_Signal,
        "HMA_Signal": HMA_Signal,
        "WMA_Signal": WMA_Signal,
        "DEMA_Signal": DEMA_Signal,
        "LSMA_Signal": LSMA_Signal,
        "KAMA_Signal": KAMA_Signal,
        "EMA_S": EMA_S,
        "HMA_S": HMA_S,
        "WMA_S": WMA_S,
        "DEMA_S": DEMA_S,
        "LSMA_S": LSMA_S,
        "KAMA_S": KAMA_S,
        "Average_Signal": Average_Signal,
    }


__all__ = [
    "compute_atc_signals",
]

