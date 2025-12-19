"""
ATR Target Calculator.

Tính toán các target prices dựa trên ATR multiples và hiển thị kết quả.
"""

from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class ATRTargetResult:
    """
    Kết quả tính toán ATR target.
    
    Attributes:
        multiple: ATR multiple (1, 2, 3, ...)
        target_price: Giá target tính toán
        delta: Delta (move_abs) - khoảng cách tuyệt đối từ giá hiện tại
        delta_pct: Delta theo phần trăm
    """
    multiple: int
    target_price: float
    delta: float
    delta_pct: float


def calculate_atr_targets(
    current_price: float,
    atr: float,
    direction: str,
    multiples: Optional[List[int]] = None,
) -> List[ATRTargetResult]:
    """
    Tính toán các target prices dựa trên ATR multiples.
    
    Args:
        current_price: Giá hiện tại
        atr: Giá trị ATR (Average True Range)
        direction: Hướng di chuyển dự kiến ("UP", "DOWN", "NEUTRAL")
        multiples: Danh sách ATR multiples để tính toán (mặc định: [1, 2, 3])
        
    Returns:
        List các ATRTargetResult chứa thông tin về target prices
        
    Examples:
        >>> results = calculate_atr_targets(
        ...     current_price=100.0,
        ...     atr=2.0,
        ...     direction="UP",
        ...     multiples=[1, 2, 3]
        ... )
        >>> len(results)
        3
        >>> results[0].target_price
        102.0
    """
    if multiples is None:
        multiples = [1, 2, 3]
    
    # Xác định sign dựa trên direction
    if direction == "UP":
        atr_sign = 1
    elif direction == "DOWN":
        atr_sign = -1
    else:
        # NEUTRAL hoặc direction không hợp lệ
        atr_sign = 0
    
    results = []
    
    for multiple in multiples:
        # Tính target price
        target_price = current_price + atr_sign * multiple * atr
        
        # Tính delta (khoảng cách tuyệt đối)
        move_abs = abs(target_price - current_price)
        
        # Tính delta theo phần trăm
        move_pct = (move_abs / current_price) * 100 if current_price > 0 else 0.0
        
        results.append(
            ATRTargetResult(
                multiple=multiple,
                target_price=target_price,
                delta=move_abs,
                delta_pct=move_pct,
            )
        )
    
    return results


def format_atr_target_display(
    result: ATRTargetResult,
    format_price_func,
) -> str:
    """
    Format một ATR target result thành string để hiển thị.
    
    Args:
        result: ATRTargetResult cần format
        format_price_func: Function để format giá (ví dụ: format_price từ utils)
        
    Returns:
        String đã được format
        
    Example:
        >>> result = ATRTargetResult(
        ...     multiple=1,
        ...     target_price=102.0,
        ...     delta=2.0,
        ...     delta_pct=2.0
        ... )
        >>> format_atr_target_display(result, lambda x: f"${x:.2f}")
        '  ATR x1: $102.00 | Delta $2.00 (2.00%)'
    """
    return (
        f"  ATR x{result.multiple}: {format_price_func(result.target_price)} | "
        f"Delta {format_price_func(result.delta)} ({result.delta_pct:.2f}%)"
    )

