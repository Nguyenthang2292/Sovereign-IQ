from dataclasses import dataclass
from typing import Callable, List, Optional

from modules.targets.core.base import TargetCalculator, TargetResult

"""
ATR Target Calculator.

Tính toán các target prices dựa trên ATR (Average True Range) multiples.
"""


@dataclass
class ATRTargetResult(TargetResult):
    """
    Kết quả tính toán ATR target.

    Attributes:
        multiple: ATR multiple (1, 2, 3, ...)
        target_price: Giá target tính toán
        delta: Delta (move_abs) - khoảng cách tuyệt đối từ giá hiện tại
        delta_pct: Delta theo phần trăm
        label: Nhãn mô tả target (ví dụ: "ATR x1")
    """

    multiple: int

    def __post_init__(self):
        """Đảm bảo label được set từ multiple nếu chưa có."""
        if not self.label:
            self.label = f"ATR x{self.multiple}"


class ATRTargetCalculator(TargetCalculator):
    """
    Calculator cho ATR-based targets.
    """

    def calculate(
        self,
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
            >>> calculator = ATRTargetCalculator()
            >>> results = calculator.calculate(
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
                    label=f"ATR x{multiple}",
                )
            )

        return results

    def format_display(
        self,
        result: ATRTargetResult,
        format_price_func: Callable[[float], str],
    ) -> str:
        """
        Format một ATR target result thành string để hiển thị.

        Args:
            result: ATRTargetResult cần format
            format_price_func: Function để format giá (ví dụ: format_price từ utils)

        Returns:
            String đã được format

        Example:
            >>> calculator = ATRTargetCalculator()
            >>> result = ATRTargetResult(
            ...     multiple=1,
            ...     target_price=102.0,
            ...     delta=2.0,
            ...     delta_pct=2.0,
            ...     label="ATR x1"
            ... )
            >>> calculator.format_display(result, lambda x: f"${x:.2f}")
            '  ATR x1: $102.00 | Delta $2.00 (2.00%)'
        """
        return (
            f"  {result.label}: {format_price_func(result.target_price)} | "
            f"Delta {format_price_func(result.delta)} ({result.delta_pct:.2f}%)"
        )


# Convenience functions để giữ backward compatibility
def calculate_atr_targets(
    current_price: float,
    atr: float,
    direction: str,
    multiples: Optional[List[int]] = None,
) -> List[ATRTargetResult]:
    """
    Tính toán các target prices dựa trên ATR multiples.

    Convenience function để giữ backward compatibility với code cũ.

    Args:
        current_price: Giá hiện tại
        atr: Giá trị ATR (Average True Range)
        direction: Hướng di chuyển dự kiến ("UP", "DOWN", "NEUTRAL")
        multiples: Danh sách ATR multiples để tính toán (mặc định: [1, 2, 3])

    Returns:
        List các ATRTargetResult chứa thông tin về target prices
    """
    calculator = ATRTargetCalculator()
    return calculator.calculate(
        current_price=current_price,
        atr=atr,
        direction=direction,
        multiples=multiples,
    )


def format_atr_target_display(
    result: ATRTargetResult,
    format_price_func: Callable[[float], str],
) -> str:
    """
    Format một ATR target result thành string để hiển thị.

    Convenience function để giữ backward compatibility với code cũ.

    Args:
        result: ATRTargetResult cần format
        format_price_func: Function để format giá (ví dụ: format_price từ utils)

    Returns:
        String đã được format
    """
    calculator = ATRTargetCalculator()
    return calculator.format_display(result, format_price_func)
