
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List

"""
Base classes và interfaces cho Target Calculator.

Cung cấp base classes để mở rộng cho các loại target khác nhau.
"""



@dataclass
class TargetResult:
    """
    Base class cho kết quả tính toán target.

    Attributes:
        target_price: Giá target tính toán
        delta: Delta (move_abs) - khoảng cách tuyệt đối từ giá hiện tại
        delta_pct: Delta theo phần trăm
        label: Nhãn mô tả target (ví dụ: "ATR x1", "Fibonacci 0.618", v.v.)
    """

    target_price: float
    delta: float
    delta_pct: float
    label: str


class TargetCalculator(ABC):
    """
    Abstract base class cho các target calculator.

    Mỗi loại target (ATR, Fibonacci, Support/Resistance, v.v.)
    sẽ implement class này.
    """

    @abstractmethod
    def calculate(self, current_price: float, direction: str, **kwargs) -> List[TargetResult]:
        """
        Tính toán các target prices.

        Args:
            current_price: Giá hiện tại
            direction: Hướng di chuyển dự kiến ("UP", "DOWN", "NEUTRAL")
            **kwargs: Các tham số cụ thể cho từng loại target

        Returns:
            List các TargetResult chứa thông tin về target prices
        """
        pass

    @abstractmethod
    def format_display(self, result: TargetResult, format_price_func: Callable[[float], str]) -> str:
        """
        Format một target result thành string để hiển thị.

        Args:
            result: TargetResult cần format
            format_price_func: Function để format giá

        Returns:
            String đã được format
        """
        pass
