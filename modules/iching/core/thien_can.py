"""
Thiên Can constants and utilities for I Ching divination.

Thiên Can (Heavenly Stems) là 10 can cơ bản trong hệ thống can chi:
- Giáp, Ất, Bính, Đinh, Mậu, Kỷ, Canh, Tân, Nhâm, Quý
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ThienCan:
    """
    Dataclass đại diện cho một Thiên Can.
    
    Attributes:
        name: Tên Thiên Can (Giáp, Ất, Bính, ...)
        yin_yang: "Dương" hoặc "Âm"
        wu_hang: Ngũ hành (Mộc, Hoả, Thổ, Kim, Thuỷ)
    """
    name: str
    yin_yang: str  # "Dương" hoặc "Âm"
    wu_hang: str  # "Mộc", "Hoả", "Thổ", "Kim", "Thuỷ"
    
    def __post_init__(self):
        """Validate yin_yang và wu_hang values."""
        if self.yin_yang not in ("Dương", "Âm"):
            raise ValueError(f"yin_yang must be 'Dương' or 'Âm', got '{self.yin_yang}'")
        if self.wu_hang not in ("Mộc", "Hoả", "Thổ", "Kim", "Thuỷ"):
            raise ValueError(f"wu_hang must be one of: Mộc, Hoả, Thổ, Kim, Thuỷ, got '{self.wu_hang}'")


# Định nghĩa 10 Thiên Can theo thứ tự cố định
GIAP = ThienCan(name="Giáp", yin_yang="Dương", wu_hang="Mộc")
AT = ThienCan(name="Ất", yin_yang="Âm", wu_hang="Mộc")
BINH = ThienCan(name="Bính", yin_yang="Dương", wu_hang="Hoả")
DINH = ThienCan(name="Đinh", yin_yang="Âm", wu_hang="Hoả")
MAU = ThienCan(name="Mậu", yin_yang="Dương", wu_hang="Thổ")
KY = ThienCan(name="Kỷ", yin_yang="Âm", wu_hang="Thổ")
CANH = ThienCan(name="Canh", yin_yang="Dương", wu_hang="Kim")
TAN = ThienCan(name="Tân", yin_yang="Âm", wu_hang="Kim")
NHAM = ThienCan(name="Nhâm", yin_yang="Dương", wu_hang="Thuỷ")
QUY = ThienCan(name="Quý", yin_yang="Âm", wu_hang="Thuỷ")

# Danh sách 10 Thiên Can theo thứ tự cố định
THIEN_CAN_LIST: Tuple[ThienCan, ...] = (
    GIAP,   # 1. Giáp - Dương - Mộc
    AT,     # 2. Ất - Âm - Mộc
    BINH,   # 3. Bính - Dương - Hoả
    DINH,   # 4. Đinh - Âm - Hoả
    MAU,    # 5. Mậu - Dương - Thổ
    KY,     # 6. Kỷ - Âm - Thổ
    CANH,   # 7. Canh - Dương - Kim
    TAN,    # 8. Tân - Âm - Kim
    NHAM,   # 9. Nhâm - Dương - Thuỷ
    QUY,    # 10. Quý - Âm - Thuỷ
)

# Dictionary mapping tên Thiên Can -> ThienCan object
THIEN_CAN_DICT: Dict[str, ThienCan] = {
    tc.name: tc for tc in THIEN_CAN_LIST
}


def get_thien_can(name: str) -> Optional[ThienCan]:
    """
    Lấy ThienCan object theo tên.
    
    Args:
        name: Tên Thiên Can (ví dụ: "Giáp", "Ất", ...)
        
    Returns:
        ThienCan object nếu tìm thấy, None nếu không tìm thấy
    """
    return THIEN_CAN_DICT.get(name)


def get_yin_yang(name: str) -> Optional[str]:
    """
    Lấy Âm/Dương của Thiên Can.
    
    Args:
        name: Tên Thiên Can
        
    Returns:
        "Dương" hoặc "Âm" nếu tìm thấy, None nếu không tìm thấy
    """
    tc = get_thien_can(name)
    return tc.yin_yang if tc else None


def get_wu_hang(name: str) -> Optional[str]:
    """
    Lấy Ngũ hành của Thiên Can.
    
    Args:
        name: Tên Thiên Can
        
    Returns:
        Ngũ hành (Mộc, Hoả, Thổ, Kim, Thuỷ) nếu tìm thấy, None nếu không tìm thấy
    """
    tc = get_thien_can(name)
    return tc.wu_hang if tc else None


def is_duong(name: str) -> bool:
    """
    Kiểm tra Thiên Can có phải Dương không.
    
    Args:
        name: Tên Thiên Can
        
    Returns:
        True nếu là Dương, False nếu không phải hoặc không tìm thấy
    """
    tc = get_thien_can(name)
    return tc.yin_yang == "Dương" if tc else False


def is_am(name: str) -> bool:
    """
    Kiểm tra Thiên Can có phải Âm không.
    
    Args:
        name: Tên Thiên Can
        
    Returns:
        True nếu là Âm, False nếu không phải hoặc không tìm thấy
    """
    tc = get_thien_can(name)
    return tc.yin_yang == "Âm" if tc else False


__all__ = [
    "ThienCan",
    "GIAP",
    "AT",
    "BINH",
    "DINH",
    "MAU",
    "KY",
    "CANH",
    "TAN",
    "NHAM",
    "QUY",
    "THIEN_CAN_LIST",
    "THIEN_CAN_DICT",
    "get_thien_can",
    "get_yin_yang",
    "get_wu_hang",
    "is_duong",
    "is_am",
]

