"""
Thiên Can constants and utilities for I Ching divination.

Thiên Can (Heavenly Stems) là 10 can cơ bản trong hệ thống can chi:
- Giáp, Ất, Bính, Đinh, Mậu, Kỷ, Canh, Tân, Nhâm, Quý
"""

from typing import Dict, Optional, Tuple

from modules.iching.predict.constants import (
    YANG,
    YIN,
    KIM,
    MOC,
    THUY,
    HOA,
    THO,
)


# ============================================================================
# Class ThienCan
# ============================================================================

class ThienCan:
    """
    Class đại diện cho một Thiên Can với các thuộc tính và methods.
    
    Attributes:
        name: Tên Thiên Can (Giáp, Ất, Bính, ...)
        yin_yang: "Dương" hoặc "Âm"
        wu_hang: Ngũ hành (Mộc, Hoả, Thổ, Kim, Thuỷ)
    """
    
    def __init__(self, name: str, yin_yang: str, wu_hang: str):
        """
        Khởi tạo Thiên Can.
        
        Args:
            name: Tên Thiên Can
            yin_yang: "Dương" hoặc "Âm"
            wu_hang: Ngũ hành (Mộc, Hoả, Thổ, Kim, Thuỷ)
            
        Raises:
            ValueError: Nếu yin_yang hoặc wu_hang không hợp lệ
        """
        # Validate yin_yang
        if yin_yang not in (YANG, YIN):
            raise ValueError(f"yin_yang must be '{YANG}' or '{YIN}', got '{yin_yang}'")
        # Validate wu_hang
        if wu_hang not in (MOC, HOA, THO, KIM, THUY):
            raise ValueError(f"wu_hang must be one of: {MOC}, {HOA}, {THO}, {KIM}, {THUY}, got '{wu_hang}'")
        
        self.name = name
        self.yin_yang = yin_yang
        self.wu_hang = wu_hang
    
    def __repr__(self) -> str:
        """String representation của Thiên Can."""
        return f"ThienCan(name='{self.name}', yin_yang='{self.yin_yang}', wu_hang='{self.wu_hang}')"
    
    def __eq__(self, other) -> bool:
        """So sánh bằng dựa trên name."""
        if not isinstance(other, ThienCan):
            return False
        return self.name == other.name
    
    def __hash__(self) -> int:
        """Hash dựa trên name để có thể dùng trong set/dict."""
        return hash(self.name)
    
    # ========================================================================
    # Instance Methods
    # ========================================================================
    
    def get_yin_yang(self) -> str:
        """
        Lấy Âm/Dương của Thiên Can này.
        
        Returns:
            "Dương" hoặc "Âm"
        """
        return self.yin_yang
    
    def get_wu_hang(self) -> str:
        """
        Lấy Ngũ hành của Thiên Can này.
        
        Returns:
            Ngũ hành (Mộc, Hoả, Thổ, Kim, Thuỷ)
        """
        return self.wu_hang
    
    def is_yang(self) -> bool:
        """
        Kiểm tra Thiên Can này có phải Dương (Yang) không.
        
        Note: Yang (Dương) là dương, tránh nhầm lẫn với Âm (Yin).
        
        Returns:
            True nếu là Dương (Yang), False nếu không
        """
        return self.yin_yang == YANG
    
    def is_yin(self) -> bool:
        """
        Kiểm tra Thiên Can này có phải Âm (Yin) không.
        
        Note: Yin (Âm) là âm, tránh nhầm lẫn với Dương (Yang).
        
        Returns:
            True nếu là Âm (Yin), False nếu không
        """
        return self.yin_yang == YIN
    
    # ========================================================================
    # Class Methods
    # ========================================================================
    
    @classmethod
    def get_by_name(cls, name: str) -> Optional['ThienCan']:
        """
        Lấy ThienCan object theo tên.
        
        Args:
            name: Tên Thiên Can (ví dụ: "Giáp", "Ất", ...)
            
        Returns:
            ThienCan object nếu tìm thấy, None nếu không tìm thấy
        """
        return THIEN_CAN_DICT.get(name)
    
    @classmethod
    def get_yin_yang_by_name(cls, name: str) -> Optional[str]:
        """
        Lấy Âm/Dương của Thiên Can theo tên.
        
        Args:
            name: Tên Thiên Can
            
        Returns:
            "Dương" hoặc "Âm" nếu tìm thấy, None nếu không tìm thấy
        """
        tc = cls.get_by_name(name)
        return tc.yin_yang if tc else None
    
    @classmethod
    def get_wu_hang_by_name(cls, name: str) -> Optional[str]:
        """
        Lấy Ngũ hành của Thiên Can theo tên.
        
        Args:
            name: Tên Thiên Can
            
        Returns:
            Ngũ hành (Mộc, Hoả, Thổ, Kim, Thuỷ) nếu tìm thấy, None nếu không tìm thấy
        """
        tc = cls.get_by_name(name)
        return tc.wu_hang if tc else None
    
    @classmethod
    def is_yang_by_name(cls, name: str) -> bool:
        """
        Kiểm tra Thiên Can có phải Dương (Yang) không theo tên.
        
        Note: Yang (Dương) là dương, tránh nhầm lẫn với Âm (Yin).
        
        Args:
            name: Tên Thiên Can
            
        Returns:
            True nếu là Dương (Yang), False nếu không phải hoặc không tìm thấy
        """
        tc = cls.get_by_name(name)
        return tc.yin_yang == YANG if tc else False
    
    @classmethod
    def is_yin_by_name(cls, name: str) -> bool:
        """
        Kiểm tra Thiên Can có phải Âm (Yin) không theo tên.
        
        Note: Yin (Âm) là âm, tránh nhầm lẫn với Dương (Yang).
        
        Args:
            name: Tên Thiên Can
            
        Returns:
            True nếu là Âm (Yin), False nếu không phải hoặc không tìm thấy
        """
        tc = cls.get_by_name(name)
        return tc.yin_yang == YIN if tc else False


# ============================================================================
# Module level: Định nghĩa 10 Thiên Can instances
# ============================================================================

GIAP = ThienCan(name="Giáp", yin_yang=YANG, wu_hang=MOC)      # 1. Giáp - Dương - Mộc
AT = ThienCan(name="Ất", yin_yang=YIN, wu_hang=MOC)          # 2. Ất - Âm - Mộc
BINH = ThienCan(name="Bính", yin_yang=YANG, wu_hang=HOA)     # 3. Bính - Dương - Hoả
DINH = ThienCan(name="Đinh", yin_yang=YIN, wu_hang=HOA)      # 4. Đinh - Âm - Hoả
MAU = ThienCan(name="Mậu", yin_yang=YANG, wu_hang=THO)      # 5. Mậu - Dương - Thổ
KY = ThienCan(name="Kỷ", yin_yang=YIN, wu_hang=THO)          # 6. Kỷ - Âm - Thổ
CANH = ThienCan(name="Canh", yin_yang=YANG, wu_hang=KIM)     # 7. Canh - Dương - Kim
TAN = ThienCan(name="Tân", yin_yang=YIN, wu_hang=KIM)        # 8. Tân - Âm - Kim
NHAM = ThienCan(name="Nhâm", yin_yang=YANG, wu_hang=THUY)    # 9. Nhâm - Dương - Thuỷ
QUY = ThienCan(name="Quý", yin_yang=YIN, wu_hang=THUY)       # 10. Quý - Âm - Thuỷ

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


__all__ = [
    # Class
    "ThienCan",
    # Instances
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
    # Constants
    "THIEN_CAN_LIST",
    "THIEN_CAN_DICT",
]
