"""
Địa Chi constants and utilities for I Ching divination.

Địa Chi (Earthly Branches) là 12 chi cơ bản trong hệ thống can chi:
- Tý, Sửu, Dần, Mão, Thìn, Tỵ, Ngọ, Mùi, Thân, Dậu, Tuất, Hợi

Trong Lục hào có địa vị vô cùng quan trọng. Nhất định phải nhớ kỹ thuộc tính
ngũ hành cùng quan hệ tương hỗ, xung hợp, sinh khắc.

Bao gồm:
- Định nghĩa 12 Địa chi với thuộc tính Âm/Dương và Ngũ hành
- Quan hệ lục hợp (6 cặp hợp) và lục xung (6 cặp xung)
- Hệ thống chấm điểm dựa trên quan hệ
"""

from typing import Dict, List, Optional, Tuple

from modules.iching.predict.constants import (
    HOA,
    KIM,
    MOC,
    THO,
    THUY,
    YANG,
    YIN,
)

# ============================================================================
# Quan hệ giữa các Địa Chi (Lục hợp và Lục xung) - Module level constants
# ============================================================================

# Lục hợp (6 cặp hợp)
LUC_HOP_PAIRS: List[Tuple[str, str]] = [
    ("Tý", "Sửu"),  # Tý hợp Sửu
    ("Dần", "Hợi"),  # Dần hợp Hợi
    ("Mão", "Tuất"),  # Mão hợp Tuất
    ("Thìn", "Dậu"),  # Thìn hợp Dậu
    ("Tỵ", "Thân"),  # Tỵ hợp Thân
    ("Ngọ", "Mùi"),  # Ngọ hợp Mùi
]

# Dictionary mapping Địa chi -> Địa chi hợp (bidirectional)
LUC_HOP_DICT: Dict[str, str] = {}
for chi1, chi2 in LUC_HOP_PAIRS:
    LUC_HOP_DICT[chi1] = chi2
    LUC_HOP_DICT[chi2] = chi1

# Lục xung (6 cặp xung)
LUC_XUNG_PAIRS: List[Tuple[str, str]] = [
    ("Tý", "Ngọ"),  # Tý xung Ngọ
    ("Sửu", "Mùi"),  # Sửu xung Mùi
    ("Dần", "Thân"),  # Dần xung Thân
    ("Mão", "Dậu"),  # Mão xung Dậu
    ("Thìn", "Tuất"),  # Thìn xung Tuất
    ("Tỵ", "Hợi"),  # Tỵ xung Hợi
]

# Dictionary mapping Địa chi -> Địa chi xung (bidirectional)
LUC_XUNG_DICT: Dict[str, str] = {}
for chi1, chi2 in LUC_XUNG_PAIRS:
    LUC_XUNG_DICT[chi1] = chi2
    LUC_XUNG_DICT[chi2] = chi1


# ============================================================================
# Class DiaChi
# ============================================================================


class DiaChi:
    """
    Class đại diện cho một Địa Chi với các thuộc tính và methods.

    Attributes:
        name: Tên Địa Chi (Tý, Sửu, Dần, ...)
        yin_yang: "Dương" hoặc "Âm"
        wu_hang: Ngũ hành (Mộc, Hoả, Thổ, Kim, Thuỷ)
    """

    def __init__(self, name: str, yin_yang: str, wu_hang: str):
        """
        Khởi tạo Địa Chi.

        Args:
            name: Tên Địa Chi
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
        """String representation của Địa Chi."""
        return f"DiaChi(name='{self.name}', yin_yang='{self.yin_yang}', wu_hang='{self.wu_hang}')"

    def __eq__(self, other) -> bool:
        """So sánh bằng dựa trên name."""
        if not isinstance(other, DiaChi):
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
        Lấy Âm/Dương của Địa Chi này.

        Returns:
            "Dương" hoặc "Âm"
        """
        return self.yin_yang

    def get_wu_hang(self) -> str:
        """
        Lấy Ngũ hành của Địa Chi này.

        Returns:
            Ngũ hành (Mộc, Hoả, Thổ, Kim, Thuỷ)
        """
        return self.wu_hang

    def is_yang(self) -> bool:
        """
        Kiểm tra Địa Chi này có phải Dương (Yang) không.

        Note: Yang (Dương) là dương, tránh nhầm lẫn với Âm (Yin).

        Returns:
            True nếu là Dương (Yang), False nếu không
        """
        return self.yin_yang == YANG

    def is_yin(self) -> bool:
        """
        Kiểm tra Địa Chi này có phải Âm (Yin) không.

        Note: Yin (Âm) là âm, tránh nhầm lẫn với Dương (Yang).

        Returns:
            True nếu là Âm (Yin), False nếu không
        """
        return self.yin_yang == YIN

    def get_luc_hop(self) -> Optional["DiaChi"]:
        """
        Lấy Địa chi hợp với Địa chi này.

        Returns:
            DiaChi object hợp với instance này, None nếu không tìm thấy
        """
        hop_name = LUC_HOP_DICT.get(self.name)
        if hop_name:
            return DIA_CHI_DICT.get(hop_name)
        return None

    def get_luc_xung(self) -> Optional["DiaChi"]:
        """
        Lấy Địa chi xung với Địa chi này.

        Returns:
            DiaChi object xung với instance này, None nếu không tìm thấy
        """
        xung_name = LUC_XUNG_DICT.get(self.name)
        if xung_name:
            return DIA_CHI_DICT.get(xung_name)
        return None

    def is_luc_hop_with(self, other: "DiaChi") -> bool:
        """
        Kiểm tra Địa chi này có hợp với Địa chi khác không.

        Args:
            other: Địa chi khác để kiểm tra

        Returns:
            True nếu hợp nhau, False nếu không
        """
        return DiaChi.is_luc_hop(self.name, other.name)

    def is_luc_xung_with(self, other: "DiaChi") -> bool:
        """
        Kiểm tra Địa chi này có xung với Địa chi khác không.

        Args:
            other: Địa chi khác để kiểm tra

        Returns:
            True nếu xung nhau, False nếu không
        """
        return DiaChi.is_luc_xung(self.name, other.name)

    # ========================================================================
    # Class Methods
    # ========================================================================

    @classmethod
    def get_by_name(cls, name: str) -> Optional["DiaChi"]:
        """
        Lấy DiaChi object theo tên.

        Args:
            name: Tên Địa Chi (ví dụ: "Tý", "Sửu", ...)

        Returns:
            DiaChi object nếu tìm thấy, None nếu không tìm thấy
        """
        return DIA_CHI_DICT.get(name)

    @classmethod
    def get_yin_yang_by_name(cls, name: str) -> Optional[str]:
        """
        Lấy Âm/Dương của Địa Chi theo tên.

        Args:
            name: Tên Địa Chi

        Returns:
            "Dương" hoặc "Âm" nếu tìm thấy, None nếu không tìm thấy
        """
        dc = cls.get_by_name(name)
        return dc.yin_yang if dc else None

    @classmethod
    def get_wu_hang_by_name(cls, name: str) -> Optional[str]:
        """
        Lấy Ngũ hành của Địa Chi theo tên.

        Args:
            name: Tên Địa Chi

        Returns:
            Ngũ hành (Mộc, Hoả, Thổ, Kim, Thuỷ) nếu tìm thấy, None nếu không tìm thấy
        """
        dc = cls.get_by_name(name)
        return dc.wu_hang if dc else None

    @classmethod
    def is_yang_by_name(cls, name: str) -> bool:
        """
        Kiểm tra Địa Chi có phải Dương (Yang) không theo tên.

        Note: Yang (Dương) là dương, tránh nhầm lẫn với Âm (Yin).

        Args:
            name: Tên Địa Chi

        Returns:
            True nếu là Dương (Yang), False nếu không phải hoặc không tìm thấy
        """
        dc = cls.get_by_name(name)
        return dc.yin_yang == YANG if dc else False

    @classmethod
    def is_yin_by_name(cls, name: str) -> bool:
        """
        Kiểm tra Địa Chi có phải Âm (Yin) không theo tên.

        Note: Yin (Âm) là âm, tránh nhầm lẫn với Dương (Yang).

        Args:
            name: Tên Địa Chi

        Returns:
            True nếu là Âm (Yin), False nếu không phải hoặc không tìm thấy
        """
        dc = cls.get_by_name(name)
        return dc.yin_yang == YIN if dc else False

    @classmethod
    def get_luc_hop_by_name(cls, name: str) -> Optional[str]:
        """
        Lấy Địa chi hợp với chi cho trước theo tên.

        Args:
            name: Tên Địa chi

        Returns:
            Tên Địa chi hợp nếu tìm thấy, None nếu không tìm thấy
        """
        return LUC_HOP_DICT.get(name)

    @classmethod
    def get_luc_xung_by_name(cls, name: str) -> Optional[str]:
        """
        Lấy Địa chi xung với chi cho trước theo tên.

        Args:
            name: Tên Địa chi

        Returns:
            Tên Địa chi xung nếu tìm thấy, None nếu không tìm thấy
        """
        return LUC_XUNG_DICT.get(name)

    @classmethod
    def is_luc_hop(cls, chi1: str, chi2: str) -> bool:
        """
        Kiểm tra 2 Địa chi có hợp (lục hợp) không.

        Args:
            chi1: Tên Địa chi thứ nhất
            chi2: Tên Địa chi thứ hai

        Returns:
            True nếu 2 Địa chi hợp nhau, False nếu không
        """
        # Kiểm tra cả hai chiều (chi1-chi2 và chi2-chi1)
        return (chi1 in LUC_HOP_DICT and LUC_HOP_DICT[chi1] == chi2) or (
            chi2 in LUC_HOP_DICT and LUC_HOP_DICT[chi2] == chi1
        )

    @classmethod
    def is_luc_xung(cls, chi1: str, chi2: str) -> bool:
        """
        Kiểm tra 2 Địa chi có xung (lục xung) không.

        Args:
            chi1: Tên Địa chi thứ nhất
            chi2: Tên Địa chi thứ hai

        Returns:
            True nếu 2 Địa chi xung nhau, False nếu không
        """
        # Kiểm tra cả hai chiều (chi1-chi2 và chi2-chi1)
        return (chi1 in LUC_XUNG_DICT and LUC_XUNG_DICT[chi1] == chi2) or (
            chi2 in LUC_XUNG_DICT and LUC_XUNG_DICT[chi2] == chi1
        )

    @classmethod
    def get_all_luc_hop(cls) -> List[Tuple[str, str]]:
        """
        Lấy tất cả các cặp hợp (lục hợp).

        Returns:
            List các tuple (chi1, chi2) đại diện cho các cặp hợp
        """
        return LUC_HOP_PAIRS.copy()

    @classmethod
    def get_all_luc_xung(cls) -> List[Tuple[str, str]]:
        """
        Lấy tất cả các cặp xung (lục xung).

        Returns:
            List các tuple (chi1, chi2) đại diện cho các cặp xung
        """
        return LUC_XUNG_PAIRS.copy()

    @classmethod
    def calculate_score(cls, chi_list: List[str]) -> int:
        """
        Tính điểm dựa trên các quan hệ lục hợp và lục xung trong danh sách Địa chi.

        Hệ thống chấm điểm:
        - 1 cặp lục hợp = +1 điểm
        - 1 cặp lục xung = -1 điểm
        - Điểm có thể âm

        Args:
            chi_list: Danh sách các tên Địa chi

        Returns:
            Tổng điểm (có thể âm)

        Examples:
            >>> DiaChi.calculate_score(["Tý", "Sửu"])  # Tý hợp Sửu
            1
            >>> DiaChi.calculate_score(["Tý", "Ngọ"])  # Tý xung Ngọ
            -1
            >>> DiaChi.calculate_score(["Tý", "Sửu", "Ngọ"])  # Tý hợp Sửu (+1), Tý xung Ngọ (-1)
            0
        """
        if len(chi_list) < 2:
            return 0

        score = 0

        # Kiểm tra tất cả các cặp trong danh sách
        for i in range(len(chi_list)):
            for j in range(i + 1, len(chi_list)):
                chi1 = chi_list[i]
                chi2 = chi_list[j]

                # Kiểm tra lục hợp
                if cls.is_luc_hop(chi1, chi2):
                    score += 1
                # Kiểm tra lục xung
                elif cls.is_luc_xung(chi1, chi2):
                    score -= 1

        return score


# ============================================================================
# Module level: Định nghĩa 12 Địa Chi instances
# ============================================================================

TY = DiaChi(name="Tý", yin_yang=YANG, wu_hang=THUY)  # 1. Tý - Dương - Thuỷ
SUU = DiaChi(name="Sửu", yin_yang=YIN, wu_hang=THO)  # 2. Sửu - Âm - Thổ
DAN = DiaChi(name="Dần", yin_yang=YANG, wu_hang=MOC)  # 3. Dần - Dương - Mộc
MAO = DiaChi(name="Mão", yin_yang=YIN, wu_hang=MOC)  # 4. Mão - Âm - Mộc
THIN = DiaChi(name="Thìn", yin_yang=YANG, wu_hang=THO)  # 5. Thìn - Dương - Thổ
TY_AM = DiaChi(name="Tỵ", yin_yang=YIN, wu_hang=HOA)  # 6. Tỵ - Âm - Hoả
NGO = DiaChi(name="Ngọ", yin_yang=YANG, wu_hang=HOA)  # 7. Ngọ - Dương - Hoả
MUI = DiaChi(name="Mùi", yin_yang=YIN, wu_hang=THO)  # 8. Mùi - Âm - Thổ
THAN = DiaChi(name="Thân", yin_yang=YANG, wu_hang=KIM)  # 9. Thân - Dương - Kim
DAU = DiaChi(name="Dậu", yin_yang=YIN, wu_hang=KIM)  # 10. Dậu - Âm - Kim
TUAT = DiaChi(name="Tuất", yin_yang=YANG, wu_hang=THO)  # 11. Tuất - Dương - Thổ
HOI = DiaChi(name="Hợi", yin_yang=YIN, wu_hang=THUY)  # 12. Hợi - Âm - Thuỷ

# Danh sách 12 Địa Chi theo thứ tự cố định
DIA_CHI_LIST: Tuple[DiaChi, ...] = (
    TY,  # 1. Tý - Dương - Thuỷ
    SUU,  # 2. Sửu - Âm - Thổ
    DAN,  # 3. Dần - Dương - Mộc
    MAO,  # 4. Mão - Âm - Mộc
    THIN,  # 5. Thìn - Dương - Thổ
    TY_AM,  # 6. Tỵ - Âm - Hoả
    NGO,  # 7. Ngọ - Dương - Hoả
    MUI,  # 8. Mùi - Âm - Thổ
    THAN,  # 9. Thân - Dương - Kim
    DAU,  # 10. Dậu - Âm - Kim
    TUAT,  # 11. Tuất - Dương - Thổ
    HOI,  # 12. Hợi - Âm - Thuỷ
)

# Dictionary mapping tên Địa Chi -> DiaChi object
DIA_CHI_DICT: Dict[str, DiaChi] = {dc.name: dc for dc in DIA_CHI_LIST}


__all__ = [
    # Class
    "DiaChi",
    # Instances
    "TY",
    "SUU",
    "DAN",
    "MAO",
    "THIN",
    "TY_AM",
    "NGO",
    "MUI",
    "THAN",
    "DAU",
    "TUAT",
    "HOI",
    # Constants
    "DIA_CHI_LIST",
    "DIA_CHI_DICT",
    "LUC_HOP_PAIRS",
    "LUC_XUNG_PAIRS",
    "LUC_HOP_DICT",
    "LUC_XUNG_DICT",
]
