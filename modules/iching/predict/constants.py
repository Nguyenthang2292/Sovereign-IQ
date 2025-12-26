"""
Các constants cơ bản cho I Ching divination.

Bao gồm:
- Âm/Dương (Yin/Yang)
- Ngũ hành (Five Elements): Kim, Mộc, Thuỷ, Hoả, Thổ
- Quan hệ Ngũ hành: Tương sinh và Tương khắc
"""

from typing import Dict, List, Tuple

# ============================================================================
# Âm/Dương Constants
# ============================================================================

YANG = "Dương"  # Yang (Dương) - masculine, active, positive
YIN = "Âm"      # Yin (Âm) - feminine, passive, negative

# Tuple chứa cả hai giá trị Âm/Dương
YIN_YANG_VALUES = (YIN, YANG)

# ============================================================================
# Ngũ hành Constants (Five Elements)
# ============================================================================

KIM = "Kim"     # Metal (Kim)
MOC = "Mộc"     # Wood (Mộc)
THUY = "Thuỷ"   # Water (Thuỷ)
HOA = "Hoả"     # Fire (Hoả)
THO = "Thổ"     # Earth (Thổ)

# Tuple chứa tất cả 5 ngũ hành theo thứ tự tương sinh
WU_HANG_LIST = (MOC, HOA, THO, KIM, THUY)

# Dictionary mapping tên ngũ hành -> giá trị constant
WU_HANG_DICT = {
    "Kim": KIM,
    "Mộc": MOC,
    "Thuỷ": THUY,
    "Hoả": HOA,
    "Thổ": THO,
}

# ============================================================================
# Ngũ hành Tương sinh (Generating Cycle)
# ============================================================================
# Thủy sinh Mộc, Mộc sinh Hỏa, Hỏa sinh Thổ, Thổ sinh Kim, Kim sinh Thủy

# Dictionary mapping ngũ hành -> ngũ hành mà nó sinh ra
WU_HANG_TUONG_SINH: Dict[str, str] = {
    THUY: MOC,  # Thủy sinh Mộc
    MOC: HOA,   # Mộc sinh Hỏa
    HOA: THO,   # Hỏa sinh Thổ
    THO: KIM,   # Thổ sinh Kim
    KIM: THUY,  # Kim sinh Thủy
}

# Dictionary mapping ngũ hành -> ngũ hành sinh ra nó
WU_HANG_DUOC_SINH: Dict[str, str] = {
    MOC: THUY,  # Mộc được Thủy sinh
    HOA: MOC,   # Hỏa được Mộc sinh
    THO: HOA,   # Thổ được Hỏa sinh
    KIM: THO,   # Kim được Thổ sinh
    THUY: KIM,  # Thủy được Kim sinh
}

# List các cặp tương sinh (tuần hoàn)
WU_HANG_TUONG_SINH_PAIRS: List[Tuple[str, str]] = [
    (THUY, MOC),  # Thủy sinh Mộc
    (MOC, HOA),   # Mộc sinh Hỏa
    (HOA, THO),   # Hỏa sinh Thổ
    (THO, KIM),   # Thổ sinh Kim
    (KIM, THUY),  # Kim sinh Thủy
]

# ============================================================================
# Ngũ hành Tương khắc (Overcoming Cycle)
# ============================================================================
# Thủy khắc Hỏa, Hỏa khắc Kim, Kim khắc Mộc, Mộc khắc Thổ, Thổ khắc Thủy

# Dictionary mapping ngũ hành -> ngũ hành mà nó khắc
WU_HANG_TUONG_KHAC: Dict[str, str] = {
    THUY: HOA,  # Thủy khắc Hỏa
    HOA: KIM,   # Hỏa khắc Kim
    KIM: MOC,   # Kim khắc Mộc
    MOC: THO,   # Mộc khắc Thổ
    THO: THUY,  # Thổ khắc Thủy
}

# Dictionary mapping ngũ hành -> ngũ hành khắc nó
WU_HANG_BI_KHAC: Dict[str, str] = {
    HOA: THUY,  # Hỏa bị Thủy khắc
    KIM: HOA,   # Kim bị Hỏa khắc
    MOC: KIM,   # Mộc bị Kim khắc
    THO: MOC,   # Thổ bị Mộc khắc
    THUY: THO,  # Thủy bị Thổ khắc
}

# List các cặp tương khắc (tuần hoàn)
WU_HANG_TUONG_KHAC_PAIRS: List[Tuple[str, str]] = [
    (THUY, HOA),  # Thủy khắc Hỏa
    (HOA, KIM),   # Hỏa khắc Kim
    (KIM, MOC),   # Kim khắc Mộc
    (MOC, THO),   # Mộc khắc Thổ
    (THO, THUY),  # Thổ khắc Thủy
]

__all__ = [
    # Âm/Dương
    "YANG",
    "YIN",
    "YIN_YANG_VALUES",
    # Ngũ hành
    "KIM",
    "MOC",
    "THUY",
    "HOA",
    "THO",
    "WU_HANG_LIST",
    "WU_HANG_DICT",
    # Ngũ hành Tương sinh
    "WU_HANG_TUONG_SINH",
    "WU_HANG_DUOC_SINH",
    "WU_HANG_TUONG_SINH_PAIRS",
    # Ngũ hành Tương khắc
    "WU_HANG_TUONG_KHAC",
    "WU_HANG_BI_KHAC",
    "WU_HANG_TUONG_KHAC_PAIRS",
]

