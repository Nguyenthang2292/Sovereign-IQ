"""I Ching prediction and divination constants."""

from modules.iching.predict.constants import (
    HOA,
    KIM,
    MOC,
    THO,
    THUY,
    WU_HANG_BI_KHAC,
    WU_HANG_DICT,
    WU_HANG_DUOC_SINH,
    WU_HANG_LIST,
    WU_HANG_TUONG_KHAC,
    WU_HANG_TUONG_KHAC_PAIRS,
    WU_HANG_TUONG_SINH,
    WU_HANG_TUONG_SINH_PAIRS,
    YANG,
    YIN,
    YIN_YANG_VALUES,
)
from modules.iching.predict.dia_chi import (
    DIA_CHI_DICT,
    DIA_CHI_LIST,
    LUC_HOP_DICT,
    LUC_HOP_PAIRS,
    LUC_XUNG_DICT,
    LUC_XUNG_PAIRS,
    DiaChi,
)
from modules.iching.predict.thien_can import (
    THIEN_CAN_DICT,
    THIEN_CAN_LIST,
    ThienCan,
)

# Note: get_yin_yang, get_wu_hang, is_yang, is_yin từ cả hai module có cùng tên
# Để tránh xung đột, sử dụng các alias với prefix:
# - get_dia_chi_yin_yang, get_dia_chi_wu_hang, is_dia_chi_yang, is_dia_chi_yin
# - get_thien_can_yin_yang, get_thien_can_wu_hang, is_thien_can_yang, is_thien_can_yin
# Hoặc import trực tiếp từ module con: from modules.iching.predict.dia_chi import get_yin_yang

__all__ = [
    # Constants
    "YANG",
    "YIN",
    "YIN_YANG_VALUES",
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
    # Địa Chi
    "DiaChi",
    "DIA_CHI_LIST",
    "DIA_CHI_DICT",
    # Địa Chi Relationships Constants
    "LUC_HOP_PAIRS",
    "LUC_XUNG_PAIRS",
    "LUC_HOP_DICT",
    "LUC_XUNG_DICT",
    # Thiên Can
    "ThienCan",
    "THIEN_CAN_LIST",
    "THIEN_CAN_DICT",
]
