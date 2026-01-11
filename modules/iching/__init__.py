
from modules.iching.core.data_models import HaoInfo, IChingResult
from modules.iching.core.hexagram import (

"""
I Ching Module.

Module for generating I Ching hexagrams and automating web form filling.
"""

    analyze_line,
    generate_ns_string,
    group_string,
    prepare_hexagram,
)
from modules.iching.core.image_generator import create_hexagram_image
from modules.iching.core.result_extractor import IChingResultExtractor
from modules.iching.core.web_automation import fill_web_form
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
from modules.iching.utils.helpers import ensure_utf8_stdout, get_font

__all__ = [
    # Core functions
    "generate_ns_string",
    "group_string",
    "analyze_line",
    "prepare_hexagram",
    # Image generation
    "create_hexagram_image",
    # Web automation
    "fill_web_form",
    # Result extraction
    "IChingResult",
    "HaoInfo",
    "IChingResultExtractor",
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
    # Utilities
    "ensure_utf8_stdout",
    "get_font",
]
