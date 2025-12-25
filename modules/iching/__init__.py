"""
I Ching Module.

Module for generating I Ching hexagrams and automating web form filling.
"""

from modules.iching.core.data_models import HaoInfo, IChingResult
from modules.iching.core.hexagram import (
    analyze_line,
    generate_ns_string,
    group_string,
    prepare_hexagram,
)
from modules.iching.core.image_generator import create_hexagram_image
from modules.iching.core.result_extractor import IChingResultExtractor
from modules.iching.core.thien_can import (
    ThienCan,
    THIEN_CAN_LIST,
    THIEN_CAN_DICT,
    get_thien_can,
    get_yin_yang,
    get_wu_hang,
    is_duong,
    is_am,
)
from modules.iching.core.web_automation import fill_web_form
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
    # Thiên Can
    "ThienCan",
    "THIEN_CAN_LIST",
    "THIEN_CAN_DICT",
    "get_thien_can",
    "get_yin_yang",
    "get_wu_hang",
    "is_duong",
    "is_am",
    # Utilities
    "ensure_utf8_stdout",
    "get_font",
]

