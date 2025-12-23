"""
Core I Ching hexagram generation logic.
"""

import random
from typing import Dict, List, Tuple

from config.iching import GROUP_SIZE, HEXAGRAM_STRING_LENGTH, NUM_LINES
from modules.common.ui.logging import log_info, log_success
from modules.iching.core.image_generator import create_hexagram_image


def generate_ns_string(length: int = HEXAGRAM_STRING_LENGTH) -> str:
    """
    Tạo chuỗi ngẫu nhiên với các ký tự N và S.
    
    Args:
        length: Độ dài chuỗi (mặc định 18)
        
    Returns:
        Chuỗi ngẫu nhiên chứa N và S
    """
    return "".join(random.choice(["N", "S"]) for _ in range(length))


def group_string(string: str, group_size: int = GROUP_SIZE) -> List[str]:
    """
    Nhóm chuỗi thành các nhóm có kích thước group_size.
    
    Args:
        string: Chuỗi cần nhóm
        group_size: Kích thước mỗi nhóm (mặc định 3)
        
    Returns:
        Danh sách các nhóm
    """
    return [string[i : i + group_size] for i in range(0, len(string), group_size)]


def analyze_line(group: str) -> Tuple[bool, bool]:
    """
    Phân tích vạch: trả về (is_solid, is_red).
    
    Args:
        group: Chuỗi 3 ký tự N/S
        
    Returns:
        Tuple (is_solid, is_red)
    """
    solid_patterns = {"NNS", "SNN", "NSN"}
    broken_patterns = {"NSS", "SSN", "SNS"}

    if group in solid_patterns:
        return True, False
    if group in broken_patterns:
        return False, False
    if group == "SSS":
        return True, True
    if group == "NNN":
        return False, True
    raise ValueError(f"Unexpected group pattern: {group}")


def prepare_hexagram() -> List[Dict[str, bool]]:
    """
    Tạo chuỗi hexagram và trả về thông tin vạch.
    
    Returns:
        Danh sách 6 dict, mỗi dict chứa {"is_solid": bool, "is_red": bool}
        
    Raises:
        ValueError: Nếu không thể tạo hexagram hợp lệ
    """
    random_string = generate_ns_string()
    grouped_string = group_string(random_string)
    log_info(f"Chuỗi ngẫu nhiên {HEXAGRAM_STRING_LENGTH} ký tự N/S: {random_string}")
    log_info(f"Chuỗi được nhóm thành {NUM_LINES} nhóm {GROUP_SIZE} ký tự: {grouped_string}")

    if len(grouped_string) < NUM_LINES:
        raise ValueError(f"Cần ít nhất {NUM_LINES} nhóm để tạo hexagram")

    try:
        create_hexagram_image(grouped_string[:NUM_LINES])
    except Exception as e:
        raise ValueError(f"Không thể tạo hình ảnh hexagram: {e}") from e
    line_info: List[Dict[str, bool]] = []
    for group in grouped_string[:NUM_LINES]:
        is_solid, is_red = analyze_line(group)
        line_info.append({"is_solid": is_solid, "is_red": is_red})

    log_success("Đã hoàn thành việc tạo hexagram và phân tích vạch.")
    return line_info

