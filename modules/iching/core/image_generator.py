"""
I Ching hexagram image generation.
"""

import os
import re
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw

from config.iching import (
    IMAGES_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LINE_HEIGHT,
    NUM_LINES,
    RECTANGLE_END_X,
    RECTANGLE_HEIGHT,
    RECTANGLE_MIDDLE_END,
    RECTANGLE_MIDDLE_START,
    RECTANGLE_START_X,
    START_Y,
)
from modules.common.ui.logging import log_info, log_success, log_warn
from modules.iching.utils.helpers import get_font


def _validate_and_sanitize_filename(filename: str) -> str:
    """
    Validate và sanitize filename để tránh path traversal attacks.
    
    Args:
        filename: Tên file cần validate
        
    Returns:
        Filename đã được sanitize (nếu hợp lệ)
        
    Raises:
        ValueError: Nếu filename không hợp lệ (absolute path, chứa path separators,
                   chứa "..", hoặc không match pattern an toàn)
    """
    # Kiểm tra absolute path
    if os.path.isabs(filename):
        raise ValueError(f"Filename không được là absolute path: {filename}")
    
    # Kiểm tra path separators
    if "/" in filename or "\\" in filename:
        raise ValueError(f"Filename không được chứa path separators: {filename}")
    
    # Kiểm tra path traversal
    if ".." in filename:
        raise ValueError(f"Filename không được chứa '..': {filename}")
    
    # Validate extension phải là .png
    if not filename.lower().endswith(".png"):
        raise ValueError(f"Filename phải có extension .png: {filename}")
    
    # Validate chỉ chứa ký tự an toàn: letters, numbers, underscore, hyphen
    # Pattern: chỉ cho phép [a-zA-Z0-9_-] trước .png (dot chỉ có trong extension)
    base_name = filename[:-4]  # Bỏ .png
    if not re.match(r"^[a-zA-Z0-9_-]+$", base_name):
        raise ValueError(
            f"Filename chỉ được chứa letters, numbers, underscore, và hyphen (trước .png): {filename}"
        )
    
    return filename


def create_hexagram_image(
    grouped_string: List[str], filename: str = "hexagram.png"
) -> str:
    """
    Create hexagram image from grouped string.
    
    Args:
        grouped_string: List of 3-character groups
        filename: Output image filename (must be valid and safe)
        
    Returns:
        Full path to the created image file
        
    Raises:
        ValueError: If grouped_string or filename is invalid
        OSError: If image file cannot be written
    """
    # Validate và sanitize filename trước khi sử dụng
    sanitized_filename = _validate_and_sanitize_filename(filename)
    
    if not grouped_string:
        raise ValueError("grouped_string không được rỗng")
    
    if len(grouped_string) < NUM_LINES:
        raise ValueError(f"Cần ít nhất {NUM_LINES} nhóm để tạo hexagram")
    
    # Validate từng nhóm: độ dài chính xác 3 ký tự và chỉ chứa N/S (case-insensitive)
    allowed_chars = {'N', 'S', 'n', 's'}
    normalized_groups = []
    for idx, group in enumerate(grouped_string):
        # Strip whitespace trước khi validate
        group_stripped = group.strip()
        
        # Kiểm tra độ dài chính xác 3 ký tự
        if len(group_stripped) != 3:
            raise ValueError(
                f"Nhóm thứ {idx + 1} phải có đúng 3 ký tự, nhưng có {len(group_stripped)} ký tự: '{group_stripped}'"
            )
        
        # Kiểm tra chỉ chứa ký tự được phép (N/S, case-insensitive)
        invalid_chars = set(group_stripped) - allowed_chars
        if invalid_chars:
            raise ValueError(
                f"Nhóm thứ {idx + 1} chứa ký tự không hợp lệ: {invalid_chars}. "
                f"Chỉ được phép: N, S (không phân biệt hoa thường)"
            )
        
        # Normalize: strip và chuyển thành chữ hoa để đảm bảo consistency
        normalized_groups.append(group_stripped.upper())
    
    # Cập nhật grouped_string với các nhóm đã được normalize
    grouped_string = normalized_groups
    
    log_info(f"Chuỗi nhóm: {grouped_string}")
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color="beige")
    draw = ImageDraw.Draw(img)
    font = get_font()

    for index in range(NUM_LINES):
        group = grouped_string[index]
        y_pos = START_Y + (NUM_LINES - 1 - index) * LINE_HEIGHT
        line_number = index + 1

        draw.text((10, y_pos - 2), str(line_number), fill="red", font=font)

        if group in {"NNS", "SNN", "NSN"}:
            draw.rectangle(
                [RECTANGLE_START_X, y_pos, RECTANGLE_END_X, y_pos + RECTANGLE_HEIGHT],
                fill="black",
            )
        elif group in {"NSS", "SSN", "SNS"}:
            draw.rectangle(
                [
                    RECTANGLE_START_X,
                    y_pos,
                    RECTANGLE_MIDDLE_START,
                    y_pos + RECTANGLE_HEIGHT,
                ],
                fill="black",
            )
            draw.rectangle(
                [
                    RECTANGLE_MIDDLE_END,
                    y_pos,
                    RECTANGLE_END_X,
                    y_pos + RECTANGLE_HEIGHT,
                ],
                fill="black",
            )
        elif group == "NNN":
            draw.rectangle(
                [
                    RECTANGLE_START_X,
                    y_pos,
                    RECTANGLE_MIDDLE_START,
                    y_pos + RECTANGLE_HEIGHT,
                ],
                fill="red",
            )
            draw.rectangle(
                [
                    RECTANGLE_MIDDLE_END,
                    y_pos,
                    RECTANGLE_END_X,
                    y_pos + RECTANGLE_HEIGHT,
                ],
                fill="red",
            )
        elif group == "SSS":
            draw.rectangle(
                [RECTANGLE_START_X, y_pos, RECTANGLE_END_X, y_pos + RECTANGLE_HEIGHT],
                fill="red",
            )
        else:
            draw.rectangle(
                [RECTANGLE_START_X, y_pos, RECTANGLE_END_X, y_pos + RECTANGLE_HEIGHT],
                fill="gray",
            )
            log_warn(f"Nhóm không hợp lệ: {group}")

    # Sử dụng project-configured output directory
    output_dir = IMAGES_DIR
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Tạo đường dẫn an toàn bằng cách join output_dir với sanitized filename
    full_path = output_dir / sanitized_filename
    img.save(full_path)
    file_size = full_path.stat().st_size
    log_success(f"Đã tạo file ảnh thành công: {full_path}, kích thước: {file_size} bytes")
    return str(full_path)

