"""File operations utilities."""

import fnmatch
import os
from typing import List, Optional

from modules.common.ui.logging import log_info, log_warn


def cleanup_old_files(directory: str, exclude_patterns: Optional[List[str]] = None) -> int:
    """
    Xóa tất cả files cũ trong thư mục.

    Args:
        directory: Đường dẫn thư mục cần cleanup
        exclude_patterns: List các patterns để giữ lại (optional)

    Returns:
        Số lượng files đã xóa
    """
    if exclude_patterns is None:
        exclude_patterns = []

    if not os.path.exists(directory):
        return 0

    deleted_count = 0
    errors = []

    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if not os.path.isfile(file_path):
                continue

            should_exclude = False
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(filename, pattern):
                    should_exclude = True
                    break
            if should_exclude:
                continue

            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                errors.append(f"{filename}: {str(e)}")

        if deleted_count > 0:
            log_info(f"Đã xóa {deleted_count} file(s) cũ trong {directory}")
        if errors:
            log_warn(f"Có {len(errors)} file(s) không thể xóa: {', '.join(errors[:3])}")
    except Exception as e:
        log_warn(f"Lỗi khi cleanup thư mục {directory}: {e}")

    return deleted_count
