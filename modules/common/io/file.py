"""File operations utilities."""

import fnmatch
import os
from typing import List, Optional

from modules.common.ui.logging import log_info, log_warn


def cleanup_old_files(directory: str, exclude_patterns: Optional[List[str]] = None) -> int:
    """
    Delete all old files in the directory.

    Args:
        directory: The directory path to clean up
        exclude_patterns: List of patterns to exclude from deletion (optional)

    Returns:
        The number of files deleted
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
            log_info(f"Deleted {deleted_count} old file(s) in {directory}")
        if errors:
            log_warn(f"{len(errors)} file(s) could not be deleted: {', '.join(errors[:3])}")
    except Exception as e:
        log_warn(f"Error while cleaning up directory {directory}: {e}")

    return deleted_count


__all__ = ["cleanup_old_files"]
