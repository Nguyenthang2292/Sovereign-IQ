"""
Utility functions for I Ching module.
"""

import io
import os
import platform
import sys
from typing import Union

from PIL import ImageFont

from config.iching import FONT_PATHS, FONT_SIZE, IMAGES_DIR


def ensure_utf8_stdout() -> None:
    """Ensure stdout uses UTF-8 encoding."""
    if not sys.stdout.encoding or sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
        )

def get_font(font_size: int = FONT_SIZE) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    """
    Get font appropriate for the platform.
    
    Args:
        font_size: Font size
        
    Returns:
        Font object from PIL
        
    Note:
        If falling back to default font, this font may not support special Unicode
        characters (such as Chinese) needed for I Ching. A warning will be logged.
    """
    # Import logging here to avoid circular import
    from modules.common.ui.logging import log_warn
    
    # Try default font first
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        pass
    
    # Try font paths by platform
    system = platform.system()
    font_paths = FONT_PATHS.get(system, [])
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except OSError:
                continue
    
    # Fallback to default font - warning because it may not support special Unicode
    log_warn(
        "No suitable TrueType font found, using default font. "
        "Default font may not support special Unicode characters (such as Chinese) "
        "needed for I Ching, which may lead to incorrect display or empty boxes."
    )
    return ImageFont.load_default()


def clean_images_folder() -> int:
    """
    Delete all image files in the images folder.
    
    Only deletes files with image extensions (.png, .jpg, .jpeg, .gif, .bmp, .svg, etc.).
    Non-image files are left untouched.
    
    Returns:
        Number of image files deleted
    """
    # Import logging here to avoid circular import when importing at the top of the file
    from modules.common.ui.logging import log_warn
    
    # Use project-configured path
    images_dir = IMAGES_DIR
    
    # Create folder if it doesn't exist (with parents=True to create intermediate directories)
    images_dir.mkdir(exist_ok=True, parents=True)
    
    # Common image file extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp', '.ico', '.tiff', '.tif'}
    
    # Count files before deletion
    deleted_count = 0
    
    # Iterate through all files in the folder and delete only image files
    try:
        for file_path in images_dir.iterdir():
            if file_path.is_file():
                # Check if file has an image extension (case-insensitive)
                file_ext = file_path.suffix.lower()
                if file_ext in image_extensions:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except OSError as e:
                        # Log warning but continue deleting other files
                        log_warn(f"Unable to delete file {file_path.name}: {e}")
    except Exception as e:
        # If there's a serious error, raise exception
        raise RuntimeError(f"Error while cleaning images folder: {e}") from e
    
    return deleted_count

