"""
I Ching Configuration.

Configuration constants for I Ching hexagram generation and web automation.
"""

import os
import warnings
from pathlib import Path
from urllib.parse import urlparse

# Project paths
# Calculate project root: config/iching.py -> config/ -> project root
_PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = _PROJECT_ROOT / "modules" / "iching" / "images"

# Hexagram generation constants
HEXAGRAM_STRING_LENGTH = 18
GROUP_SIZE = 3
NUM_LINES = 6

# Image generation constants
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 600
LINE_HEIGHT = 80
START_Y = 50
RECTANGLE_START_X = 50
RECTANGLE_END_X = 350
RECTANGLE_MIDDLE_START = 170
RECTANGLE_MIDDLE_END = 230
RECTANGLE_HEIGHT = 20
FONT_SIZE = 24

# Font paths for different platforms
FONT_PATHS = {
    "Windows": [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
    "Darwin": [  # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ],
    "Linux": [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
}

# Selenium wait times
WAIT_TIMEOUT = 10
SCROLL_DELAY_DEFAULT = 0.3
SCROLL_DELAY_LONG = 0.5
CLICK_DELAY = 0.2
SELECT_DELAY = 0.3
SUBMIT_DELAY = 2.0

# Web automation settings
# Default URL with fallback
_DEFAULT_ICHING_URL = "https://simkinhdich.com/boi-dich/luc-hao"
_FALLBACK_ICHING_URL = "https://simkinhdich.com/boi-dich/luc-hao"  # Same as default, can be changed if needed


def _validate_url(url: str) -> bool:
    """
    Validate URL for security: must be HTTPS and have valid scheme.

    Args:
        url: URL string to validate

    Returns:
        True if URL is valid and secure (HTTPS), False otherwise
    """
    try:
        parsed = urlparse(url)
        # Must be HTTPS for security
        if parsed.scheme != "https":
            return False
        # Must have netloc (domain)
        if not parsed.netloc:
            return False
        return True
    except Exception:
        return False


def _get_iching_url() -> str:
    """
    Get I Ching URL with validation and fallback mechanism.

    Returns:
        Valid HTTPS URL string

    Raises:
        ValueError: If no valid URL can be determined
    """
    url = os.getenv("ICHING_URL", _DEFAULT_ICHING_URL)

    # Validate URL
    if not _validate_url(url):
        warning_msg = (
            f"ICHING_URL từ environment variable không hợp lệ hoặc không an toàn (không phải HTTPS): {url}. "
            f"Sử dụng URL mặc định: {_DEFAULT_ICHING_URL}"
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
        url = _DEFAULT_ICHING_URL

    # Double-check default URL
    if not _validate_url(url):
        # Try fallback
        if _validate_url(_FALLBACK_ICHING_URL):
            warning_msg = f"URL mặc định không hợp lệ: {url}. Sử dụng URL dự phòng: {_FALLBACK_ICHING_URL}"
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
            url = _FALLBACK_ICHING_URL
        else:
            raise ValueError(
                f"Không thể xác định URL hợp lệ cho I Ching. URL hiện tại: {url}, URL dự phòng: {_FALLBACK_ICHING_URL}"
            )

    return url


# Get validated I Ching URL
ICHING_URL = _get_iching_url()
