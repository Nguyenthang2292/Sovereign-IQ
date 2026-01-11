
from pathlib import Path
import os

"""
I Ching Configuration.

Configuration constants for I Ching hexagram generation and web automation.
"""


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
ICHING_URL = os.getenv("ICHING_URL", "https://simkinhdich.com/boi-dich/luc-hao")
