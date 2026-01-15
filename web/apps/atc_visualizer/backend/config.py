"""
Configuration for ATC Visualizer app.
"""

from pathlib import Path

# Get paths
BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# Go up from: web/apps/atc_visualizer/backend -> project root
PROJECT_ROOT = BASE_DIR.parent.parent.parent

# App settings
APP_TITLE = "ATC Visualizer API"
APP_DESCRIPTION = "REST API for Adaptive Trend Classification visualization"
APP_VERSION = "1.0.0"

# Port configuration
BACKEND_PORT = 8002
FRONTEND_DEV_PORT = 5174

# CORS origins (allow all for visualization tool)
CORS_ORIGINS = ["*"]

# API settings
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "15m"
DEFAULT_LIMIT = 1500
MIN_LIMIT = 100
MAX_LIMIT = 5000

# ATC Parameters defaults
DEFAULT_MA_LENGTH = 28
DEFAULT_ROBUSTNESS = "Medium"
DEFAULT_LAMBDA = 0.02
DEFAULT_DECAY = 0.03
DEFAULT_CUTOUT = 0

# Available timeframes
TIMEFRAMES = [
    {"value": "1m", "label": "1 Minute"},
    {"value": "5m", "label": "5 Minutes"},
    {"value": "15m", "label": "15 Minutes"},
    {"value": "30m", "label": "30 Minutes"},
    {"value": "1h", "label": "1 Hour"},
    {"value": "4h", "label": "4 Hours"},
    {"value": "1d", "label": "1 Day"},
]
