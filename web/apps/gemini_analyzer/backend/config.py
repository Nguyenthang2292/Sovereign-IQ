"""
Configuration for Gemini Analyzer app.

This module contains app-specific configuration settings.
"""

from pathlib import Path
from typing import List


# Get paths
BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
VUE_DIST_DIR = FRONTEND_DIR / "dist"

# Get module paths for serving charts and results
# Go up from: web/apps/gemini_analyzer/backend -> project root
PROJECT_ROOT = BASE_DIR.parent.parent.parent
MODULE_ROOT = PROJECT_ROOT / "modules" / "gemini_chart_analyzer"
CHARTS_DIR = MODULE_ROOT / "charts"
RESULTS_DIR = MODULE_ROOT / "analysis_results"

# Ensure directories exist
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# App settings
APP_TITLE = "Gemini Chart Analyzer API"
APP_DESCRIPTION = "REST API for Gemini Chart Analyzer and Batch Scanner"
APP_VERSION = "1.0.0"

# Port configuration
BACKEND_PORT = 8001
FRONTEND_DEV_PORT = 5173

# CORS origins
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
    "http://localhost:8001",
]

# API prefixes
API_PREFIX = "/api"

# Static file routes
STATIC_CHARTS_ROUTE = "/static/charts"
STATIC_RESULTS_ROUTE = "/static/results"
STATIC_VUE_ROUTE = "/static/vue"

# Task settings
TASK_CLEANUP_HOURS = 1
LOG_MAX_AGE_HOURS = 24

# Chart settings
CHART_DPI = 100
CHART_FIGSIZE = (14, 7)

# Scanner settings
DEFAULT_CHARTS_PER_BATCH = 50
DEFAULT_COOLDOWN_SECONDS = 2.5
DEFAULT_QUOTE_CURRENCY = "USDT"
DEFAULT_EXCHANGE = "binance"
