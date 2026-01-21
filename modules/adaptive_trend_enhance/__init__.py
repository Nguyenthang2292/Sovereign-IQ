"""
Adaptive Trend Classification (ATC) Module.

This module provides Adaptive Trend Classification analysis for cryptocurrency trading,
including signal computation, symbol analysis, scanning, and CLI utilities.

Core Components:
- Signal Computation: Calculate ATC signals from price data
- Symbol Analysis: Analyze individual symbols
- Scanner: Scan multiple symbols for trading signals
- Configuration: ATCConfig and configuration utilities
- CLI: Command-line interface components
"""

from modules.adaptive_trend_enhance.cli import (
    display_atc_signals,
    display_scan_results,
    list_futures_symbols,
    parse_args,
    prompt_interactive_mode,
    prompt_timeframe,
)
from modules.adaptive_trend_enhance.core.analyzer import analyze_symbol
from modules.adaptive_trend_enhance.core.compute_atc_signals import compute_atc_signals

# Hardware and Memory Management
from modules.adaptive_trend_enhance.core.hardware_manager import (
    HardwareManager,
    get_hardware_manager,
)
from modules.adaptive_trend_enhance.core.memory_manager import (
    MemoryManager,
    get_memory_manager,
    track_memory,
)
from modules.adaptive_trend_enhance.core.scanner import scan_all_symbols

# Utility functions
from modules.adaptive_trend_enhance.utils import (
    diflen,
    exp_growth,
    rate_of_change,
)

# Configuration
from modules.adaptive_trend_enhance.utils.config import (
    ATCConfig,
    create_atc_config_from_dict,
)

__all__ = [
    # Core analysis
    "analyze_symbol",
    "scan_all_symbols",
    "compute_atc_signals",
    # Configuration
    "ATCConfig",
    "create_atc_config_from_dict",
    # Hardware and Memory Management
    "HardwareManager",
    "get_hardware_manager",
    "MemoryManager",
    "get_memory_manager",
    "track_memory",
    # Utilities
    "rate_of_change",
    "diflen",
    "exp_growth",
    # CLI
    "parse_args",
    "prompt_timeframe",
    "prompt_interactive_mode",
    "display_atc_signals",
    "display_scan_results",
    "list_futures_symbols",
]
