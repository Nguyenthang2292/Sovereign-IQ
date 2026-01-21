"""
Core computation modules for Adaptive Trend Classification Enhanced (ATC Enhanced).

ENHANCED VERSION with:
- Hardware management (CPU/GPU/RAM auto-detection)
- Memory management (automatic cleanup, leak prevention)
- Numba JIT optimizations for MA calculations
- Intelligent caching
- GPU acceleration support

This package provides core computational functions for ATC:
- Signal computation and processing
- Moving average calculations (ENHANCED)
- Equity curve calculations
- Symbol analysis and scanning
- Signal detection and generation
- Hardware and memory management (NEW)
"""

from modules.adaptive_trend_enhance.core.analyzer import analyze_symbol
from modules.adaptive_trend_enhance.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_enhance.core.compute_equity import equity_series
from modules.adaptive_trend_enhance.core.compute_moving_averages import (
    calculate_kama_atc,
    ma_calculation,
    ma_calculation_enhanced,
    set_of_moving_averages,
    set_of_moving_averages_enhanced,
)
from modules.adaptive_trend_enhance.core.process_layer1 import (
    _layer1_signal_for_ma,
    cut_signal,
    trend_sign,
    weighted_signal,
)
from modules.adaptive_trend_enhance.core.scanner import scan_all_symbols
from modules.adaptive_trend_enhance.core.signal_detection import (
    crossover,
    crossunder,
    generate_signal_from_ma,
)

# NEW: Hardware and Memory Management
from modules.adaptive_trend_enhance.core.hardware_manager import (
    HardwareManager,
    HardwareResources,
    WorkloadConfig,
    get_hardware_manager,
    reset_hardware_manager,
)
from modules.adaptive_trend_enhance.core.memory_manager import (
    MemoryManager,
    MemorySnapshot,
    get_memory_manager,
    reset_memory_manager,
    track_memory,
)

__all__ = [
    # Analysis
    "analyze_symbol",
    "scan_all_symbols",
    # Signal computation
    "compute_atc_signals",
    # Equity calculations
    "equity_series",
    # Moving averages (ENHANCED)
    "calculate_kama_atc",
    "ma_calculation",
    "ma_calculation_enhanced",
    "set_of_moving_averages",
    "set_of_moving_averages_enhanced",
    # Layer 1 processing
    "weighted_signal",
    "cut_signal",
    "trend_sign",
    "_layer1_signal_for_ma",
    # Signal detection
    "crossover",
    "crossunder",
    "generate_signal_from_ma",
    # Hardware Management (NEW)
    "HardwareManager",
    "HardwareResources",
    "WorkloadConfig",
    "get_hardware_manager",
    "reset_hardware_manager",
    # Memory Management (NEW)
    "MemoryManager",
    "MemorySnapshot",
    "get_memory_manager",
    "reset_memory_manager",
    "track_memory",
]
