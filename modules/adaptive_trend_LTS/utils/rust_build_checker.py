"""
Rust Backend Build Checker for ATC LTS.

Provides utilities to check if Rust backend is available and give
user-friendly instructions for building it if not.
"""

from pathlib import Path
from typing import Dict


def check_rust_backend() -> Dict[str, any]:
    """
    Check Rust backend availability and provide build instructions if missing.

    Returns:
        dict with keys:
        - 'available': bool - True if Rust backend is available
        - 'message': str - User-friendly status message
        - 'build_command': str - Command to build Rust backend if not available
    """
    # Define build command logic relative to this file
    project_root = Path(__file__).parent.parent.parent.parent
    rust_dir = project_root / "modules" / "adaptive_trend_LTS" / "rust_extensions"
    build_cmd = (
        f'cd "{rust_dir}" && maturin develop --release\n'
        f"  OR from project root: .\\build_rust.bat (Windows) / ./build_rust.ps1"
    )

    try:
        from modules.adaptive_trend_LTS.core.rust_backend import RUST_AVAILABLE

        if RUST_AVAILABLE:
            return {
                "available": True,
                "message": "✓ Rust backend is available and will be used for optimal ATC performance.",
                "build_command": build_cmd,
            }
    except ImportError:
        pass

    # Rust backend not available
    return {
        "available": False,
        "message": (
            "Rust backend is NOT available. ATC will use slower Numba/pandas_ta fallback.\n"
            "For optimal performance (2-5x faster), build the Rust backend."
        ),
        "build_command": build_cmd,
    }


def is_rust_available() -> bool:
    """
    Quick check if Rust backend is available.

    Returns:
        bool: True if Rust backend is available, False otherwise
    """
    return check_rust_backend()["available"]


__all__ = ["check_rust_backend", "is_rust_available"]
