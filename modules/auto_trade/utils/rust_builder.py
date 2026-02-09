"""Auto-build Rust extensions for ATC and XGBoost modules."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def _build_env_for_maturin() -> dict:
    """On Windows, set LIB so the linker finds the current Python's python3XX.lib (avoids ServBay/other LIB paths)."""
    env = os.environ.copy()
    if sys.platform == "win32":
        # Use base prefix so venv finds the base Python's libs (e.g. python313.lib)
        base = getattr(sys, "base_prefix", sys.prefix)
        libs_dir = os.path.join(base, "libs")
        if os.path.isdir(libs_dir):
            # Prepend so linker finds python313.lib from the Python we're building for
            env["LIB"] = libs_dir + os.pathsep + env.get("LIB", "")
    return env


def build_rust_extension(module_path: Path, module_name: str) -> Tuple[bool, str]:
    """
    Build a single Rust extension using maturin.

    Args:
        module_path: Path to the rust_extensions directory
        module_name: Name of the module for logging

    Returns:
        Tuple of (success: bool, message: str)
    """
    cargo_toml = module_path / "Cargo.toml"

    if not cargo_toml.exists():
        return False, f"Cargo.toml not found at {module_path}"

    print(f"Building {module_name} Rust extension...")

    try:
        env = _build_env_for_maturin()
        # Use maturin develop for development builds
        result = subprocess.run(
            [sys.executable, "-m", "maturin", "develop", "--release"],
            cwd=module_path,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # Replace unencodable characters instead of crashing
            timeout=300,  # 5 minutes timeout
        )

        if result.returncode == 0:
            return True, f"  {module_name} built successfully"
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            return False, f"  {module_name} build failed:\n{error_msg}"

    except subprocess.TimeoutExpired:
        return False, f"  {module_name} build timeout (>5 minutes)"
    except FileNotFoundError:
        return False, "  maturin not found. Install with: pip install maturin"
    except Exception as e:
        return False, f"  {module_name} build error: {str(e)}"


def build_all_rust_extensions(verbose: bool = False) -> bool:
    """
    Build all required Rust extensions for the auto-trade system.

    Args:
        verbose: Print detailed build output

    Returns:
        True if all builds succeeded, False otherwise
    """
    project_root = Path(__file__).resolve().parents[3]  # Go up to crypto-probability/

    # Define modules to build
    modules_to_build: List[Tuple[Path, str]] = [
        (
            project_root / "modules" / "adaptive_trend_LTS_mini" / "rust_extensions",
            "ATC LTS Mini",
        ),
        (
            project_root / "modules" / "xgboost_LTS" / "rust_extensions",
            "XGBoost LTS",
        ),
    ]

    print("\n" + "=" * 60)
    print("  Building Rust Extensions for Auto-Trade System")
    print("=" * 60 + "\n")

    all_success = True
    results = []

    for module_path, module_name in modules_to_build:
        success, message = build_rust_extension(module_path, module_name)
        results.append((module_name, success, message))
        all_success &= success

        if verbose or not success:
            print(message)
        else:
            # Show only summary for successful builds
            print(f"  {module_name}")

    # Print summary
    print("\n" + "=" * 60)
    print("  Build Summary")
    print("=" * 60)

    successful = sum(1 for _, success, _ in results if success)
    total = len(results)

    for module_name, success, message in results:
        status = "  PASS" if success else "  FAIL"
        print(f"{status}: {module_name}")

    print(f"\nTotal: {successful}/{total} successful")
    print("=" * 60 + "\n")

    if not all_success:
        print("  Some Rust extensions failed to build.")
        print("The system will fall back to Python implementations (slower).\n")

    return all_success


def check_rust_toolchain() -> bool:
    """
    Check if Rust toolchain is installed.

    Returns:
        True if Rust is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["rustc", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ensure_maturin_installed() -> bool:
    """
    Ensure maturin is installed in the current environment.

    Returns:
        True if maturin is available, False otherwise
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "maturin", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def auto_build_with_checks(verbose: bool = False, clear_cache: bool = True) -> bool:
    """
    Build Rust extensions with prerequisite checks.

    Args:
        verbose: Print detailed output
        clear_cache: Clear Python bytecode cache after building (default: True)

    Returns:
        True if build succeeded or not needed, False if failed
    """
    # Check Rust toolchain
    if not check_rust_toolchain():
        print("  Rust toolchain not found. Skipping Rust builds.")
        print("   Install from: https://rustup.rs/")
        print("   Using Python fallback implementations.\n")
        return False

    # Check maturin
    if not ensure_maturin_installed():
        print("  maturin not found. Installing...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "maturin"],
                check=True,
                capture_output=not verbose,
            )
            print("  maturin installed successfully\n")
        except subprocess.CalledProcessError:
            print("  Failed to install maturin. Skipping Rust builds.\n")
            return False

    # Build all extensions
    build_success = build_all_rust_extensions(verbose=verbose)

    # Clear Python cache after successful build
    if build_success and clear_cache:
        print("\nClearing Python bytecode cache...")
        try:
            from modules.auto_trade.utils.cache_cleaner import clear_module_cache

            clear_module_cache(
                module_names=["adaptive_trend_LTS_mini", "xgboost_LTS"],
                verbose=verbose,
            )
        except Exception as e:
            print(f"  Warning: Failed to clear cache: {e}")
            print("  You may need to manually restart Python to use new extensions.\n")

    return build_success


if __name__ == "__main__":
    # Allow running as standalone script
    import argparse

    parser = argparse.ArgumentParser(description="Build Rust extensions for auto-trade")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--no-cache-clear",
        action="store_true",
        help="Skip clearing Python bytecode cache after build",
    )
    args = parser.parse_args()

    success = auto_build_with_checks(verbose=args.verbose, clear_cache=not args.no_cache_clear)
    sys.exit(0 if success else 1)
