"""Build utilities for ensuring Rust and CUDA extensions are built."""

import subprocess
import sys
from pathlib import Path

from modules.common.utils import log_error, log_info, log_success, log_warn


def ensure_rust_extensions_built():
    """Ensure Rust extensions are built and up-to-date."""
    log_info("Ensuring Rust extensions are built...")
    try:
        # Locate rust_extensions directory
        # From benchmark_comparison/ -> ../ -> ../rust_extensions
        script_dir = Path(__file__).parent.parent
        rust_ext_dir = script_dir.parent / "rust_extensions"

        if not rust_ext_dir.exists():
            log_error(f"Rust extensions directory not found at {rust_ext_dir}")
            return

        log_info(f"Building Rust extensions in {rust_ext_dir}...")

        # Run maturin develop --release
        # Use sys.executable to ensure we use the same python environment
        cmd = [sys.executable, "-m", "maturin", "develop", "--release"]

        subprocess.run(cmd, cwd=str(rust_ext_dir), capture_output=True, text=True, check=True, encoding="utf-8")

        log_success("Rust extensions built successfully")

    except subprocess.CalledProcessError as e:
        log_error(f"Failed to build Rust extensions: {e}")
        log_error(f"Stdout: {e.stdout}")
        log_error(f"Stderr: {e.stderr}")
        # We don't exit here, allowing the script to try running anyway (maybe old binary works)
    except Exception as e:
        log_error(f"Error building Rust extensions: {e}")


def ensure_cuda_extensions_built():
    """Ensure CUDA extensions are built and up-to-date."""
    log_info("Ensuring CUDA extensions are built...")
    try:
        script_dir = Path(__file__).parent.parent
        rust_ext_dir = script_dir.parent / "rust_extensions"
        build_script = rust_ext_dir / "build_cuda.ps1"

        if not build_script.exists():
            log_warn(f"CUDA build script not found at {build_script}, skipping CUDA build")
            return

        log_info(f"Building CUDA extensions using {build_script}...")

        # Run PowerShell script
        cmd = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(build_script)]
        result = subprocess.run(cmd, cwd=str(rust_ext_dir), capture_output=True, text=True, encoding="utf-8")

        if result.returncode == 0:
            log_success("CUDA extensions built successfully")
        else:
            log_warn("CUDA build completed with warnings or errors")
            log_warn(f"Stdout: {result.stdout}")
            log_warn(f"Stderr: {result.stderr}")

    except Exception as e:
        log_warn(f"Error building CUDA extensions: {e}")
        log_warn("CUDA benchmarks may not be available")
