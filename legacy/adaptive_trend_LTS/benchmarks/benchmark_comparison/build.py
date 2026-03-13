"""Build utilities for ensuring Rust and CUDA extensions are built."""

import subprocess
import sys
from pathlib import Path

from modules.common.utils import log_error, log_info, log_success, log_warn


def ensure_rust_extensions_built(clean_build: bool = True):
    """Ensure Rust extensions are built and up-to-date.

    Args:
        clean_build: If True, run cargo clean before building to ensure fresh compilation.
                    This is important when CUDA kernel headers (.h files) are modified.
    """
    log_info("=" * 60)
    log_info("Building Rust Extensions (including CUDA kernels)")
    log_info("=" * 60)

    try:
        # Locate rust_extensions directory
        # From benchmark_comparison/ -> ../ -> ../rust_extensions
        script_dir = Path(__file__).parent.parent
        rust_ext_dir = script_dir.parent / "rust_extensions"

        if not rust_ext_dir.exists():
            log_error(f"Rust extensions directory not found at {rust_ext_dir}")
            return

        log_info(f"Rust extensions directory: {rust_ext_dir}")
        log_info("")

        # Clean previous build if requested
        if clean_build:
            log_info("Cleaning previous build (cargo clean)...")
            try:
                clean_cmd = ["cargo", "clean"]
                subprocess.run(
                    clean_cmd,
                    cwd=str(rust_ext_dir),
                    capture_output=True,
                    text=True,
                    check=True,
                    encoding="utf-8"
                )
                log_success("Previous build cleaned")
            except subprocess.CalledProcessError as e:
                log_warn(f"cargo clean failed (continuing anyway): {e}")
            except FileNotFoundError:
                log_warn("cargo not found in PATH (continuing anyway)")
            log_info("")

        log_info("Building Rust extensions in release mode...")
        log_info("This will:")
        log_info("  - Compile Rust code with optimizations")
        log_info("  - Embed CUDA kernels from core/gpu_backend/*.cu")
        log_info("  - Include gpu_common.h with NVRTC compatibility fixes")
        log_info("  - Install atc_rust module to current Python environment")
        log_info("")
        log_info("This may take 1-3 minutes on first build...")
        log_info("")

        # Run maturin develop --release
        # Use sys.executable to ensure we use the same python environment
        cmd = [sys.executable, "-m", "maturin", "develop", "--release"]

        result = subprocess.run(
            cmd,
            cwd=str(rust_ext_dir),
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8"
        )

        log_success("Rust extensions built successfully!")
        log_info("")
        log_info("Build output:")
        if result.stdout:
            for line in result.stdout.splitlines()[-10:]:  # Show last 10 lines
                log_info(f"  {line}")
        log_info("")

    except subprocess.CalledProcessError as e:
        log_error("=" * 60)
        log_error("RUST BUILD FAILED")
        log_error("=" * 60)
        log_error(f"Return code: {e.returncode}")
        log_error("")
        if e.stdout:
            log_error("Build stdout:")
            for line in e.stdout.splitlines()[-20:]:  # Show last 20 lines
                log_error(f"  {line}")
        log_error("")
        if e.stderr:
            log_error("Build stderr:")
            for line in e.stderr.splitlines()[-20:]:  # Show last 20 lines
                log_error(f"  {line}")
        log_error("")
        log_error("Common issues:")
        log_error("  1. Rust toolchain not installed - run: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
        log_error("  2. maturin not installed - run: pip install maturin")
        log_error("  3. CUDA toolkit not found (if using CUDA features)")
        log_error("")
        log_warn("Benchmark will attempt to run with existing binary (may have outdated code)")
        log_error("=" * 60)

    except Exception as e:
        log_error(f"Unexpected error building Rust extensions: {e}")
        log_warn("Benchmark will attempt to run with existing binary")


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
