"""
Gemini Chart Batch Scanner Entry Point

This script serves as the main entry point for running the Gemini batch chart scanner
from the command line. It adjusts the Python path to ensure proper imports, handles necessary
stdin and encoding workarounds on Windows, configures standard I/O, and delegates execution
to the batch scanner's CLI main function.

Behavior:
- Adds the project root to sys.path for reliable import resolution.
- On Windows, ensures stdin is available and encoding issues are addressed prior to further imports.
- Calls `configure_windows_stdio` after preparing stdin.
- Imports and invokes the `main` function from the batch scanner CLI module.

Usage:
    python main_gemini_chart_batch_scanner.py [args]

This centralizes environment bootstrapping and dispatches the batch chart analysis job.
"""

import io
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
if "__file__" in globals():
    project_root = Path(__file__).parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

LOG_DIR = Path(__file__).parent / "logs"


def _cleanup_old_logs(log_dir: Path, keep_count: int = 5) -> None:
    """Keep only the N most recent log files in the logs directory."""
    try:
        if not log_dir.exists():
            return

        # Get all log files matching the pattern
        log_files = list(log_dir.glob("gemini_batch_scan_*.log"))

        # Sort by modification time (newest last) or name (since we use YYYYMMDD_HHMMSS)
        log_files.sort()

        # If we have more than keep_count, delete the oldest ones
        if len(log_files) >= keep_count:
            # We keep (keep_count - 1) because we are about to create a new one
            to_delete = log_files[: len(log_files) - (keep_count - 1)]
            for f in to_delete:
                try:
                    f.unlink()
                except Exception:
                    pass
    except Exception:
        # Silently fail for cleanup operations
        pass


def _build_log_file_path() -> Path:
    """Return a timestamped log file path inside the local logs directory."""
    # Ensure directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Cleanup old logs before creating a new one
    _cleanup_old_logs(LOG_DIR, keep_count=5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"gemini_batch_scan_{timestamp}.log"


@contextmanager
def _tee_output(log_file: Path):
    """Duplicate stdout/stderr to the console and the given log file."""
    original_stdout, original_stderr = sys.stdout, sys.stderr
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as fh:

        class _Tee(io.TextIOBase):
            def __init__(self, streams):
                self.streams = streams

            def write(self, data: str) -> int:
                for stream in self.streams:
                    stream.write(data)
                return len(data)

            def flush(self) -> None:
                for stream in self.streams:
                    stream.flush()

        sys.stdout = _Tee([original_stdout, fh])
        sys.stderr = _Tee([original_stderr, fh])
        try:
            yield log_file
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            fh.flush()


# Ensure stdin is available on Windows BEFORE any imports
# This is critical when running the file directly (not via wrapper)
# This must happen BEFORE configure_windows_stdio() is called
if sys.platform == "win32":
    try:
        if sys.stdin is None or (hasattr(sys.stdin, "closed") and sys.stdin.closed):
            sys.stdin = open("CON", "r", encoding="utf-8", errors="replace")
    except (OSError, IOError, AttributeError):
        # Continue if we can't fix stdin - may occur in non-console contexts
        # or when console access is restricted (e.g., running as a service)
        pass

# Fix encoding issues on Windows
# This must be called AFTER stdin is opened
from modules.common.utils import configure_windows_stdio

configure_windows_stdio()

# Check Rust backend availability for optimal ATC performance
from modules.adaptive_trend_LTS.utils.rust_build_checker import check_rust_backend

rust_status = check_rust_backend()
if not rust_status["available"]:
    print(f"\n{'=' * 60}")
    print("⚠️  PERFORMANCE WARNING")
    print(f"{'=' * 60}")
    print(rust_status["message"])
    print("\nTo build Rust backend:")
    print(f"  {rust_status['build_command']}")
    print(f"\n{'=' * 60}\n")
else:
    print(f"\n{'=' * 60}")
    print("✅ Rust backend is ACTIVE (Optimal performance)")
    print(f"   Tip: To rebuild, run: {rust_status.get('build_command', '.\\build_rust.bat').split('\\n')[0]}")
    print(f"{'=' * 60}\n")

if not rust_status["available"]:
    # Ask user if they want to auto-build Rust backend
    try:
        response = input("Auto-build Rust backend now? (y/n) [y]: ").lower()
        if not response or response in ["y", "yes"]:
            print("\n🔨 Building Rust backend... (this may take 1-2 minutes)")
            print("=" * 60)

            import subprocess
            from pathlib import Path

            rust_dir = Path(__file__).parent / "modules" / "adaptive_trend_LTS" / "rust_extensions"

            try:
                # Run maturin develop --release
                result = subprocess.run(
                    ["maturin", "develop", "--release"],
                    cwd=str(rust_dir),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=300,  # 5 minutes timeout
                )

                if result.returncode == 0:
                    print("✅ Rust backend built successfully!")
                    print("=" * 60)
                    print("\nRESTARTING scanner to enable Rust backend functionality...\n")

                    # Real restart of the current process to load the new dynamic library
                    import os

                    os.execl(sys.executable, sys.executable, *sys.argv)
                else:
                    print(f"❌ Build failed with return code {result.returncode}")
                    print("\nBuild output:")
                    print(result.stdout)
                    if result.stderr:
                        print("\nErrors:")
                        print(result.stderr)
                    print("\n" + "=" * 60)

                    cont = input("\nContinue without Rust backend? (y/n) [n]: ").lower()
                    if cont not in ["y", "yes"]:
                        print("Exiting. Please fix build errors and try again.")
                        sys.exit(1)

            except FileNotFoundError:
                print("❌ Error: 'maturin' not found in PATH")
                print("   Install with: pip install maturin")
                print("   Or use: .\\build_rust.bat (Windows)")
                print("=" * 60)

                cont = input("\nContinue without Rust backend? (y/n) [n]: ").lower()
                if cont not in ["y", "yes"]:
                    print("Exiting. Please install maturin and try again.")
                    sys.exit(1)

            except subprocess.TimeoutExpired:
                print("❌ Build timeout (exceeded 5 minutes)")
                print("=" * 60)

                cont = input("\nContinue without Rust backend? (y/n) [n]: ").lower()
                if cont not in ["y", "yes"]:
                    print("Exiting.")
                    sys.exit(1)

            except Exception as e:
                print(f"❌ Unexpected error during build: {e}")
                print("=" * 60)

                cont = input("\nContinue without Rust backend? (y/n) [n]: ").lower()
                if cont not in ["y", "yes"]:
                    print("Exiting.")
                    sys.exit(1)
        else:
            # User chose not to build
            cont = input("\nContinue without Rust backend? (y/n) [y]: ").lower()
            if cont and cont not in ["y", "yes", ""]:
                print("Exiting. Please build Rust backend and try again.")
                sys.exit(0)

    except (EOFError, KeyboardInterrupt):
        # If input fails (e.g., non-interactive), continue anyway
        print("\nNon-interactive mode detected, continuing without Rust backend...")

# Now import and call main
from modules.gemini_chart_analyzer.cli.batch_scanner.main import main

if __name__ == "__main__":
    log_file = _build_log_file_path()
    with _tee_output(log_file):
        main()
        print(f"\nLog saved to: {log_file}")
