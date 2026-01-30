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
- If Rust backend is missing: auto-builds with `maturin develop --release` (no cargo clean),
  then restarts so the new extension is loaded. Set GEMINI_SCANNER_SKIP_RUST_BUILD=1 to skip.
- Imports and invokes the `main` function from the batch scanner CLI module.

Usage:
    python main_gemini_chart_batch_scanner.py [args]

    # Skip automatic Rust build (e.g. in CI):
    set GEMINI_SCANNER_SKIP_RUST_BUILD=1
    python main_gemini_chart_batch_scanner.py [args]

This centralizes environment bootstrapping and dispatches the batch chart analysis job.
"""

import io
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# Set to "1" to skip automatic Rust build when backend is missing (e.g. in CI)
SKIP_AUTO_RUST_BUILD = os.environ.get("GEMINI_SCANNER_SKIP_RUST_BUILD", "").strip().lower() in ("1", "true", "yes")

# Add project root to sys.path
if "__file__" in globals():
    project_root = Path(__file__).parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

LOG_DIR = Path(__file__).parent / "logs"
MODEL_CLEANUP_DIR = Path(__file__).parent / "artifacts" / "models" / "random_forest"


def _cleanup_old_models(model_dir: Path, keep_count: int = 5) -> None:
    """Keep only the N most recent model files in the models directory."""
    try:
        if not model_dir.exists():
            return

        # Get all model files (joblib)
        model_files = list(model_dir.glob("*.joblib"))

        # Sort by modification time (newest last)
        model_files.sort(key=lambda x: x.stat().st_mtime)

        # If we have more than keep_count, delete the oldest ones
        if len(model_files) > keep_count:
            to_delete = model_files[: len(model_files) - keep_count]
            print(f"🧹 Cleaning up {len(to_delete)} old model(s) in {model_dir.name}...")
            for f in to_delete:
                try:
                    f.unlink()
                except Exception:
                    pass
    except Exception:
        # Silently fail for cleanup operations
        pass


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

# Check Rust backends availability
# For XGBoost, we'll check if the module can be imported

from modules.adaptive_trend_LTS.utils.rust_build_checker import check_rust_backend as check_atc_rust


def check_xgboost_rust():
    try:
        import importlib.util

        if importlib.util.find_spec("xgboost_rust") is not None:
            return {"available": True}
    except ImportError:
        pass
    return {"available": False}


def load_config_for_backend_check():
    """Load config to check which performance modules are enabled."""
    import yaml

    config_path = Path(__file__).parent / "standard_batch_scan_config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config
        except Exception:
            pass
    return {}


atc_status = check_atc_rust()
xgb_status = check_xgboost_rust()

# Load config to check if performance modules are enabled
config = load_config_for_backend_check()
use_atc_performance = config.get("use_atc_performance", True)
use_xgboost_performance = config.get("use_xgboost_performance", True)

backends_missing = []
backends_disabled = []

# Only warn about ATC if it's enabled in config
if use_atc_performance and not atc_status["available"]:
    backends_missing.append("Adaptive Trend LTS")
elif not use_atc_performance:
    backends_disabled.append("Adaptive Trend LTS (using legacy module)")

# Only warn about XGBoost if it's enabled in config
if use_xgboost_performance and not xgb_status["available"]:
    backends_missing.append("XGBoost LTS")
elif not use_xgboost_performance:
    backends_disabled.append("XGBoost LTS (using legacy module)")

if backends_missing:
    print(f"\n{'=' * 60}")
    print("⚠️  PERFORMANCE WARNING")
    print(f"{'=' * 60}")
    print(f"The following Rust backends are MISSING: {', '.join(backends_missing)}")
    print("\nTo build all backends:")
    print("  .\\build_rust.ps1")
    print(f"{'=' * 60}\n")
elif backends_disabled:
    print(f"\n{'=' * 60}")
    print("ℹ️  Performance modules disabled in config")
    print(f"{'=' * 60}")
    for msg in backends_disabled:
        print(f"  • {msg}")
    print(f"{'=' * 60}\n")
else:
    print(f"\n{'=' * 60}")
    print("✅ All Rust backends are ACTIVE (Optimal performance)")
    print(f"{'=' * 60}\n")

if backends_missing and not SKIP_AUTO_RUST_BUILD:
    import subprocess

    _do_auto_build = True

    def _run_rust_build(module_rel_path: str) -> bool:
        """Run maturin develop --release; return True on success."""
        rust_dir = Path(__file__).parent / module_rel_path
        print(f"🔨 Building {module_rel_path}...")
        try:
            result = subprocess.run(
                ["maturin", "develop", "--release"],
                cwd=str(rust_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
            if result.returncode == 0:
                return True
            print(f"❌ Build failed for {module_rel_path} (exit {result.returncode})")
            return False
        except Exception as e:
            print(f"❌ Build error: {e}")
            return False

    def _ask_continue() -> bool:
        try:
            r = input("\nContinue without missing Rust backends? (y/n) [y]: ").lower()
            return not r or r in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return True

    try:
        any_built = False
        # Only build if enabled in config
        if use_atc_performance and not atc_status["available"]:
            if _run_rust_build("modules/adaptive_trend_LTS/rust_extensions"):
                any_built = True

        if use_xgboost_performance and not xgb_status["available"]:
            if _run_rust_build("modules/xgboost_LTS/rust_extensions"):
                any_built = True

        if any_built:
            print("✅ Rust backends built. Restarting to load them...\n")
            os.execl(sys.executable, sys.executable, *sys.argv)

        if not _ask_continue():
            sys.exit(0)

    except (EOFError, KeyboardInterrupt):
        print("\nNon-interactive: continuing without some Rust backends...")

# Now import and call main
from modules.gemini_chart_analyzer.cli.batch_scanner.main import main

if __name__ == "__main__":
    # Cleanup Random Forest models on startup
    _cleanup_old_models(MODEL_CLEANUP_DIR, keep_count=5)

    log_file = _build_log_file_path()
    with _tee_output(log_file):
        main()
        print(f"\nLog saved to: {log_file}")
