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


def _build_log_file_path() -> Path:
    """Return a timestamped log file path inside the local logs directory."""
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

# Now import and call main
from modules.gemini_chart_analyzer.cli.batch_scanner_main import main

if __name__ == "__main__":
    log_file = _build_log_file_path()
    with _tee_output(log_file):
        main()
        print(f"\nLog saved to: {log_file}")
