
from pathlib import Path
import sys

"""
Main entry point for Market Batch Scanner.

Run batch market scanning with Google Gemini AI.
"""


# Add project root to sys.path
if "__file__" in globals():
    project_root = Path(__file__).parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

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
    main()
