"""Entry point for both module and direct-script execution.

Supports:
    python -m modules.adaptive_trend_LTS_serverless.visualizer
    python modules/adaptive_trend_LTS_serverless/visualizer/__main__.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .cli import main
except ImportError:
    # Direct script execution has no package context for relative imports.
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from modules.adaptive_trend_LTS_serverless.visualizer.cli import main


if __name__ == "__main__":
    main()
