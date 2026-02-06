#!/usr/bin/env python3
"""
Entry point at project root for Auto-Trade GUI.
Delegates to modules.auto_trade.run_gui.main().
Run from project root: python run_auto_trade_gui.py [--clear-cache] [--no-rust-build]
"""

import sys
from pathlib import Path

# Ensure project root is on path so "modules.auto_trade.run_gui" resolves
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from modules.auto_trade.run_gui import main

if __name__ == "__main__":
    main()
