#!/usr/bin/env python3
"""
Entry point at project root for Auto-Trade GUI.
Delegates to modules.auto_trade.run_gui.main().
Run from project root: python run_auto_trade_gui.py [--clear-cache] [--no-rust-build]
"""

import multiprocessing as mp
import sys
from pathlib import Path

# Ensure project root is on path so "modules.auto_trade.run_gui" resolves
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Load .env BEFORE importing any modules that use AWS / Binance credentials.
# This ensures boto3, ccxt, and other clients receive keys from env vars
# instead of reporting "Unable to locate credentials" errors.
from dotenv import load_dotenv  # noqa: E402

# Priority: modules/auto_trade/.env (contains auto-trade specific credentials)
_auto_trade_env = _root / "modules" / "auto_trade" / ".env"
if _auto_trade_env.exists():
    load_dotenv(dotenv_path=_auto_trade_env, override=True)

# Fallback: project-root .env for any missing variables
load_dotenv(dotenv_path=_root / ".env", override=False)

from modules.auto_trade.run_gui import main  # noqa: E402

if __name__ == "__main__":
    mp.freeze_support()
    main()

