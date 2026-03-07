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

import customtkinter as ctk  # noqa: E402

# ------------- Monkey Patch CTkButton Hover Text Color -------------
original_on_enter = ctk.CTkButton._on_enter
original_on_leave = ctk.CTkButton._on_leave


def patched_on_enter(self, *args, **kwargs):
    if getattr(self, "_state", "normal") == "normal":
        if not hasattr(self, "_original_text_color"):
            self._original_text_color = self.cget("text_color")
        self.configure(text_color="#1a1a1a")

        current_img = self.cget("image")
        if current_img is not None and getattr(current_img, "_hover_variant", None) is not None:
            if not hasattr(self, "_original_image"):
                self._original_image = current_img
            self.configure(image=current_img._hover_variant)

    original_on_enter(self, *args, **kwargs)


def patched_on_leave(self, *args, **kwargs):
    if hasattr(self, "_original_text_color"):
        self.configure(text_color=self._original_text_color)

    if hasattr(self, "_original_image"):
        self.configure(image=self._original_image)

    original_on_leave(self, *args, **kwargs)


ctk.CTkButton._on_enter = patched_on_enter
ctk.CTkButton._on_leave = patched_on_leave
# -------------------------------------------------------------------

if __name__ == "__main__":
    mp.freeze_support()
    from modules.auto_trade.run_gui import main  # noqa: E402

    main()
