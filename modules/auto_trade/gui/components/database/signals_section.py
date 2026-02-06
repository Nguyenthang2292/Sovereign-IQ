"""Signals Section Component for Database Panel."""

import customtkinter as ctk
import uuid
import logging
from typing import Callable

from modules.auto_trade.database import (
    session_scope,
    save_signal,
    get_recent_signals,
    get_signal_performance_stats,
)

logger = logging.getLogger(__name__)


class SignalsSection:
    """Signals testing section component."""

    def __init__(self, parent: ctk.CTkFrame, log_callback: Callable, refresh_callback: Callable):
        self.parent = parent
        self.log_callback = log_callback
        self.refresh_callback = refresh_callback
        self._create_ui()

    def _create_ui(self):
        """Create the signals section UI."""
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(frame, text="🎯 Signals Testing", font=("Roboto", 14, "bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        input_frame = ctk.CTkFrame(frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(input_frame, text="Symbol:").pack(side="left", padx=(0, 5))
        self.signal_symbol = ctk.CTkEntry(input_frame, width=100)
        self.signal_symbol.pack(side="left", padx=(0, 10))
        self.signal_symbol.insert(0, "BTCUSDT")

        ctk.CTkLabel(input_frame, text="Confidence:").pack(side="left", padx=(0, 5))
        self.signal_confidence = ctk.CTkEntry(input_frame, width=100)
        self.signal_confidence.pack(side="left", padx=(0, 10))
        self.signal_confidence.insert(0, "0.85")

        ctk.CTkButton(input_frame, text="Create Test Signal", command=self._create_test_signal).pack(side="right")

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(btn_frame, text="📊 Get Recent Signals", command=self._get_recent_signals).pack(
            side="left", padx=(0, 5), fill="x", expand=True
        )
        ctk.CTkButton(btn_frame, text="📈 Signal Performance Stats", command=self._get_signal_stats).pack(
            side="left", padx=(5, 0), fill="x", expand=True
        )

    def _create_test_signal(self):
        """Create a test signal."""
        symbol = self.signal_symbol.get()
        try:
            confidence = float(self.signal_confidence.get())
            if not (0 <= confidence <= 1):
                raise ValueError("Confidence must be between 0 and 1")

            with session_scope() as session:
                signal_data = {
                    "correlation_id": f"SIG_{uuid.uuid4().hex[:8]}",
                    "symbol": symbol,
                    "signal_type": "LONG",
                    "confidence": confidence,
                    "executed": False,
                }

                save_signal(session, **signal_data)

                self.log_callback(f"Created test signal for {symbol}", "SUCCESS")
                self.refresh_callback()

        except ValueError as ve:
            self.log_callback(str(ve), "WARNING")
        except Exception as e:
            self.log_callback(f"Failed to create test signal: {e}", "ERROR")

    def _get_recent_signals(self):
        """Get recent signals."""
        try:
            with session_scope() as session:
                signals = get_recent_signals(session, limit=50)

                output = "Recent Signals:\n"
                output += "-" * 50 + "\n"
                for sig in signals:
                    output += f"{sig.created_at} | {sig.symbol} | {sig.signal_type} | Conf: {sig.confidence}\n"

                self._show_in_data_viewer(output)
                self.log_callback("Retrieved recent signals", "INFO")

        except Exception as e:
            self.log_callback(f"Failed to get recent signals: {e}", "ERROR")

    def _get_signal_stats(self):
        """Get signal performance statistics."""
        try:
            with session_scope() as session:
                stats = get_signal_performance_stats(session, days=30)

                output = "Signal Performance (30d):\n"
                output += "=" * 30 + "\n"
                for key, value in stats.items():
                    output += f"{key.replace('_', ' ').title()}: {value}\n"

                self._show_in_data_viewer(output)
                self.log_callback("Retrieved signal stats", "INFO")

        except Exception as e:
            self.log_callback(f"Failed to get signal stats: {e}", "ERROR")

    def _show_in_data_viewer(self, content: str):
        """Show content in data viewer."""
        if hasattr(self.parent, "data_viewer_callback"):
            self.parent.data_viewer_callback(content)
