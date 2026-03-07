"""Signals Section Component for Database Panel."""

import uuid
from typing import Callable

import customtkinter as ctk

from modules.auto_trade.database.repository.context import RepositoryContext
from modules.auto_trade.gui.config.database_panel_config import DatabasePanelConfig
from modules.auto_trade.gui.utils.svg_icons import get_icon


class SignalsSection:
    """Signals testing section component."""

    def __init__(
        self,
        parent: ctk.CTkFrame | ctk.CTkScrollableFrame,
        log_callback: Callable,
        refresh_callback: Callable,
    ):
        self.parent = parent
        self.log_callback = log_callback
        self.refresh_callback = refresh_callback
        self._create_ui()

    def _create_ui(self):
        """Create the signals section UI."""
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(
            frame,
            text="  Signals Testing",
            font=DatabasePanelConfig.TITLE_FONT,
            image=get_icon("target", size=(20, 20)),
            compound="left",
        ).pack(
            anchor="w",
            padx=DatabasePanelConfig.PADX_MEDIUM,
            pady=(DatabasePanelConfig.PADX_MEDIUM, DatabasePanelConfig.PADY_SMALL),
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

        ctk.CTkButton(
            btn_frame,
            text="  Get Recent Signals",
            command=self._get_recent_signals,
            image=get_icon("database", size=(16, 16)),
            compound="left",
        ).pack(side="left", padx=(0, 5), fill="x", expand=True)

        ctk.CTkButton(
            btn_frame,
            text="  Signal Performance Stats",
            command=self._get_signal_stats,
            image=get_icon("bar_chart_2", size=(16, 16)),
            compound="left",
        ).pack(side="left", padx=(5, 0), fill="x", expand=True)

    def _create_test_signal(self):
        """Create a test signal via RepositoryContext."""
        symbol = self.signal_symbol.get()
        try:
            confidence = float(self.signal_confidence.get())
            if not (0 <= confidence <= 1):
                raise ValueError("Confidence must be between 0 and 1")

            ctx = RepositoryContext.from_env()
            signal_data = {
                "correlation_id": f"SIG_{uuid.uuid4().hex[:8]}",
                "symbol": symbol,
                "signal_type": "LONG",
                "confidence": confidence,
                "executed": False,
            }
            ctx.signals.save_signal(signal_data)

            self.log_callback(f"Created test signal for {symbol}", "SUCCESS")
            self.refresh_callback()

        except ValueError as ve:
            self.log_callback(str(ve), "WARNING")
        except Exception as e:
            self.log_callback(f"Failed to create test signal: {e}", "ERROR")

    def _get_recent_signals(self):
        """Get recent signals via RepositoryContext."""
        try:
            ctx = RepositoryContext.from_env()
            signals = ctx.signals.get_recent_signals(limit=50)

            output = "Recent Signals:\n"
            output += "-" * 50 + "\n"
            for sig in signals:
                if isinstance(sig, dict):
                    output += (
                        f"{sig.get('created_at', '')} | {sig.get('symbol')} | "
                        f"{sig.get('signal_type')} | Conf: {sig.get('confidence')}\n"
                    )
                else:
                    output += f"{sig.created_at} | {sig.symbol} | {sig.signal_type} | Conf: {sig.confidence}\n"

            self._show_in_data_viewer(output)
            self.log_callback("Retrieved recent signals", "INFO")

        except Exception as e:
            self.log_callback(f"Failed to get recent signals: {e}", "ERROR")

    def _get_signal_stats(self):
        """Get signal performance statistics via RepositoryContext."""
        try:
            ctx = RepositoryContext.from_env()
            signals = ctx.signals.get_recent_signals(limit=99999)

            # Aggregate in Python (backend-agnostic)
            from datetime import datetime, timedelta, timezone

            cutoff = datetime.now(timezone.utc) - timedelta(days=30)

            total = len(signals)
            executed = 0
            recent = 0
            confidence_sum = 0.0

            for sig in signals:
                if isinstance(sig, dict):
                    conf = float(sig.get("confidence") or 0)
                    exc = sig.get("executed", False)
                    created_raw = sig.get("created_at", "")
                else:
                    conf = float(getattr(sig, "confidence", 0) or 0)
                    exc = getattr(sig, "executed", False)
                    created_raw = getattr(sig, "created_at", "")

                confidence_sum += conf
                if exc:
                    executed += 1
                try:
                    if isinstance(created_raw, str):
                        dt = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                    else:
                        dt = created_raw
                    if dt and dt >= cutoff:
                        recent += 1
                except Exception:
                    pass

            avg_conf = (confidence_sum / total) if total > 0 else 0.0
            stats = {
                "total_signals_all_time": total,
                "signals_last_30d": recent,
                "executed_signals": executed,
                "avg_confidence": f"{avg_conf:.4f}",
            }

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
        callback = getattr(self.parent, "data_viewer_callback", None)
        if callable(callback):
            callback(content)
