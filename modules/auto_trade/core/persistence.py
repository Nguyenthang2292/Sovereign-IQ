"""
Signal Persistence Module

Handles saving trade signals for historical analysis and accuracy tracking.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from modules.auto_trade.core.signal_selector import FinalSignal
from modules.common.ui.logging import log_error, log_info


class SignalPersistence:
    """Manages storage of trading signals."""

    def __init__(self, storage_dir: str = "data/signals"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.filename = self.storage_dir / "signal_history.jsonl"

    def save_signal(self, signal: FinalSignal) -> bool:
        """
        Append a signal to the history file.

        Args:
            signal: The FinalSignal to save.

        Returns:
            True if successful, False otherwise.
        """
        try:
            record = {
                "timestamp": datetime.fromtimestamp(signal.timestamp).isoformat(),
                "symbol": signal.symbol,
                "type": signal.signal_type,
                "confidence": signal.confidence,
                "entry": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "sources": signal.sources,
            }

            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            log_info(f"Saved signal for {signal.symbol} to history.")
            return True

        except Exception as e:
            log_error(f"Failed to save signal history: {e}")
            return False
