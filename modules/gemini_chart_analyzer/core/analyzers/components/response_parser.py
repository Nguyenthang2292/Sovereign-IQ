"""Helpers for parsing Gemini responses into structured signals."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class TradingSignal:
    direction: str = ""  # LONG, SHORT, or NEUTRAL
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    confidence: Optional[str] = None
    reasoning: Optional[str] = None


def parse_trading_signal(response_text: str) -> TradingSignal:
    """Convert Gemini response text into a TradingSignal data object."""
    signal = TradingSignal()
    text_lower = response_text.lower()

    # Check Vietnamese patterns separately to avoid encoding issues
    vietnamese_patterns = {
        "tang": r"t\u0103ng",  # Tang
        "giam": r"gi\u1ea3m",  # Giam
        "mua": r"mua",  # Mua
        "ban": r"b\u00e1n",  # Ban
    }

    for vi_key, pattern in vietnamese_patterns.items():
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            if vi_key in ["tang", "mua"]:
                signal.direction = "LONG"
                break
            elif vi_key in ["giam", "ban"]:
                signal.direction = "SHORT"
                break

    if signal.direction == "":  # Still NEUTRAL, check English
        if "long" in text_lower:
            signal.direction = "LONG"
        elif "short" in text_lower:
            signal.direction = "SHORT"
        else:
            signal.direction = "NEUTRAL"

    price_patterns = {
        "entry_price": [
            r"entry:\s*([0-9]+\.?[0-9]*)",
            r"(?:into|at):\s*([0-9]+\.?[0-9]*)",
        ],
        "stop_loss": [
            r"(?:stop\s*loss|sl):\s*([0-9]+\.?[0-9]*)",
            r"sl:\s*([0-9]+\.?[0-9]*)",
        ],
        "take_profit_1": [
            r"(?:take\s*profit|tp)\s*1?:\s*([0-9]+\.?[0-9]*)",
            r"tp1?\s*([0-9]+\.?[0-9]*)",
        ],
        "take_profit_2": [
            r"(?:take\s*profit|tp)\s*2?:\s*([0-9]+\.?[0-9]*)",
            r"tp2?\s*([0-9]+\.?[0-9]*)",
        ],
    }

    for field, patterns in price_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    setattr(signal, field, float(match.group(1)))
                    break
                except (ValueError, IndexError):
                    pass

    confidence_patterns = [
        r"(?:confidence):\s*(high|medium|low|cao|trung bình|thấp)",
        r"(?:probability):\s*([0-9]+%)",
    ]

    for pattern in confidence_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            signal.confidence = match.group(1)
            break

    signal.reasoning = response_text
    return signal
