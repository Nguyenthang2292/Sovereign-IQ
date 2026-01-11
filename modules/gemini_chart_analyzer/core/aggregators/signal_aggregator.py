
from typing import Any, Dict, List, Optional, Tuple
import math

from config.gemini_chart_analyzer import TIMEFRAME_WEIGHTS
from config.gemini_chart_analyzer import TIMEFRAME_WEIGHTS

"""
Signal Aggregator for combining signals from multiple timeframes.

Aggregate signals from multiple timeframes using weighted aggregation.
"""


# Import timeframe weights from config
from config.gemini_chart_analyzer import TIMEFRAME_WEIGHTS


class SignalAggregator:
    """Aggregate signals from multiple timeframes using weighted aggregation."""

    # Default weight for unknown timeframes when not found in timeframe_weights
    DEFAULT_UNKNOWN_TIMEFRAME_WEIGHT = 0.1
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        timeframe_weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize SignalAggregator.

        Args:
            timeframe_weights: Dict mapping timeframe -> weight (if None, use TIMEFRAME_WEIGHTS)
            confidence_threshold: Threshold for determining clear signals (default: 0.5)
        """
        self.timeframe_weights = timeframe_weights or TIMEFRAME_WEIGHTS.copy()
        self.confidence_threshold = confidence_threshold

        # Normalize weights so total = 1.0
        total_weight = sum(self.timeframe_weights.values())
        if total_weight > 0:
            self.timeframe_weights = {tf: w / total_weight for tf, w in self.timeframe_weights.items()}

    def aggregate_signals(self, timeframe_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate signals from multiple timeframes.

        Args:
            timeframe_signals: Dict mapping timeframe -> signal dict
                Format: {
                    '15m': {'signal': 'LONG', 'confidence': 0.7},
                    '1h': {'signal': 'LONG', 'confidence': 0.8},
                    ...
                }

        Returns:
            Dict with aggregated signal:
            {
                'signal': 'LONG' | 'SHORT' | 'NONE',
                'confidence': float (0.0-1.0),
                'timeframe_breakdown': {...},
                'weights_used': {...}
            }
        """
        if not timeframe_signals:
            return {"signal": "NONE", "confidence": 0.0, "timeframe_breakdown": {}, "weights_used": {}}

        # Group signals by type
        long_signals = []
        short_signals = []
        none_signals = []

        timeframe_breakdown = {}
        weights_used = {}

        for timeframe, signal_data in timeframe_signals.items():
            signal = signal_data.get("signal", "NONE").upper()
            confidence = float(signal_data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

            # Get weight for this timeframe
            weight = self.timeframe_weights.get(timeframe, self.DEFAULT_UNKNOWN_TIMEFRAME_WEIGHT)
            weights_used[timeframe] = weight

            # Store breakdown
            timeframe_breakdown[timeframe] = {"signal": signal, "confidence": confidence, "weight": weight}

            # Group by signal type
            if signal == "LONG":
                long_signals.append((timeframe, confidence, weight))
            elif signal == "SHORT":
                short_signals.append((timeframe, confidence, weight))
            else:  # NONE
                none_signals.append((timeframe, confidence, weight))

        # Calculate weighted confidence for each signal type
        long_weighted_conf = self._calculate_weighted_confidence(long_signals)
        short_weighted_conf = self._calculate_weighted_confidence(short_signals)
        none_weighted_conf = self._calculate_weighted_confidence(none_signals)

        # Determine final signal
        final_signal, final_confidence = self._determine_final_signal(
            long_weighted_conf, short_weighted_conf, none_weighted_conf
        )

        return {
            "signal": final_signal,
            "confidence": final_confidence,
            "timeframe_breakdown": timeframe_breakdown,
            "weights_used": weights_used,
            "long_weighted_conf": long_weighted_conf,
            "short_weighted_conf": short_weighted_conf,
            "none_weighted_conf": none_weighted_conf,
        }

    def _calculate_weighted_confidence(self, signals: List[Tuple[str, float, float]]) -> float:
        """
        Calculate the weighted confidence for a group of signals.

        Args:
            signals: List of (timeframe, confidence, weight) tuples

        Returns:
            Weighted confidence (0.0-1.0), clamped and validated
        """
        if not signals:
            return 0.0

        total_weighted_conf = 0.0
        total_weight = 0.0

        for _, confidence, weight in signals:
            # Validate inputs are finite numbers
            if not (isinstance(confidence, (int, float)) and isinstance(weight, (int, float))):
                continue

            # Skip NaN or Infinity values
            if math.isnan(confidence) or math.isinf(confidence):
                continue
            if math.isnan(weight) or math.isinf(weight):
                continue

            total_weighted_conf += confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        try:
            result = total_weighted_conf / total_weight

            # Validate result is finite
            if math.isnan(result) or math.isinf(result):
                return 0.0

            # Clamp to [0.0, 1.0]
            return max(0.0, min(1.0, result))
        except (ZeroDivisionError, TypeError, ValueError):
            return 0.0

    def _determine_final_signal(
        self, long_weighted_conf: float, short_weighted_conf: float, none_weighted_conf: float
    ) -> Tuple[str, float]:
        """
        Determine the final signal based on weighted confidences.

        Args:
            long_weighted_conf: Weighted confidence for LONG
            short_weighted_conf: Weighted confidence for SHORT
            none_weighted_conf: Weighted confidence for NONE

        Returns:
            Tuple of (signal, confidence)
        """
        # Compare LONG and SHORT
        if long_weighted_conf > short_weighted_conf:
            if long_weighted_conf > self.confidence_threshold:  # Threshold to identify a clear signal
                return ("LONG", long_weighted_conf)
            else:
                return ("NONE", max(long_weighted_conf, none_weighted_conf))
        elif short_weighted_conf > long_weighted_conf:
            if short_weighted_conf > self.confidence_threshold:
                return ("SHORT", short_weighted_conf)
            else:
                return ("NONE", max(short_weighted_conf, none_weighted_conf))
        else:
            # Equal: long_weighted_conf == short_weighted_conf
            # When confidences are equal, return NONE to avoid conflict
            return ("NONE", max(none_weighted_conf, long_weighted_conf, short_weighted_conf))
