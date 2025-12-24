"""
Signal Aggregator for combining signals from multiple timeframes.

Tổng hợp signals từ nhiều timeframes sử dụng weighted aggregation.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# Default timeframe weights (timeframe lớn hơn = weight cao hơn)
DEFAULT_TIMEFRAME_WEIGHTS = {
    '15m': 0.1,
    '30m': 0.15,
    '1h': 0.2,
    '4h': 0.25,
    '1d': 0.3,
    '1w': 0.35
}


class SignalAggregator:
    """Tổng hợp signals từ nhiều timeframes sử dụng weighted aggregation."""
    
    def __init__(self, timeframe_weights: Optional[Dict[str, float]] = None):
        """
        Khởi tạo SignalAggregator.
        
        Args:
            timeframe_weights: Dict mapping timeframe -> weight (nếu None, dùng DEFAULT_TIMEFRAME_WEIGHTS)
        """
        self.timeframe_weights = timeframe_weights or DEFAULT_TIMEFRAME_WEIGHTS.copy()
        # Normalize weights để tổng = 1.0
        total_weight = sum(self.timeframe_weights.values())
        if total_weight > 0:
            self.timeframe_weights = {
                tf: w / total_weight 
                for tf, w in self.timeframe_weights.items()
            }
    
    def aggregate_signals(
        self,
        timeframe_signals: Dict[str, Dict[str, any]]
    ) -> Dict[str, any]:
        """
        Tổng hợp signals từ nhiều timeframes.
        
        Args:
            timeframe_signals: Dict mapping timeframe -> signal dict
                Format: {
                    '15m': {'signal': 'LONG', 'confidence': 0.7},
                    '1h': {'signal': 'LONG', 'confidence': 0.8},
                    ...
                }
        
        Returns:
            Dict với aggregated signal:
            {
                'signal': 'LONG' | 'SHORT' | 'NONE',
                'confidence': float (0.0-1.0),
                'timeframe_breakdown': {...},
                'weights_used': {...}
            }
        """
        if not timeframe_signals:
            return {
                'signal': 'NONE',
                'confidence': 0.0,
                'timeframe_breakdown': {},
                'weights_used': {}
            }
        
        # Group signals by type
        long_signals = []
        short_signals = []
        none_signals = []
        
        timeframe_breakdown = {}
        weights_used = {}
        
        for timeframe, signal_data in timeframe_signals.items():
            signal = signal_data.get('signal', 'NONE').upper()
            confidence = float(signal_data.get('confidence', 0.0))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            
            # Get weight for this timeframe
            weight = self.timeframe_weights.get(timeframe, 0.1)
            weights_used[timeframe] = weight
            
            # Store breakdown
            timeframe_breakdown[timeframe] = {
                'signal': signal,
                'confidence': confidence,
                'weight': weight
            }
            
            # Group by signal type
            if signal == 'LONG':
                long_signals.append((timeframe, confidence, weight))
            elif signal == 'SHORT':
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
            'signal': final_signal,
            'confidence': final_confidence,
            'timeframe_breakdown': timeframe_breakdown,
            'weights_used': weights_used,
            'long_weighted_conf': long_weighted_conf,
            'short_weighted_conf': short_weighted_conf,
            'none_weighted_conf': none_weighted_conf
        }
    
    def _calculate_weighted_confidence(
        self,
        signals: List[Tuple[str, float, float]]
    ) -> float:
        """
        Tính weighted confidence cho một nhóm signals.
        
        Args:
            signals: List of (timeframe, confidence, weight) tuples
        
        Returns:
            Weighted confidence (0.0-1.0)
        """
        if not signals:
            return 0.0
        
        total_weighted_conf = 0.0
        total_weight = 0.0
        
        for timeframe, confidence, weight in signals:
            total_weighted_conf += confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_weighted_conf / total_weight
    
    def _determine_final_signal(
        self,
        long_weighted_conf: float,
        short_weighted_conf: float,
        none_weighted_conf: float
    ) -> Tuple[str, float]:
        """
        Xác định signal cuối cùng dựa trên weighted confidences.
        
        Args:
            long_weighted_conf: Weighted confidence cho LONG
            short_weighted_conf: Weighted confidence cho SHORT
            none_weighted_conf: Weighted confidence cho NONE
        
        Returns:
            Tuple of (signal, confidence)
        """
        # So sánh LONG và SHORT
        if long_weighted_conf > short_weighted_conf:
            if long_weighted_conf > 0.5:  # Threshold để xác định signal rõ ràng
                return ('LONG', long_weighted_conf)
            else:
                return ('NONE', max(long_weighted_conf, none_weighted_conf))
        elif short_weighted_conf > long_weighted_conf:
            if short_weighted_conf > 0.5:
                return ('SHORT', short_weighted_conf)
            else:
                return ('NONE', max(short_weighted_conf, none_weighted_conf))
        else:
            # Equal hoặc cả hai đều thấp
            return ('NONE', max(none_weighted_conf, long_weighted_conf, short_weighted_conf))

