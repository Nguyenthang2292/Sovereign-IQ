"""
HMM Signal Combiner Module

Combines multiple HMM strategies to generate trading signals using
a registry-based approach for scalability.
"""

from typing import Dict, Optional, Any
import pandas as pd

from modules.hmm.signals.strategy import HMMStrategyResult
from modules.hmm.signals.registry import HMMStrategyRegistry, get_default_registry
from modules.hmm.signals.voting import VotingMechanism
from modules.hmm.signals.utils import validate_dataframe
from modules.common.indicators import calculate_returns_volatility
from modules.hmm.signals.scoring import (
    calculate_strategy_scores,
    normalize_strategy_scores,
)
from modules.hmm.signals.confidence import calculate_kama_confidence, calculate_combined_confidence
from modules.hmm.signals.resolution import (
    calculate_dynamic_threshold,
    resolve_multi_strategy_conflicts,
    Signal,
    LONG,
    HOLD,
    SHORT,
)
from modules.common.utils import log_error, log_info

# Export Signal type for backward compatibility
__all__ = ['HMMSignalCombiner', 'combine_signals', 'Signal']

from config import (
    HMM_SIGNAL_MIN_THRESHOLD,
    HMM_HIGH_ORDER_MAX_SCORE,
    HMM_FEATURES,
    HMM_STRATEGIES,
    HMM_VOTING_MECHANISM,
    HMM_VOTING_THRESHOLD,
)


class HMMSignalCombiner:
    """
    Combines signals from multiple HMM strategies using registry-based approach.
    
    Supports any number of strategies, configurable voting mechanisms,
    and dynamic strategy management.
    """
    
    def __init__(self, registry: Optional[HMMStrategyRegistry] = None):
        """
        Initialize HMM Signal Combiner.
        
        Args:
            registry: Strategy registry (default: uses global default registry)
        """
        self.registry = registry if registry is not None else get_default_registry()
        
        # Load strategies from config if registry is empty
        if len(self.registry) == 0:
            self.registry.load_from_config(HMM_STRATEGIES)
    
    def combine(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Combine signals from all enabled strategies.
        
        Args:
            df: DataFrame containing OHLCV data
            **kwargs: Additional parameters (may override strategy params)
            
        Returns:
            Dictionary with:
            - signals: Dict[str, Signal] - Signal from each strategy
            - combined_signal: Signal - Final combined signal
            - confidence: float - Combined confidence
            - votes: Dict[str, int] - Vote counts (LONG/SHORT/HOLD)
            - metadata: Dict[str, Any] - Additional info
        """
        # Input validation
        if not validate_dataframe(df):
            return self._empty_result()
        
        # Get enabled strategies
        strategies = self.registry.get_enabled()
        if not strategies:
            log_error("No enabled HMM strategies found")
            return self._empty_result()
        
        # Run all strategies
        results: Dict[str, HMMStrategyResult] = {}
        errors: Dict[str, str] = {}
        
        for strategy in strategies:
            try:
                # Merge strategy params with kwargs (kwargs take precedence)
                strategy_params = {**strategy.params, **kwargs}
                result = strategy.analyze(df, **strategy_params)
                results[strategy.name] = result
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                log_error(f"Error in strategy '{strategy.name}': {error_msg}")
                errors[strategy.name] = error_msg
                # Create HOLD result for failed strategy
                results[strategy.name] = HMMStrategyResult(
                    signal=HOLD,
                    probability=0.0,
                    state=0,
                    metadata={"error": error_msg}
                )
        
        # Extract signals
        signals = {name: result.signal for name, result in results.items()}
        
        # Calculate scores
        score_long, score_short = calculate_strategy_scores(strategies, results)
        
        # Normalize scores
        normalized_long, normalized_short = normalize_strategy_scores(
            score_long, score_short, strategies, results
        )
        
        # Apply voting mechanism
        voting_mechanism = VotingMechanism()
        voting_method = getattr(voting_mechanism, HMM_VOTING_MECHANISM, voting_mechanism.confidence_weighted)
        
        if HMM_VOTING_MECHANISM == "threshold_based":
            combined_signal = voting_method(strategies, results, HMM_VOTING_THRESHOLD)
        else:
            combined_signal = voting_method(strategies, results)
        
        # Calculate confidence
        # Average probability of all strategies
        probabilities = [r.probability for r in results.values() if r.probability > 0]
        avg_probability = sum(probabilities) / len(probabilities) if probabilities else 0.5
        
        # Calculate KAMA-style confidence from scores
        kama_confidence = calculate_kama_confidence(score_long, score_short)
        
        # Check signal agreement
        signal_agreement = self._check_signal_agreement(signals, combined_signal)
        
        # Combined confidence
        combined_confidence = calculate_combined_confidence(
            avg_probability, kama_confidence, signal_agreement
        )
        
        # Calculate market volatility for dynamic threshold
        volatility = calculate_returns_volatility(df)
        
        # Calculate base threshold
        if HMM_FEATURES.get("normalization_enabled", True):
            # Calculate max possible score from all strategies
            max_possible = sum(s.weight * HMM_HIGH_ORDER_MAX_SCORE for s in strategies)
            base_threshold = (HMM_SIGNAL_MIN_THRESHOLD / max_possible * 100) if max_possible > 0 else HMM_SIGNAL_MIN_THRESHOLD
        else:
            base_threshold = HMM_SIGNAL_MIN_THRESHOLD
        
        # Apply dynamic threshold adjustment
        adjusted_threshold = calculate_dynamic_threshold(base_threshold, volatility)
        
        # Final signal decision with threshold
        final_score_long = normalized_long if HMM_FEATURES.get("normalization_enabled", True) else score_long
        final_score_short = normalized_short if HMM_FEATURES.get("normalization_enabled", True) else score_short
        
        final_signal = combined_signal
        if final_score_long >= adjusted_threshold and final_score_long > final_score_short:
            final_signal = LONG
        elif final_score_short >= adjusted_threshold and final_score_short > final_score_long:
            final_signal = SHORT
        else:
            final_signal = HOLD
        
        # Conflict resolution
        original_signals = signals.copy()
        if HMM_FEATURES.get("conflict_resolution_enabled", True):
            resolved_signals = resolve_multi_strategy_conflicts(strategies, results)
            signals.update(resolved_signals)
        
        # Count votes
        votes = {LONG: 0, SHORT: 0, HOLD: 0}
        for signal in signals.values():
            votes[signal] += 1
        
        # Log results
        self._log_results(signals, original_signals, results, final_signal, combined_confidence, 
                         normalized_long, normalized_short, score_long, score_short,
                         adjusted_threshold, base_threshold, volatility)
        
        # Build metadata
        metadata = {
            "strategies_count": len(strategies),
            "enabled_strategies": [s.name for s in strategies],
            "errors": errors,
            "voting_mechanism": HMM_VOTING_MECHANISM,
            "volatility": volatility,
            "threshold": adjusted_threshold,
            "base_threshold": base_threshold,
        }
        
        return {
            "signals": signals,
            "combined_signal": final_signal,
            "confidence": combined_confidence,
            "votes": votes,
            "metadata": metadata,
            "results": results,  # Include raw results for advanced usage
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "signals": {},
            "combined_signal": HOLD,
            "confidence": 0.0,
            "votes": {LONG: 0, SHORT: 0, HOLD: 0},
            "metadata": {},
            "results": {},
        }
    
    def _check_signal_agreement(self, signals: Dict[str, Signal], combined_signal: Signal) -> bool:
        """Check if signals agree with combined signal."""
        if not signals:
            return False
        
        # Count agreements
        agreements = sum(1 for s in signals.values() if s == combined_signal)
        total = len(signals)
        
        # Consider agreement if majority agrees or all are HOLD
        return agreements >= (total / 2) or all(s == HOLD for s in signals.values())
    
    def _log_results(
        self,
        signals: Dict[str, Signal],
        original_signals: Dict[str, Signal],
        results: Dict[str, HMMStrategyResult],
        final_signal: Signal,
        combined_confidence: float,
        normalized_long: float,
        normalized_short: float,
        score_long: float,
        score_short: float,
        adjusted_threshold: float,
        base_threshold: float,
        volatility: float,
    ) -> None:
        """Log combined signal results."""
        if all(s == HOLD for s in signals.values()) and final_signal == HOLD:
            return  # Skip logging if all HOLD
        
        signal_map: Dict[Signal, str] = {LONG: "LONG", HOLD: "HOLD", SHORT: "SHORT"}
        
        # Build signal info string
        signal_info_parts = []
        for name, signal in signals.items():
            result = results.get(name)
            if result:
                signal_info_parts.append(
                    f"{name}: {signal_map[signal]} "
                    f"(state: {result.state}, prob: {result.probability:.3f})"
                )
        
        signal_info = ", ".join(signal_info_parts)
        
        # Conflict info
        conflict_info = ""
        if HMM_FEATURES.get("conflict_resolution_enabled", True):
            conflicts = [
                name for name in signals.keys()
                if original_signals.get(name) != signals.get(name)
            ]
            if conflicts:
                conflict_info = f" [CONFLICT RESOLVED: {', '.join(conflicts)}]"
        
        # Confidence info
        confidence_info = ""
        if HMM_FEATURES.get("combined_confidence_enabled", True):
            confidence_info = f", Combined Confidence: {combined_confidence:.3f}"
        
        # Score display
        score_display = (
            f"(normalized L:{normalized_long:.1f}/S:{normalized_short:.1f})"
            if HMM_FEATURES.get("normalization_enabled", True)
            else f"(raw L:{score_long:.1f}/S:{score_short:.1f})"
        )
        
        # Threshold info
        threshold_info = ""
        if HMM_FEATURES.get("dynamic_threshold_enabled", True):
            if adjusted_threshold != base_threshold:
                threshold_info = f", Threshold: {adjusted_threshold:.2f} (adj from {base_threshold:.2f})"
            else:
                threshold_info = f", Threshold: {adjusted_threshold:.2f}"
        
        # Volatility info
        volatility_info = ""
        if HMM_FEATURES.get("dynamic_threshold_enabled", True):
            volatility_info = f", Volatility: {volatility:.4f}"
        
        log_info(
            f"HMM Signals{conflict_info} - {signal_info}, "
            f"Combined: {signal_map[final_signal]} {score_display}"
            f"{confidence_info}{threshold_info}{volatility_info}"
        )


def combine_signals(
    df: pd.DataFrame,
    registry: Optional[HMMStrategyRegistry] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to combine HMM signals.
    
    Args:
        df: DataFrame containing OHLCV data
        registry: Strategy registry (default: uses global default registry)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with combined signal results
    """
    combiner = HMMSignalCombiner(registry=registry)
    return combiner.combine(df, **kwargs)
