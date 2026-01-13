"""
Signal Combiner Mixin for Hybrid Signal Calculator.

This module provides signal combination functionality using majority vote
and confidence weighting approaches.
"""

from typing import Dict, List, Tuple


class SignalCombinerMixin:
    """
    Mixin class providing signal combination functionality.

    Combines signals from multiple indicators using majority vote with
    optional confidence weighting.

    Configurable Attributes (optional, with defaults):
        use_confidence_weighting (bool): Whether to weight votes by confidence scores.
            If True, uses confidence values to weight votes; if False, uses simple
            vote counting. Consuming classes may override the default value of True.
        min_indicators_agreement (int): Minimum number of indicators that must agree
            for a signal to be valid. Must be >= 1. If fewer indicators agree than
            this threshold, returns neutral signal (0, 0.0). Consuming classes may
            override the default value of 3.
        default_confidence (float): Fallback confidence value used when indicators lack
            explicit confidence scores. DEFAULT_CONFIDENCE (0.5) is the class-level
            default value, representing neutral confidence - neither high nor low conviction.
            Consumers may override this per-instance by setting a lowercase default_confidence
            attribute to tune the fallback confidence used when indicators lack explicit scores.

    Note:
        These attributes can be set as class attributes (for defaults) or instance
        attributes (for per-instance configuration). If not set, defaults will be used.

    Tie-breaking: When long and short votes are equal, the combiner returns
    a neutral signal (0) with confidence computed from neutral votes. This
    conservative approach indicates insufficient consensus for directional
    action. See combine_signals_majority_vote() docstring for details.
    """

    # Default class attributes (can be overridden by consuming classes)
    use_confidence_weighting: bool = True
    min_indicators_agreement: int = 3

    # Default confidence value used when indicator signals lack a confidence field.
    # The midpoint (0.5) represents neutral confidence - neither high nor low conviction.
    # This value can be adjusted per-instance via the default_confidence attribute
    # to tune behavior when indicators don't provide explicit confidence scores.
    DEFAULT_CONFIDENCE: float = 0.5

    def combine_signals_majority_vote(
        self,
        indicator_signals: List[Dict],
        expected_signal_type: str = "LONG",
    ) -> Tuple[int, float]:
        """
        Combine signals from multiple indicators using majority vote.

        Args:
            indicator_signals: List of dicts with 'indicator', 'signal', 'confidence'
            expected_signal_type: "LONG" or "SHORT" - the expected signal direction
                (case-insensitive). Raises ValueError for invalid values.

        Returns:
            Tuple of (combined_signal, combined_confidence)

        Raises:
            ValueError: If expected_signal_type is not "LONG" or "SHORT" (case-insensitive)
            AttributeError: If required attributes (use_confidence_weighting or
                min_indicators_agreement) are not set or have invalid types/values.

        Note:
            Tie-breaking behavior: When long_votes equals short_votes (or when neither
            long nor short has a clear majority over neutral), the function falls through
            to the else branch and returns a neutral signal (0). The confidence in this
            case is computed from neutral_votes, which may be 0 if no indicators voted
            neutral, resulting in a confidence of 0.0.

            Consumers should interpret a neutral result from an equal long/short tie as
            indicating insufficient consensus for directional action. Consider:
            - Using the neutral signal to reduce position size or exit trades
            - Implementing additional tie-breaking logic (e.g., confidence-weighted
              comparison, recent signal precedence, or market context) if different
              behavior is desired
            - Treating (0, 0.0) as a "no-op" signal that preserves current position
        """
        # Validate required attributes
        if not isinstance(self.use_confidence_weighting, bool):
            raise AttributeError(
                f"use_confidence_weighting must be a boolean, got {type(self.use_confidence_weighting).__name__}"
            )

        if not isinstance(self.min_indicators_agreement, int):
            raise AttributeError(
                f"min_indicators_agreement must be an integer, got {type(self.min_indicators_agreement).__name__}"
            )
        elif self.min_indicators_agreement < 1:
            raise AttributeError(f"min_indicators_agreement must be >= 1, got {self.min_indicators_agreement}")

        if not indicator_signals:
            return (0, 0.0)

        # Validate and convert expected signal type to int
        expected_signal_type_upper = str(expected_signal_type).upper()
        if expected_signal_type_upper == "LONG":
            expected_signal = 1
        elif expected_signal_type_upper == "SHORT":
            expected_signal = -1
        else:
            raise ValueError(
                f"Invalid expected_signal_type: '{expected_signal_type}'. Must be 'LONG' or 'SHORT' (case-insensitive)."
            )

        # Count votes for each signal direction
        long_votes = 0
        short_votes = 0
        neutral_votes = 0

        long_confidence_sum = 0.0
        short_confidence_sum = 0.0
        neutral_confidence_sum = 0.0

        # Get default confidence: use instance attribute if set, otherwise use class constant
        default_confidence = getattr(self, "default_confidence", self.DEFAULT_CONFIDENCE)

        for indicator in indicator_signals:
            signal = indicator.get("signal", 0)  # Default to NEUTRAL if missing
            confidence = indicator.get("confidence", default_confidence)
            if signal == 1:  # LONG
                long_votes += 1
                if self.use_confidence_weighting:
                    long_confidence_sum += confidence
            elif signal == -1:  # SHORT
                short_votes += 1
                if self.use_confidence_weighting:
                    short_confidence_sum += confidence
            else:  # NEUTRAL/HOLD (0)
                neutral_votes += 1
                if self.use_confidence_weighting:
                    neutral_confidence_sum += confidence

        # Determine majority signal
        total_votes = len(indicator_signals)

        # Majority vote - determine winning direction first
        if long_votes > short_votes and long_votes > neutral_votes:
            # LONG wins - check if we have minimum directional agreement
            if long_votes < self.min_indicators_agreement:
                # Not enough indicators agree on LONG direction, return neutral
                return (0, 0.0)
            combined_signal = 1
            combined_confidence = (
                long_confidence_sum / max(long_votes, 1) if self.use_confidence_weighting else long_votes / total_votes
            )
        elif short_votes > long_votes and short_votes > neutral_votes:
            # SHORT wins - check if we have minimum directional agreement
            if short_votes < self.min_indicators_agreement:
                # Not enough indicators agree on SHORT direction, return neutral
                return (0, 0.0)
            combined_signal = -1
            combined_confidence = (
                short_confidence_sum / max(short_votes, 1)
                if self.use_confidence_weighting
                else short_votes / total_votes
            )
        else:
            # Tie-breaking behavior: This branch handles several cases:
            # 1. When neutral_votes is the maximum (neutral wins)
            # 2. When long_votes == short_votes (equal votes tie - falls through here)
            # 3. When there's a three-way tie or no clear majority
            #
            # For equal long/short ties specifically: The function returns neutral (0)
            # as a conservative default, indicating insufficient consensus for directional
            # action. The confidence is computed from neutral_votes, which may be 0 if
            # no indicators voted neutral, resulting in (0, 0.0). This signals that
            # there is no clear directional bias and no neutral consensus either.
            combined_signal = 0
            combined_confidence = (
                neutral_confidence_sum / max(neutral_votes, 1)
                if self.use_confidence_weighting
                else neutral_votes / total_votes
            )

        # Filter by expected signal type: only return signal if it matches expected direction
        if expected_signal == 1 and combined_signal != 1:
            return (0, 0.0)  # Expected LONG but got something else
        elif expected_signal == -1 and combined_signal != -1:
            return (0, 0.0)  # Expected SHORT but got something else

        return (combined_signal, min(combined_confidence, 1.0))

    def _select_highest_confidence_signal(
        self,
        non_zero_signals: List[Dict],
    ) -> Tuple[int, float]:
        """
        Select the signal with highest confidence from filtered non-zero signals.

        Filters out neutral signals (signal == 0) and selects the signal with
        the highest confidence. If multiple signals have the same confidence,
        prefers LONG (signal == 1) over SHORT (signal == -1).

        This method is used by HybridSignalCalculator in:
        - calculate_single_signal_highest_confidence()
        - _calculate_signal_from_precomputed_highest_confidence()

        Args:
            non_zero_signals: List of signal dicts with 'signal' and 'confidence' keys.
                Should already be filtered to exclude neutral signals (signal != 0).

        Returns:
            Tuple of (signal, confidence) where:
            - signal: 1 (LONG), -1 (SHORT), or 0 if no valid signal found
            - confidence: Confidence score of selected signal (0.0 to 1.0)

            Returns (0, 0.0) if:
            - non_zero_signals is empty
            - No valid signal is found
        """
        if not non_zero_signals:
            return (0, 0.0)

        # Select signal with highest confidence
        # If tie, prefer LONG (1) over SHORT (-1)
        best_signal = None
        best_confidence = -1.0

        # Get default confidence: use instance attribute if set, otherwise use class constant
        # This matches the behavior in combine_signals_majority_vote for consistency
        default_confidence = getattr(self, "default_confidence", self.DEFAULT_CONFIDENCE)

        for sig in non_zero_signals:
            signal_val = sig.get("signal", 0)
            confidence_val = sig.get("confidence", default_confidence)

            if confidence_val > best_confidence:
                best_signal = signal_val
                best_confidence = confidence_val
            elif confidence_val == best_confidence and signal_val == 1 and best_signal == -1:
                # Tie in confidence, prefer LONG over SHORT
                best_signal = signal_val
                best_confidence = confidence_val

        if best_signal is None:
            return (0, 0.0)

        return (best_signal, best_confidence)
