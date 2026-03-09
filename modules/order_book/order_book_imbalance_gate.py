import time
from typing import Optional

from modules.common.ui.logging import log_info, log_warn

from .market_data_fetcher import fetch_agg_trades, fetch_depth
from .models import CombinedResult, OBIDecision
from .order_book_imbalance_calculator import calculate_combined_score


class OrderBookImbalanceGate:
    """
    Order Book Imbalance confirmation gate with delay-retry logic.

    Behavior:
    - Fail-open: any fetch/calculation failure returns PASS.
    - Neutral scores (|score| < threshold) are PASS.
    - Directional conflict retries up to max_retries, then SKIP.
    """

    def __init__(
        self,
        threshold: float = 0.15,
        retry_wait_seconds: int = 30,
        max_retries: int = 2,
        depth_limit: int = 100,
        delta_window_minutes: int = 5,
        testnet: bool = False,
        enabled: bool = True,
    ):
        self.threshold = float(threshold)
        self.retry_wait_seconds = int(max(retry_wait_seconds, 0))
        self.max_retries = int(max(max_retries, 0))
        self.depth_limit = int(max(depth_limit, 1))
        self.delta_window_minutes = int(max(delta_window_minutes, 1))
        self.testnet = bool(testnet)
        self.enabled = bool(enabled)

    def check(self, symbol: str, signal_type: str) -> tuple[OBIDecision, Optional[CombinedResult]]:
        """
        Validate order direction by OBI + cumulative delta with retry policy.

        Returns:
            (OBIDecision.PASS, result_or_none) when aligned/neutral/fail-open
            (OBIDecision.SKIP, last_result) when still conflicting after retries
        """
        if not self.enabled:
            return OBIDecision.PASS, None

        normalized_signal = (signal_type or "LONG").upper()
        if normalized_signal not in ("LONG", "SHORT"):
            log_warn("[OrderBookImbalanceGate] Unsupported signal_type=%s. Default to LONG.", normalized_signal)
            normalized_signal = "LONG"

        last_result: Optional[CombinedResult] = None
        total_attempts = self.max_retries + 1

        for attempt_idx in range(total_attempts):
            if attempt_idx > 0:
                time.sleep(self.retry_wait_seconds)

            snapshot = fetch_depth(symbol=symbol, limit=self.depth_limit, testnet=self.testnet)
            trades = fetch_agg_trades(
                symbol=symbol,
                window_minutes=self.delta_window_minutes,
                testnet=self.testnet,
            )

            if snapshot is None or trades is None:
                log_warn(
                    "[OrderBookImbalanceGate] Fail-open for %s (%s): market data unavailable.",
                    symbol,
                    normalized_signal,
                )
                return OBIDecision.PASS, None

            try:
                result = calculate_combined_score(snapshot=snapshot, trades=trades)
            except Exception as exc:
                log_warn(
                    "[OrderBookImbalanceGate] Fail-open for %s (%s): calculation error: %s",
                    symbol,
                    normalized_signal,
                    exc,
                )
                return OBIDecision.PASS, None

            last_result = result
            score = result.combined_score

            if _is_aligned_or_neutral(normalized_signal, score, self.threshold):
                log_info(
                    "[OrderBookImbalanceGate] %s %s PASSED (Combined=%.3f, attempt=%d/%d)",
                    symbol,
                    normalized_signal,
                    score,
                    attempt_idx + 1,
                    total_attempts,
                )
                log_info(
                    "[OrderBook] +- OBI   : %+.4f (raw=%+.4f)",
                    result.obi_score,
                    result.obi_raw,
                )
                log_info(
                    "[OrderBook] +- Delta : %+.4f (raw=%+.4f)",
                    result.delta_score,
                    result.delta_raw,
                )
                log_info(
                    "[OrderBook] -- Mid   : %.4f (weighted)",
                    result.weighted_mid,
                )
                return OBIDecision.PASS, result

            if attempt_idx < self.max_retries:
                log_warn(
                    "[OrderBookImbalanceGate] %s %s conflict (Score=%.3f). Retry %d/%d in %ss.",
                    symbol,
                    normalized_signal,
                    score,
                    attempt_idx + 1,
                    self.max_retries,
                    self.retry_wait_seconds,
                )

        final_score = last_result.combined_score if last_result is not None else 0.0
        log_warn(
            "[OrderBookImbalanceGate] %s %s SKIPPED after retry (Combined=%.3f)",
            symbol,
            normalized_signal,
            final_score,
        )
        if last_result is not None:
            log_warn(
                "[OrderBook] +- OBI   : %+.4f (raw=%+.4f)",
                last_result.obi_score,
                last_result.obi_raw,
            )
            log_warn(
                "[OrderBook] +- Delta : %+.4f (raw=%+.4f)",
                last_result.delta_score,
                last_result.delta_raw,
            )
            log_warn(
                "[OrderBook] -- Mid   : %.4f (weighted)",
                last_result.weighted_mid,
            )
        return OBIDecision.SKIP, last_result


def _is_aligned_or_neutral(signal_type: str, score: float, threshold: float) -> bool:
    if abs(score) < threshold:
        return True

    if signal_type == "LONG":
        return score > threshold

    return score < -threshold
