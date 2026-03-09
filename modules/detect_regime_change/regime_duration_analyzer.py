"""
detect_regime_change/regime_duration_analyzer.py
================================================
Main engine: combines PELT + HMM to compute recommended regime duration per symbol.

This is the only entry point called by the auto_trade module.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from modules.common.utils import log_error, log_info, log_warn
from modules.detect_regime_change.hmm_regime_bridge import estimate_hmm_regime_duration
from modules.detect_regime_change.models import RegimeDurationResult
from modules.detect_regime_change.pelt_detector import (
    calculate_pelt_avg_duration,
    detect_change_points_pelt,
)


class RegimeDurationAnalyzer:
    """
    Analyze regime duration for one symbol.

    Combines two methods:
    1. PELT (offline) - detects change points in historical data
    2. HMM (real-time) - predicts duration of current/next state

    Combination formula:
        recommended = w_pelt * pelt_avg + w_hmm * hmm_duration

    Weights depend on HMM confidence:
    - High HMM probability (>0.7) -> w_hmm = 0.6, w_pelt = 0.4
    - Medium/low HMM probability -> w_hmm = 0.3, w_pelt = 0.7
    - HMM failure -> recommended = pelt_avg (100% PELT)
    """

    def __init__(
        self,
        lookback_days: int = 60,
        pelt_model: str = "rbf",
        pelt_penalty: Optional[float] = None,
        pelt_min_segment: int = 10,
        pelt_trim_pct: float = 0.1,
        hmm_train_ratio: float = 0.8,
        hmm_high_confidence_threshold: float = 0.7,
        w_pelt_high_conf: float = 0.4,
        w_hmm_high_conf: float = 0.6,
        w_pelt_low_conf: float = 0.7,
        w_hmm_low_conf: float = 0.3,
    ):
        self.lookback_days = lookback_days
        self.pelt_model = pelt_model
        self.pelt_penalty = pelt_penalty
        self.pelt_min_segment = pelt_min_segment
        self.pelt_trim_pct = pelt_trim_pct
        self.hmm_train_ratio = hmm_train_ratio
        self.hmm_high_confidence_threshold = hmm_high_confidence_threshold
        self.w_pelt_high_conf = w_pelt_high_conf
        self.w_hmm_high_conf = w_hmm_high_conf
        self.w_pelt_low_conf = w_pelt_low_conf
        self.w_hmm_low_conf = w_hmm_low_conf

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "15m",
    ) -> RegimeDurationResult:
        """
        Main entry point: analyze regime duration for one symbol.

        Args:
            df: OHLCV DataFrame with DatetimeIndex (already fetched, ideally >= 30 days)
            symbol: Symbol name (e.g., "BTC/USDT")
            timeframe: Data timeframe (e.g., "15m", "1h")

        Returns:
            RegimeDurationResult with recommended_duration_hours
        """
        start_time = time.time()

        result = RegimeDurationResult(
            symbol=symbol,
            timeframe=timeframe,
            data_points_analyzed=len(df),
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        try:
            # === 1. Prepare returns ===
            close_prices = df["close"].values.astype(float)
            log_returns = np.diff(np.log(close_prices + 1e-10))

            timestamps = None
            if isinstance(df.index, pd.DatetimeIndex):
                timestamps = df.index[1:].values  # Align with returns

            # === 2. PELT Change Point Detection ===
            try:
                change_points, segments = detect_change_points_pelt(
                    returns=log_returns,
                    timestamps=timestamps,
                    penalty=self.pelt_penalty,
                    model=self.pelt_model,
                    min_segment_length=self.pelt_min_segment,
                )
                result.pelt_change_points = change_points
                result.pelt_segments = segments

                avg_hrs, median_hrs = calculate_pelt_avg_duration(
                    segments, trim_pct=self.pelt_trim_pct,
                )
                result.pelt_avg_duration_hours = avg_hrs
                result.pelt_median_duration_hours = median_hrs

                if avg_hrs and median_hrs:
                    log_info(
                        f"PELT [{symbol}]: {len(change_points)} change points, "
                        f"avg={avg_hrs:.2f}h, median={median_hrs:.2f}h"
                    )
                else:
                    log_info(
                        f"PELT [{symbol}]: {len(change_points)} change points"
                    )
            except Exception as pelt_err:
                log_warn(f"PELT analysis failed for {symbol}: {pelt_err}")

            # === 3. HMM Regime Duration ===
            try:
                hmm_duration, hmm_state, hmm_prob = estimate_hmm_regime_duration(
                    df=df,
                    train_ratio=self.hmm_train_ratio,
                )
                result.hmm_next_state_duration_hours = hmm_duration
                result.hmm_state = hmm_state
                result.hmm_state_probability = hmm_prob

                if hmm_duration is not None:
                    log_info(
                        f"HMM [{symbol}]: state={hmm_state}, "
                        f"duration={hmm_duration:.2f}h, prob={hmm_prob:.3f}"
                    )
            except Exception as hmm_err:
                log_warn(f"HMM analysis failed for {symbol}: {hmm_err}")

            # === 4. Combine PELT + HMM ===
            result.recommended_duration_hours = self._combine_results(result)

            if result.recommended_duration_hours is not None:
                log_info(
                    f"Regime Duration [{symbol}]: "
                    f"recommended={result.recommended_duration_hours:.2f}h"
                )
            elif result.error is None:
                result.error = (
                    "No valid regime duration estimate from PELT or HMM"
                )

        except Exception as e:
            result.error = str(e)
            log_error(f"Regime analysis failed for {symbol}: {e}")

        result.computation_time_ms = (time.time() - start_time) * 1000
        return result

    def _combine_results(self, result: RegimeDurationResult) -> Optional[float]:
        """
        Combine PELT and HMM using weighted average.

        Strategy:
        - If both are available -> weighted average based on HMM confidence
        - If only PELT is available -> use PELT avg (or median)
        - If only HMM is available -> use HMM duration
        - If both fail -> None (auto_trade layer should fallback)
        """
        pelt_val = result.pelt_avg_duration_hours
        hmm_val = result.hmm_next_state_duration_hours
        hmm_prob = result.hmm_state_probability or 0.0

        # Both available
        if pelt_val is not None and hmm_val is not None and hmm_val > 0:
            if hmm_prob >= self.hmm_high_confidence_threshold:
                w_pelt = self.w_pelt_high_conf
                w_hmm = self.w_hmm_high_conf
            else:
                w_pelt = self.w_pelt_low_conf
                w_hmm = self.w_hmm_low_conf

            return w_pelt * pelt_val + w_hmm * hmm_val

        # PELT only
        if pelt_val is not None:
            return pelt_val

        # HMM only
        if hmm_val is not None and hmm_val > 0:
            return hmm_val

        return None
