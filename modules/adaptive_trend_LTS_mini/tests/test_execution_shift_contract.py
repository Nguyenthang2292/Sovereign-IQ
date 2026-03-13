"""Regression tests for raw/execution signal contract (Task 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import (
    IncrementalATC,
    compute_atc_signals,
)


def _build_prices(n_bars: int = 220) -> pd.Series:
    rng = np.random.default_rng(20260314)
    trend = np.linspace(100.0, 132.0, n_bars)
    seasonal = np.sin(np.linspace(0.0, 10.0 * np.pi, n_bars)) * 2.8
    noise = rng.normal(0.0, 0.5, n_bars)
    prices = trend + seasonal + noise
    return pd.Series(prices, index=pd.RangeIndex(n_bars), dtype="float64")


def _base_compute_kwargs() -> dict[str, object]:
    return {
        "ema_len": 20,
        "hma_len": 20,
        "wma_len": 20,
        "dema_len": 20,
        "lsma_len": 20,
        "kama_len": 20,
        "lambda_param": 0.02,
        "decay": 0.03,
        "robustness": "Medium",
        "use_rust_backend": False,
        "use_cache": False,
    }


def test_batch_strategy_mode_preserves_raw_and_adds_single_shift_exec() -> None:
    prices = _build_prices()

    base_kwargs = _base_compute_kwargs()

    raw_result = compute_atc_signals(
        prices,
        **base_kwargs,
        strategy_mode=False,
    )
    strategy_result = compute_atc_signals(
        prices,
        **base_kwargs,
        strategy_mode=True,
    )

    assert "Average_Signal" in raw_result
    assert "Average_Signal" in strategy_result
    assert "Average_Signal_Exec" in strategy_result

    pd.testing.assert_series_equal(
        strategy_result["Average_Signal"],
        raw_result["Average_Signal"],
        check_names=False,
        rtol=1e-12,
        atol=1e-12,
    )

    expected_exec = raw_result["Average_Signal"].shift(1).fillna(0.0)
    pd.testing.assert_series_equal(
        strategy_result["Average_Signal_Exec"],
        expected_exec,
        check_names=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_incremental_strategy_mode_outputs_single_shift_without_raw_drift() -> None:
    prices = _build_prices(190)

    base_kwargs = _base_compute_kwargs()
    incremental_config = {
        **base_kwargs,
        "strategy_mode": True,
        "use_rust_incremental": False,
        "use_o1_mas": False,
    }

    batch_result = compute_atc_signals(
        prices,
        **base_kwargs,
        strategy_mode=True,
    )
    batch_raw = batch_result["Average_Signal"]
    batch_exec = batch_result["Average_Signal_Exec"]

    init_bars = 90
    atc = IncrementalATC(incremental_config)
    atc.initialize(prices.iloc[:init_bars])

    init_raw = float(atc.state["average_signal"])
    expected_init_raw = float(batch_raw.iloc[init_bars - 1])
    assert abs(init_raw - expected_init_raw) < 0.25

    prev_raw = init_raw
    for idx in range(init_bars, len(prices)):
        output_signal = float(atc.update(float(prices.iloc[idx])))
        state_raw = float(atc.state["average_signal"])
        state_raw_from_alias = float(atc.state["signal_raw"])

        expected_raw = float(batch_raw.iloc[idx])
        expected_exec = float(batch_exec.iloc[idx])

        # Raw state must stay aligned with batch raw (no internal strategy shift).
        assert abs(state_raw - expected_raw) < 0.25
        assert abs(state_raw_from_alias - state_raw) < 1e-12

        # Strategy output must be exactly single-shifted execution view.
        assert abs(output_signal - prev_raw) < 1e-12
        assert abs(output_signal - expected_exec) < 0.25

        prev_raw = state_raw
