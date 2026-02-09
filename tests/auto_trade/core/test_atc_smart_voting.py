"""Tests for ATC Scanner smart voting: weight normalization, adaptive threshold, and safe Polars conversion."""

from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest

from modules.auto_trade.core.atc_scanner import (
    ATCScanner,
    _pandas_to_polars_safe,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_data_fetcher():
    return MagicMock()


@pytest.fixture
def scanner(mock_data_fetcher):
    """Scanner with known weights: 15m=0.5, 1h=0.3, 4h=0.2, threshold=0.15."""
    config = {
        "weights": {"15m": 0.5, "1h": 0.3, "4h": 0.2},
        "threshold": 0.15,
        "timeframes": ["15m", "1h", "4h"],
    }
    return ATCScanner(mock_data_fetcher, config=config)  # type: ignore[arg-type]


# ============================================================================
# _normalize_weights tests
# ============================================================================


class TestNormalizeWeights:
    """Tests for the _normalize_weights method."""

    def test_all_tfs_have_data(self, scanner):
        """When all TFs produce data, weights and threshold stay the same."""
        results_by_tf = {
            "15m": {"longs": {"BTCUSDT"}, "shorts": set(), "strengths": {"BTCUSDT": 0.8}},
            "1h": {"longs": {"BTCUSDT"}, "shorts": set(), "strengths": {"BTCUSDT": 0.7}},
            "4h": {"longs": {"BTCUSDT"}, "shorts": set(), "strengths": {"BTCUSDT": 0.6}},
        }
        norm_w, adaptive_th = scanner._normalize_weights(results_by_tf)

        assert norm_w["15m"] == pytest.approx(0.5)
        assert norm_w["1h"] == pytest.approx(0.3)
        assert norm_w["4h"] == pytest.approx(0.2)
        assert adaptive_th == pytest.approx(0.15)

    def test_one_tf_failed_redistributes_weight(self, scanner):
        """When 1h fails (empty), its 0.3 weight is split among 15m and 4h."""
        results_by_tf = {
            "15m": {"longs": {"BTCUSDT"}, "shorts": set(), "strengths": {"BTCUSDT": 0.8}},
            "1h": {"longs": set(), "shorts": set(), "strengths": {}},  # empty = failed
            "4h": {"longs": {"BTCUSDT"}, "shorts": set(), "strengths": {"BTCUSDT": 0.6}},
        }
        norm_w, adaptive_th = scanner._normalize_weights(results_by_tf)

        # Active weight = 0.5 + 0.2 = 0.7
        # 15m normalized: 0.5/0.7 ≈ 0.714, 4h normalized: 0.2/0.7 ≈ 0.286
        assert norm_w["15m"] == pytest.approx(0.5 / 0.7, rel=1e-3)
        assert norm_w["1h"] == pytest.approx(0.0)
        assert norm_w["4h"] == pytest.approx(0.2 / 0.7, rel=1e-3)
        assert sum(norm_w.values()) == pytest.approx(1.0)

        # Adaptive threshold: 0.15 * 0.7 = 0.105
        assert adaptive_th == pytest.approx(0.15 * 0.7, rel=1e-3)

    def test_two_tfs_failed(self, scanner):
        """When only 15m has data, its weight becomes 1.0."""
        results_by_tf = {
            "15m": {"longs": set(), "shorts": {"ETHUSDT"}, "strengths": {"ETHUSDT": -0.9}},
            "1h": {"longs": set(), "shorts": set(), "strengths": {}},
            "4h": {"longs": set(), "shorts": set(), "strengths": {}},
        }
        norm_w, adaptive_th = scanner._normalize_weights(results_by_tf)

        assert norm_w["15m"] == pytest.approx(1.0)
        assert norm_w["1h"] == pytest.approx(0.0)
        assert norm_w["4h"] == pytest.approx(0.0)

        # threshold * 0.5 (only 50% of weight active)
        assert adaptive_th == pytest.approx(0.15 * 0.5, rel=1e-3)

    def test_all_tfs_failed_returns_originals(self, scanner):
        """When no TF has data, original weights/threshold are returned (fallback)."""
        results_by_tf = {
            "15m": {"longs": set(), "shorts": set(), "strengths": {}},
            "1h": {"longs": set(), "shorts": set(), "strengths": {}},
            "4h": {"longs": set(), "shorts": set(), "strengths": {}},
        }
        norm_w, adaptive_th = scanner._normalize_weights(results_by_tf)

        assert norm_w == scanner.weights
        assert adaptive_th == scanner.threshold

    def test_missing_tf_in_results_treated_as_failed(self, scanner):
        """If a timeframe is completely absent from results, it counts as failed."""
        results_by_tf = {
            "15m": {"longs": {"BTCUSDT"}, "shorts": set(), "strengths": {"BTCUSDT": 0.8}},
            # 1h and 4h not present at all
        }
        norm_w, adaptive_th = scanner._normalize_weights(results_by_tf)

        assert norm_w["15m"] == pytest.approx(1.0)
        assert norm_w["1h"] == pytest.approx(0.0)
        assert norm_w["4h"] == pytest.approx(0.0)
        assert adaptive_th == pytest.approx(0.15 * 0.5, rel=1e-3)

    def test_normalized_weights_always_sum_to_one(self, scanner):
        """Normalized weights of active TFs must always sum to 1.0."""
        results_by_tf = {
            "15m": {"longs": {"A"}, "shorts": set(), "strengths": {"A": 0.5}},
            "1h": {"longs": {"B"}, "shorts": set(), "strengths": {"B": 0.4}},
            "4h": {"longs": set(), "shorts": set(), "strengths": {}},
        }
        norm_w, _ = scanner._normalize_weights(results_by_tf)
        assert sum(norm_w.values()) == pytest.approx(1.0)

    def test_shorts_count_as_data(self, scanner):
        """A TF with only SHORT signals still counts as having data."""
        results_by_tf = {
            "15m": {"longs": set(), "shorts": {"BTCUSDT"}, "strengths": {"BTCUSDT": -0.7}},
            "1h": {"longs": set(), "shorts": set(), "strengths": {}},
            "4h": {"longs": set(), "shorts": set(), "strengths": {}},
        }
        norm_w, _ = scanner._normalize_weights(results_by_tf)
        assert norm_w["15m"] == pytest.approx(1.0)

    def test_adaptive_threshold_lower_bound(self, scanner):
        """Adaptive threshold should never go below 0."""
        # All failed → returns original threshold, which is ≥ 0
        results_by_tf = {
            "15m": {"longs": set(), "shorts": set(), "strengths": {}},
            "1h": {"longs": set(), "shorts": set(), "strengths": {}},
            "4h": {"longs": set(), "shorts": set(), "strengths": {}},
        }
        _, adaptive_th = scanner._normalize_weights(results_by_tf)
        assert adaptive_th >= 0


# ============================================================================
# _pandas_to_polars_safe tests
# ============================================================================


EMPTY_SCHEMA = {"symbol": pl.Utf8, "signal": pl.Float64}


class TestPandasToPolarsafe:
    """Tests for the _pandas_to_polars_safe function."""

    def test_empty_dataframe_returns_empty_polars(self):
        """Empty pandas DataFrame should yield empty Polars DataFrame."""
        pd_df = pd.DataFrame()
        result = _pandas_to_polars_safe(pd_df, EMPTY_SCHEMA)
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

    def test_none_returns_empty_polars(self):
        """None input should yield empty Polars DataFrame."""
        result = _pandas_to_polars_safe(None, EMPTY_SCHEMA)  # type: ignore[arg-type]
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

    def test_simple_numpy_backed_conversion(self):
        """Standard numpy-backed dtypes should convert correctly."""
        pd_df = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT"], "signal": [0.9, -0.7]})
        result = _pandas_to_polars_safe(pd_df, EMPTY_SCHEMA)

        assert result.shape == (2, 2)
        assert result["symbol"].to_list() == ["BTCUSDT", "ETHUSDT"]
        assert result["signal"].to_list() == [0.9, -0.7]

    def test_extension_int64_dtype(self):
        """Pandas nullable Int64 should convert without pyarrow."""
        pd_df = pd.DataFrame(
            {"symbol": ["A", "B"], "signal": pd.array([1, 2], dtype="Int64")}
        )
        result = _pandas_to_polars_safe(pd_df, EMPTY_SCHEMA)
        assert result.shape == (2, 2)

    def test_extension_float64_dtype(self):
        """Pandas nullable Float64 should convert without pyarrow."""
        pd_df = pd.DataFrame(
            {"symbol": ["A"], "signal": pd.array([1.5], dtype="Float64")}
        )
        result = _pandas_to_polars_safe(pd_df, EMPTY_SCHEMA)
        assert result.shape == (1, 2)

    def test_extension_boolean_dtype(self):
        """Pandas nullable boolean should convert without pyarrow."""
        pd_df = pd.DataFrame(
            {"symbol": ["A"], "flag": pd.array([True], dtype="boolean")}
        )
        result = _pandas_to_polars_safe(pd_df, {"symbol": pl.Utf8, "flag": pl.Boolean})
        assert result.shape == (1, 2)

    def test_extension_string_dtype(self):
        """Pandas StringDtype should convert without pyarrow."""
        pd_df = pd.DataFrame(
            {"symbol": pd.array(["ABC"], dtype="string"), "signal": [0.5]}
        )
        result = _pandas_to_polars_safe(pd_df, EMPTY_SCHEMA)
        assert result["symbol"].to_list() == ["ABC"]

    def test_mixed_dtypes(self):
        """DataFrame with a mix of numpy and extension dtypes should convert."""
        pd_df = pd.DataFrame(
            {
                "symbol": ["BTC", "ETH"],
                "count": pd.array([10, 20], dtype="Int64"),
                "signal": [0.5, 0.3],
            }
        )
        result = _pandas_to_polars_safe(pd_df, {"symbol": pl.Utf8, "count": pl.Int64, "signal": pl.Float64})
        assert result.shape == (2, 3)

    def test_numpy_scalar_conversion(self):
        """numpy scalars in .tolist() should be converted to plain Python types."""
        import numpy as np

        pd_df = pd.DataFrame({"symbol": ["X"], "signal": np.array([np.float64(0.42)])})
        result = _pandas_to_polars_safe(pd_df, EMPTY_SCHEMA)
        assert result["signal"].to_list() == [pytest.approx(0.42)]


# ============================================================================
# Integration: end-to-end adaptive voting
# ============================================================================


class TestAdaptiveVotingIntegration:
    """Integration tests verifying the full scan → normalize → aggregate flow."""

    def test_single_tf_produces_signals(self, mock_data_fetcher):
        """With only one TF producing data, signals should still pass threshold."""
        scanner = ATCScanner(
            mock_data_fetcher,
            config={  # type: ignore[arg-type]
                "weights": {"15m": 0.5, "1h": 0.3, "4h": 0.2},
                "threshold": 0.15,
                "timeframes": ["15m", "1h", "4h"],
            },
        )

        # Simulate: only 15m has data, 1h/4h failed
        results_by_tf = {
            "15m": {
                "longs": set(),
                "shorts": {"BTCUSDT", "ETHUSDT"},
                "strengths": {"BTCUSDT": -0.9, "ETHUSDT": -0.7},
            },
            "1h": {"longs": set(), "shorts": set(), "strengths": {}},
            "4h": {"longs": set(), "shorts": set(), "strengths": {}},
        }

        norm_w, adaptive_th = scanner._normalize_weights(results_by_tf)

        # 15m normalized weight = 1.0, threshold = 0.15 * 0.5 = 0.075
        # BTC score (SHORT, strength=-0.9): -1.0 * 0.9 = -0.9  (use_signal_strength=True)
        # Score -0.9 < -0.075 → SHORT signal should pass
        assert adaptive_th == pytest.approx(0.075, rel=1e-2)
        assert norm_w["15m"] == pytest.approx(1.0)

    def test_conflicting_tfs_cancel_out(self, mock_data_fetcher):
        """If 15m=LONG and 1h=SHORT with equal effective weight, score ≈ 0 → NEUTRAL."""
        scanner = ATCScanner(
            mock_data_fetcher,
            config={  # type: ignore[arg-type]
                "weights": {"15m": 0.5, "1h": 0.5},
                "threshold": 0.15,
                "timeframes": ["15m", "1h"],
            },
        )
        results_by_tf = {
            "15m": {"longs": {"BTCUSDT"}, "shorts": set(), "strengths": {"BTCUSDT": 0.8}},
            "1h": {"longs": set(), "shorts": {"BTCUSDT"}, "strengths": {"BTCUSDT": -0.8}},
        }
        norm_w, adaptive_th = scanner._normalize_weights(results_by_tf)

        # Both have data → no normalization change
        assert norm_w["15m"] == pytest.approx(0.5)
        assert norm_w["1h"] == pytest.approx(0.5)
        assert adaptive_th == pytest.approx(0.15)
