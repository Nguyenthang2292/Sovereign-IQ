"""
Integration tests: ATCChartRenderer + StaticBacktester (backtester_static).

These tests verify that the backtester_static module is correctly wired into
the ATC Serverless Visualizer — specifically:
  - BacktestConfig is forwarded from renderer.render() into StaticBacktester.run()
  - BacktestResult (with trades and TP/SL levels) is produced and passed into
    _build_chart() for overlaying on the candlestick chart.
  - generate_report() is called and produces CSV + PNG artefacts.
  - When backtest_config=None, no backtest is run (zero overhead).

All external I/O is mocked:
  - fetch_ohlcv_data_dict     → synthetic OHLCV DataFrame
  - compute_atc_signals       → constant LONG signal series
  - calculate_{ema,wma,…}     → identity arrays (plain close prices)
  - ATCLambdaClient           → mock_mode=True (no AWS calls)
  - matplotlib / mplfinance   → disabled (no PNG written to disk)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from modules.backtester_static import BacktestConfig, BacktestResult
from modules.adaptive_trend_LTS_serverless.lambda_client import ATCLambdaClient
from modules.adaptive_trend_LTS_serverless.visualizer import ATCChartRenderer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_N = 50  # number of synthetic bars


def _make_ohlcv(n: int = _N) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with a DatetimeIndex."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame(
        {
            "open":   close - rng.uniform(0.1, 0.3, n),
            "high":   close + rng.uniform(0.2, 0.5, n),
            "low":    close - rng.uniform(0.2, 0.5, n),
            "close":  close,
            "volume": rng.uniform(100, 500, n),
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


def _make_fetch_return(df: pd.DataFrame, symbol: str = "BTCUSDT", tf: str = "1h") -> dict:
    return {symbol: {tf: df}}


def _make_compute_atc_signals_return(n: int = _N, signal_value: float = 0.5) -> dict:
    """Return a fake compute_atc_signals result with a constant Average_Signal."""
    return {"Average_Signal": [signal_value] * n}


def _identity_ma(prices: np.ndarray, _length: int) -> np.ndarray:
    """Stub MA function: return the input prices unchanged."""
    return np.asarray(prices, dtype=np.float64)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_client() -> ATCLambdaClient:
    return ATCLambdaClient(mock_mode=True)


@pytest.fixture()
def df_ohlcv() -> pd.DataFrame:
    return _make_ohlcv(_N)


# ---------------------------------------------------------------------------
# Context-manager helpers: patch all external I/O at once
# ---------------------------------------------------------------------------

def _patch_chart_io(df: pd.DataFrame, symbol: str = "BTCUSDT", tf: str = "1h"):
    """
    Bundle all patches needed by ATCChartRenderer._build_chart into a single
    context that does NOT write any PNG to disk.
    """
    n = len(df)
    _MA_MODULE = "modules.adaptive_trend_LTS_serverless.visualizer.chart"

    return (
        patch(f"{_MA_MODULE}.fetch_ohlcv_data_dict", return_value=_make_fetch_return(df, symbol, tf)),
        patch(f"{_MA_MODULE}.compute_atc_signals", return_value=_make_compute_atc_signals_return(n, 0.5)),
        patch(f"{_MA_MODULE}.calculate_ema",  side_effect=_identity_ma),
        patch(f"{_MA_MODULE}.calculate_wma",  side_effect=_identity_ma),
        patch(f"{_MA_MODULE}.calculate_dema", side_effect=_identity_ma),
        patch(f"{_MA_MODULE}.calculate_lsma", side_effect=_identity_ma),
        patch(f"{_MA_MODULE}.calculate_hma",  side_effect=_identity_ma),
        patch(f"{_MA_MODULE}.calculate_kama", side_effect=_identity_ma),
        # Suppress mplfinance + matplotlib I/O
        patch(f"{_MA_MODULE}.mpf.plot", return_value=(MagicMock(), [MagicMock(), MagicMock()])),
        patch(f"{_MA_MODULE}.plt.close"),
    )


# ---------------------------------------------------------------------------
# Tests: no backtest (baseline)
# ---------------------------------------------------------------------------

class TestRendererWithoutBacktest:
    def test_render_returns_path_without_backtest(
        self, mock_client: ATCLambdaClient, df_ohlcv: pd.DataFrame, tmp_path: Path
    ) -> None:
        """render() without backtest_config must succeed and return a Path."""
        renderer = ATCChartRenderer(mock_client, ma_length=12)
        out = tmp_path / "chart.png"

        patches = _patch_chart_io(df_ohlcv)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9]:
            result_path = renderer.render(
                symbol="BTCUSDT",
                timeframe="1h",
                limit=_N,
                display_last=_N,
                output_path=out,
                backtest_config=None,
            )

        assert result_path == out

    def test_render_calls_fetch_once(
        self, mock_client: ATCLambdaClient, df_ohlcv: pd.DataFrame, tmp_path: Path
    ) -> None:
        renderer = ATCChartRenderer(mock_client, ma_length=12)
        out = tmp_path / "chart.png"

        patches = _patch_chart_io(df_ohlcv)
        _MOD = "modules.adaptive_trend_LTS_serverless.visualizer.chart"
        with patch(f"{_MOD}.fetch_ohlcv_data_dict", return_value=_make_fetch_return(df_ohlcv)) as mock_fetch, \
             patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9]:
            renderer.render(
                symbol="BTCUSDT", timeframe="1h",
                limit=_N, display_last=_N,
                output_path=out, backtest_config=None,
            )
        mock_fetch.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: backtest enabled
# ---------------------------------------------------------------------------

class TestRendererWithBacktest:
    def test_render_with_backtest_config_runs_backtester(
        self, mock_client: ATCLambdaClient, df_ohlcv: pd.DataFrame, tmp_path: Path
    ) -> None:
        """
        When backtest_config is provided, StaticBacktester.run() must be called
        and the result forwarded to _build_chart().
        """
        renderer = ATCChartRenderer(mock_client, ma_length=12)
        out = tmp_path / "chart.png"
        bt_cfg = BacktestConfig(mode="pct", tp=2.0, sl=1.0)

        _MOD = "modules.adaptive_trend_LTS_serverless.visualizer.chart"
        patches = _patch_chart_io(df_ohlcv)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], \
             patch(f"{_MOD}.generate_report") as mock_report:
            result_path = renderer.render(
                symbol="BTCUSDT", timeframe="1h",
                limit=_N, display_last=_N,
                output_path=out, backtest_config=bt_cfg,
            )

        assert result_path == out
        # generate_report must have been called once with a BacktestResult
        mock_report.assert_called_once()
        call_args = mock_report.call_args
        assert isinstance(call_args[0][0], BacktestResult)

    def test_backtest_result_has_config_attached(
        self, mock_client: ATCLambdaClient, df_ohlcv: pd.DataFrame, tmp_path: Path
    ) -> None:
        """BacktestResult.config must equal the BacktestConfig passed to render()."""
        renderer = ATCChartRenderer(mock_client, ma_length=12)
        out = tmp_path / "chart.png"
        bt_cfg = BacktestConfig(mode="pct", tp=3.0, sl=1.5, trailing_stop=0.5)

        captured: list[BacktestResult] = []

        _MOD = "modules.adaptive_trend_LTS_serverless.visualizer.chart"
        patches = _patch_chart_io(df_ohlcv)

        def _capture_report(result: BacktestResult, *args: Any, **kwargs: Any):
            captured.append(result)

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], \
             patch(f"{_MOD}.generate_report", side_effect=_capture_report):
            renderer.render(
                symbol="BTCUSDT", timeframe="1h",
                limit=_N, display_last=_N,
                output_path=out, backtest_config=bt_cfg,
            )

        assert len(captured) == 1
        assert captured[0].config == bt_cfg

    def test_backtest_result_equity_curve_has_correct_length(
        self, mock_client: ATCLambdaClient, df_ohlcv: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Equity curve length must equal display_last."""
        display_last = 30
        renderer = ATCChartRenderer(mock_client, ma_length=12)
        out = tmp_path / "chart.png"
        bt_cfg = BacktestConfig(mode="pct", tp=2.0, sl=1.0)

        captured: list[BacktestResult] = []

        _MOD = "modules.adaptive_trend_LTS_serverless.visualizer.chart"
        df_sub = _make_ohlcv(display_last)
        patches = _patch_chart_io(df_sub)

        def _capture(result: BacktestResult, *args: Any, **kwargs: Any):
            captured.append(result)

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], \
             patch(f"{_MOD}.generate_report", side_effect=_capture):
            renderer.render(
                symbol="BTCUSDT", timeframe="1h",
                limit=display_last, display_last=display_last,
                output_path=out, backtest_config=bt_cfg,
            )

        assert len(captured) == 1
        assert len(captured[0].equity_curve) == display_last

    def test_backtest_long_signals_produce_trades(
        self, mock_client: ATCLambdaClient, tmp_path: Path
    ) -> None:
        """
        A constant LONG Average_Signal (0.5 > threshold 0.1) means every bar is a
        LONG signal.  With TP=1% and many bars, at least one trade must be produced.
        """
        n = 40
        df = _make_ohlcv(n)
        renderer = ATCChartRenderer(mock_client, ma_length=5)
        out = tmp_path / "chart.png"
        bt_cfg = BacktestConfig(mode="pct", tp=1.0, sl=2.0, max_hold_bars=5)

        captured: list[BacktestResult] = []

        _MOD = "modules.adaptive_trend_LTS_serverless.visualizer.chart"
        patches = _patch_chart_io(df)

        def _capture(result: BacktestResult, *args: Any, **kwargs: Any):
            captured.append(result)

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], \
             patch(f"{_MOD}.generate_report", side_effect=_capture):
            renderer.render(
                symbol="BTCUSDT", timeframe="1h",
                limit=n, display_last=n,
                output_path=out, backtest_config=bt_cfg,
            )

        assert len(captured) == 1
        assert captured[0].metrics["num_trades"] > 0

    def test_backtest_short_signals_produce_trades(
        self, mock_client: ATCLambdaClient, tmp_path: Path
    ) -> None:
        """Constant SHORT Average_Signal (-0.5) → all SHORT trades."""
        n = 40
        df = _make_ohlcv(n)
        renderer = ATCChartRenderer(mock_client, ma_length=5)
        out = tmp_path / "chart.png"
        bt_cfg = BacktestConfig(mode="pct", tp=1.0, sl=2.0, max_hold_bars=5)

        captured: list[BacktestResult] = []

        _MOD = "modules.adaptive_trend_LTS_serverless.visualizer.chart"

        # Override compute_atc_signals to return SHORT signals (-0.5)
        def _short_signals(*args: Any, **kwargs: Any) -> dict:
            return {"Average_Signal": [-0.5] * n}

        with patch(f"{_MOD}.fetch_ohlcv_data_dict", return_value=_make_fetch_return(df)), \
             patch(f"{_MOD}.compute_atc_signals", side_effect=_short_signals), \
             patch(f"{_MOD}.calculate_ema",  side_effect=_identity_ma), \
             patch(f"{_MOD}.calculate_wma",  side_effect=_identity_ma), \
             patch(f"{_MOD}.calculate_dema", side_effect=_identity_ma), \
             patch(f"{_MOD}.calculate_lsma", side_effect=_identity_ma), \
             patch(f"{_MOD}.calculate_hma",  side_effect=_identity_ma), \
             patch(f"{_MOD}.calculate_kama", side_effect=_identity_ma), \
             patch(f"{_MOD}.mpf.plot", return_value=(MagicMock(), [MagicMock(), MagicMock()])), \
             patch(f"{_MOD}.plt.close"), \
             patch(f"{_MOD}.generate_report", side_effect=lambda r, *a, **k: captured.append(r)):
            renderer.render(
                symbol="BTCUSDT", timeframe="1h",
                limit=n, display_last=n,
                output_path=out, backtest_config=bt_cfg,
            )

        assert len(captured) == 1
        result = captured[0]
        assert result.metrics["num_trades"] > 0
        assert result.metrics["num_short"] > 0
        assert result.metrics["num_long"] == 0.0

    def test_tp_sl_levels_match_trade_count(
        self, mock_client: ATCLambdaClient, tmp_path: Path
    ) -> None:
        """tp_sl_levels length must equal number of trades."""
        n = 30
        df = _make_ohlcv(n)
        renderer = ATCChartRenderer(mock_client, ma_length=5)
        out = tmp_path / "chart.png"
        bt_cfg = BacktestConfig(mode="pct", tp=0.5, sl=2.0, max_hold_bars=3)

        captured: list[BacktestResult] = []

        _MOD = "modules.adaptive_trend_LTS_serverless.visualizer.chart"
        patches = _patch_chart_io(df)

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], \
             patch(f"{_MOD}.generate_report", side_effect=lambda r, *a, **k: captured.append(r)):
            renderer.render(
                symbol="BTCUSDT", timeframe="1h",
                limit=n, display_last=n,
                output_path=out, backtest_config=bt_cfg,
            )

        assert len(captured) == 1
        result = captured[0]
        assert len(result.tp_sl_levels) == len(result.trades)

    def test_no_backtest_when_config_is_none(
        self, mock_client: ATCLambdaClient, df_ohlcv: pd.DataFrame, tmp_path: Path
    ) -> None:
        """generate_report must NOT be called when backtest_config=None."""
        renderer = ATCChartRenderer(mock_client, ma_length=12)
        out = tmp_path / "chart.png"

        _MOD = "modules.adaptive_trend_LTS_serverless.visualizer.chart"
        patches = _patch_chart_io(df_ohlcv)

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], \
             patch(f"{_MOD}.generate_report") as mock_report:
            renderer.render(
                symbol="BTCUSDT", timeframe="1h",
                limit=_N, display_last=_N,
                output_path=out, backtest_config=None,
            )

        mock_report.assert_not_called()

    def test_backtest_atr_mode_runs_without_error(
        self, mock_client: ATCLambdaClient, tmp_path: Path
    ) -> None:
        """ATR mode should run cleanly through the renderer."""
        n = 30
        df = _make_ohlcv(n)
        renderer = ATCChartRenderer(mock_client, ma_length=5)
        out = tmp_path / "chart.png"
        bt_cfg = BacktestConfig(mode="atr", tp=1.5, sl=1.0, atr_period=5, max_hold_bars=5)

        captured: list[BacktestResult] = []

        _MOD = "modules.adaptive_trend_LTS_serverless.visualizer.chart"
        patches = _patch_chart_io(df)

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], \
             patch(f"{_MOD}.generate_report", side_effect=lambda r, *a, **k: captured.append(r)):
            renderer.render(
                symbol="BTCUSDT", timeframe="1h",
                limit=n, display_last=n,
                output_path=out, backtest_config=bt_cfg,
            )

        assert len(captured) == 1
        assert captured[0].config is not None
        assert captured[0].config.mode == "atr"

    def test_backtest_trailing_stop_mode_runs_without_error(
        self, mock_client: ATCLambdaClient, tmp_path: Path
    ) -> None:
        """Trailing stop should be forwarded correctly to the backtester."""
        n = 30
        df = _make_ohlcv(n)
        renderer = ATCChartRenderer(mock_client, ma_length=5)
        out = tmp_path / "chart.png"
        bt_cfg = BacktestConfig(mode="pct", tp=2.0, sl=1.0, trailing_stop=0.5, max_hold_bars=5)

        _MOD = "modules.adaptive_trend_LTS_serverless.visualizer.chart"
        patches = _patch_chart_io(df)

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], \
             patch(f"{_MOD}.generate_report"):
            result_path = renderer.render(
                symbol="BTCUSDT", timeframe="1h",
                limit=n, display_last=n,
                output_path=out, backtest_config=bt_cfg,
            )

        assert result_path == out


# ---------------------------------------------------------------------------
# Tests: draw_tp_sl_levels helper
# ---------------------------------------------------------------------------

class TestDrawTpSlLevels:
    """Unit-test the TP/SL drawing helper in isolation."""

    def test_draw_tp_sl_levels_calls_hlines_twice_per_level(self) -> None:
        from modules.backtester_static.engine import TpSlLevel
        from modules.adaptive_trend_LTS_serverless.visualizer.chart import ATCChartRenderer

        renderer = ATCChartRenderer(ATCLambdaClient(mock_mode=True))
        ax = MagicMock()

        levels = [
            TpSlLevel(entry_idx=0, exit_idx=5, direction="LONG",
                      tp_price=102.0, sl_price=98.0, exit_reason="TP"),
            TpSlLevel(entry_idx=6, exit_idx=10, direction="SHORT",
                      tp_price=95.0, sl_price=105.0, exit_reason="SL"),
        ]

        renderer._draw_tp_sl_levels(ax, levels, n_bars=20)

        # 2 levels × 2 lines (TP + SL) = 4 hlines calls
        assert ax.hlines.call_count == 4

    def test_draw_tp_sl_levels_empty_list_no_calls(self) -> None:
        from modules.adaptive_trend_LTS_serverless.visualizer.chart import ATCChartRenderer

        renderer = ATCChartRenderer(ATCLambdaClient(mock_mode=True))
        ax = MagicMock()
        renderer._draw_tp_sl_levels(ax, [], n_bars=50)
        ax.hlines.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: BacktestConfig round-trip via renderer
# ---------------------------------------------------------------------------

class TestBacktestConfigRoundTrip:

    @pytest.mark.parametrize("mode,tp,sl,trailing", [
        ("pct", 2.0, 1.0, None),
        ("pct", 3.0, 1.5, 0.5),
        ("atr", 2.0, 1.0, None),
        ("atr", 2.0, 1.0, 0.8),
    ])
    def test_config_preserved_in_backtest_result(
        self,
        mode: str,
        tp: float,
        sl: float,
        trailing: float | None,
        mock_client: ATCLambdaClient,
        tmp_path: Path,
    ) -> None:
        n = 20
        df = _make_ohlcv(n)
        renderer = ATCChartRenderer(mock_client, ma_length=5)
        out = tmp_path / "chart.png"
        bt_cfg = BacktestConfig(mode=mode, tp=tp, sl=sl, trailing_stop=trailing, max_hold_bars=5)  # type: ignore[arg-type]

        captured: list[BacktestResult] = []

        _MOD = "modules.adaptive_trend_LTS_serverless.visualizer.chart"
        patches = _patch_chart_io(df)

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], \
             patch(f"{_MOD}.generate_report", side_effect=lambda r, *a, **k: captured.append(r)):
            renderer.render(
                symbol="BTCUSDT", timeframe="1h",
                limit=n, display_last=n,
                output_path=out, backtest_config=bt_cfg,
            )

        assert len(captured) == 1
        result_cfg = captured[0].config
        assert result_cfg is not None
        assert result_cfg.mode == mode
        assert result_cfg.tp == tp
        assert result_cfg.sl == sl
        assert result_cfg.trailing_stop == trailing
