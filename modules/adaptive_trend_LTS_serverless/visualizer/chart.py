"""
ATCChartRenderer - Candlestick chart with ATC signals and all 6 MA lines.

MA types mirrored from the ATC Rust engine (DEFAULT_ATC_CONFIG):
  EMA · WMA · DEMA · LSMA · HMA · KAMA  (default length = 12)

Signal marker is drawn on the LAST candle:
  LONG  → green  ▲ below bar
  SHORT → red    ▼ above bar
  NEUTRAL → score annotation only
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mplfinance as mpf  # type: ignore[import-untyped,no-redef]
import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.compute_atc_signals import (
    compute_atc_signals,
)
from modules.adaptive_trend_LTS_mini.core.rust_backend import (
    calculate_dema,
    calculate_ema,
    calculate_hma,
    calculate_kama,
    calculate_lsma,
    calculate_wma,
)
from modules.adaptive_trend_LTS_serverless.lambda_client import (
    DEFAULT_ATC_CONFIG,
    ATCLambdaClient,
)
from modules.backtester_static import BacktestConfig, BacktestResult, StaticBacktester, TpSlLevel
from modules.backtester_static.report import generate_report
from modules.common.data.fetchers import fetch_ohlcv_data_dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MA palette – matches TradingView-style dark theme, each clearly distinct
# ---------------------------------------------------------------------------
_MA_STYLES: dict[str, dict[str, Any]] = {
    "EMA":  {"color": "#FF9800", "width": 1.2, "linestyle": "-"},   # amber
    "WMA":  {"color": "#42A5F5", "width": 1.2, "linestyle": "-"},   # blue
    "DEMA": {"color": "#CE93D8", "width": 1.2, "linestyle": "--"},  # purple
    "LSMA": {"color": "#26C6DA", "width": 1.2, "linestyle": "--"},  # cyan
    "HMA":  {"color": "#66BB6A", "width": 1.5, "linestyle": "-"},   # green (thicker = faster)
    "KAMA": {"color": "#EF5350", "width": 1.5, "linestyle": "-"},   # red   (thicker = adaptive)
}

_MA_FUNCS = {
    "EMA":  calculate_ema,
    "WMA":  calculate_wma,
    "DEMA": calculate_dema,
    "LSMA": calculate_lsma,
    "HMA":  calculate_hma,
    "KAMA": calculate_kama,
}

# Signal colors
_SIG_COLOR = {
    "LONG":    "#00e676",  # neon green
    "SHORT":   "#ff1744",  # neon red
    "NEUTRAL": "#90a4ae",  # grey
}

# mplfinance dark market colors (TradingView palette)
_MARKET_COLORS = mpf.make_marketcolors(
    up="#26a69a",
    down="#ef5350",
    wick={"up": "#26a69a", "down": "#ef5350"},
    edge={"up": "#26a69a", "down": "#ef5350"},
    volume={"up": "#26a69a55", "down": "#ef535055"},
)
_CHART_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=_MARKET_COLORS,
    gridstyle=":",
    gridcolor="#2a2a3e",
    facecolor="#0d0d1a",
    figcolor="#0d0d1a",
    rc={
        "axes.labelcolor": "#cfd8dc",
        "axes.edgecolor": "#2a2a3e",
        "xtick.color": "#90a4ae",
        "ytick.color": "#90a4ae",
        "font.size": 9,
    },
)


class ATCChartRenderer:
    """
    Renders a high-quality candlestick chart with:
      - All 6 ATC MA lines overlaid on the price panel
      - LONG / SHORT signal marker on the last candle
      - Volume panel below
      - Score annotation (top-right of price panel)

    Args:
        lambda_client: Configured ATCLambdaClient instance.
        ma_length: MA period (default 12, matching DEFAULT_ATC_CONFIG).
    """

    def __init__(
        self,
        lambda_client: ATCLambdaClient,
        ma_length: int = 12,
    ) -> None:
        self.client = lambda_client
        self.ma_length = ma_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 300,
        display_last: int = 150,
        output_path: Optional[Union[str, Path]] = None,
        config: Optional[dict[str, Any]] = None,
        backtest_config: Optional[BacktestConfig] = None,
    ) -> Path:
        """
        Fetch data → run Lambda → draw chart → save PNG.

        Args:
            symbol:       Trading pair, e.g. ``"BTCUSDT"``.
            timeframe:    Candle timeframe, e.g. ``"1h"``, ``"4h"``, ``"15m"``.
            limit:        Total bars to fetch for MA warmup (default 300).
            display_last: Candles shown in the chart window (default 150).
            output_path:  Path to save ``.png``.  Auto-generated under
                          ``./charts/`` when *None*.
            config:            Override ATC config; uses DEFAULT_ATC_CONFIG if *None*.
            backtest_config:   When provided, runs :class:`StaticBacktester` on the
                               displayed bars and overlays TP/SL levels on the chart.
                               Also saves a trade-log CSV and a table PNG.

        Returns:
            :class:`Path` to the saved PNG file.
        """
        logger.info("Fetching %s / %s (%d bars)…", symbol, timeframe, limit)
        df_full = self._fetch_data(symbol, timeframe, limit)

        effective_config = self._build_effective_config(config, timeframe)

        logger.info("Computing ATC signal via Lambda…")
        signal = self._compute_signal(df_full, symbol, timeframe, effective_config)
        logger.info(
            "Signal: %s  score: %+.4f",
            signal.get("signal_type", "?"),
            signal.get("score", 0.0),
        )

        logger.info("Computing historical ATC signals for past candles…")
        historical_types_full = self._compute_historical_signal_types(df_full, effective_config)

        # Compute MA across the full series for proper warmup, then trim
        logger.debug("Computing MA series (length=%d)…", self.ma_length)
        ma_full = self._compute_mas(df_full["close"])

        # Trim to display window
        n = min(display_last, len(df_full))
        df_display = df_full.tail(n).copy()
        ma_display = {name: arr[-n:] for name, arr in ma_full.items()}
        historical_types_display = historical_types_full.tail(n)

        out_path = self._resolve_output_path(symbol, timeframe, output_path)

        # Optional backtest run
        backtest_result: Optional[BacktestResult] = None
        if backtest_config is not None:
            logger.info("Running static backtest (%s)…", backtest_config.summary())
            bt = StaticBacktester(config=backtest_config)
            backtest_result = bt.run(df_display, historical_types_display)
            generate_report(
                backtest_result, symbol, timeframe, backtest_config, out_path.parent
            )

        logger.info("Rendering chart → %s", out_path)
        self._build_chart(
            df_display,
            ma_display,
            signal,
            historical_types_display,
            symbol,
            timeframe,
            out_path,
            backtest_result=backtest_result,
        )
        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_data(self, symbol: str, tf: str, limit: int) -> pd.DataFrame:
        """Fetch OHLCV and return a DataFrame with DatetimeIndex."""
        data = fetch_ohlcv_data_dict([symbol], [tf], limit=limit)
        if not data or symbol not in data or tf not in data[symbol]:
            raise ValueError(f"No OHLCV data returned for {symbol}/{tf}")

        df: pd.DataFrame = data[symbol][tf].copy()

        # Normalise index to DatetimeIndex (mplfinance requirement)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
            elif "datetime" in df.columns:
                df = df.set_index(pd.to_datetime(df["datetime"], utc=True))
            else:
                raise ValueError("Cannot determine datetime column from fetched DataFrame")

        df.index.name = "Date"
        df.columns = [c.lower() for c in df.columns]

        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' missing in fetched DataFrame")

        return df.sort_index()

    def _compute_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        tf: str,
        config: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Call Lambda and return the signal dict for *symbol*."""
        if config is None:
            config = DEFAULT_ATC_CONFIG.copy()
            # Single-TF chart: give 100 % weight to this TF
            config["weights"] = {tf: 1.0}

        # Convert DatetimeIndex → epoch ms for Lambda payload
        ts_ms: list[int] = (df.index.astype("int64") // 10**6).tolist()

        symbols_payload = [
            {
                "symbol": symbol,
                "timeframes": {
                    tf: {
                        "timestamp": ts_ms,
                        "open":      df["open"].tolist(),
                        "high":      df["high"].tolist(),
                        "low":       df["low"].tolist(),
                        "close":     df["close"].tolist(),
                        "volume":    df["volume"].tolist(),
                    }
                },
            }
        ]

        result = self.client.invoke(symbols_payload, config)

        results_list: list[dict[str, Any]] = result.get("results", [])
        if results_list:
            return results_list[0]

        errors: list[Any] = result.get("errors", [])
        if errors:
            raise RuntimeError(f"Lambda returned error: {errors[0]}")

        # Fallback: Lambda returned no data yet. In mock mode this is expected.
        if self.client.mock_mode:
            logger.info("Mock mode: Lambda returned empty results list; defaulting to NEUTRAL.")
        else:
            logger.warning("Lambda returned empty results list; defaulting to NEUTRAL.")
        return {"signal_type": "NEUTRAL", "score": 0.0, "symbol": symbol}

    def _build_effective_config(self, config: Optional[dict[str, Any]], tf: str) -> dict[str, Any]:
        """Build an effective config used consistently for Lambda + historical plotting."""
        if config is None:
            effective = DEFAULT_ATC_CONFIG.copy()
            effective["weights"] = {tf: 1.0}
            return effective
        return config.copy()

    def _compute_historical_signal_types(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.Series:
        """Compute historical LONG/SHORT/NEUTRAL series from local ATC Average_Signal."""
        prices = pd.Series(df["close"].values, index=df.index)

        threshold = float(config.get("threshold", 0.1))
        ma_cfgs = config.get("ma_configs", DEFAULT_ATC_CONFIG["ma_configs"])

        ma_defaults: dict[str, tuple[int, float]] = {
            "EMA": (12, 1.0),
            "HMA": (12, 1.0),
            "WMA": (12, 1.0),
            "DEMA": (12, 1.0),
            "LSMA": (12, 1.0),
            "KAMA": (12, 1.0),
        }

        for ma in ma_cfgs:
            ma_type = str(ma.get("ma_type", "")).upper()
            if ma_type in ma_defaults:
                ma_defaults[ma_type] = (int(ma.get("length", 12)), float(ma.get("weight", 1.0)))

        try:
            atc_result = compute_atc_signals(
                prices,
                ema_len=ma_defaults["EMA"][0],
                hma_len=ma_defaults["HMA"][0],
                wma_len=ma_defaults["WMA"][0],
                dema_len=ma_defaults["DEMA"][0],
                lsma_len=ma_defaults["LSMA"][0],
                kama_len=ma_defaults["KAMA"][0],
                ema_w=ma_defaults["EMA"][1],
                hma_w=ma_defaults["HMA"][1],
                wma_w=ma_defaults["WMA"][1],
                dema_w=ma_defaults["DEMA"][1],
                lsma_w=ma_defaults["LSMA"][1],
                kama_w=ma_defaults["KAMA"][1],
                robustness=str(config.get("robustness", "Medium")),
                lambda_param=float(config.get("lambda_param", 0.02)),
                decay=float(config.get("decay", 0.03)),
                cutout=int(config.get("cutout", 0)),
                long_threshold=threshold,
                short_threshold=-threshold,
                equity_floor=float(config.get("equity_floor", 0.25)),
                use_rust_backend=True,
                fast_mode=True,
                use_cache=True,
            )
            avg_signal = pd.Series(atc_result["Average_Signal"], index=df.index)
        except Exception as exc:
            logger.warning("Failed to compute historical ATC signals locally: %s", exc)
            return pd.Series(["NEUTRAL"] * len(df), index=df.index)

        signal_type_arr = np.full(len(avg_signal), "NEUTRAL", dtype=object)
        avg_vals = avg_signal.to_numpy(dtype=np.float64)
        signal_type_arr[avg_vals > threshold] = "LONG"
        signal_type_arr[avg_vals < -threshold] = "SHORT"
        return pd.Series(signal_type_arr, index=df.index)

    def _compute_mas(self, close: pd.Series) -> dict[str, np.ndarray]:
        """Compute all 6 MA series from the close price array."""
        prices: np.ndarray = np.asarray(close.values, dtype=np.float64)
        return {
            name: fn(prices, self.ma_length)  # type: ignore[arg-type]
            for name, fn in _MA_FUNCS.items()
        }

    # ------------------------------------------------------------------
    # Chart construction
    # ------------------------------------------------------------------

    def _build_chart(
        self,
        df: pd.DataFrame,
        ma_series: dict[str, np.ndarray],
        signal: dict[str, Any],
        historical_signal_types: pd.Series,
        symbol: str,
        timeframe: str,
        out_path: Path,
        backtest_result: Optional[BacktestResult] = None,
    ) -> None:
        signal_type: str = str(signal.get("signal_type", "NEUTRAL")).upper()
        score: float = float(signal.get("score", 0.0))
        sig_color = _SIG_COLOR.get(signal_type, _SIG_COLOR["NEUTRAL"])

        addplots = self._build_ma_addplots(ma_series, len(df))
        addplots += self._build_historical_signal_markers(df, historical_signal_types)
        addplots += self._build_signal_marker(df, signal_type)

        title = (
            f"{symbol}  /  {timeframe}"
            f"    |    Signal: {signal_type}"
            f"    |    Score: {score:+.4f}"
        )

        fig, axes = mpf.plot(
            df,
            type="candle",
            style=_CHART_STYLE,
            title=title,
            volume=True,
            addplot=addplots if addplots else None,
            returnfig=True,
            figsize=(20, 11),
            panel_ratios=(4, 1),
            tight_layout=True,
        )

        ax_price = axes[0]

        # TP/SL overlay (drawn before legend so they sit behind annotations)
        if backtest_result is not None and backtest_result.tp_sl_levels:
            self._draw_tp_sl_levels(ax_price, backtest_result.tp_sl_levels, len(df))

        # MA legend (top-left)
        legend_handles = [
            mpatches.Patch(
                color=_MA_STYLES[name]["color"],
                label=f"{name}({self.ma_length})",
            )
            for name in _MA_FUNCS
        ]
        # Signal badge in legend
        sig_label = f"{'▲' if signal_type == 'LONG' else '▼' if signal_type == 'SHORT' else '●'} {signal_type}"
        legend_handles.append(mpatches.Patch(color=sig_color, label=sig_label))

        ax_price.legend(
            handles=legend_handles,
            loc="upper left",
            fontsize=8,
            framealpha=0.25,
            facecolor="#1a1a2e",
            edgecolor="#2a2a3e",
            labelcolor="#cfd8dc",
            ncol=2,
        )

        # Score badge (top-right)
        ax_price.text(
            0.995, 0.975,
            f"Score  {score:+.4f}",
            transform=ax_price.transAxes,
            ha="right", va="top",
            color=sig_color,
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#0d0d1a",
                edgecolor=sig_color,
                alpha=0.85,
                linewidth=1.2,
            ),
        )

        # Watermark
        ax_price.text(
            0.5, 0.5,
            "ATC Serverless",
            transform=ax_price.transAxes,
            ha="center", va="center",
            fontsize=28, color="#ffffff08",
            fontweight="bold", rotation=30,
        )

        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Chart saved → %s", out_path)

    # ------------------------------------------------------------------
    # addplot helpers / TP-SL overlay
    # ------------------------------------------------------------------

    def _draw_tp_sl_levels(
        self,
        ax: Any,
        levels: list[TpSlLevel],
        n_bars: int,
    ) -> None:
        """Draw semi-transparent TP (green) and SL (red) horizontal segments."""
        for lvl in levels:
            x1 = lvl.entry_idx
            x2 = min(lvl.exit_idx, n_bars - 1)
            # TP line — dashed green
            ax.hlines(
                y=lvl.tp_price,
                xmin=x1, xmax=x2,
                colors="#00e67666",
                linestyles="dashed",
                linewidth=1.2,
                zorder=2,
            )
            # SL line — dashed red
            ax.hlines(
                y=lvl.sl_price,
                xmin=x1, xmax=x2,
                colors="#ff174466",
                linestyles="dashed",
                linewidth=1.2,
                zorder=2,
            )

    def _build_ma_addplots(
        self,
        ma_series: dict[str, np.ndarray],
        n: int,
    ) -> list[Any]:
        """Return one addplot per MA line."""
        plots = []
        for name, arr in ma_series.items():
            style = _MA_STYLES[name]
            # Guard: NaN-only series → skip to avoid mplfinance warnings
            if np.all(np.isnan(arr)):
                logger.debug("MA %s is entirely NaN – skipping addplot.", name)
                continue
            plots.append(
                mpf.make_addplot(
                    arr,
                    color=style["color"],
                    width=style["width"],
                    linestyle=style["linestyle"],
                    panel=0,
                    secondary_y=False,
                )
            )
        return plots

    def _build_signal_marker(
        self, df: pd.DataFrame, signal_type: str
    ) -> list[Any]:
        """Return a scatter addplot for the signal arrow on the last candle."""
        if signal_type not in ("LONG", "SHORT"):
            return []

        n = len(df)
        marker_arr = np.full(n, np.nan)
        last = n - 1

        if signal_type == "LONG":
            # Place arrow below the low with a small gap
            marker_arr[last] = df["low"].iloc[last] * 0.9985
            return [
                mpf.make_addplot(
                    marker_arr,
                    type="scatter",
                    markersize=200,
                    marker="^",
                    color="#00e676",
                    panel=0,
                    secondary_y=False,
                )
            ]
        else:  # SHORT
            marker_arr[last] = df["high"].iloc[last] * 1.0015
            return [
                mpf.make_addplot(
                    marker_arr,
                    type="scatter",
                    markersize=200,
                    marker="v",
                    color="#ff1744",
                    panel=0,
                    secondary_y=False,
                )
            ]

    def _build_historical_signal_markers(
        self,
        df: pd.DataFrame,
        historical_signal_types: pd.Series,
    ) -> list[Any]:
        """Return scatter addplots for historical LONG/SHORT markers."""
        n = len(df)
        long_arr = np.full(n, np.nan)
        short_arr = np.full(n, np.nan)

        sig_np = historical_signal_types.astype(str).str.upper().to_numpy()
        long_mask = sig_np == "LONG"
        short_mask = sig_np == "SHORT"

        low_np = df["low"].to_numpy(dtype=np.float64)
        high_np = df["high"].to_numpy(dtype=np.float64)

        long_arr[long_mask] = low_np[long_mask] * 0.9992
        short_arr[short_mask] = high_np[short_mask] * 1.0008

        plots: list[Any] = []
        if np.any(long_mask):
            plots.append(
                mpf.make_addplot(
                    long_arr,
                    type="scatter",
                    markersize=34,
                    marker="^",
                    color="#00e67699",
                    panel=0,
                    secondary_y=False,
                )
            )
        if np.any(short_mask):
            plots.append(
                mpf.make_addplot(
                    short_arr,
                    type="scatter",
                    markersize=34,
                    marker="v",
                    color="#ff174499",
                    panel=0,
                    secondary_y=False,
                )
            )
        return plots

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_output_path(
        symbol: str,
        timeframe: str,
        output_path: Optional[Union[str, Path]],
    ) -> Path:
        if output_path is not None:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        charts_dir = Path("charts")
        charts_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return charts_dir / f"{symbol}_{timeframe}_{ts}.png"
