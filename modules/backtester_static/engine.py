"""
StaticBacktester – generic signal-consumer backtesting engine.

Accepts any OHLCV DataFrame + a signal Series (LONG/SHORT/NEUTRAL or 1/-1/0)
and simulates trades with configurable TP, SL, and optional trailing stop.

Designed to be reusable across multiple signal modules (ATC, XGBoost, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import BacktestConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Single completed trade."""

    idx_entry: int              # bar index in input df
    idx_exit: int               # bar index in input df
    time_entry: Any             # timestamp at entry
    time_exit: Any              # timestamp at exit
    direction: str              # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    tp_price: float
    sl_price: float
    exit_reason: str            # "TP" | "SL" | "TRAIL" | "TIME" | "SIGNAL"
    pnl_pct: float              # percent PnL from entry
    bars_held: int


@dataclass
class TpSlLevel:
    """TP/SL horizontal segment for chart overlay."""

    entry_idx: int
    exit_idx: int
    direction: str
    tp_price: float
    sl_price: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Output of StaticBacktester.run()."""

    trades: list[TradeRecord] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    tp_sl_levels: list[TpSlLevel] = field(default_factory=list)
    config: Optional[BacktestConfig] = None


# ---------------------------------------------------------------------------
# ATR helper
# ---------------------------------------------------------------------------

def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's ATR — returns array aligned to input length."""
    n = len(high)
    if n == 0:
        return np.array([], dtype=np.float64)

    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        h = high[i]
        lo = low[i]
        pc = close[i - 1]
        tr[i] = max(h - lo, abs(h - pc), abs(lo - pc))

    atr = np.zeros(n, dtype=np.float64)
    initial = min(period, n)
    atr[initial - 1] = float(np.mean(tr[:initial]))
    alpha = 1.0 / period
    for i in range(initial, n):
        atr[i] = (1.0 - alpha) * atr[i - 1] + alpha * tr[i]
    return atr


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class StaticBacktester:
    """
    Generic static backtester.

    Accepts any DataFrame with OHLCV columns (case-insensitive) and a signal
    Series whose values are ``"LONG"`` / ``"SHORT"`` / ``"NEUTRAL"`` or numeric
    ``1`` / ``-1`` / ``0``.

    Args:
        config: Backtest configuration. Defaults to ``BacktestConfig()``.

    Example::

        from modules.backtester_static import StaticBacktester, BacktestConfig

        cfg = BacktestConfig(mode="pct", tp=2.0, sl=1.0, trailing_stop=0.5)
        bt = StaticBacktester(config=cfg)
        result = bt.run(df=df_ohlcv, signals=signal_series)
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
    ) -> BacktestResult:
        """
        Run backtest simulation.

        Args:
            df:      OHLCV DataFrame (must have open/high/low/close/volume columns,
                     case-insensitive). Index should be DatetimeIndex.
            signals: Series aligned to ``df.index`` with values
                     ``"LONG"`` / ``"SHORT"`` / ``"NEUTRAL"`` or numeric 1/-1/0.

        Returns:
            :class:`BacktestResult` with trades, metrics, equity curve, and TP/SL levels.
        """
        df = self._normalise_df(df)
        sig = self._normalise_signals(signals, df.index)

        cfg = self.config

        # Pre-extract numpy arrays for performance
        close = df["close"].to_numpy(dtype=np.float64)
        high  = df["high"].to_numpy(dtype=np.float64)
        low   = df["low"].to_numpy(dtype=np.float64)
        n = len(df)

        # ATR (computed always; used only when mode="atr")
        atr: Optional[np.ndarray] = None
        if cfg.mode == "atr":
            atr = _compute_atr(high, low, close, cfg.atr_period)

        trades: list[TradeRecord] = []
        tp_sl_levels: list[TpSlLevel] = []

        position: Optional[dict[str, Any]] = None

        for i in range(n):
            price = close[i]
            h = high[i]
            lo = low[i]
            sig_i = sig[i]
            closed_this_bar = False

            # ----------------------------------------------------------------
            # Check exit conditions for open position
            # ----------------------------------------------------------------
            if position is not None:
                direction = position["direction"]
                ep = position["entry_price"]
                tp = position["tp_price"]
                sl = position["sl_price"]
                entry_i = position["entry_idx"]
                bars_held = i - entry_i

                exit_reason: Optional[str] = None
                exit_price = price

                if direction == "LONG":
                    # Update trailing stop
                    if cfg.trailing_stop is not None:
                        peak = position.get("peak_price", ep)
                        peak = max(peak, h)
                        position["peak_price"] = peak
                        if cfg.mode == "pct":
                            new_sl = peak * (1.0 - cfg.trailing_stop / 100.0)
                        else:
                            atri = float(atr[i]) if atr is not None else 0.0  # type: ignore[index]
                            new_sl = peak - cfg.trailing_stop * atri
                        if new_sl > sl:  # only move stop up
                            position["sl_price"] = new_sl
                            sl = new_sl

                    # Check SL first (conservative — low could have hit SL)
                    if lo <= sl:
                        exit_reason = "TRAIL" if cfg.trailing_stop else "SL"
                        exit_price = sl
                    elif h >= tp:
                        exit_reason = "TP"
                        exit_price = tp
                    elif bars_held >= cfg.max_hold_bars:
                        exit_reason = "TIME"
                        exit_price = price

                else:  # SHORT
                    # Update trailing stop for short
                    if cfg.trailing_stop is not None:
                        trough = position.get("trough_price", ep)
                        trough = min(trough, lo)
                        position["trough_price"] = trough
                        if cfg.mode == "pct":
                            new_sl = trough * (1.0 + cfg.trailing_stop / 100.0)
                        else:
                            atri = float(atr[i]) if atr is not None else 0.0  # type: ignore[index]
                            new_sl = trough + cfg.trailing_stop * atri
                        if new_sl < sl:  # only move stop down
                            position["sl_price"] = new_sl
                            sl = new_sl

                    # Check SL first
                    if h >= sl:
                        exit_reason = "TRAIL" if cfg.trailing_stop else "SL"
                        exit_price = sl
                    elif lo <= tp:
                        exit_reason = "TP"
                        exit_price = tp
                    elif bars_held >= cfg.max_hold_bars:
                        exit_reason = "TIME"
                        exit_price = price

                if exit_reason is not None:
                    pnl_pct = (
                        (exit_price / ep - 1.0) * 100.0
                        if direction == "LONG"
                        else (1.0 - exit_price / ep) * 100.0
                    )
                    trades.append(TradeRecord(
                        idx_entry=entry_i,
                        idx_exit=i,
                        time_entry=df.index[entry_i],
                        time_exit=df.index[i],
                        direction=direction,
                        entry_price=ep,
                        exit_price=exit_price,
                        tp_price=position["tp_price"],
                        sl_price=position["sl_price"],
                        exit_reason=exit_reason,
                        pnl_pct=pnl_pct,
                        bars_held=bars_held,
                    ))
                    tp_sl_levels.append(TpSlLevel(
                        entry_idx=entry_i,
                        exit_idx=i,
                        direction=direction,
                        tp_price=position["tp_price"],
                        sl_price=position["sl_price"],
                        exit_reason=exit_reason,
                    ))
                    position = None
                    closed_this_bar = True

            # ----------------------------------------------------------------
            # Check entry conditions (only when flat)
            # ----------------------------------------------------------------
            if position is None and not closed_this_bar and sig_i in ("LONG", "SHORT"):
                entry_price = price

                if cfg.mode == "pct":
                    if sig_i == "LONG":
                        tp_price = entry_price * (1.0 + cfg.tp / 100.0)
                        sl_price = entry_price * (1.0 - cfg.sl / 100.0)
                    else:
                        tp_price = entry_price * (1.0 - cfg.tp / 100.0)
                        sl_price = entry_price * (1.0 + cfg.sl / 100.0)
                else:
                    atri = float(atr[i]) if atr is not None else 0.0  # type: ignore[index]
                    if sig_i == "LONG":
                        tp_price = entry_price + cfg.tp * atri
                        sl_price = entry_price - cfg.sl * atri
                    else:
                        tp_price = entry_price - cfg.tp * atri
                        sl_price = entry_price + cfg.sl * atri

                position = {
                    "entry_idx":   i,
                    "direction":   sig_i,
                    "entry_price": entry_price,
                    "tp_price":    tp_price,
                    "sl_price":    sl_price,
                }

        # Close any open position at end of data
        if position is not None:
            ep = position["entry_price"]
            direction = position["direction"]
            entry_i = position["entry_idx"]
            exit_price = close[-1]
            pnl_pct = (
                (exit_price / ep - 1.0) * 100.0
                if direction == "LONG"
                else (1.0 - exit_price / ep) * 100.0
            )
            trades.append(TradeRecord(
                idx_entry=entry_i,
                idx_exit=n - 1,
                time_entry=df.index[entry_i],
                time_exit=df.index[n - 1],
                direction=direction,
                entry_price=ep,
                exit_price=exit_price,
                tp_price=position["tp_price"],
                sl_price=position["sl_price"],
                exit_reason="TIME",
                pnl_pct=pnl_pct,
                bars_held=n - 1 - entry_i,
            ))
            tp_sl_levels.append(TpSlLevel(
                entry_idx=entry_i,
                exit_idx=n - 1,
                direction=direction,
                tp_price=position["tp_price"],
                sl_price=position["sl_price"],
                exit_reason="TIME",
            ))

        # Compute equity curve and metrics
        from .metrics import calculate_metrics
        equity_curve = self._build_equity_curve(trades, n, df.index, cfg.initial_capital)
        metrics = calculate_metrics(trades, equity_curve, cfg.initial_capital)

        logger.info(
            "Backtest done: %d trades | win=%.1f%% | return=%+.2f%% | maxDD=%.2f%%",
            metrics.get("num_trades", 0),
            metrics.get("win_rate", 0.0) * 100,
            metrics.get("total_return_pct", 0.0),
            metrics.get("max_drawdown_pct", 0.0),
        )

        return BacktestResult(
            trades=trades,
            metrics=metrics,
            equity_curve=equity_curve,
            tp_sl_levels=tp_sl_levels,
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' missing from DataFrame")
        return df

    @staticmethod
    def _normalise_signals(signals: pd.Series, idx: pd.Index) -> np.ndarray:
        """Convert signal Series to string ndarray ('LONG'/'SHORT'/'NEUTRAL')."""
        arr = signals.reindex(idx).fillna(0)
        out = np.full(len(arr), "NEUTRAL", dtype=object)
        vals = arr.values
        for i, v in enumerate(vals):
            sv = str(v).upper()
            if sv in ("LONG", "1", "1.0"):
                out[i] = "LONG"
            elif sv in ("SHORT", "-1", "-1.0"):
                out[i] = "SHORT"
        return out

    @staticmethod
    def _build_equity_curve(
        trades: list[TradeRecord],
        n_bars: int,
        index: pd.Index,
        initial_capital: float,
    ) -> pd.Series:
        """Compounded equity curve using realized trade PnL on exit bars."""
        capital = float(initial_capital)
        cap_arr = np.full(n_bars, capital, dtype=np.float64)
        for t in trades:
            capital *= 1.0 + (t.pnl_pct / 100.0)
            cap_arr[t.idx_exit:] = capital
        return pd.Series(cap_arr, index=index)
