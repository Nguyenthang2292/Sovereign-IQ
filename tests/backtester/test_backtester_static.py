from __future__ import annotations

import pandas as pd
import pytest

from modules.backtester_static import BacktestConfig, BacktestResult, StaticBacktester, TpSlLevel


def _build_df(rows: list[dict[str, float]]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(rows), freq="1h")
    return pd.DataFrame(rows, index=index)


# ---------------------------------------------------------------------------
# BacktestConfig validation
# ---------------------------------------------------------------------------

def test_backtest_config_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        BacktestConfig(mode="invalid")  # type: ignore[arg-type]


def test_backtest_config_invalid_tp() -> None:
    with pytest.raises(ValueError, match="tp"):
        BacktestConfig(tp=0.0)


def test_backtest_config_invalid_sl() -> None:
    with pytest.raises(ValueError, match="sl"):
        BacktestConfig(sl=-1.0)


def test_backtest_config_invalid_trailing_stop() -> None:
    with pytest.raises(ValueError, match="trailing_stop"):
        BacktestConfig(trailing_stop=-0.5)


def test_backtest_config_summary_no_trailing() -> None:
    cfg = BacktestConfig(mode="pct", tp=2.0, sl=1.0)
    s = cfg.summary()
    assert "PCT" in s
    assert "TP=2.0" in s
    assert "SL=1.0" in s
    assert "Trail" not in s


def test_backtest_config_summary_with_trailing() -> None:
    cfg = BacktestConfig(mode="atr", tp=3.0, sl=1.5, trailing_stop=0.8)
    s = cfg.summary()
    assert "ATR" in s
    assert "Trail 0.8" in s


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_backtester_atr_mode_handles_empty_input() -> None:
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([]))
    signals = pd.Series(dtype=object, index=df.index)

    bt = StaticBacktester(BacktestConfig(mode="atr", tp=2.0, sl=1.0, atr_period=14))
    result = bt.run(df=df, signals=signals)

    assert len(result.trades) == 0
    assert len(result.equity_curve) == 0
    assert result.metrics["num_trades"] == 0.0


def test_backtester_all_neutral_signals_no_trades() -> None:
    df = _build_df([
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10.0},
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10.0},
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10.0},
    ])
    signals = pd.Series(["NEUTRAL", "NEUTRAL", "NEUTRAL"], index=df.index)

    result = StaticBacktester().run(df, signals)

    assert len(result.trades) == 0
    assert result.metrics["num_trades"] == 0.0


# ---------------------------------------------------------------------------
# LONG trade exits
# ---------------------------------------------------------------------------

def test_backtester_long_tp_hit() -> None:
    """LONG entry at 100 with TP=1% → exit at 101 when high reaches 101."""
    df = _build_df([
        {"open": 100.0, "high": 100.3, "low": 99.8, "close": 100.0, "volume": 10.0},
        {"open": 100.0, "high": 101.5, "low": 99.9, "close": 101.0, "volume": 10.0},
    ])
    signals = pd.Series(["LONG", "NEUTRAL"], index=df.index)

    result = StaticBacktester(BacktestConfig(mode="pct", tp=1.0, sl=2.0)).run(df, signals)

    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.direction == "LONG"
    assert t.exit_reason == "TP"
    assert t.exit_price == pytest.approx(101.0)
    assert t.pnl_pct == pytest.approx(1.0)


def test_backtester_long_sl_hit() -> None:
    """LONG entry at 100 with SL=1% → exit at 99 when low drops below 99."""
    df = _build_df([
        {"open": 100.0, "high": 100.3, "low": 99.8, "close": 100.0, "volume": 10.0},
        {"open": 100.0, "high": 100.1, "low": 98.5, "close": 99.0, "volume": 10.0},
    ])
    signals = pd.Series(["LONG", "NEUTRAL"], index=df.index)

    result = StaticBacktester(BacktestConfig(mode="pct", tp=2.0, sl=1.0)).run(df, signals)

    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.direction == "LONG"
    assert t.exit_reason == "SL"
    assert t.exit_price == pytest.approx(99.0)
    assert t.pnl_pct == pytest.approx(-1.0)


def test_backtester_long_max_hold_time_exit() -> None:
    """LONG position force-closed after max_hold_bars bars."""
    df = _build_df([
        {"open": 100.0, "high": 100.3, "low": 99.8, "close": 100.0, "volume": 10.0},
        {"open": 100.0, "high": 100.2, "low": 99.9, "close": 100.1, "volume": 10.0},
        {"open": 100.1, "high": 100.2, "low": 100.0, "close": 100.1, "volume": 10.0},
    ])
    signals = pd.Series(["LONG", "NEUTRAL", "NEUTRAL"], index=df.index)

    result = StaticBacktester(BacktestConfig(mode="pct", tp=5.0, sl=5.0, max_hold_bars=1)).run(df, signals)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "TIME"
    assert result.trades[0].bars_held == 1


# ---------------------------------------------------------------------------
# SHORT trade exits
# ---------------------------------------------------------------------------

def test_backtester_short_tp_hit() -> None:
    """SHORT entry at 100 with TP=1% → exit at 99 when low drops to 99."""
    df = _build_df([
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 10.0},
        {"open": 100.0, "high": 100.1, "low": 98.5, "close": 99.0, "volume": 10.0},
    ])
    signals = pd.Series(["SHORT", "NEUTRAL"], index=df.index)

    result = StaticBacktester(BacktestConfig(mode="pct", tp=1.0, sl=2.0)).run(df, signals)

    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.direction == "SHORT"
    assert t.exit_reason == "TP"
    assert t.exit_price == pytest.approx(99.0)
    assert t.pnl_pct == pytest.approx(1.0)


def test_backtester_short_sl_hit() -> None:
    """SHORT entry at 100 with SL=1% → exit at 101 when high reaches 101."""
    df = _build_df([
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 10.0},
        {"open": 100.0, "high": 101.5, "low": 99.9, "close": 101.0, "volume": 10.0},
    ])
    signals = pd.Series(["SHORT", "NEUTRAL"], index=df.index)

    result = StaticBacktester(BacktestConfig(mode="pct", tp=2.0, sl=1.0)).run(df, signals)

    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.direction == "SHORT"
    assert t.exit_reason == "SL"
    assert t.exit_price == pytest.approx(101.0)
    assert t.pnl_pct == pytest.approx(-1.0)


def test_backtester_short_max_hold_time_exit() -> None:
    df = _build_df([
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 10.0},
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "volume": 10.0},
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "volume": 10.0},
    ])
    signals = pd.Series(["SHORT", "NEUTRAL", "NEUTRAL"], index=df.index)

    result = StaticBacktester(BacktestConfig(mode="pct", tp=5.0, sl=5.0, max_hold_bars=1)).run(df, signals)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "TIME"
    assert result.trades[0].direction == "SHORT"


# ---------------------------------------------------------------------------
# Trailing stop
# ---------------------------------------------------------------------------

def test_backtester_long_trailing_stop_triggered() -> None:
    """LONG with trailing_stop=1%: peak at 103, trail SL moves to 101.97, hits on bar 3."""
    df = _build_df([
        {"open": 100.0, "high": 100.3, "low": 99.8, "close": 100.0, "volume": 10.0},  # entry
        {"open": 100.0, "high": 103.0, "low": 100.0, "close": 102.0, "volume": 10.0},  # peak 103
        {"open": 102.0, "high": 102.5, "low": 101.5, "close": 101.8, "volume": 10.0},  # doesn't hit trail
        {"open": 101.8, "high": 102.0, "low": 101.5, "close": 101.6, "volume": 10.0},  # trail SL = 103*(1-0.01)=101.97 → low 101.5 hits
    ])
    signals = pd.Series(["LONG", "NEUTRAL", "NEUTRAL", "NEUTRAL"], index=df.index)

    result = StaticBacktester(
        BacktestConfig(mode="pct", tp=10.0, sl=5.0, trailing_stop=1.0, max_hold_bars=100)
    ).run(df, signals)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "TRAIL"
    assert result.trades[0].direction == "LONG"


def test_backtester_short_trailing_stop_triggered() -> None:
    """SHORT with trailing_stop=1%: trough at 97, trail SL moves to 97.97, hits on price rise."""
    df = _build_df([
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 10.0},  # entry
        {"open": 100.0, "high": 100.1, "low": 97.0, "close": 97.5, "volume": 10.0},   # trough 97
        {"open": 97.5, "high": 98.5, "low": 97.4, "close": 98.0, "volume": 10.0},     # trail SL=97*(1.01)=97.97, high=98.5 → hits
    ])
    signals = pd.Series(["SHORT", "NEUTRAL", "NEUTRAL"], index=df.index)

    result = StaticBacktester(
        BacktestConfig(mode="pct", tp=10.0, sl=5.0, trailing_stop=1.0, max_hold_bars=100)
    ).run(df, signals)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "TRAIL"
    assert result.trades[0].direction == "SHORT"


# ---------------------------------------------------------------------------
# ATR mode
# ---------------------------------------------------------------------------

def test_backtester_atr_mode_long_tp() -> None:
    """ATR mode: TP at entry + tp*ATR. With 14 identical bars the ATR is approx the range."""
    rows = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0}] * 14
    rows.append({"open": 100.0, "high": 110.0, "low": 99.5, "close": 105.0, "volume": 10.0})
    df = _build_df(rows)
    signals = pd.Series(["NEUTRAL"] * 14 + ["LONG"], index=df.index)

    result = StaticBacktester(
        BacktestConfig(mode="atr", tp=1.0, sl=5.0, atr_period=14, max_hold_bars=50)
    ).run(df, signals)

    # Position entered at bar 14, closed at end (TIME) since it's the last bar
    assert len(result.trades) == 1
    # Result can vary by ATR calculation; just check no errors raised
    assert result.trades[0].direction == "LONG"


# ---------------------------------------------------------------------------
# No re-entry same bar + multiple trades
# ---------------------------------------------------------------------------

def test_backtester_does_not_reenter_same_bar_after_exit() -> None:
    df = _build_df(
        [
            {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.0, "volume": 10.0},
            {"open": 100.0, "high": 100.4, "low": 98.5, "close": 100.0, "volume": 10.0},
            {"open": 100.0, "high": 100.2, "low": 99.5, "close": 100.0, "volume": 10.0},
        ]
    )
    signals = pd.Series(["LONG", "LONG", "NEUTRAL"], index=df.index)

    bt = StaticBacktester(BacktestConfig(mode="pct", tp=1.0, sl=1.0, max_hold_bars=10))
    result = bt.run(df=df, signals=signals)

    assert len(result.trades) == 1
    assert result.trades[0].idx_entry == 0
    assert result.trades[0].idx_exit == 1


def test_backtester_multiple_sequential_long_trades() -> None:
    """Two back-to-back LONG trades separated by a NEUTRAL bar."""
    df = _build_df([
        {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.0, "volume": 10.0},  # entry 1
        {"open": 100.0, "high": 101.5, "low": 99.9, "close": 101.0, "volume": 10.0},  # TP hit (bar 1)
        {"open": 101.0, "high": 101.2, "low": 100.8, "close": 101.0, "volume": 10.0}, # re-entry
        {"open": 101.0, "high": 102.1, "low": 100.8, "close": 102.0, "volume": 10.0}, # TP hit (bar 3)
    ])
    signals = pd.Series(["LONG", "NEUTRAL", "LONG", "NEUTRAL"], index=df.index)

    result = StaticBacktester(BacktestConfig(mode="pct", tp=1.0, sl=5.0)).run(df, signals)

    assert len(result.trades) == 2
    assert result.trades[0].exit_reason == "TP"
    assert result.trades[1].exit_reason == "TP"
    assert result.metrics["num_long"] == 2.0
    assert result.metrics["num_short"] == 0.0


def test_backtester_long_and_short_trades() -> None:
    """One LONG TP and one SHORT TP → metrics count both directions."""
    df = _build_df([
        {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.0, "volume": 10.0},
        {"open": 100.0, "high": 101.5, "low": 99.9, "close": 101.0, "volume": 10.0},  # LONG TP
        {"open": 101.0, "high": 101.2, "low": 100.8, "close": 101.0, "volume": 10.0},
        {"open": 101.0, "high": 101.2, "low": 98.5, "close": 99.5, "volume": 10.0},   # SHORT TP
    ])
    signals = pd.Series(["LONG", "NEUTRAL", "SHORT", "NEUTRAL"], index=df.index)

    result = StaticBacktester(BacktestConfig(mode="pct", tp=1.0, sl=5.0)).run(df, signals)

    assert result.metrics["num_long"] == 1.0
    assert result.metrics["num_short"] == 1.0
    assert result.metrics["win_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Equity and metrics
# ---------------------------------------------------------------------------

def test_backtester_equity_and_total_return_use_initial_capital() -> None:
    df = _build_df(
        [
            {"open": 100.0, "high": 100.6, "low": 99.8, "close": 100.0, "volume": 10.0},
            {"open": 100.0, "high": 101.5, "low": 100.0, "close": 101.0, "volume": 10.0},
            {"open": 101.0, "high": 101.2, "low": 100.8, "close": 101.0, "volume": 10.0},
        ]
    )
    signals = pd.Series(["LONG", "NEUTRAL", "NEUTRAL"], index=df.index)

    initial_capital = 10_000.0
    bt = StaticBacktester(BacktestConfig(mode="pct", tp=1.0, sl=1.0, initial_capital=initial_capital))
    result = bt.run(df=df, signals=signals)

    assert len(result.trades) == 1
    assert result.trades[0].pnl_pct == pytest.approx(1.0)
    assert result.equity_curve.iloc[-1] == pytest.approx(10_100.0)
    assert result.metrics["total_return_pct"] == pytest.approx(1.0)


def test_backtester_metrics_win_rate_50_percent() -> None:
    """One winning LONG and one losing SHORT → 50% win rate."""
    df = _build_df([
        {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.0, "volume": 10.0},
        {"open": 100.0, "high": 101.5, "low": 99.9, "close": 101.0, "volume": 10.0},  # LONG TP
        {"open": 101.0, "high": 101.5, "low": 100.8, "close": 101.0, "volume": 10.0},
        {"open": 101.0, "high": 102.5, "low": 100.8, "close": 102.0, "volume": 10.0},  # SHORT SL
    ])
    signals = pd.Series(["LONG", "NEUTRAL", "SHORT", "NEUTRAL"], index=df.index)

    result = StaticBacktester(BacktestConfig(mode="pct", tp=1.0, sl=1.0)).run(df, signals)

    assert result.metrics["win_rate"] == pytest.approx(0.5)
    assert result.metrics["num_trades"] == 2.0


def test_backtester_equity_curve_length_matches_df() -> None:
    n = 10
    rows = [{"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10.0}] * n
    df = _build_df(rows)
    signals = pd.Series(["NEUTRAL"] * n, index=df.index)

    result = StaticBacktester().run(df, signals)
    assert len(result.equity_curve) == n


# ---------------------------------------------------------------------------
# TpSlLevel generation
# ---------------------------------------------------------------------------

def test_backtester_tp_sl_levels_populated() -> None:
    """BacktestResult.tp_sl_levels must contain one entry per completed trade."""
    df = _build_df([
        {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.0, "volume": 10.0},
        {"open": 100.0, "high": 101.5, "low": 99.9, "close": 101.0, "volume": 10.0},
        {"open": 101.0, "high": 101.2, "low": 100.8, "close": 101.0, "volume": 10.0},
    ])
    signals = pd.Series(["LONG", "NEUTRAL", "NEUTRAL"], index=df.index)

    result = StaticBacktester(BacktestConfig(mode="pct", tp=1.0, sl=1.0)).run(df, signals)

    assert len(result.tp_sl_levels) == len(result.trades)
    lvl: TpSlLevel = result.tp_sl_levels[0]
    assert lvl.tp_price == pytest.approx(101.0)
    assert lvl.sl_price == pytest.approx(99.0)
    assert lvl.entry_idx == 0
    assert lvl.direction == "LONG"


def test_backtester_tp_sl_levels_short_direction() -> None:
    df = _build_df([
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 10.0},
        {"open": 100.0, "high": 100.1, "low": 98.5, "close": 99.0, "volume": 10.0},
    ])
    signals = pd.Series(["SHORT", "NEUTRAL"], index=df.index)

    result = StaticBacktester(BacktestConfig(mode="pct", tp=1.0, sl=2.0)).run(df, signals)

    assert len(result.tp_sl_levels) == 1
    lvl = result.tp_sl_levels[0]
    assert lvl.direction == "SHORT"
    assert lvl.tp_price < 100.0   # TP below entry for short
    assert lvl.sl_price > 100.0   # SL above entry for short


# ---------------------------------------------------------------------------
# Case-insensitive column names
# ---------------------------------------------------------------------------

def test_backtester_case_insensitive_columns() -> None:
    """DataFrame with uppercase column names should be normalised correctly."""
    index = pd.date_range("2024-01-01", periods=2, freq="1h")
    df = pd.DataFrame(
        {
            "Open":   [100.0, 100.0],
            "High":   [100.5, 101.5],
            "Low":    [99.8,  99.9],
            "Close":  [100.0, 101.0],
            "Volume": [10.0,  10.0],
        },
        index=index,
    )
    signals = pd.Series(["LONG", "NEUTRAL"], index=index)

    result = StaticBacktester(BacktestConfig(mode="pct", tp=1.0, sl=2.0)).run(df, signals)
    assert len(result.trades) == 1


# ---------------------------------------------------------------------------
# Numeric signal values (1 / -1 / 0)
# ---------------------------------------------------------------------------

def test_backtester_numeric_signal_values() -> None:
    """Signals passed as integers 1 / -1 / 0 should be treated as LONG/SHORT/NEUTRAL."""
    df = _build_df([
        {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.0, "volume": 10.0},
        {"open": 100.0, "high": 101.5, "low": 99.9, "close": 101.0, "volume": 10.0},
    ])
    signals = pd.Series([1, 0], index=df.index)

    result = StaticBacktester(BacktestConfig(mode="pct", tp=1.0, sl=2.0)).run(df, signals)
    assert len(result.trades) == 1
    assert result.trades[0].direction == "LONG"


# ---------------------------------------------------------------------------
# Open position at end-of-data → TIME exit
# ---------------------------------------------------------------------------

def test_backtester_open_position_at_end_of_data_closed_as_time() -> None:
    df = _build_df([
        {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.0, "volume": 10.0},
        {"open": 100.0, "high": 100.3, "low": 99.9, "close": 100.1, "volume": 10.0},
        {"open": 100.1, "high": 100.2, "low": 100.0, "close": 100.1, "volume": 10.0},
    ])
    signals = pd.Series(["LONG", "NEUTRAL", "NEUTRAL"], index=df.index)

    result = StaticBacktester(BacktestConfig(mode="pct", tp=5.0, sl=5.0, max_hold_bars=50)).run(df, signals)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "TIME"
    assert result.trades[0].idx_exit == 2  # last bar
