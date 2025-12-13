import numpy as np
import pandas as pd

from main.main_hmm import _compute_std_targets, _print_summary
from modules.hmm.signal_resolution import LONG, HOLD, SHORT


def _sample_close_dataframe(length: int = 60) -> pd.DataFrame:
    """Helper tạo chuỗi giá đơn giản cho các phép tính rolling."""
    idx = pd.date_range("2024-01-01", periods=length, freq="h")
    prices = np.linspace(100.0, 110.0, length) + np.random.default_rng(0).normal(
        0, 0.1, length
    )
    return pd.DataFrame({"close": prices}, index=idx)


def test_compute_std_targets_returns_expected_levels():
    df = _sample_close_dataframe()

    targets = _compute_std_targets(df, window=50)

    assert targets is not None
    expected_keys = {
        "window", "basis", "std",
        "bearish_1σ", "bearish_2σ", "bearish_3σ",
        "bullish_1σ", "bullish_2σ", "bullish_3σ"
    }
    assert set(targets.keys()) == expected_keys

    closes = df["close"].tail(50)
    expected_mean = closes.mean()
    expected_std = closes.std(ddof=0)

    assert targets["window"] == 50
    assert np.isclose(targets["basis"], expected_mean)
    assert np.isclose(targets["std"], expected_std)
    assert np.isclose(targets["bearish_1σ"], expected_mean - expected_std)
    assert np.isclose(targets["bearish_3σ"], expected_mean - 3 * expected_std)
    assert np.isclose(targets["bullish_1σ"], expected_mean + expected_std)
    assert np.isclose(targets["bullish_3σ"], expected_mean + 3 * expected_std)


def test_print_summary_includes_std_targets(capsys):
    signal_high_order = LONG
    signal_kama = SHORT
    std_targets = {
        "window": 50,
        "basis": 105.0,
        "std": 2.0,
        "bearish_1σ": 103.0,
        "bearish_2σ": 101.0,
        "bearish_3σ": 99.0,
        "bullish_1σ": 107.0,
        "bullish_2σ": 109.0,
        "bullish_3σ": 111.0,
    }

    _print_summary("BTC/USDT", "binance", signal_high_order, signal_kama, std_targets)
    captured = capsys.readouterr().out

    assert "HMM SIGNAL ANALYSIS" in captured
    assert "High-Order HMM Signal" in captured
    assert "HMM-KAMA Signal" in captured
    assert "Combined Recommendation" in captured
    # Should show both bullish and bearish targets due to conflict (LONG vs SHORT)
    assert "Price Targets" in captured
    assert "Mean: 105.00" in captured
    assert "Bullish:" in captured
    assert "Bearish:" in captured


def test_print_summary_shows_bullish_targets_for_long(capsys):
    """Test that LONG signal shows bullish targets."""
    signal_high_order = LONG
    signal_kama = LONG
    std_targets = {
        "window": 50,
        "basis": 105.0,
        "std": 2.0,
        "bearish_1σ": 103.0,
        "bearish_2σ": 101.0,
        "bearish_3σ": 99.0,
        "bullish_1σ": 107.0,
        "bullish_2σ": 109.0,
        "bullish_3σ": 111.0,
    }

    _print_summary("BTC/USDT", "binance", signal_high_order, signal_kama, std_targets)
    captured = capsys.readouterr().out

    assert "Bullish Targets" in captured
    assert "+1σ 107.00" in captured
    assert "+2σ 109.00" in captured
    assert "+3σ 111.00" in captured


def test_print_summary_shows_bearish_targets_for_short(capsys):
    """Test that SHORT signal shows bearish targets."""
    signal_high_order = SHORT
    signal_kama = SHORT
    std_targets = {
        "window": 50,
        "basis": 105.0,
        "std": 2.0,
        "bearish_1σ": 103.0,
        "bearish_2σ": 101.0,
        "bearish_3σ": 99.0,
        "bullish_1σ": 107.0,
        "bullish_2σ": 109.0,
        "bullish_3σ": 111.0,
    }

    _print_summary("BTC/USDT", "binance", signal_high_order, signal_kama, std_targets)
    captured = capsys.readouterr().out

    assert "Bearish Targets" in captured
    assert "-1σ 103.00" in captured
    assert "-2σ 101.00" in captured
    assert "-3σ 99.00" in captured

