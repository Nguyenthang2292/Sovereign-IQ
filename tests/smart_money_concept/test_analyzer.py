import pandas as pd
import numpy as np
from modules.smart_money_concept.core.analyzer import SMCAnalyzer, SMCState, _last_break_direction
from modules.smart_money_concept.core.bos import BosChochResult
from modules.smart_money_concept.core.trend import BEARISH

def test_analyzer_run():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=300, tz="UTC")
    df = pd.DataFrame({
        "Date": dates,
        "Open": np.random.rand(300) * 100,
        "High": np.random.rand(300) * 100,
        "Low": np.random.rand(300) * 100,
        "Close": np.random.rand(300) * 100
    })
    
    analyzer = SMCAnalyzer()
    state = analyzer.run(df)
    
    # Assert return type
    assert isinstance(state, SMCState)
    
    # Assert fields presence
    assert hasattr(state, "ohlcv")
    assert hasattr(state, "swings")
    assert hasattr(state, "trend")
    assert hasattr(state, "internal_structure")
    assert hasattr(state, "swing_structure")
    assert hasattr(state, "equal_hl")
    assert hasattr(state, "ob_internal")
    assert hasattr(state, "ob_swing")
    
    # Assert export()
    exported = analyzer.export(df)
    assert isinstance(exported, tuple)
    assert len(exported) == 15


def test_last_break_direction_uses_latest_crossing_time():
    structure = BosChochResult(
        bullish_bos=pd.DataFrame(
            [
                {
                    "Pivot_level": 10.0,
                    "Pivot_bullishBos_Time": pd.Timestamp("2023-01-01", tz="UTC"),
                    "Crossing_Time": pd.Timestamp("2023-01-03", tz="UTC"),
                    "event_type": "BOS",
                }
            ]
        ),
        bearish_bos=pd.DataFrame(
            [
                {
                    "Pivot_level": 7.0,
                    "Pivot_bearishBos_Time": pd.Timestamp("2023-01-02", tz="UTC"),
                    "Crossing_Time": pd.Timestamp("2023-01-04", tz="UTC"),
                    "event_type": "BOS",
                }
            ]
        ),
        bullish_choch=pd.DataFrame(),
        bearish_choch=pd.DataFrame(),
    )

    assert _last_break_direction(structure) == BEARISH
