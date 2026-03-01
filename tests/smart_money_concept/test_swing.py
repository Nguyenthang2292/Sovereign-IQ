import pandas as pd
import numpy as np
from modules.smart_money_concept.core.swing import detect_swings, classify_swing_types
from modules.smart_money_concept.core.analyzer import SMCAnalyzer
from modules.smart_money_concept.models.pivot import Pivot

def test_detect_swings():
    # Setup mock dataframe with alternating highs and lows
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100)
    # Create sine wave pattern
    highs = np.sin(np.linspace(0, 10*np.pi, 100)) * 10 + 100
    lows = highs - 5
    df = pd.DataFrame({
        "High": highs,
        "Low": lows,
    }, index=dates)

    result = detect_swings(df, internal_order=2, external_order=4)
    
    # Assert return type and fields
    assert hasattr(result, "internal_highs")
    assert hasattr(result, "internal_lows")
    assert hasattr(result, "swing_highs")
    assert hasattr(result, "swing_lows")
    
    # Assert some swings were detected
    assert len(result.internal_highs) > 0
    assert len(result.internal_lows) > 0

def test_classify_swing_types():
    highs = [Pivot(10, pd.Timestamp("2023-01-01")), 
             Pivot(20, pd.Timestamp("2023-01-02")), 
             Pivot(15, pd.Timestamp("2023-01-03"))]
    lows = [Pivot(5, pd.Timestamp("2023-01-01")), 
            Pivot(10, pd.Timestamp("2023-01-02")), 
            Pivot(2, pd.Timestamp("2023-01-03"))]
            
    classified_highs, classified_lows = classify_swing_types(highs, lows)
    
    assert len(classified_highs) == 3
    assert len(classified_lows) == 3
    
    # Middle is HH
    assert classified_highs[1][1] == "HH"
    assert classified_highs[0][1] == "HL"
    
    assert classified_lows[1][1] == "LH"
    assert classified_lows[0][1] == "LL"


def test_analyzer_default_external_order_is_50():
    analyzer = SMCAnalyzer()
    assert analyzer.external_order == 50
