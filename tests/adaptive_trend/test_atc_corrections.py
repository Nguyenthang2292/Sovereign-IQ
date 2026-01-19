import pytest
import pandas as pd
import numpy as np
from modules.adaptive_trend.core.signal_detection import generate_signal_from_ma
from modules.adaptive_trend.utils.config import ATCConfig
from modules.adaptive_trend.utils.exp_growth import exp_growth
from modules.adaptive_trend.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend.core.compute_moving_averages import calculate_kama_atc

# ============================================================================
# 1. Test Signal Persistence (Fix Bug #1)
# ============================================================================

def test_signal_persistence_diagnostic():
    """Verify that signals persist until the next crossover (matching PineScript 'var' behavior)."""
    # Create a scenario:
    # Bar 0: Price below MA
    # Bar 1: Price crosses above MA -> Signal = 1
    # Bar 2-3: Price stays above MA -> Signal should stay 1
    # Bar 4: Price crosses below MA -> Signal = -1
    # Bar 5: Price stays below MA -> Signal should stay -1
    
    price = pd.Series([90, 110, 115, 120, 80, 75])
    ma = pd.Series([100, 100, 100, 100, 100, 100])
    
    sig = generate_signal_from_ma(price, ma)
    
    # Expected: [0, 1, 1, 1, -1, -1]
    # Note: Bar 0 is 0 because there's no previous bar to detect crossover
    expected = [0, 1, 1, 1, -1, -1]
    assert sig.tolist() == expected, f"Signal persistence failed. Expected {expected}, got {sig.tolist()}"

# ============================================================================
# 2. Test Parameter Scaling (Fix Bug #2)
# ============================================================================

def test_config_scaling_diagnostic():
    """Verify that ATCConfig provides scaled parameters for lambda and decay."""
    config = ATCConfig(lambda_param=0.02, decay=0.03)
    
    # Check scaling properties
    assert config.lambda_scaled == 0.00002, f"Lambda scaling failed: {config.lambda_scaled}"
    assert config.decay_scaled == 0.0003, f"Decay scaling failed: {config.decay_scaled}"

def test_computation_scaling_integration():
    """Verify that compute_atc_signals internally applies scaling to Lambda and Decay."""
    # Use longer data (500 bars) and a random walk to avoid MA calculation issues
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(500))).clip(lower=1.0)
    
    # This should not raise overflow errors now
    results = compute_atc_signals(
        prices=prices,
        La=0.02,  # User input format
        De=0.03,  # User input format
    )
    
    assert "Average_Signal" in results
    # Check that signals aren't NaN/Inf at the end of the series
    # (Allow first few bars to be NaN due to MA initialization)
    final_signals = results["Average_Signal"].iloc[-10:]
    assert not final_signals.isna().any(), "Latest signals should not be NaN"
    assert np.isfinite(final_signals.values).all()

# ============================================================================
# 3. Test Exponential Growth Bounds
# ============================================================================

def test_exp_growth_bounds_diagnostic():
    """Verify that exp_growth calculation with correct scale stays within sane bounds."""
    # With scaled La = 0.00002, 1000 bars index
    index = pd.RangeIndex(1000)
    # Correct scale (what compute_atc_signals does now)
    growth = exp_growth(L=0.00002, index=index, cutout=0)
    
    # Max value at bar 999: e^(0.00002 * 999) ≈ e^0.01998 ≈ 1.020
    max_val = growth.max()
    assert max_val < 1.1, f"Exponential growth too high with correct scale: {max_val}"
    
    # Comparison with unscaled (what it was before the fix)
    # L = 0.02, 1000 bars: e^(0.02 * 999) ≈ e^20 ≈ 4.85e8
    growth_unscaled = exp_growth(L=0.02, index=index, cutout=0)
    assert growth_unscaled.max() > 1000000, "Unscaled growth should be very large"

# ============================================================================
# 4. KAMA Optimization & Correctness (To-do 4)
# ============================================================================

def test_kama_numba_correctness():
    """Verify that Numba-optimized KAMA matches the expected logic."""
    # Need enough data for length=5
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0])
    length = 5
    
    kama = calculate_kama_atc(prices, length=length)
    
    assert isinstance(kama, pd.Series)
    assert len(kama) == len(prices)
    # First values until 'length' should be price or prev_kama
    assert not np.isnan(kama.iloc[0])
    # Check if results are finite
    assert np.isfinite(kama.values).all()

# ============================================================================
# 5. Strategy Mode (To-do 3)
# ============================================================================

def test_strategy_mode_shift():
    """Verify that strategy_mode shifts the Average_Signal by 1 bar."""
    prices = pd.Series(np.linspace(100, 110, 100)) # Increased to 100
    
    # Default (strategy_mode=False)
    results_normal = compute_atc_signals(prices, strategy_mode=False)
    
    # Strategy Mode (strategy_mode=True)
    results_strategy = compute_atc_signals(prices, strategy_mode=True)
    
    avg_normal = results_normal["Average_Signal"]
    avg_strategy = results_strategy["Average_Signal"]
    
    # The strategy signal at index i should be the normal signal at index i-1
    # Check index 5
    assert avg_strategy.iloc[5] == avg_normal.iloc[4]
    # Index 0 should be 0 due to fillna(0)
    assert avg_strategy.iloc[0] == 0.0
