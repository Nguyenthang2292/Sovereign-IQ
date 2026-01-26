"""
Tests for core ATC LTS functionality (migrated from adaptive_trend_enhance).
"""

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS.core.process_layer1 import cut_signal, trend_sign, weighted_signal
from modules.common.system import get_hardware_manager, get_memory_manager


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    n = 200
    prices = pd.Series(
        100 * (1 + np.random.randn(n).cumsum() * 0.01), index=pd.date_range("2023-01-01", periods=n, freq="H")
    )
    return prices


def test_weighted_signal_basic(sample_data):
    """Test basic functionality of weighted_signal."""
    s1 = pd.Series(1.0, index=sample_data.index)
    s2 = pd.Series(-1.0, index=sample_data.index)
    w1 = pd.Series(1.0, index=sample_data.index)
    w2 = pd.Series(1.0, index=sample_data.index)

    result = weighted_signal([s1, s2], [w1, w2])

    # (1*1 + -1*1) / (1 + 1) = 0
    assert result.iloc[0] == 0.0
    assert len(result) == len(sample_data)


def test_cut_signal(sample_data):
    """Test cut_signal discretization."""
    s = pd.Series([0.6, 0.4, 0.0, -0.4, -0.6], index=pd.date_range("2023-01-01", periods=5, freq="H"))
    result = cut_signal(s, threshold=0.49)

    assert result.iloc[0] == 1
    assert result.iloc[1] == 0
    assert result.iloc[2] == 0
    assert result.iloc[3] == 0
    assert result.iloc[4] == -1


def test_trend_sign():
    """Test trend_sign direction."""
    s = pd.Series([1.0, 0.5, 0.0, -0.5, -1.0])
    result = trend_sign(s)

    assert result.iloc[0] == 1
    assert result.iloc[2] == 0
    assert result.iloc[4] == -1


def test_compute_atc_signals_basic(sample_data):
    """Test full ATC signal computation."""
    results = compute_atc_signals(sample_data)

    assert "Average_Signal" in results
    assert len(results["Average_Signal"]) == len(sample_data)
    assert not results["Average_Signal"].isna().all()


def test_hardware_manager_singleton():
    """Test HardwareManager singleton access."""
    hw1 = get_hardware_manager()
    hw2 = get_hardware_manager()
    assert hw1 is hw2


def test_memory_manager_tracking(sample_data):
    """Test MemoryManager tracking functionality."""
    mem = get_memory_manager()
    with mem.track_memory("test_op"):
        _ = compute_atc_signals(sample_data)

    stats = mem.get_current_usage()
    assert stats.ram_percent >= 0
