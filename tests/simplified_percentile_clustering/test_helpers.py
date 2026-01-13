"""
Tests for helper utility functions.
"""

import numpy as np
import pandas as pd
import pytest

from modules.simplified_percentile_clustering.utils.helpers import (
    normalize_cluster_name,
    safe_isfinite,
    safe_isna,
    vectorized_min_and_second_min,
    vectorized_min_distance,
)


def test_safe_isna_with_scalar():
    """Test safe_isna with scalar values."""
    assert not safe_isna(1.0)
    assert not safe_isna(0.0)
    assert safe_isna(np.nan)
    assert safe_isna(pd.NA)
    # Note: pd.isna(None) returns True, which is the expected behavior
    assert safe_isna(None)


def test_safe_isna_with_series():
    """Test safe_isna with pandas Series."""
    series = pd.Series([1.0, 2.0, np.nan, 4.0])
    result = safe_isna(series)

    assert isinstance(result, pd.Series)
    assert not result.iloc[0]
    assert not result.iloc[1]
    assert result.iloc[2]
    assert not result.iloc[3]


def test_safe_isna_with_array():
    """Test safe_isna with numpy array."""
    arr = np.array([1.0, 2.0, np.nan, 4.0])
    result = safe_isna(arr)

    assert isinstance(result, np.ndarray) or isinstance(result, pd.Series)
    assert not result[0]
    assert not result[1]
    assert result[2]
    assert not result[3]


def test_safe_isfinite_with_scalar():
    """Test safe_isfinite with scalar values."""
    assert safe_isfinite(1.0)
    assert safe_isfinite(0.0)
    assert not safe_isfinite(np.nan)
    assert not safe_isfinite(np.inf)
    assert not safe_isfinite(-np.inf)


def test_safe_isfinite_with_series():
    """Test safe_isfinite with pandas Series."""
    series = pd.Series([1.0, 2.0, np.nan, np.inf, 4.0])
    result = safe_isfinite(series)

    assert isinstance(result, pd.Series)
    assert result.iloc[0]
    assert result.iloc[1]
    assert not result.iloc[2]
    assert not result.iloc[3]
    assert result.iloc[4]


def test_vectorized_min_distance_basic():
    """Test vectorized_min_distance with basic data."""
    feature_val = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    centers = pd.DataFrame(
        {
            "k0": [0.5, 1.5, 2.5, 3.5, 4.5],
            "k1": [5.5, 6.5, 7.5, 8.5, 9.5],
        }
    )

    result = vectorized_min_distance(feature_val, centers)

    assert isinstance(result, pd.Series)
    assert len(result) == len(feature_val)
    # First value (1.0) is closer to k0 (0.5) than k1 (5.5)
    assert result.iloc[0] == pytest.approx(0.5, abs=0.1)
    # Last value (5.0) is closer to k1 (4.5) than k0 (3.5)
    assert result.iloc[-1] == pytest.approx(0.5, abs=0.1)


def test_vectorized_min_distance_with_nan():
    """Test vectorized_min_distance with NaN values."""
    feature_val = pd.Series([1.0, np.nan, 3.0, 4.0])
    centers = pd.DataFrame(
        {
            "k0": [0.5, 1.5, 2.5, 3.5],
            "k1": [5.5, 6.5, 7.5, 8.5],
        }
    )

    result = vectorized_min_distance(feature_val, centers)

    assert isinstance(result, pd.Series)
    assert pd.isna(result.iloc[1])  # NaN input should produce NaN output


def test_vectorized_min_distance_empty():
    """Test vectorized_min_distance with empty data."""
    feature_val = pd.Series(dtype=float)
    centers = pd.DataFrame(columns=["k0", "k1"])

    result = vectorized_min_distance(feature_val, centers)

    assert isinstance(result, pd.Series)
    assert len(result) == 0


def test_vectorized_min_distance_different_indices():
    """Test vectorized_min_distance with different indices (should align)."""
    feature_val = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    centers = pd.DataFrame(
        {
            "k0": [0.5, 1.5, 2.5],
            "k1": [5.5, 6.5, 7.5],
        },
        index=[0, 1, 2],
    )

    result = vectorized_min_distance(feature_val, centers)

    assert isinstance(result, pd.Series)
    assert len(result) == 3
    assert result.index.equals(feature_val.index)


def test_vectorized_min_and_second_min_basic():
    """Test vectorized_min_and_second_min with basic data."""
    dist_array = np.array(
        [
            [1.0, 2.0, 3.0],  # Min: 1.0 (k0), Second: 2.0 (k1)
            [3.0, 1.0, 2.0],  # Min: 1.0 (k1), Second: 2.0 (k2)
            [2.0, 3.0, 1.0],  # Min: 1.0 (k2), Second: 2.0 (k0)
        ]
    )

    min_dist, second_min_dist, cluster_val, second_cluster_val = vectorized_min_and_second_min(dist_array)

    assert len(min_dist) == 3
    assert len(second_min_dist) == 3
    assert len(cluster_val) == 3
    assert len(second_cluster_val) == 3

    # First row
    assert min_dist[0] == pytest.approx(1.0)
    assert second_min_dist[0] == pytest.approx(2.0)
    assert cluster_val[0] == 0  # k0
    assert second_cluster_val[0] == 1  # k1

    # Second row
    assert min_dist[1] == pytest.approx(1.0)
    assert second_min_dist[1] == pytest.approx(2.0)
    assert cluster_val[1] == 1  # k1
    assert second_cluster_val[1] == 2  # k2

    # Third row
    assert min_dist[2] == pytest.approx(1.0)
    assert second_min_dist[2] == pytest.approx(2.0)
    assert cluster_val[2] == 2  # k2
    assert second_cluster_val[2] == 0  # k0


def test_vectorized_min_and_second_min_with_nan():
    """Test vectorized_min_and_second_min with NaN values."""
    dist_array = np.array(
        [
            [1.0, np.nan, 3.0],  # Min: 1.0 (k0), Second: 3.0 (k2)
            [np.nan, np.nan, 2.0],  # Only one valid: 2.0 (k2), no second
            [1.0, 2.0, np.nan],  # Min: 1.0 (k0), Second: 2.0 (k1)
        ]
    )

    min_dist, second_min_dist, cluster_val, second_cluster_val = vectorized_min_and_second_min(dist_array)

    # First row
    assert min_dist[0] == pytest.approx(1.0)
    assert second_min_dist[0] == pytest.approx(3.0)
    assert cluster_val[0] == 0
    assert second_cluster_val[0] == 2

    # Second row - only one valid value
    assert min_dist[1] == pytest.approx(2.0)
    assert pd.isna(second_min_dist[1])
    assert cluster_val[1] == 2
    assert pd.isna(second_cluster_val[1])


def test_vectorized_min_and_second_min_all_nan():
    """Test vectorized_min_and_second_min with all NaN row."""
    dist_array = np.array(
        [
            [1.0, 2.0, 3.0],
            [np.nan, np.nan, np.nan],  # All NaN
            [2.0, 1.0, 3.0],
        ]
    )

    min_dist, second_min_dist, cluster_val, second_cluster_val = vectorized_min_and_second_min(dist_array)

    # Second row should have NaN results
    assert pd.isna(min_dist[1])
    assert pd.isna(second_min_dist[1])
    assert pd.isna(cluster_val[1])
    assert pd.isna(second_cluster_val[1])


def test_vectorized_min_and_second_min_single_center():
    """Test vectorized_min_and_second_min with single center (k=1)."""
    dist_array = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    min_dist, second_min_dist, cluster_val, second_cluster_val = vectorized_min_and_second_min(dist_array)

    # Should have min but no second min
    assert min_dist[0] == pytest.approx(1.0)
    assert pd.isna(second_min_dist[0])
    assert cluster_val[0] == 0
    assert pd.isna(second_cluster_val[0])


def test_normalize_cluster_name_valid():
    """Test normalize_cluster_name with valid values."""
    assert normalize_cluster_name(0.0) == "k0"
    assert normalize_cluster_name(1.0) == "k1"
    assert normalize_cluster_name(2.0) == "k2"
    assert normalize_cluster_name(0) == "k0"
    assert normalize_cluster_name(1) == "k1"
    assert normalize_cluster_name(2) == "k2"


def test_normalize_cluster_name_nan():
    """Test normalize_cluster_name with NaN."""
    assert normalize_cluster_name(np.nan) is None
    assert normalize_cluster_name(pd.NA) is None


def test_normalize_cluster_name_invalid():
    """Test normalize_cluster_name with invalid values."""
    assert normalize_cluster_name(3.0) is None
    assert normalize_cluster_name(-1.0) is None
    # Note: int(1.5) = 1, so 1.5 will be converted to "k1"
    # This is expected behavior - function converts float to int
    assert normalize_cluster_name(1.5) == "k1"  # int(1.5) = 1


def test_normalize_cluster_name_with_series():
    """Test normalize_cluster_name can handle Series values."""
    # Function should work with scalar values from Series
    series = pd.Series([0.0, 1.0, 2.0, np.nan])
    results = [normalize_cluster_name(val) for val in series]

    assert results[0] == "k0"
    assert results[1] == "k1"
    assert results[2] == "k2"
    assert results[3] is None
