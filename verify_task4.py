import os
import sys

import numpy as np

# Add project root to path to import the module
sys.path.append(os.getcwd())

# We need to import the rust extension.
try:
    from modules.xgboost_LTS.rust_extensions import xgboost_rust
except ImportError:
    import xgboost_rust


def test_add_price_derived_features():
    # open, high, low, close, volume
    n = 6
    open_p = np.array([100.0, 102.0, 104.0, 103.0, 105.0, 106.0], dtype=np.float64)
    high_p = np.array([101.0, 104.0, 105.0, 105.0, 107.0, 108.0], dtype=np.float64)
    low_p = np.array([99.0, 101.0, 103.0, 102.0, 104.0, 105.0], dtype=np.float64)
    close_p = np.array([100.0, 103.0, 104.0, 104.0, 106.0, 107.0], dtype=np.float64)
    volume_p = np.array([1000.0, 1100.0, 1200.0, 1150.0, 1300.0, 1400.0], dtype=np.float64)

    # Run rust function
    features = xgboost_rust.add_price_derived_features_rust(open_p, high_p, low_p, close_p, volume_p)

    # Expected keys
    expected_keys = ["returns_1", "returns_5", "log_volume", "high_low_range", "close_open_diff"]
    for key in expected_keys:
        assert key in features, f"Missing key {key}"
        print(f"Found {key}: {features[key]}")

    # Verify values (roughly)
    # returns_1[1] = (103-100)/100 = 0.03
    np.testing.assert_almost_equal(features["returns_1"][1], 0.03)

    # returns_5[5] = (107-100)/100 = 0.07
    np.testing.assert_almost_equal(features["returns_5"][5], 0.07)

    # log_volume[0] = ln(1001)
    np.testing.assert_almost_equal(features["log_volume"][0], np.log(1001.0))

    print("Test Passed!")


if __name__ == "__main__":
    test_add_price_derived_features()
