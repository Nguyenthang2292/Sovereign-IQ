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


def test_add_advanced_features():
    n = 30
    close_p = np.linspace(100, 130, n)
    volume_p = np.ones(n) * 1000
    returns_1_p = np.diff(close_p, prepend=close_p[0]) / close_p

    # Just basic test, no optional args for now
    features = xgboost_rust.add_advanced_features_rust(close_p, volume_p, returns_1_p, None, None, None, None, None)

    print("Keys found:", features.keys())

    # Check ROC_3
    # roc_3[3] = (close[3]-close[0])/close[0]
    expected_roc3 = (close_p[3] - close_p[0]) / close_p[0]
    found_roc3 = features["roc_3"][3]
    np.testing.assert_almost_equal(found_roc3, expected_roc3)

    # Check rolling_std_10
    # Window 10
    # idx 9 -> returns_1[0:10] (first element 0 is 0.0)
    # just check it exists and is not nan at index 9
    assert not np.isnan(features["rolling_std_10"][9])

    # Check lag
    # returns_1_lag_1[1] = returns_1[0]
    np.testing.assert_almost_equal(features["returns_1_lag_1"][1], returns_1_p[0])

    print("Test Passed!")


if __name__ == "__main__":
    test_add_advanced_features()
