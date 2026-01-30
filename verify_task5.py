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


def test_rolling_std_skew():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    window = 3

    # STD
    # 0: [1] -> NaN
    # 1: [1, 2] -> NaN
    # 2: [1, 2, 3] -> std([1,2,3], ddof=1) = 1.0
    # 3: [2, 3, 4] -> std([2,3,4], ddof=1) = 1.0
    # 4: [3, 4, 5] -> std([3,4,5], ddof=1) = 1.0

    std_result = xgboost_rust.rolling_std_rust(data, window)
    print("Input:", data)
    print("Std Result:", std_result)

    expected_std = np.array([np.nan, np.nan, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(std_result[2:], expected_std[2:])

    # Skew
    # [1, 2, 3] -> symmetric -> 0.0

    skew_result = xgboost_rust.rolling_skew_rust(data, window)
    print("Skew Result:", skew_result)

    expected_skew = np.array([np.nan, np.nan, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(skew_result[2:], expected_skew[2:], atol=1e-10)

    print("Test Passed!")


if __name__ == "__main__":
    test_rolling_std_skew()
