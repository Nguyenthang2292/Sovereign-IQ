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


def test_rolling_mean():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    window = 3

    # Expected:
    # 0: [1] -> NaN
    # 1: [1, 2] -> NaN
    # 2: [1, 2, 3] -> 2.0
    # 3: [2, 3, 4] -> 3.0
    # 4: [3, 4, 5] -> 4.0

    result = xgboost_rust.rolling_mean_rust(data, window)
    print("Input:", data)
    print("Result:", result)

    expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0])

    np.testing.assert_allclose(result[2:], expected[2:])
    print("Test Passed!")


if __name__ == "__main__":
    test_rolling_mean()
