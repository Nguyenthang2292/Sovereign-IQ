import os
import sys

import numpy as np

# Add project root to path to import the module
sys.path.append(os.getcwd())

# We need to import the rust extension.
# It seems it is installed as a package 'xgboost_rust' or inside 'modules.xgboost_LTS.rust_extensions'
try:
    from modules.xgboost_LTS.rust_extensions import xgboost_rust
except ImportError:
    import xgboost_rust


def test_rolling_quantile():
    data = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 9.0], dtype=np.float64)
    window = 3
    q = 0.5

    # Expected:
    # 0: [1] -> NaN (window=3)
    # 1: [1, 5] -> NaN
    # 2: [1, 5, 2] -> sort [1, 2, 5] -> 2.0
    # 3: [5, 2, 8] -> sort [2, 5, 8] -> 5.0
    # 4: [2, 8, 3] -> sort [2, 3, 8] -> 3.0
    # 5: [8, 3, 9] -> sort [3, 8, 9] -> 8.0

    result = xgboost_rust.rolling_quantile_rust(data, window, q)
    print("Input:", data)
    print("Result:", result)

    expected = np.array([np.nan, np.nan, 2.0, 5.0, 3.0, 8.0])

    np.testing.assert_allclose(result[2:], expected[2:])
    print("Test Passed!")


if __name__ == "__main__":
    test_rolling_quantile()
