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


def test_apply_directional_labels():
    # close prices: [100, 102, 101, 105, 100]
    # target_horizon = 2
    # base_threshold = 0.02

    # 0: close[0]=100, future[2]=101 -> 1% -> NEUTRAL (1)
    # 1: close[1]=102, future[3]=105 -> (105-102)/102 = 2.94% -> UP (2)
    # 2: close[2]=101, future[4]=100 -> (100-101)/101 = -0.99% -> NEUTRAL (1)
    # 3: close[3]=105, future[5] -> Invalid -> -1
    # 4: close[4]=100, future[6] -> Invalid -> -1

    close = np.array([100.0, 102.0, 101.0, 105.0, 100.0], dtype=np.float64)
    target_horizon = 2
    base_threshold = 0.02

    labels, thresholds = xgboost_rust.apply_directional_labels_rust(close, target_horizon, base_threshold)

    print("Close:", close)
    print("Labels:", labels)

    expected_labels = np.array([1, 2, 1, -1, -1], dtype=np.int32)

    np.testing.assert_array_equal(labels, expected_labels)
    print("Test Passed!")


if __name__ == "__main__":
    test_apply_directional_labels()
