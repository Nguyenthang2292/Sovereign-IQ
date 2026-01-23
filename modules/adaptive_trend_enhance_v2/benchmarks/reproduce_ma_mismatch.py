import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

# Add project root to path
try:
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
except NameError:
    pass

from modules.adaptive_trend_enhance_v2.core.compute_moving_averages.calculate_kama_atc import calculate_kama_atc
from modules.adaptive_trend_enhance_v2.core.rust_backend import (
    calculate_dema,
    calculate_ema,
    calculate_hma,
    calculate_kama,
    calculate_lsma,
    calculate_wma,
)


def test_ma_consistency():
    with open("ma_verification.log", "w", encoding="utf-8") as f:
        f.write("Testing MA consistency between Rust and Python/Pandas_TA...\n")

        # Generate random data
        np.random.seed(42)
        length = 1000
        prices = 100 + np.cumsum(np.random.randn(length))
        series = pd.Series(prices)
        ma_len = 28

        def check_ma(name, py_vals, ru_vals, offset):
            f.write(f"\n--- {name} ---\n")
            diff = np.abs(py_vals[offset:] - ru_vals[offset:])

            # Handle all NaNs case
            if np.all(np.isnan(diff)):
                f.write("All diffs are NaN (unexpected)\n")
                max_diff = 0.0
            else:
                max_diff = np.nanmax(diff)

            f.write(f"Max Diff: {max_diff:.2e}\n")
            f.write(f"Rust (sample): {ru_vals[offset : offset + 3]}\n")
            f.write(f"Py   (sample): {py_vals[offset : offset + 3]}\n")

            if max_diff > 1e-6:
                f.write(f"MISMATCH in {name}!\n")
            else:
                f.write(f"{name} MATCHED ✅\n")
            f.flush()

        # 1. EMA
        py_ema = ta.ema(series, length=ma_len).values
        ru_ema = calculate_ema(prices, length=ma_len, use_rust=True)
        check_ma("EMA", py_ema, ru_ema, ma_len)

        # 2. HMA
        # Note: Source uses SMA for HMA
        py_hma = ta.sma(series, length=ma_len).values
        ru_hma = calculate_hma(prices, length=ma_len, use_rust=True)
        check_ma("HMA", py_hma, ru_hma, ma_len)

        # 3. WMA
        py_wma = ta.wma(series, length=ma_len).values
        ru_wma = calculate_wma(prices, length=ma_len, use_rust=True)
        check_ma("WMA", py_wma, ru_wma, ma_len)

        # 4. DEMA
        py_dema = ta.dema(series, length=ma_len).values
        ru_dema = calculate_dema(prices, length=ma_len, use_rust=True)
        check_ma("DEMA", py_dema, ru_dema, ma_len)

        # 5. LSMA
        py_lsma = ta.linreg(series, length=ma_len).values
        ru_lsma = calculate_lsma(prices, length=ma_len, use_rust=True)
        check_ma("LSMA", py_lsma, ru_lsma, ma_len)

        # 6. KAMA
        py_kama = calculate_kama_atc(series, length=ma_len).values
        ru_kama = calculate_kama(prices, length=ma_len, use_rust=True)
        check_ma("KAMA", py_kama, ru_kama, ma_len)


if __name__ == "__main__":
    test_ma_consistency()
