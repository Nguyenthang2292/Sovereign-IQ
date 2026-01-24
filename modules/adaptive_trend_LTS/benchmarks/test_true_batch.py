import time

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import process_symbols_batch_cuda


def test_true_batch():
    print("Testing True Batch CUDA Processing...")

    # Create fake data that actually moves
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    bars = 1000
    data = {}
    for s in symbols:
        # Create a trend to get non-zero signals
        t = np.linspace(0, 10, bars)
        trend = 100 + 5 * t + 10 * np.sin(t)
        noise = np.random.randn(bars) * 2
        data[s] = pd.Series(trend + noise, name="Close")

    config = {
        "ema_len": 20,  # Shorter for more signals
        "robustness": "Medium",
        "La": 0.02,
        "De": 0.03,
        "long_threshold": 0.05,
        "short_threshold": -0.05,
        "use_cuda": True,
    }

    start_time = time.time()
    results = process_symbols_batch_cuda(data, config)
    end_time = time.time()

    print(f"Batch processing took {end_time - start_time:.4f}s")

    if results:
        print(f"Successfully processed {len(results)} symbols.")
        for s in symbols:
            if s in results:
                sig = results[s]["Average_Signal"]
                print(f"Symbol {s} result shape: {len(sig)}")
                print(f"Non-zero signals: {np.count_nonzero(sig)}")
            else:
                print(f"Symbol {s} MISSING in results")
    else:
        print("FAILED: No results returned")


if __name__ == "__main__":
    test_true_batch()
