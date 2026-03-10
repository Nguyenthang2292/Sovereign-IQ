import time

import numpy as np


def benchmark():
    np.random.seed(42)
    r1 = np.random.normal(0, 0.001, 1000)
    r2 = np.random.normal(0, 0.01, 1000)
    r3 = np.random.normal(0.001, 0.005, 1000)
    r4 = np.random.normal(-0.001, 0.02, 1000)
    r5 = np.random.normal(0, 0.002, 1000)
    returns = np.concatenate([r1, r2, r3, r4, r5])  # 5000 points

    n = len(returns)
    penalty = np.log(n) * returns.var()
    min_segment_length = 20

    # 1. Rust
    start = time.time()
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    try:
        import rust_extensions

        for _ in range(5):
            bps_rust = rust_extensions.detect_change_points_pelt_rs(
                returns.tolist(), float(penalty), int(min_segment_length)
            )
        rust_time = time.time() - start
        print(f"Rust PELT (L2) time (5 runs): {rust_time:.4f}s")
        print(f"Rust result: {bps_rust}")
    except Exception as e:
        print(f"Rust error: {e}")
        rust_time = float("inf")

    # 2. Python (ruptures L2)
    try:
        import ruptures as rpt

        start = time.time()
        for _ in range(5):
            algo = rpt.Pelt(model="l2", min_size=min_segment_length).fit(returns)
            bps_py = algo.predict(pen=penalty)
        py_time = time.time() - start
        print(f"Python ruptures PELT (L2) time (5 runs): {py_time:.4f}s")
        print(f"Python result: {bps_py}")

        # 3. Python (ruptures RBF)
        start = time.time()
        for _ in range(5):
            algo = rpt.Pelt(model="rbf", min_size=min_segment_length).fit(returns)
            bps_rbf = algo.predict(pen=penalty)
        rbf_time = time.time() - start
        print(f"Python ruptures PELT (RBF) time (5 runs): {rbf_time:.4f}s")
        print(f"RBF result: {bps_rbf}")

        if rust_time > 0:
            print(f"Speedup: {py_time / rust_time:.2f}x vs Python L2, {rbf_time / rust_time:.2f}x vs Python RBF")
    except ImportError:
        print("Ruptures not installed")


if __name__ == "__main__":
    benchmark()
