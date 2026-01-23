import numpy as np

from modules.adaptive_trend_enhance_v2.core.rust_backend import RUST_AVAILABLE, calculate_equity


def test_rust_equity():
    print(f"Rust Available: {RUST_AVAILABLE}")
    r = np.array([0.01, 0.02, -0.01], dtype=np.float64)
    sig = np.array([1.0, 1.0, -1.0], dtype=np.float64)

    e = calculate_equity(r, sig, 100.0, 1.0, 0, use_rust=True)
    print(f"Result (Rust): {e}")

    expected = [100.0, 102.0, 103.02]
    np.testing.assert_allclose(e, expected, atol=1e-6)
    print("Test passed!")


if __name__ == "__main__":
    test_rust_equity()
