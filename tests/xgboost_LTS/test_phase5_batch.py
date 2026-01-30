import pandas as pd
import numpy as np
import time
from modules.xgboost_LTS.utils.batch_symbols import batch_train_symbols
from config import MODEL_FEATURES


def mock_train_func(df, use_cache=True):
    """Mock training function that sleeps to simulate work"""
    time.sleep(0.1)
    return "model_artifact"


def failing_func(df, use_cache=True):
    """Mock training function that fails for empty dataframes"""
    if len(df) == 0:
        raise ValueError("Empty dataframe")
    return "success"


def test_batch_train_symbols():
    # Create dummy data
    n_samples = 100
    data = {
        "Target": np.random.randint(0, 3, n_samples),
    }
    for feature in MODEL_FEATURES:
        data[feature] = np.random.randn(n_samples)

    df = pd.DataFrame(data)

    symbols_data = {"BTCUSDT": df.copy(), "ETHUSDT": df.copy(), "SOLUSDT": df.copy()}

    # Test batch training
    start = time.perf_counter()
    results = batch_train_symbols(symbols_data, train_and_predict_fn=mock_train_func, max_workers=2, use_cache=False)
    duration = time.perf_counter() - start

    assert len(results) == 3
    assert "BTCUSDT" in results
    assert "ETHUSDT" in results
    assert "SOLUSDT" in results

    for symbol, res in results.items():
        assert res["ok"] is True
        assert res["result"] == "model_artifact"

    print(f"Batch processing took {duration:.4f}s")


def test_batch_train_error_handling():
    symbols_data = {"GOOD": pd.DataFrame({"a": [1]}), "BAD": pd.DataFrame({})}

    results = batch_train_symbols(symbols_data, train_and_predict_fn=failing_func, max_workers=2)

    assert results["GOOD"]["ok"] is True
    assert results["GOOD"]["result"] == "success"

    assert results["BAD"]["ok"] is False
    assert "Empty dataframe" in results["BAD"]["error"]
