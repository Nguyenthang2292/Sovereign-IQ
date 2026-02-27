import numpy as np
import pandas as pd

from modules.xgboost_LTS.utils import features as feature_utils
from modules.xgboost_LTS.utils.cache_manager import CacheManager
from modules.xgboost_LTS.utils.memory_map import dataframe_to_memmap, load_memmap


def _sample_ohlcv(rows: int = 32) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 100 + rows - 1, rows),
            "high": np.linspace(101, 101 + rows - 1, rows),
            "low": np.linspace(99, 99 + rows - 1, rows),
            "close": np.linspace(100, 100 + rows - 1, rows),
            "volume": np.linspace(10, 10 + rows - 1, rows),
        }
    )
    df.index = pd.date_range("2026-01-01", periods=rows, freq="h")
    return df


def test_compute_features_lazy_with_selected_features(monkeypatch):
    monkeypatch.setattr(feature_utils, "RUST_AVAILABLE", False)

    df = _sample_ohlcv()
    result = feature_utils.compute_features_lazy(df, selected_features=["returns_1", "hour"], importance_threshold=0.2)

    assert "returns_1" in result.columns
    assert "hour" in result.columns


def test_memory_map_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    mmap_path = tmp_path / "labels.mmap"

    mapped, used_columns = dataframe_to_memmap(df, mmap_path, columns=["a", "b"], dtype=np.float32)

    assert used_columns == ["a", "b"]
    assert mapped.shape == (3, 2)
    np.testing.assert_allclose(mapped[:, 0], np.array([1.0, 2.0, 3.0], dtype=np.float32))

    reloaded = load_memmap(mmap_path, shape=(3, 2), dtype=np.float32)
    np.testing.assert_allclose(reloaded[:, 1], np.array([10.0, 20.0, 30.0], dtype=np.float32))


def test_cache_manager_load_labels_memmap(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.xgboost_LTS.utils.cache_manager.ARTIFACTS_DIR", tmp_path)

    cache = CacheManager(subsystem="xgboost_test")
    source_df = pd.DataFrame({"close": [100.0, 101.0, 102.0], "volume": [1.0, 2.0, 3.0]})
    config = {"threshold": 0.01}

    fake_cache_path = cache.get_labels_path(source_df, config)
    fake_cache_path.parent.mkdir(parents=True, exist_ok=True)
    fake_cache_path.touch()

    stored_df = pd.DataFrame({"Target": [0.0, 1.0, 2.0], "score": [0.1, 0.2, 0.3]})

    monkeypatch.setattr(
        "modules.xgboost_LTS.utils.cache_manager.pd.read_parquet",
        lambda *_args, **_kwargs: stored_df,
    )

    result = cache.load_labels_memmap(source_df, config, columns=["Target", "score"], dtype=np.float32)
    assert result is not None
    mapped, used_columns = result

    assert used_columns == ["Target", "score"]
    assert mapped.shape == (3, 2)
    np.testing.assert_allclose(mapped[:, 0], np.array([0.0, 1.0, 2.0], dtype=np.float32))
