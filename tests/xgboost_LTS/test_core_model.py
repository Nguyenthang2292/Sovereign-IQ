import numpy as np
import pandas as pd
import pytest

from modules.xgboost_LTS.core import model as model_module


def _make_training_df(rows=200, feature_count=3):
    features = [f"f_{i}" for i in range(feature_count)]
    data = {col: np.random.randn(rows) for col in features}
    df = pd.DataFrame(data)
    return df, features


def test_train_and_predict_missing_class_zero(monkeypatch):
    df, features = _make_training_df()
    df["Target"] = np.random.choice([1, 2], size=len(df))

    monkeypatch.setattr(model_module, "MODEL_FEATURES", features)
    monkeypatch.setattr(model_module, "TARGET_HORIZON", 1)
    monkeypatch.setattr(model_module, "XGBOOST_TRAIN_TEST_SPLIT", 0.8)
    monkeypatch.setattr(model_module, "XGBOOST_MIN_TRAIN_FRACTION", 0.5)

    with pytest.raises(model_module.ClassDiversityError) as exc:
        model_module.train_and_predict(df, use_cache=False)

    assert "missing class 0" in str(exc.value).lower()


def test_train_and_predict_insufficient_class_diversity(monkeypatch):
    df, features = _make_training_df()
    df["Target"] = 0

    monkeypatch.setattr(model_module, "MODEL_FEATURES", features)
    monkeypatch.setattr(model_module, "TARGET_HORIZON", 1)
    monkeypatch.setattr(model_module, "XGBOOST_TRAIN_TEST_SPLIT", 0.8)
    monkeypatch.setattr(model_module, "XGBOOST_MIN_TRAIN_FRACTION", 0.5)

    with pytest.raises(model_module.ClassDiversityError):
        model_module.train_and_predict(df, use_cache=False)


def test_train_and_predict_drops_non_finite_target_and_succeeds(monkeypatch):
    """Rows with NaN/inf Target (e.g. from labeling warmup) are dropped before training."""
    df, features = _make_training_df(rows=300)
    # Valid labels 0, 1, 2; last 30 rows NaN (simulating no future price)
    df["Target"] = np.random.choice([0, 1, 2], size=len(df)).astype(float)
    df.loc[df.index[-30:], "Target"] = np.nan

    monkeypatch.setattr(model_module, "MODEL_FEATURES", features)
    monkeypatch.setattr(model_module, "TARGET_HORIZON", 1)
    monkeypatch.setattr(model_module, "XGBOOST_TRAIN_TEST_SPLIT", 0.8)
    monkeypatch.setattr(model_module, "XGBOOST_MIN_TRAIN_FRACTION", 0.1)

    result = model_module.train_and_predict(df, use_cache=False)

    assert result is not None
    assert hasattr(result, "predict")


def test_predict_next_move_accepts_series_and_dataframe(monkeypatch):
    df, features = _make_training_df(rows=1)
    monkeypatch.setattr(model_module, "MODEL_FEATURES", features)

    class DummyModel:
        def predict_proba(self, X):
            return np.array([[0.1, 0.2, 0.7]])

    dummy = DummyModel()
    row_series = df.iloc[0]
    row_df = df.iloc[[0]]

    proba_series = model_module.predict_next_move(dummy, row_series)
    proba_df = model_module.predict_next_move(dummy, row_df)

    assert proba_series.shape == (3,)
    assert proba_df.shape == (3,)
    np.testing.assert_allclose(proba_series, [0.1, 0.2, 0.7])
    np.testing.assert_allclose(proba_df, [0.1, 0.2, 0.7])
