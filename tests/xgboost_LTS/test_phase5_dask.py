import pytest
import pandas as pd
import numpy as np
from config import MODEL_FEATURES

# Skip tests if dask is not installed
dask = pytest.importorskip("dask")
dd = pytest.importorskip("dask.dataframe")
try:
    import dask_ml
    import xgboost
except ImportError:
    pytest.skip("dask-ml or xgboost not installed", allow_module_level=True)

from modules.xgboost_LTS.core.model_dask import train_and_predict_dask


def test_train_and_predict_dask():
    # Create dummy data
    n_samples = 100
    data = {
        "Target": np.random.randint(0, 3, n_samples),
    }
    # Add model features
    for feature in MODEL_FEATURES:
        data[feature] = np.random.randn(n_samples)

    pdf = pd.DataFrame(data)

    # Create Dask DataFrame
    df_dask = dd.from_pandas(pdf, npartitions=2)

    # Train model
    # Use small params for speed
    params = {"n_estimators": 2, "max_depth": 2, "objective": "multi:softprob", "num_class": 3}

    try:
        model = train_and_predict_dask(df_dask, model_features=MODEL_FEATURES, params=params)

        assert model is not None
        # Check if it has predict method (either sklearn wrapper or dask wrapper)
        assert hasattr(model, "predict")

    except ImportError as e:
        pytest.skip(f"Skipping dask test due to missing dependencies: {e}")
    except Exception as e:
        pytest.fail(f"Dask training failed: {e}")
