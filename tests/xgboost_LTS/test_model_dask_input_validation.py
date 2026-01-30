import pytest

from modules.xgboost_LTS.core.model_dask import train_and_predict_dask


def test_train_and_predict_dask_missing_dependencies():
    try:
        import dask  # noqa: F401
        import dask.dataframe  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            train_and_predict_dask(df_dask=None, model_features=[])
        return

    pytest.skip("Dask is installed; missing-dependency test not applicable")
