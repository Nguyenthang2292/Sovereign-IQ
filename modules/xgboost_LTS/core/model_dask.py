"""
Dask-based training for XGBoost (optional, for large datasets).
Requires: dask[dataframe], dask-ml, dask-xgboost (or dask_ml.xgboost).
"""

from typing import Any, Optional


def train_and_predict_dask(
    df_dask: "dd.DataFrame",
    model_features: list,
    target_col: str = "Target",
    **kwargs: Any,
) -> Any:
    """
    Train XGBoost with Dask for out-of-core / distributed data.

    Args:
        df_dask: Dask DataFrame with model_features and target_col.
        model_features: List of feature column names.
        target_col: Target column name.
        **kwargs: Passed to XGBoost (e.g. num_class, params).

    Returns:
        Trained model (or Dask-specific wrapper) and optional metrics.
    """
    try:
        import dask.array as da
        import dask.dataframe as dd
        from dask_ml.model_selection import train_test_split

        # Try importing XGBoost for Dask
        # dask-xgboost is older, dask_ml.xgboost or xgboost.dask is newer
        try:
            from xgboost import dask as dxgb
        except ImportError:
            try:
                import dask_ml.xgboost as dxgb
            except ImportError:
                raise ImportError(
                    "Could not import xgboost.dask or dask_ml.xgboost. Please install xgboost>=1.0 or dask-ml."
                )
    except ImportError as e:
        raise ImportError(f"Dask dependencies missing: {e}. Install dask[dataframe] and dask-ml.")

    # Select features and target
    X = df_dask[model_features]
    y = df_dask[target_col]

    # Split (Dask-friendly)
    # Note: dask_ml.model_selection.train_test_split works on Dask arrays and DataFrames
    # shuffle=False is important for time-series
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    params = kwargs.get("params", {}).copy()

    # XGBoost Dask API usually uses a client or just functional API
    # Using the Scikit-Learn wrapper interface if available in dask_ml
    # OR using xgboost.dask.DaskXGBClassifier which is the standard now

    try:
        from xgboost.dask import DaskXGBClassifier

        # Remove n_jobs from params as Dask handles parallelism
        if "n_jobs" in params:
            del params["n_jobs"]

        model = DaskXGBClassifier(**params)
        model.client = None  # Let it find the default client

        model.fit(X_train, y_train)

        # We can evaluate if needed, but return model for now
        # score = model.score(X_test, y_test)

    except ImportError:
        # Fallback to dask_ml.xgboost if installed (older)
        import dask_ml.xgboost as dml_xgb

        model = dml_xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

    return model
