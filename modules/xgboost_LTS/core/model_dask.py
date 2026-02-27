"""
Dask-based training for XGBoost (optional, for large datasets).
Requires: dask[dataframe], xgboost (with dask support).
"""

# pyright: reportMissingImports=false

from typing import Any, Optional
import importlib


def _resolve_dask_client(
    client: Optional[Any] = None,
    scheduler_address: Optional[str] = None,
    use_cuda: bool = False,
    n_workers: Optional[int] = None,
    threads_per_worker: Optional[int] = None,
    memory_limit: Optional[str] = None,
) -> tuple[Any, bool]:
    """
    Resolve a Dask client.

    Returns:
        (client, managed) where managed=True means this function created the client
        and the caller is responsible for closing it.
    """
    if client is not None:
        return client, False

    from dask.distributed import Client, LocalCluster, get_client

    if scheduler_address:
        return Client(scheduler_address), True

    if use_cuda:
        try:
            dask_cuda_mod = importlib.import_module("dask_cuda")
            LocalCUDACluster = getattr(dask_cuda_mod, "LocalCUDACluster")

            cluster_kwargs: dict[str, Any] = {}
            if n_workers is not None:
                cluster_kwargs["n_workers"] = n_workers
            if threads_per_worker is not None:
                cluster_kwargs["threads_per_worker"] = threads_per_worker
            if memory_limit is not None:
                cluster_kwargs["device_memory_limit"] = memory_limit
            cluster = LocalCUDACluster(**cluster_kwargs)
            return Client(cluster), True
        except ImportError as exc:
            raise ImportError(
                "GPU Dask requested but 'dask-cuda' is not installed. Install dask-cuda or disable use_cuda."
            ) from exc

    # Reuse existing active client if present, else create local CPU cluster.
    try:
        return get_client(), False
    except ValueError:
        cluster_kwargs = {
            "processes": True,
            "n_workers": n_workers,
            "threads_per_worker": threads_per_worker,
            "memory_limit": memory_limit,
        }
        cluster = LocalCluster(**{k: v for k, v in cluster_kwargs.items() if v is not None})
        return Client(cluster), True


def train_and_predict_dask(
    df_dask: Any,
    model_features: list,
    target_col: str = "Target",
    client: Optional[Any] = None,
    scheduler_address: Optional[str] = None,
    use_cuda: bool = False,
    n_workers: Optional[int] = None,
    threads_per_worker: Optional[int] = None,
    memory_limit: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Train XGBoost with Dask for out-of-core / distributed data.

    Args:
        df_dask: Dask DataFrame with model_features and target_col.
        model_features: List of feature column names.
        target_col: Target column name.
        client: Optional existing Dask client.
        scheduler_address: Optional remote scheduler address (e.g. tcp://host:8786).
        use_cuda: If True, create CUDA-backed local cluster via dask-cuda.
        n_workers: Optional Dask workers for locally managed cluster.
        threads_per_worker: Optional threads per Dask worker.
        memory_limit: Optional memory limit per worker (e.g., "4GB").
        **kwargs: Passed to XGBoost (e.g. num_class, params).

    Returns:
        Trained model (or Dask-specific wrapper) and optional metrics.
    """
    try:
        import dask.dataframe as dd
    except ImportError as e:
        raise ImportError(f"Dask dependencies missing: {e}. Install dask[dataframe].")

    managed_client = False
    dask_client = None
    try:
        dask_client, managed_client = _resolve_dask_client(
            client=client,
            scheduler_address=scheduler_address,
            use_cuda=use_cuda,
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
        )

        # Select features and target
        X = df_dask[model_features]
        y = df_dask[target_col]

        # Split without shuffle (time-series safe) using Dask arrays
        X_array = X.to_dask_array(lengths=True)
        y_array = y.to_dask_array(lengths=True)
        total_rows = int(X_array.shape[0])
        split_idx = max(1, int(total_rows * 0.8))
        if split_idx >= total_rows:
            split_idx = total_rows - 1

        X_train = X_array[:split_idx]
        X_test = X_array[split_idx:]
        y_train = y_array[:split_idx]
        y_test = y_array[split_idx:]

        params = kwargs.get("params", {}).copy()
        if use_cuda and "device" not in params:
            params["device"] = "cuda"

        try:
            from xgboost.dask import DaskXGBClassifier

            # Remove n_jobs from params as Dask handles parallelism
            if "n_jobs" in params:
                del params["n_jobs"]

            model = DaskXGBClassifier(**params)
            model.client = dask_client

            model.fit(X_train, y_train)

            # We can evaluate if needed, but return model for now
            # score = model.score(X_test, y_test)

        except ImportError as exc:
            raise ImportError(
                "xgboost.dask is unavailable. Install a compatible xgboost build with Dask support."
            ) from exc

        return model
    finally:
        if managed_client and dask_client is not None:
            dask_client.close()
