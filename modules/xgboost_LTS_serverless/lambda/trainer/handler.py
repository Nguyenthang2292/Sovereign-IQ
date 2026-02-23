"""
XGBoost Trainer Lambda Handler

Triggered async by XGBoostServerlessFilter khi model thiếu trong S3.
Tái dùng toàn bộ xgboost_LTS training pipeline.

Event format:
{
    "symbol": "BTC/USDT",
    "timeframe": "15m",
    "model_version": "v1",
    "s3_bucket": "xgboost-models-store",
    "fetch_limit": 1000
}
"""

import time


# Lazy import — tránh cold start nặng
def _imports():
    from modules.common.core.data_fetcher import DataFetcher
    from modules.common.core.exchange_manager import ExchangeManager
    from modules.common.core.indicator_engine import (
        IndicatorConfig,
        IndicatorEngine,
        IndicatorProfile,
    )
    from modules.xgboost_LTS.core.labeling import apply_directional_labels
    from modules.xgboost_LTS.core.model import train_and_predict
    from modules.xgboost_LTS.utils.features import add_advanced_features

    return (
        DataFetcher,
        ExchangeManager,
        IndicatorConfig,
        IndicatorEngine,
        IndicatorProfile,
        apply_directional_labels,
        train_and_predict,
        add_advanced_features,
    )


def _normalize(symbol: str) -> str:
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


def handler(event, context):
    """Lambda entrypoint."""
    t0 = time.perf_counter()

    symbol = event["symbol"]
    timeframe = event.get("timeframe", "15m")
    version = event.get("model_version", "v1")
    s3_bucket = event["s3_bucket"]
    fetch_limit = int(event.get("fetch_limit", 1000))

    print(f"[trainer] START symbol={symbol} tf={timeframe} ver={version}")

    # ── 0. AWS Lambda / Container Fixes ────────────────────────────────────────
    # Disable ProcessPoolExecutor CV which crashes in Lambda emulator
    import config

    config.XGBOOST_USE_PARALLEL_CV = False

    # ── 1. Import heavy deps ───────────────────────────────────────────────────
    (
        DataFetcher,
        ExchangeManager,
        IndicatorConfig,
        IndicatorEngine,
        IndicatorProfile,
        apply_directional_labels,
        train_and_predict,
        add_advanced_features,
    ) = _imports()

    # ── 2. Exchange + DataFetcher (dùng env vars từ Lambda env) ───────────────
    exchange = ExchangeManager()
    fetcher = DataFetcher(exchange)

    df = fetcher.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        limit=fetch_limit,
        check_freshness=False,
    )
    if df is None or df.empty:
        raise ValueError(f"fetch_ohlcv returned empty data for {symbol}")
    print(f"[trainer] fetched {len(df)} candles")

    # ── 3. Indicators ──────────────────────────────────────────────────────────
    engine = IndicatorEngine(IndicatorConfig.for_profile(IndicatorProfile.XGBOOST))
    result = engine.compute_features(df)
    df = result[0] if isinstance(result, tuple) else result
    df = add_advanced_features(df)

    # ── 4. Labels ──────────────────────────────────────────────────────────────
    df = apply_directional_labels(df, use_cache=False)
    df = df.dropna(subset=["Target"])

    # ── 5. Train ───────────────────────────────────────────────────────────────
    model = train_and_predict(df, use_cache=False)
    print(f"[trainer] training done in {time.perf_counter() - t0:.1f}s")

    # ── 6+7. Serialize in-memory → upload thẳng S3 (bypass /tmp) ─────────────
    import boto3

    normalized = _normalize(symbol)
    filename = f"{normalized}_{timeframe}_{version}.json"

    booster = model.get_booster()
    # save_raw() returns bytes directly -- works on all XGBoost versions
    # (save_model(BytesIO) was added only in XGBoost 2.1+, save_raw is stable)
    model_bytes = booster.save_raw(raw_format="json")

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=s3_bucket,
        Key=filename,  # bare key — khớp với Lambda Rust handler
        Body=model_bytes,
        ContentType="application/json",
        Metadata={
            "symbol": symbol,
            "timeframe": timeframe,
            "version": version,
            "trained_at": str(int(time.time())),
        },
    )
    elapsed = time.perf_counter() - t0
    print(f"[trainer] uploaded s3://{s3_bucket}/{filename} ({len(model_bytes)} bytes) in {elapsed:.1f}s total")

    return {
        "status": "ok",
        "symbol": symbol,
        "s3_key": filename,
        "size_bytes": len(model_bytes),
        "elapsed_s": round(elapsed, 1),
    }
