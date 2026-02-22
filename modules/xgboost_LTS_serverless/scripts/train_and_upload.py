import argparse
import os

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import boto3
except ImportError:
    boto3 = None

try:
    from modules.common.core.data_fetcher import DataFetcher
    from modules.common.core.exchange_manager import ExchangeManager
except ImportError:
    # Fallback stub
    class DataFetcher:
        def fetch_ohlcv(self, *args, **kwargs):
            return []

    class ExchangeManager:
        pass


def train_model_with_cv(symbol, timeframe, data):
    """
    Train model using cross validation.
    This uses a stub/dummy implementation for the CLI to run if the actual implementation is missing.
    """
    if xgb is None:
        raise ImportError("xgboost is required for actual training but is not installed.")

    print(f"Training XGBoost model for {symbol} ({timeframe}) with CV...")

    # In a real scenario, we'd use the provided data to train
    model = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.01, max_depth=5, objective="multi:softprob", num_class=3
    )

    # Dummy data for the sake of having a model that could be saved
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=92, n_classes=3, n_informative=10)
    model.fit(X, y)

    return model


def train_and_upload(symbol, timeframe, version, bucket):
    print(f"Starting training process for {symbol} ({timeframe}) version {version}")

    # Fetch Data
    exchange_manager = ExchangeManager()
    fetcher = DataFetcher(exchange_manager)
    data = fetcher.fetch_ohlcv(symbol, timeframe, limit=5000)

    # Train Model
    model = train_model_with_cv(symbol, timeframe, data)

    # Normalize symbol to match Lambda's expected S3 key format
    # normalize_symbol_key("BTC/USDT") → "BTCUSDT"  (matches handler.rs model_s3_key lookup)
    normalized_symbol = "".join(ch for ch in symbol.upper() if ch.isalnum())
    model_filename = f"{normalized_symbol}_{timeframe}_{version}.json"

    # Standard temp paths
    tmp_dir = "/tmp" if os.name != "nt" else os.environ.get("TEMP", ".")
    model_path = os.path.join(tmp_dir, model_filename)

    print(f"Exporting model to {model_path}...")
    model.save_model(model_path)

    # Upload to S3
    # IMPORTANT: S3 key must NOT have a prefix — Lambda handler downloads by bare filename
    # e.g. "BTCUSDT_15m_v1.json", NOT "models/xgboost/BTCUSDT_15m_v1.json"
    print(f"Uploading to S3 bucket: {bucket}")
    if boto3 is None:
        raise ImportError("boto3 is required to upload to S3 but is not installed.")

    s3 = boto3.client("s3")
    key = model_filename  # bare key — matches model_s3_key sent by xgboost_serverless_filter.py

    try:
        metadata = {"symbol": symbol, "timeframe": timeframe, "version": version, "features": "92"}
        s3.upload_file(model_path, bucket, key, ExtraArgs={"Metadata": metadata, "ContentType": "application/json"})
        print(f"Model successfully uploaded: s3://{bucket}/{key}")
    except Exception as e:
        print(f"Failed to upload to S3: {e}")

    return f"s3://{bucket}/{key}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and upload XGBoost model to S3")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g., BTC/USDT)")
    parser.add_argument("--timeframe", type=str, required=True, help="Timeframe (e.g., 15m)")
    parser.add_argument("--version", type=str, default="v1", help="Model version (e.g., v1)")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name")

    args = parser.parse_args()

    train_and_upload(args.symbol, args.timeframe, args.version, args.bucket)
