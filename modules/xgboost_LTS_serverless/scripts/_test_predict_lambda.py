"""Manual smoke script for invoking xgboost-serverless-predict with real Binance OHLCV data.

Run:
    python modules/xgboost_LTS_serverless/scripts/_test_predict_lambda.py
"""

import json
import shutil
import subprocess
from pathlib import Path

import ccxt

FUNCTION_NAME = "xgboost-serverless-predict"
SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"
FETCH_LIMIT = 1500


def fetch_data_and_predict():
    if shutil.which("aws") is None:
        raise RuntimeError("AWS CLI not found in PATH. Install AWS CLI v2 and configure credentials first.")

    print(f"Fetching {FETCH_LIMIT} real candles for {SYMBOL} {TIMEFRAME} from Binance...")
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=FETCH_LIMIT)

    data = {"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
    for row in ohlcv:
        data["timestamp"].append(int(row[0]))
        data["open"].append(float(row[1]))
        data["high"].append(float(row[2]))
        data["low"].append(float(row[3]))
        data["close"].append(float(row[4]))
        data["volume"].append(float(row[5]))

    script_dir = Path(__file__).resolve().parent
    payload_file = script_dir / "payload_predict_real.json"
    response_file = script_dir / "response_predict_real.json"

    payload = {
        "requests": [
            {
                "symbol": "BTCUSDT",
                "timeframe": TIMEFRAME,
                "model_version": "v1",
                "model_s3_key": f"BTCUSDT_{TIMEFRAME}_v1.json",
                "data": data,
            }
        ]
    }

    with payload_file.open("w", encoding="utf-8") as file:
        json.dump(payload, file)

    print(f"Payload saved to {payload_file} (Length: {len(data['timestamp'])} rows)")

    print(f"Invoking Lambda {FUNCTION_NAME}...")
    cmd = [
        "aws",
        "lambda",
        "invoke",
        "--function-name",
        FUNCTION_NAME,
        "--payload",
        f"fileb://{payload_file}",
        "--cli-binary-format",
        "raw-in-base64-out",
        str(response_file),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Lambda invoke command failed with exit code {exc.returncode}") from exc

    print("Invocation returned. Reading response...")
    with response_file.open("r", encoding="utf-8") as file:
        print(file.read())


if __name__ == "__main__":
    fetch_data_and_predict()
