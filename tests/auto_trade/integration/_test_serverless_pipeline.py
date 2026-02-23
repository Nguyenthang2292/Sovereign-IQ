"""Manual integration smoke test for XGBoost serverless filter in auto-trade pipeline.

Run:
    python tests/auto_trade/integration/_test_serverless_pipeline.py
"""

import logging
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.auto_trade.core.atc_scanner import SignalResult
from modules.auto_trade.core.xgboost_serverless_filter import XGBoostServerlessFilter
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager

logging.basicConfig(level=logging.INFO)


def run_test():
    exchange_manager = ExchangeManager()
    fetcher = DataFetcher(exchange_manager)

    mock_signal = SignalResult(
        symbol="BTC/USDT",
        score=0.8,
        signal_type="LONG",
        details={"15m": "LONG"},
        strengths={"15m": 0.8},
    )

    config = {
        "xgboost_serverless_function_name": "xgboost-serverless-predict",
        "xgboost_serverless_timeframe": "15m",
        "xgboost_serverless_model_version": "v1",
        "xgboost_serverless_s3_bucket": "xgboost-models-store",
    }

    filter_adapter = XGBoostServerlessFilter(data_fetcher=fetcher, config=config)
    signals = filter_adapter.filter_signals([mock_signal])

    print("\n--- TEST DONE ---")
    if signals:
        print("Signal passed the filter!")
        print("Details =", signals[0].details)
    else:
        print("Signal did NOT pass the filter. (Maybe prediction was opposite or confidence too low)")


if __name__ == "__main__":
    run_test()
