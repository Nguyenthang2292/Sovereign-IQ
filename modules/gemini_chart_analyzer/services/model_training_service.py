"""Service for machine learning model training operations."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from modules.common.core.data_fetcher import DataFetcher
from modules.common.ui.logging import log_error, log_info, log_success, log_progress, log_warn
from modules.gemini_chart_analyzer.cli.models.random_forest_manager import (
    check_random_forest_model_status,
    delete_old_model,
)
from modules.random_forest import train_and_save_global_rf_model


def run_rf_model_training(
    data_fetcher: DataFetcher, symbols: List[str], timeframe: str = "1h", limit: int = 1500, delete_old: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Execute Random Forest model training.

    Args:
        data_fetcher: Initialized DataFetcher
        symbols: List of symbols for training data
        timeframe: Data timeframe
        limit: Candles per symbol
        delete_old: Whether to delete the existing model first

    Returns:
        Tuple of (success_status, saved_model_path)
    """
    try:
        log_info(f"Starting model training service for {len(symbols)} symbols...")

        if delete_old:
            delete_old_model()

        # Fetch training data
        all_dfs = []
        for idx, symbol in enumerate(symbols, 1):
            try:
                log_progress(f"Fetching training data: {symbol} ({idx}/{len(symbols)})...")
                df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                    symbol, timeframe=timeframe, limit=limit, check_freshness=False
                )
                if df is not None and not df.empty:
                    df_copy = df.copy()
                    df_copy["symbol"] = symbol
                    all_dfs.append(df_copy)
                else:
                    log_warn(f"No data for {symbol}")
            except Exception as e:
                log_warn(f"Error fetching {symbol}: {e}")

        if not all_dfs:
            log_error("No training data collected.")
            return False, None

        # Combine and train
        combined_df = pd.concat(all_dfs, ignore_index=True)
        log_info(f"Training on combined dataset of {len(combined_df)} rows...")

        model, model_path = train_and_save_global_rf_model(combined_df)

        if model is not None and model_path:
            log_success(f"Model trained and saved to: {model_path}")
            return True, str(model_path)

        return False, None

    except Exception as e:
        log_error(f"Error in model training service: {e}")
        return False, None


def get_model_status() -> Dict[str, Any]:
    """Get current status of the Random Forest model."""
    return check_random_forest_model_status()
