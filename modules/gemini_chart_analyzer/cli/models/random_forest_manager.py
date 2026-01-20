"""Random Forest model manager for batch scanner."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.random_forest import MODELS_DIR, RANDOM_FOREST_MODEL_FILENAME
from modules.common.ui.logging import log_info, log_progress
from modules.common.utils import log_error, log_success, log_warn
from modules.random_forest import train_and_save_global_rf_model
from modules.random_forest.signal_random_forest import validate_model


def delete_old_model(model_path: Optional[str] = None) -> bool:
    """Delete old Random Forest model file.

    Args:
        model_path: Optional path to model file (default: uses default from config)

    Returns:
        True if model was deleted or didn't exist, False if deletion failed
    """
    try:
        if model_path:
            model_path_obj = Path(model_path)
        else:
            model_path_obj = MODELS_DIR / RANDOM_FOREST_MODEL_FILENAME

        if model_path_obj.exists():
            log_info(f"Deleting old model: {model_path_obj}")
            model_path_obj.unlink()
            log_success("Old model deleted successfully")
            return True
        else:
            log_info("No old model found to delete")
            return True
    except Exception as e:
        log_error(f"Failed to delete old model: {type(e).__name__}: {e}")
        return False


def check_random_forest_model_status(model_path: Optional[str] = None) -> Dict[str, Any]:
    """Check Random Forest model status and compatibility.

    Args:
        model_path: Optional path to model file (default: uses default from config)

    Returns:
        Dictionary with keys:
        - exists: bool
        - compatible: bool (only if exists=True)
        - model_path: str
        - modification_date: Optional[str]
        - error_message: Optional[str]
        - uses_deprecated_features: bool
    """
    status = {
        "exists": False,
        "compatible": False,
        "model_path": "",
        "modification_date": None,
        "error_message": None,
        "uses_deprecated_features": False,
    }

    try:
        # Determine model path
        if model_path:
            model_path_obj = Path(model_path)
        else:
            model_path_obj = MODELS_DIR / RANDOM_FOREST_MODEL_FILENAME

        status["model_path"] = str(model_path_obj)

        # Check if model file exists
        if not model_path_obj.exists():
            status["error_message"] = f"Model file not found: {model_path_obj}"
            return status

        status["exists"] = True

        # Get file modification date
        try:
            modification_time = model_path_obj.stat().st_mtime
            modification_date = datetime.fromtimestamp(modification_time)
            status["modification_date"] = modification_date.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass  # Could not get modification date

        # Validate model compatibility
        is_valid, error_msg = validate_model(str(model_path_obj))
        status["compatible"] = is_valid

        if not is_valid:
            status["error_message"] = error_msg
            # Check if error is about deprecated features
            if error_msg and isinstance(error_msg, str) and "deprecated raw ohlcv features" in error_msg.lower():
                status["uses_deprecated_features"] = True

    except Exception as e:
        status["error_message"] = f"Error checking model status: {type(e).__name__}: {e}"

    return status


def train_random_forest_model_interactive(
    data_fetcher,
    symbols: List[str],
    timeframe: str,
    limit: int,
) -> Tuple[bool, Optional[str]]:
    """Train Random Forest model interactively.

    Args:
        data_fetcher: DataFetcher instance
        symbols: List of symbols to fetch data for
        timeframe: Timeframe for data
        limit: Number of candles per symbol

    Returns:
        Tuple of (success, model_path). model_path is None if failed.
    """
    try:
        log_info("=" * 60)
        log_info("RANDOM FOREST MODEL TRAINING")
        log_info("=" * 60)

        # Delete old model before training new one
        if not delete_old_model():
            log_warn("Failed to delete old model. Continuing training anyway...")

        # Fetch data for all symbols
        all_dataframes = []
        total_symbols = len(symbols)

        log_info(f"Fetching data for {total_symbols} symbols...")
        for idx, symbol in enumerate(symbols, 1):
            try:
                log_progress(f"Fetching data for {symbol} ({idx}/{total_symbols})...")
                df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                    symbol, timeframe=timeframe, limit=limit, check_freshness=False
                )

                if df is not None and not df.empty:
                    # Add symbol column for identification
                    df_copy = df.copy()
                    df_copy["symbol"] = symbol
                    all_dataframes.append(df_copy)
                    log_progress(f"  ✓ Fetched {len(df)} candles for {symbol}")
                else:
                    log_warn(f"  ✗ Failed to fetch data for {symbol}")

            except Exception as e:
                log_warn(f"  ✗ Error fetching {symbol}: {type(e).__name__}: {e}")

        if not all_dataframes:
            log_error("No data fetched for any symbols. Cannot train model.")
            return False, None

        # Combine all DataFrames
        log_info(f"\nCombining data from {len(all_dataframes)} symbols...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        log_success(f"Combined dataset: {len(combined_df)} rows")

        # Train model
        log_info("\nTraining Random Forest model...")
        log_info("This may take several minutes...")

        model, model_path = train_and_save_global_rf_model(combined_df)

        if model is not None and model_path:
            log_success(f"Model trained successfully!")
            log_success(f"Model saved to: {model_path}")
            return True, model_path
        else:
            log_error("Model training failed.")
            return False, None

    except Exception as e:
        log_error(f"Error during model training: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def display_model_status(rf_model_status: Dict[str, Any]) -> None:
    """Display Random Forest model status to user.

    Args:
        rf_model_status: Dictionary containing model status information
    """
    print("\nRandom Forest Model Status:")
    if rf_model_status.get("exists", False):
        print(f"  Model file: {rf_model_status.get('model_path', 'Unknown')}")
        if rf_model_status.get("modification_date"):
            print(f"  Last modified: {rf_model_status['modification_date']}")

        if rf_model_status.get("compatible", False):
            print("  Status: Compatible")
        else:
            print("  Status: Incompatible")
            if rf_model_status.get("uses_deprecated_features"):
                print("    (uses deprecated features)")

        if rf_model_status.get("error_message"):
            print(f"  Warning: {rf_model_status['error_message']}")
    else:
        print("  Model file not found")
        if rf_model_status and rf_model_status.get("error_message"):
            print(f"  Error: {rf_model_status['error_message']}")
