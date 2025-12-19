"""
Main CLI function for Random Forest model training and signal generation.

This module provides the main entry point for training Random Forest models
and generating trading signals from the command line.
"""

import pandas as pd
import time

from config import MODEL_FEATURES
from modules.random_forest import (
    get_latest_random_forest_signal,
    train_and_save_global_rf_model,
)
from modules.random_forest.cli.argument_parser import (
    parse_args,
    DEFAULT_CRYPTO_SYMBOLS,
)
from modules.common.ui.logging import (
    log_info,
    log_model,
    log_error,
    log_warn,
    log_success,
)


def main():
    """
    Main function to train Random Forest model and test signal generation on crypto data.
    
    This function:
    1. Initializes Binance tick processor
    2. Loads market data for specified crypto pairs
    3. Trains Random Forest model on combined data
    4. Tests signal generation on a sample pair
    """
    DATAFRAME_COLUMNS = ['Pair', 'FinalSignal', 'SignalTimeframe']
    start_time = time.time()
    
    log_model("Starting Random Forest model training for crypto signals")
    
    args = parse_args()
    
    # NOTE: This script requires data fetching components that may not be available.
    # For now, we'll use a simplified version that expects combined_df to be provided.
    # In a full implementation, you would use DataFetcher from modules.common.core.data_fetcher
    
    log_warn(
        "This script requires data fetching components. "
        "Please provide a combined_df DataFrame with OHLCV data for training."
    )
    
    # Determine crypto pairs to analyze
    crypto_pairs = DEFAULT_CRYPTO_SYMBOLS.copy()
    
    if args.pairs:
        custom_pairs = [pair.strip() for pair in args.pairs.split(',')]
        if custom_pairs:
            crypto_pairs = custom_pairs
            log_info(f"Using custom pairs: {crypto_pairs}")
    else:
        log_info(f"Using default pairs: {crypto_pairs}")
            
    log_info(f"Final pairs list ({len(crypto_pairs)} pairs): {crypto_pairs[:5]}{'...' if len(crypto_pairs) > 5 else ''}")
    
    # TODO: Implement data loading using DataFetcher from modules.common.core.data_fetcher
    # For now, we'll require the user to provide data
    log_error(
        "Data loading not implemented. "
        "Please use DataFetcher from modules.common.core.data_fetcher to load data, "
        "then call train_and_save_global_rf_model(combined_df) directly."
    )
    
    # Placeholder for combined_df - in real usage, this would come from DataFetcher
    combined_df = pd.DataFrame()
    
    # Train Random Forest model
    model, model_path = None, ""
    try:
        model, model_path = train_and_save_global_rf_model(combined_df)
    except Exception as e:
        log_error(f"Error during model training: {e}")
    
    # Display model information if training successful
    if model is not None and model_path:
        log_success(f"Model trained successfully! Saved at: {model_path}")
        
        log_model("=" * 80)
        log_model("RANDOM FOREST MODEL DETAILS".center(80))
        log_model("=" * 80)
        
        # Model general information
        model_params = model.get_params()
        log_model(f"\nGENERAL INFORMATION:")
        log_model(f"- Model type: {type(model).__name__}")
        log_model(f"- Trees (n_estimators): {model_params['n_estimators']}")
        log_model(f"- Max depth: {'unlimited' if model_params['max_depth'] is None else model_params['max_depth']}")
        
        # Check if model has been fitted before accessing n_features_in_
        if hasattr(model, 'n_features_in_'):
            log_model(f"- Number of features: {model.n_features_in_}")  # type: ignore
        else:
            log_model(f"- Number of features: Not available (model not fitted)")
            
        log_model(f"- Random state: {model_params['random_state']}")
        
        # Feature importance analysis
        if hasattr(model, 'feature_importances_'):
            log_model(f"\nFEATURE IMPORTANCE:")
            feature_importance = pd.DataFrame({
                'Feature': MODEL_FEATURES,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            for _, row in feature_importance.iterrows():
                log_model(f"- {row['Feature']}: {row['Importance']:.4f}")
        
        # First tree analysis
        if hasattr(model, 'estimators_') and model.estimators_:
            first_tree = model.estimators_[0]
            log_model(f"\nFIRST TREE INFORMATION:")
            log_model(f"- Nodes: {first_tree.tree_.node_count}")
            log_model(f"- Depth: {first_tree.get_depth()}")
            log_model(f"- Leaves: {first_tree.get_n_leaves()}")
        
        # Additional parameters
        log_model(f"\nOTHER PARAMETERS:")
        if hasattr(model, 'classes_'):
            log_model(f"- Classes: {model.classes_}")
        else:
            log_model(f"- Classes: Not available (model not fitted)")
        log_model(f"- OOB score enabled: {model_params.get('oob_score', False)}")
        
        log_model("=" * 80)
        log_model("END OF MODEL INFORMATION".center(80))
        log_model("=" * 80)
    else:
        log_error("Model training failed!")
    
    # Test signal generation on sample pair
    # NOTE: This requires data fetching. In a full implementation, use DataFetcher
    if model is not None:
        log_info("Model training completed. To test signal generation, provide market data DataFrame.")
        log_info("Example: signal, confidence = get_latest_random_forest_signal(df_market_data, model)")
    else:
        log_error("Model is not available, cannot generate signal")

    elapsed_time = time.time() - start_time
    log_success(f"Process completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

