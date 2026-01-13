"""
Signal generation workflow for LSTM CLI.

This module provides the workflow for generating trading signals using
trained LSTM models.
"""

from pathlib import Path
from typing import Optional, Union

from colorama import Fore, Style

from config.lstm import MODELS_DIR
from modules.common.ui.logging import log_error, log_info
from modules.common.utils import color_text, initialize_components
from modules.common.utils.data import fetch_ohlcv_data_dict
from modules.lstm.cli.interactive import prompt_symbol, prompt_timeframe
from modules.lstm.models.model_utils import get_latest_signal, load_model_and_scaler


def generate_signal_workflow(
    model_path: Optional[Union[Path, str]] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    limit: int = 1500,
) -> None:
    """
    Generate trading signal workflow.

    Args:
        model_path: Path to model checkpoint (can be Path object or string)
        symbol: Trading symbol
        timeframe: Timeframe
        limit: Number of candles to fetch
    """
    # Get symbol and timeframe (from args or prompt)
    if symbol is None:
        symbol = prompt_symbol()

    if timeframe is None:
        timeframe = prompt_timeframe()

    # Normalize symbol format (remove / if present)
    symbol = symbol.replace("/", "").upper()

    log_info("=" * 80)
    log_info("LSTM SIGNAL GENERATOR")
    log_info("=" * 80)
    log_info(f"Symbol: {symbol}")
    log_info(f"Timeframe: {timeframe}")
    log_info("=" * 80)

    # Load model and scaler
    log_info("\nLoading LSTM model...")
    model, scaler, look_back = load_model_and_scaler(model_path)

    if model is None:
        log_error("Failed to load model. Please train a model first or specify correct model path.")
        return

    # Initialize components
    log_info("\nInitializing exchange components...")
    exchange_manager, data_fetcher = initialize_components()

    # Fetch OHLCV data
    log_info(f"\nFetching OHLCV data for {symbol} {timeframe}...")
    try:
        all_data = fetch_ohlcv_data_dict(
            symbols=[symbol],
            timeframes=[timeframe],
            exchange_manager=exchange_manager,
            data_fetcher=data_fetcher,
            limit=limit,
        )

        if not all_data or symbol not in all_data:
            log_error(f"Failed to fetch data for {symbol}")
            return

        symbol_data = all_data[symbol]
        if not symbol_data or timeframe not in symbol_data:
            log_error(f"Failed to fetch data for {symbol} {timeframe}")
            return

        df_market_data = symbol_data[timeframe]

        if df_market_data.empty:
            log_error(f"Empty DataFrame for {symbol} {timeframe}")
            return

        log_info(f"Fetched {len(df_market_data)} candles")

    except Exception as e:
        log_error(f"Error fetching data: {e}")
        import traceback

        log_error(f"Traceback: {traceback.format_exc()}")
        return

    # Generate signal
    log_info("\nGenerating trading signal...")
    try:
        signal = get_latest_signal(df_market_data=df_market_data, model=model, scaler=scaler, look_back=look_back)

        # Display result
        log_info("\n" + "=" * 80)
        log_info("SIGNAL RESULT")
        log_info("=" * 80)

        if signal == "BUY":
            signal_color = Fore.GREEN
            signal_icon = "🟢"
        elif signal == "SELL":
            signal_color = Fore.RED
            signal_icon = "🔴"
        else:
            signal_color = Fore.YELLOW
            signal_icon = "🟡"

        print(color_text(f"\n{signal_icon} Signal: {signal}", signal_color, Style.BRIGHT))
        print(color_text(f"Symbol: {symbol}", Fore.CYAN))
        print(color_text(f"Timeframe: {timeframe}", Fore.CYAN))
        print(color_text(f"Model: {model_path or MODELS_DIR / 'lstm_model.pth'}", Fore.CYAN))
        log_info("=" * 80)

    except Exception as e:
        log_error(f"Error generating signal: {e}")
        import traceback

        log_error(f"Traceback: {traceback.format_exc()}")
