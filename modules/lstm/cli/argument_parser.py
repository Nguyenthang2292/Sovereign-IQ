"""
Command-line argument parser for LSTM Model Manager.

This module provides the main argument parser for the LSTM CLI,
defining all command-line options and their default values.
"""

import argparse
from pathlib import Path

from config.lstm import MODELS_DIR
from config.common import DEFAULT_SYMBOL, DEFAULT_TIMEFRAME


def parse_args():
    """
    Parse command-line arguments for LSTM Model Manager.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate trading signals using trained LSTM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate signal with default model
  python main_lstm.py --symbol BTCUSDT --timeframe 1h
  
  # Generate signal with custom model path
  python main_lstm.py --symbol ETHUSDT --timeframe 15m --model-path artifacts/models/lstm/cnn_lstm_attention_model.pth
  
  # Interactive mode (prompt for symbol and timeframe)
  python main_lstm.py
        """
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default=None,
        help=f'Trading symbol (e.g., BTCUSDT, ETHUSDT). Default: {DEFAULT_SYMBOL}'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default=None,
        help=f'Timeframe (e.g., 15m, 1h, 4h). Default: {DEFAULT_TIMEFRAME}'
    )
    
    parser.add_argument(
        '--model-path',
        type=Path,
        default=None,
        help=f'Path to model checkpoint file. Default: {MODELS_DIR / "lstm_model.pth"}'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=1500,
        help='Number of candles to fetch (default: 1500)'
    )
    
    return parser.parse_args()

