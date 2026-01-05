"""
CLI entry points for LSTM module.
"""

from .argument_parser import parse_args
from .interactive import (
    prompt_symbol,
    prompt_timeframe,
    display_main_menu,
    prompt_menu_choice,
    select_model_components,
    manage_symbols_menu,
    train_model_menu,
)
from .workflow import generate_signal_workflow

__all__ = [
    'parse_args',
    'prompt_symbol',
    'prompt_timeframe',
    'display_main_menu',
    'prompt_menu_choice',
    'select_model_components',
    'manage_symbols_menu',
    'train_model_menu',
    'generate_signal_workflow',
]
