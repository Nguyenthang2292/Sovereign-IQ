"""
Display utilities for position sizing CLI.

This module provides formatted display functions for position sizing results.
"""

import pandas as pd
from typing import Optional
from colorama import Fore, Style

from modules.common.utils import (
    color_text,
    format_price,
)


def display_position_sizing_results(df: pd.DataFrame) -> None:
    """
    Display position sizing results in a formatted table.
    
    Args:
        df: DataFrame with position sizing results
    """
    if df is None or df.empty:
        print(color_text("\nNo position sizing results to display.", Fore.YELLOW))
        return
    
    print("\n" + color_text("=" * 120, Fore.CYAN, Style.BRIGHT))
    print(color_text("POSITION SIZING RESULTS", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 120, Fore.CYAN, Style.BRIGHT))
    
    # Select columns to display
    display_columns = [
        'symbol', 'signal_type', 'regime', 'position_size_usdt',
        'position_size_pct', 'kelly_fraction', 'adjusted_kelly_fraction',
    ]
    
    # Add metrics columns if available
    if 'metrics' in df.columns:
        # We'll display metrics separately
        pass
    
    # Create display DataFrame
    display_df = df[display_columns].copy() if all(col in df.columns for col in display_columns) else df.copy()
    
    # Format columns
    if 'position_size_usdt' in display_df.columns:
        display_df['position_size_usdt'] = display_df['position_size_usdt'].apply(lambda x: f"{x:.2f}")
    if 'position_size_pct' in display_df.columns:
        display_df['position_size_pct'] = display_df['position_size_pct'].apply(lambda x: f"{x:.2f}%")
    if 'kelly_fraction' in display_df.columns:
        display_df['kelly_fraction'] = display_df['kelly_fraction'].apply(lambda x: f"{x:.4f}")
    if 'adjusted_kelly_fraction' in display_df.columns:
        display_df['adjusted_kelly_fraction'] = display_df['adjusted_kelly_fraction'].apply(lambda x: f"{x:.4f}")
    
    # Rename columns for display
    display_df.columns = [
        'Symbol', 'Signal', 'Regime', 'Position Size (USDT)',
        'Position Size (%)', 'Kelly Fraction', 'Adjusted Kelly',
    ]
    
    print(display_df.to_string(index=False))
    print(color_text("=" * 120, Fore.CYAN, Style.BRIGHT))
    
    # Display metrics summary
    if 'metrics' in df.columns:
        print("\n" + color_text("PERFORMANCE METRICS", Fore.CYAN, Style.BRIGHT))
        print(color_text("-" * 120, Fore.CYAN))
        
        metrics_rows = []
        for _, row in df.iterrows():
            metrics = row.get('metrics', {})
            if isinstance(metrics, dict):
                metrics_row = {
                    'Symbol': row.get('symbol', 'N/A'),
                    'Win Rate': f"{metrics.get('win_rate', 0.0)*100:.2f}%",
                    'Avg Win': f"{metrics.get('avg_win', 0.0)*100:.2f}%",
                    'Avg Loss': f"{metrics.get('avg_loss', 0.0)*100:.2f}%",
                    'Sharpe': f"{metrics.get('sharpe_ratio', 0.0):.2f}",
                    'Max DD': f"{metrics.get('max_drawdown', 0.0)*100:.2f}%",
                    'Trades': metrics.get('num_trades', 0),
                }
                metrics_rows.append(metrics_row)
        
        if metrics_rows:
            metrics_df = pd.DataFrame(metrics_rows)
            print(metrics_df.to_string(index=False))
            print(color_text("-" * 120, Fore.CYAN))
    
    # Display portfolio summary
    if 'position_size_usdt' in df.columns:
        total_position_size = df['position_size_usdt'].sum()
        total_exposure_pct = df['position_size_pct'].sum() if 'position_size_pct' in df.columns else 0.0
        
        print("\n" + color_text("PORTFOLIO SUMMARY", Fore.CYAN, Style.BRIGHT))
        print(color_text("-" * 120, Fore.CYAN))
        print(f"Total Position Size: {color_text(f'{total_position_size:.2f} USDT', Fore.GREEN)}")
        print(f"Total Exposure: {color_text(f'{total_exposure_pct:.2f}%', Fore.YELLOW)}")
        print(f"Number of Positions: {color_text(str(len(df)), Fore.WHITE)}")
        print(color_text("-" * 120, Fore.CYAN))


def display_configuration(config: dict) -> None:
    """
    Display configuration information.
    
    Args:
        config: Dictionary with configuration values
    """
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("POSITION SIZING CONFIGURATION", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    
    print(color_text("Account Balance:", Fore.WHITE), f"{config.get('account_balance', 0.0):.2f} USDT")
    print(color_text("Timeframe:", Fore.WHITE), config.get('timeframe', 'N/A'))
    print(color_text("Lookback Days:", Fore.WHITE), config.get('lookback_days', 0))
    print(color_text("Max Position Size:", Fore.WHITE), f"{config.get('max_position_size', 0.0)*100:.1f}% of account balance")
    
    if 'source' in config:
        print(color_text("Source:", Fore.WHITE), config['source'])
    elif 'symbols_file' in config:
        print(color_text("Symbols File:", Fore.WHITE), config['symbols_file'])
    elif 'symbols' in config:
        print(color_text("Symbols:", Fore.WHITE), config['symbols'])
    
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

