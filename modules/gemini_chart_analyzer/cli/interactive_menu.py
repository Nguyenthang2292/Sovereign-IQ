"""
Interactive menu for Gemini Chart Analyzer CLI.

This module provides interactive configuration menu for chart analysis
with Google Gemini AI, following the pattern from other modules.
"""

import sys
import argparse
from typing import Optional, Dict, Tuple
from colorama import Fore, Style

from modules.common.utils import (
    color_text,
    prompt_user_input,
    normalize_timeframe,
)
from modules.common.ui.formatting import prompt_user_input_with_backspace


def _format_current_value(value) -> str:
    """Format current value for display in menu."""
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        if 'periods' in value:
            return f"periods={value['periods']}"
        if 'period' in value:
            return f"period={value['period']}"
        return str(value)
    return str(value)


def _display_main_menu(config):
    """Display main menu with current configuration values."""
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("GEMINI CHART ANALYZER - Configuration Menu", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print("\n" + color_text("MAIN MENU", Fore.YELLOW, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))
    
    # Format current values
    symbol_val = _format_current_value(getattr(config, 'symbol', None))
    timeframe_val = _format_current_value(getattr(config, 'timeframe', '1h'))
    timeframes_val = _format_current_value(getattr(config, 'timeframes_list', None))
    
    # Indicators
    ma_val = "disabled"
    if hasattr(config, 'indicators') and 'MA' in config.indicators:
        ma_val = f"periods={config.indicators['MA']['periods']}"
    elif not getattr(config, 'no_ma', False):
        ma_val = "periods=[20, 50, 200]"
    
    rsi_val = "disabled"
    if hasattr(config, 'indicators') and 'RSI' in config.indicators:
        rsi_val = f"period={config.indicators['RSI']['period']}"
    elif not getattr(config, 'no_rsi', False):
        rsi_val = "period=14"
    
    macd_val = "disabled" if getattr(config, 'no_macd', False) else "enabled"
    bb_val = "disabled"
    if hasattr(config, 'indicators') and 'BB' in config.indicators:
        bb_val = f"period={config.indicators['BB']['period']}"
    elif getattr(config, 'enable_bb', False):
        bb_val = "period=20"
    
    prompt_type_val = _format_current_value(getattr(config, 'prompt_type', 'detailed'))
    custom_prompt_val = _format_current_value(getattr(config, 'custom_prompt', None))
    
    if timeframes_val and timeframes_val != "not set":
        tf_display = f'{symbol_val} / Multi-TF: {timeframes_val}'
    else:
        tf_display = f'{symbol_val} / {timeframe_val}'
    print(f"  1. Symbol & Timeframe [{color_text(tf_display, Fore.GREEN)}]")
    print(f"  2. Indicators Configuration [{color_text(f'MA={ma_val}, RSI={rsi_val}, MACD={macd_val}, BB={bb_val}', Fore.GREEN)}]")
    print(f"  3. Gemini Prompt Configuration [{color_text(f'{prompt_type_val}', Fore.GREEN)}]")
    print(f"  4. Review and Confirm")
    print(f"  5. Exit")
    print(color_text("-" * 80, Fore.CYAN))


def _prompt_with_back(prompt: str, default: str = None, allow_back: bool = True) -> Tuple[Optional[str], str]:
    """
    Prompt user with backspace key for back navigation.
    
    Returns:
        (value, action) where action is 'main' or 'continue'
    """
    if allow_back:
        back_prompt = f"{prompt} (press Backspace to go back): "
    else:
        back_prompt = prompt
    
    if allow_back:
        user_input, is_back = prompt_user_input_with_backspace(back_prompt, default=default)
        
        if is_back:
            return (None, 'main')
        
        user_input = user_input.strip() if user_input is not None else None
        
        return (user_input, 'continue')
    else:
        user_input = prompt_user_input(back_prompt, default=default).strip()
        return (user_input, 'continue')
def _configure_symbol_timeframe(config):
    """Configure symbol and timeframe with back option."""
    print("\n" + color_text("1. SYMBOL & TIMEFRAME", Fore.YELLOW, Style.BRIGHT))
    print(color_text("   b) Back to main menu", Fore.CYAN))
    print()
    
    # Symbol
    current_symbol = getattr(config, 'symbol', None)
    symbol_input, action = _prompt_with_back(
        f"Enter symbol (e.g., BTC/USDT) [{current_symbol or 'BTC/USDT'}]: ",
        default=current_symbol or "BTC/USDT"
    )
    if action == 'main':
        return ('main', False)
    
    if not symbol_input:
        symbol_input = "BTC/USDT"
    
    # Choose single or multi-timeframe mode
    current_timeframes_list = getattr(config, 'timeframes_list', None)
    use_multi_tf = current_timeframes_list is not None and len(current_timeframes_list) > 0
    
    print("\nAnalysis mode:")
    print("  1. Single timeframe")
    print("  2. Multi-timeframe (recommended)")
    mode_input, action = _prompt_with_back(
        f"Select mode (1/2) [{('2' if use_multi_tf else '1')}]: ",
        default='2' if use_multi_tf else '1'
    )
    if action == 'main':
        return ('main', False)
    
    if not mode_input:
        mode_input = '2' if use_multi_tf else '1'
    
    changed = (symbol_input != current_symbol)
    
    if mode_input == '2':
        # Multi-timeframe mode
        from modules.gemini_chart_analyzer.core.utils import DEFAULT_TIMEFRAMES, normalize_timeframes
        
        current_tfs_str = ", ".join(current_timeframes_list) if current_timeframes_list else ", ".join(DEFAULT_TIMEFRAMES)
        print(f"\nDefault timeframes: {', '.join(DEFAULT_TIMEFRAMES)}")
        print("Timeframes: 15m, 30m, 1h, 4h, 1d, 1w (comma-separated)")
        timeframes_input, action = _prompt_with_back(
            f"Enter timeframes (comma-separated) [{current_tfs_str}]: ",
            default=current_tfs_str
        )
        if action == 'main':
            return ('main', False)
        
        if not timeframes_input:
            timeframes_input = ", ".join(DEFAULT_TIMEFRAMES)
        
        try:
            timeframes_list = [tf.strip() for tf in timeframes_input.split(',') if tf.strip()]
            normalized_tfs = normalize_timeframes(timeframes_list)
            if normalized_tfs:
                config.timeframes_list = normalized_tfs
                config.timeframe = None  # Clear single timeframe
                if normalized_tfs != (current_timeframes_list or []):
                    changed = True
            else:
                print(color_text("Warning: No valid timeframes. Using single timeframe mode.", Fore.YELLOW))
                config.timeframes_list = None
                config.timeframe = '1h'
        except Exception as e:
            print(color_text(f"Warning: Error parsing timeframes: {e}. Using single timeframe mode.", Fore.YELLOW))
            config.timeframes_list = None
            config.timeframe = '1h'
    else:
        # Single timeframe mode
        current_timeframe = getattr(config, 'timeframe', '1h')
        print("\nTimeframes: 15m (or m15), 30m (or m30), 1h (or h1), 4h (or h4), 1d (or d1), 1w (or w1)")
        timeframe_input, action = _prompt_with_back(
            f"Enter timeframe [{current_timeframe}]: ",
            default=current_timeframe
        )
        if action == 'main':
            return ('main', False)
        
        if not timeframe_input:
            timeframe_input = '1h'
        
        # Normalize timeframe (accept both '15m' and 'm15', '1h' and 'h1', etc.)
        timeframe_input = normalize_timeframe(timeframe_input)
        
        config.timeframe = timeframe_input
        config.timeframes_list = None  # Clear multi-timeframe
        if timeframe_input != current_timeframe:
            changed = True
    
    config.symbol = symbol_input
    
    return ('main', changed)


def _configure_indicators(config):
    """Configure indicators with back option."""
    print("\n" + color_text("2. INDICATORS CONFIGURATION", Fore.YELLOW, Style.BRIGHT))
    print(color_text("   b) Back to main menu", Fore.CYAN))
    print()
    
    if not hasattr(config, 'indicators'):
        config.indicators = {}
    
    changed = False
    
    # Moving Averages
    current_ma = config.indicators.get('MA', {'periods': [20, 50, 200]}) if not getattr(config, 'no_ma', False) else None
    ma_periods_str = ",".join(map(str, current_ma['periods'])) if current_ma else ""
    
    use_ma_input, action = _prompt_with_back(
        f"Use Moving Averages? (y/n) [{('y' if current_ma else 'n')}]: ",
        default='y' if current_ma else 'n'
    )
    if action == 'main':
        return ('main', False)
    
    use_ma = use_ma_input.lower() in ['y', 'yes', '']
    if not use_ma:
        config.no_ma = True
        config.indicators.pop('MA', None)
        if current_ma:
            changed = True
    else:
        config.no_ma = False
        ma_periods_input, action = _prompt_with_back(
            f"MA periods (comma-separated, e.g., 20,50,200) [{ma_periods_str}]: ",
            default=ma_periods_str
        )
        if action == 'main':
            return ('main', False)
        
        if ma_periods_input:
            try:
                ma_periods = [int(p.strip()) for p in ma_periods_input.split(',')]
                config.indicators['MA'] = {'periods': ma_periods}
                if not current_ma or list(ma_periods) != list(current_ma.get('periods', [])):
                    changed = True
            except ValueError:
                print(color_text("Invalid input, using default: [20, 50, 200]", Fore.YELLOW))
                config.indicators['MA'] = {'periods': [20, 50, 200]}
        else:
            config.indicators['MA'] = {'periods': [20, 50, 200]}
    
    # RSI
    current_rsi = config.indicators.get('RSI', {'period': 14}) if not getattr(config, 'no_rsi', False) else None
    
    use_rsi_input, action = _prompt_with_back(
        f"Use RSI? (y/n) [{('y' if current_rsi else 'n')}]: ",
        default='y' if current_rsi else 'n'
    )
    if action == 'main':
        return ('main', False)
    
    use_rsi = use_rsi_input.lower() in ['y', 'yes', '']
    if not use_rsi:
        config.no_rsi = True
        config.indicators.pop('RSI', None)
        if current_rsi:
            changed = True
    else:
        config.no_rsi = False
        rsi_period_input, action = _prompt_with_back(
            f"RSI period [{current_rsi['period'] if current_rsi else 14}]: ",
            default=str(current_rsi['period']) if current_rsi else "14"
        )
        if action == 'main':
            return ('main', False)
        
        if rsi_period_input:
            try:
                rsi_period = int(rsi_period_input)
                config.indicators['RSI'] = {'period': rsi_period}
                if not current_rsi or rsi_period != current_rsi.get('period', 14):
                    changed = True
            except ValueError:
                print(color_text("Invalid input, using default: 14", Fore.YELLOW))
                config.indicators['RSI'] = {'period': 14}
        else:
            config.indicators['RSI'] = {'period': 14}
    
    # MACD
    current_macd = not getattr(config, 'no_macd', False)
    
    use_macd_input, action = _prompt_with_back(
        f"Use MACD? (y/n) [{('y' if current_macd else 'n')}]: ",
        default='y' if current_macd else 'n'
    )
    if action == 'main':
        return ('main', False)
    
    use_macd = use_macd_input.lower() in ['y', 'yes', '']
    config.no_macd = not use_macd
    if use_macd:
        config.indicators['MACD'] = {'fast': 12, 'slow': 26, 'signal': 9}
        if not current_macd:
            changed = True
    else:
        config.indicators.pop('MACD', None)
        if current_macd:
            changed = True
    
    # Bollinger Bands
    current_bb = config.indicators.get('BB') if getattr(config, 'enable_bb', False) else None
    
    use_bb_input, action = _prompt_with_back(
        f"Use Bollinger Bands? (y/n) [{('y' if current_bb else 'n')}]: ",
        default='y' if current_bb else 'n'
    )
    if action == 'main':
        return ('main', False)
    
    use_bb = use_bb_input.lower() in ['y', 'yes', '']
    if use_bb:
        bb_period_input, action = _prompt_with_back(
            f"BB period [{current_bb['period'] if current_bb else 20}]: ",
            default=str(current_bb['period']) if current_bb else "20"
        )
        if action == 'main':
            return ('main', False)
        
        if bb_period_input:
            try:
                bb_period = int(bb_period_input)
                config.indicators['BB'] = {'period': bb_period, 'std': 2}
                if not current_bb or bb_period != current_bb.get('period', 20):
                    changed = True
            except ValueError:
                print(color_text("Invalid input, using default: 20", Fore.YELLOW))
                config.indicators['BB'] = {'period': 20, 'std': 2}
        else:
            config.indicators['BB'] = {'period': 20, 'std': 2}
    else:
        config.indicators.pop('BB', None)
        if current_bb:
            changed = True
    
    return ('main', changed)


def _configure_prompt(config):
    """Configure Gemini prompt type with back option."""
    print("\n" + color_text("3. GEMINI PROMPT CONFIGURATION", Fore.YELLOW, Style.BRIGHT))
    print("  1. Detailed - Phân tích chi tiết với cấu trúc đầy đủ (mặc định)")
    print("  2. Simple - Phân tích đơn giản")
    print("  3. Custom - Nhập prompt tùy chỉnh")
    print(color_text("  b) Back to main menu", Fore.CYAN))
    print()
    
    current_prompt_type = getattr(config, 'prompt_type', 'detailed')
    current_custom = getattr(config, 'custom_prompt', None)
    
    choice_input, action = _prompt_with_back(
        f"Select prompt type (1/2/3) [{('1' if current_prompt_type == 'detailed' else '2' if current_prompt_type == 'simple' else '3')}]: ",
        default='1' if current_prompt_type == 'detailed' else '2' if current_prompt_type == 'simple' else '3'
    )
    if action == 'main':
        return ('main', False)
    
    if not choice_input:
        choice_input = '1'
    
    changed = False
    if choice_input == '2':
        config.prompt_type = 'simple'
        config.custom_prompt = None
        if current_prompt_type != 'simple':
            changed = True
    elif choice_input == '3':
        config.prompt_type = 'custom'
        custom_prompt_input, action = _prompt_with_back(
            f"Enter custom prompt [{current_custom or ''}]: ",
            default=current_custom or ""
        )
        if action == 'main':
            return ('main', False)
        
        if not custom_prompt_input:
            print(color_text("Warning: Custom prompt cannot be empty. Using 'detailed' instead.", Fore.YELLOW))
            config.prompt_type = 'detailed'
            config.custom_prompt = None
        else:
            config.custom_prompt = custom_prompt_input
            if custom_prompt_input != current_custom:
                changed = True
    else:
        config.prompt_type = 'detailed'
        config.custom_prompt = None
        if current_prompt_type != 'detailed':
            changed = True
    
    return ('main', changed)


def _review_and_confirm(config):
    """Review configuration and confirm."""
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("CONFIGURATION REVIEW", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    
    print(f"\nSymbol: {getattr(config, 'symbol', 'not set')}")
    timeframes_list = getattr(config, 'timeframes_list', None)
    if timeframes_list:
        print(f"Timeframes: {', '.join(timeframes_list)} (Multi-timeframe mode)")
    else:
        print(f"Timeframe: {getattr(config, 'timeframe', '1h')} (Single timeframe mode)")
    
    print(f"\nIndicators:")
    indicators = getattr(config, 'indicators', {})
    if 'MA' in indicators:
        print(f"  - Moving Averages: periods={indicators['MA']['periods']}")
    else:
        print(f"  - Moving Averages: disabled")
    
    if 'RSI' in indicators:
        print(f"  - RSI: period={indicators['RSI']['period']}")
    else:
        print(f"  - RSI: disabled")
    
    if 'MACD' in indicators:
        print(f"  - MACD: enabled")
    else:
        print(f"  - MACD: disabled")
    
    if 'BB' in indicators:
        print(f"  - Bollinger Bands: period={indicators['BB']['period']}")
    else:
        print(f"  - Bollinger Bands: disabled")
    
    print(f"\nGemini Prompt Type: {getattr(config, 'prompt_type', 'detailed')}")
    if getattr(config, 'prompt_type', 'detailed') == 'custom':
        custom_prompt = getattr(config, 'custom_prompt', '')
        print(f"Custom Prompt: {custom_prompt[:50]}{'...' if len(custom_prompt) > 50 else ''}")
    
    print("\n" + color_text("-" * 80, Fore.CYAN))
    confirm = prompt_user_input("Confirm this configuration? (y/n) [y]: ", default="y").strip().lower()
    
    if confirm in ['y', 'yes', '']:
        return 'done'
    else:
        return 'main'


def interactive_config_menu():
    """
    Interactive menu for configuring Gemini Chart Analyzer.
    
    Returns:
        argparse.Namespace object with all configuration values
    """
    # Create namespace object with defaults
    config = argparse.Namespace()
    
    # Initialize default values
    config.symbol = None
    config.timeframe = '1h'
    config.indicators = {}
    config.no_ma = False
    config.no_rsi = False
    config.no_macd = False
    config.enable_bb = False
    config.prompt_type = 'detailed'
    config.custom_prompt = None
    config.limit = 500
    config.chart_figsize_tuple = (16, 10)
    config.chart_dpi = 150
    config.no_cleanup = False
    
    # Track unsaved changes
    has_unsaved_changes = False
    
    # Main menu loop
    while True:
        _display_main_menu(config)
        
        choice = prompt_user_input("\nSelect option [1-5]: ").strip()
        
        if choice == '1':
            _, changed = _configure_symbol_timeframe(config)
            if changed:
                has_unsaved_changes = True
        elif choice == '2':
            _, changed = _configure_indicators(config)
            if changed:
                has_unsaved_changes = True
        elif choice == '3':
            _, changed = _configure_prompt(config)
            if changed:
                has_unsaved_changes = True
        elif choice == '4':
            result = _review_and_confirm(config)
            if result == 'done':
                has_unsaved_changes = False  # Changes are confirmed/saved
                break
        elif choice == '5':
            # Prompt for confirmation before exiting
            if has_unsaved_changes:
                confirm_msg = color_text(
                    "\n⚠️  Are you sure you want to exit? Unsaved changes will be lost. (y/N): ",
                    Fore.YELLOW
                )
            else:
                confirm_msg = color_text(
                    "\nAre you sure you want to exit? (y/N): ",
                    Fore.YELLOW
                )
            
            confirm = prompt_user_input(confirm_msg, default="n").strip().lower()
            
            if confirm in ['y', 'yes']:
                print(color_text("\nExiting configuration menu.", Fore.YELLOW))
                sys.exit(0)
            else:
                # User chose not to exit, return to menu
                continue
        else:
            print(color_text("Invalid choice. Please select 1-5.", Fore.RED))
    
    return config

