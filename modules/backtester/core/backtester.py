"""
Full Backtester for trading strategy simulation.

This module simulates trading with entry/exit rules based on signals
from indicators (ATC, Oscillator, SPC, etc.) and calculates performance metrics.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import traceback
import time
import pickle
from multiprocessing import Pool, cpu_count
import functools

from modules.common.core.data_fetcher import DataFetcher
from modules.common.quantitative_metrics.risk import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)
from config.position_sizing import (
    BACKTEST_STOP_LOSS_PCT,
    BACKTEST_TAKE_PROFIT_PCT,
    BACKTEST_TRAILING_STOP_PCT,
    BACKTEST_MAX_HOLD_PERIODS,
    ENABLED_INDICATORS,
    USE_CONFIDENCE_WEIGHTING,
    MIN_INDICATORS_AGREEMENT,
    HYBRID_OSC_LENGTH,
    HYBRID_OSC_MULT,
    HYBRID_OSC_STRATEGIES,
    HYBRID_SPC_PARAMS,
    ENABLE_PARALLEL_PROCESSING,
    NUM_WORKERS,
    BATCH_SIZE,
    ENABLE_PERFORMANCE_PROFILING,
    LOG_PERFORMANCE_METRICS,
)
from modules.common.utils import (
    log_error,
    log_warn,
    log_progress,
)


class FullBacktester:
    """
    Full backtester that simulates trading based on indicator signals.
    
    Simulates entry/exit rules, tracks trades, calculates PnL,
    and computes performance metrics (win rate, Sharpe ratio, max drawdown, etc.).
    """
    
    def __init__(
        self,
        data_fetcher: DataFetcher,
        stop_loss_pct: float = BACKTEST_STOP_LOSS_PCT,
        take_profit_pct: float = BACKTEST_TAKE_PROFIT_PCT,
        trailing_stop_pct: float = BACKTEST_TRAILING_STOP_PCT,
        max_hold_periods: int = BACKTEST_MAX_HOLD_PERIODS,
        enabled_indicators: Optional[List[str]] = None,
        use_confidence_weighting: bool = USE_CONFIDENCE_WEIGHTING,
        min_indicators_agreement: int = MIN_INDICATORS_AGREEMENT,
    ):
        """
        Initialize Full Backtester.
        
        Args:
            data_fetcher: DataFetcher instance for fetching OHLCV data
            stop_loss_pct: Stop loss percentage (default: 2%)
            take_profit_pct: Take profit percentage (default: 4%)
            trailing_stop_pct: Trailing stop percentage (default: 1.5%)
            max_hold_periods: Maximum periods to hold a position (default: 100)
            enabled_indicators: List of enabled indicators (default: from config)
            use_confidence_weighting: Whether to weight votes by confidence (default: True)
            min_indicators_agreement: Minimum indicators that must agree (default: 3)
        """
        self.data_fetcher = data_fetcher
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_hold_periods = max_hold_periods
        
        # Lazy import to avoid circular dependency
        from modules.position_sizing.core.hybrid_signal_calculator import HybridSignalCalculator
        
        # Initialize Hybrid Signal Calculator
        self.hybrid_signal_calculator = HybridSignalCalculator(
            data_fetcher=data_fetcher,
            enabled_indicators=enabled_indicators or ENABLED_INDICATORS,
            use_confidence_weighting=use_confidence_weighting,
            min_indicators_agreement=min_indicators_agreement,
        )
    
    def backtest(
        self,
        symbol: str,
        timeframe: str,
        lookback: int,
        signal_type: str,
        initial_capital: float = 10000.0,
    ) -> Dict:
        """
        Run full backtest simulation for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            lookback: Number of candles to look back (should be converted from days using days_to_candles)
            signal_type: "LONG" or "SHORT"
            initial_capital: Initial capital for backtesting (default: 10000)
            
        Returns:
            Dictionary with backtest results:
            {
                'trades': List[Dict],  # List of trade records
                'equity_curve': pd.Series,  # Cumulative equity over time
                'metrics': {
                    'win_rate': float,
                    'avg_win': float,
                    'avg_loss': float,
                    'total_return': float,
                    'sharpe_ratio': float,
                    'max_drawdown': float,
                    'num_trades': int,
                    'profit_factor': float,
                }
            }
        """
        try:
            # Fetch historical data
            df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=lookback,
                timeframe=timeframe,
                check_freshness=False,
            )
            
            if df is None or df.empty:
                log_warn(f"No data available for {symbol}")
                return self._empty_backtest_result()
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                log_error(f"Missing required columns for {symbol}: {missing_columns}")
                return self._empty_backtest_result()
            
            # Calculate signals for each period
            start_time = time.time() if (ENABLE_PERFORMANCE_PROFILING or LOG_PERFORMANCE_METRICS) else None
            if ENABLE_PARALLEL_PROCESSING and len(df) > 100:  # Only use parallel for large datasets
                signals = self._calculate_signals_parallel(df, symbol, timeframe, lookback, signal_type)
            else:
                signals = self._calculate_signals(df, symbol, timeframe, lookback, signal_type)
            
            if start_time and LOG_PERFORMANCE_METRICS:
                elapsed = time.time() - start_time
                log_progress(f"  Signal calculation took {elapsed:.2f} seconds")
            
            # Simulate trades
            trades = self._simulate_trades(df, signals, signal_type, initial_capital)
            
            if not trades:
                log_warn(f"No trades generated for {symbol}")
                return self._empty_backtest_result()
            
            # Calculate equity curve
            equity_curve = self._calculate_equity_curve(trades, initial_capital, len(df))
            
            # Calculate performance metrics
            metrics = self._calculate_metrics(trades, equity_curve)
            
            return {
                'trades': trades,
                'equity_curve': equity_curve,
                'metrics': metrics,
            }
            
        except Exception as e:
            log_error(f"Error backtesting {symbol}: {e}")
            log_error(f"Traceback: {traceback.format_exc()}")
            return self._empty_backtest_result()
    
    def _calculate_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        limit: int,
        signal_type: str,
    ) -> pd.Series:
        """
        Calculate signals for each period in the DataFrame using hybrid approach.
        
        Uses rolling window: for each period, calculates signals using only
        historical data up to that period (walk-forward testing).
        
        Returns:
            Series with signal values (1 for LONG entry, -1 for SHORT entry, 0 for no signal)
        """
        signals = pd.Series(0, index=df.index)
        
        # Calculate signals for each period using rolling window
        # This ensures we only use data available at that time (walk-forward testing)
        log_progress(f"  Calculating hybrid signals for {len(df)} periods...")
        
        # Calculate signals for each period (caching is handled internally by HybridSignalCalculator)
        for i in range(len(df)):
            try:
                # Calculate hybrid signal for this period
                signal, confidence = self.hybrid_signal_calculator.calculate_hybrid_signal(
                    df=df,
                    symbol=symbol,
                    timeframe=timeframe,
                    period_index=i,
                    signal_type=signal_type,
                    osc_length=HYBRID_OSC_LENGTH,
                    osc_mult=HYBRID_OSC_MULT,
                    osc_strategies=HYBRID_OSC_STRATEGIES,
                    spc_params=HYBRID_SPC_PARAMS,
                )
                
                # Store signal (confidence is not used in trade simulation, but could be)
                signals.iloc[i] = signal
                
            except Exception as e:
                log_warn(f"Error calculating signal for period {i}: {e}")
                signals.iloc[i] = 0  # Default to no signal on error
        
        # Log signal statistics
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        neutral_signals = (signals == 0).sum()
        log_progress(f"  Signal distribution: LONG={long_signals}, SHORT={short_signals}, NEUTRAL={neutral_signals}")
        
        # Log cache statistics
        cache_stats = self.hybrid_signal_calculator.get_cache_stats()
        log_progress(f"  Cache stats: {cache_stats['signal_cache_size']}/{cache_stats['signal_cache_max_size']} signals cached")
        
        return signals
    
    def _calculate_signals_parallel(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        limit: int,
        signal_type: str,
    ) -> pd.Series:
        """
        Calculate signals for each period using multiprocessing (batch processing).
        
        Divides periods into batches and processes them in parallel.
        
        Returns:
            Series with signal values (1 for LONG entry, -1 for SHORT entry, 0 for no signal)
        """
        signals = pd.Series(0, index=df.index)
        
        # Determine number of workers
        num_workers = NUM_WORKERS if NUM_WORKERS is not None else cpu_count()
        num_workers = min(num_workers, len(df))  # Don't use more workers than periods
        
        # Calculate batch size
        if BATCH_SIZE is not None:
            batch_size = BATCH_SIZE
        else:
            batch_size = max(1, len(df) // num_workers)
        
        # Create batches
        batches = []
        for i in range(0, len(df), batch_size):
            end_idx = min(i + batch_size, len(df))
            batches.append((i, end_idx))
        
        log_progress(f"  Calculating signals in parallel: {len(batches)} batches, {num_workers} workers")
        
        # Prepare arguments for worker function
        # We need to pass the hybrid_signal_calculator config, not the object itself
        # (multiprocessing can't pickle complex objects)
        calc_args = {
            'symbol': symbol,
            'timeframe': timeframe,
            'limit': limit,
            'signal_type': signal_type,
            'osc_length': HYBRID_OSC_LENGTH,
            'osc_mult': HYBRID_OSC_MULT,
            'osc_strategies': HYBRID_OSC_STRATEGIES,
            'spc_params': HYBRID_SPC_PARAMS,
            'enabled_indicators': self.hybrid_signal_calculator.enabled_indicators,
            'use_confidence_weighting': self.hybrid_signal_calculator.use_confidence_weighting,
            'min_indicators_agreement': self.hybrid_signal_calculator.min_indicators_agreement,
        }
        
        # Pickle DataFrame for passing to workers
        # (DataFrame can be large but pickle is efficient for this)
        df_bytes = pickle.dumps(df)
        
        # Use multiprocessing Pool
        try:
            with Pool(processes=num_workers) as pool:
                # Create partial function with fixed arguments
                worker_func = functools.partial(
                    _calculate_signal_batch_worker,
                    df_bytes=df_bytes,
                    **calc_args
                )
                
                # Process batches in parallel
                results = pool.starmap(worker_func, batches)
            
            # Merge results
            for batch_signals in results:
                if batch_signals is not None:
                    for idx, signal_val in batch_signals.items():
                        signals.iloc[idx] = signal_val
            
        except Exception as e:
            log_error(f"Error in parallel signal calculation: {e}")
            log_warn("Falling back to sequential calculation")
            signals = self._calculate_signals(df, symbol, timeframe, limit, signal_type)
        
        # Log signal statistics
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        neutral_signals = (signals == 0).sum()
        log_progress(f"  Signal distribution: LONG={long_signals}, SHORT={short_signals}, NEUTRAL={neutral_signals}")
        
        return signals
    
    def _simulate_trades(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        signal_type: str,
        initial_capital: float,
    ) -> List[Dict]:
        """
        Simulate trades based on signals.
        
        Returns:
            List of trade dictionaries with entry/exit information
        """
        trades = []
        position = None  # Current position: None, or dict with entry info
        
        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            
            # Check if we should exit current position
            if position is not None:
                exit_reason = None
                exit_price = None
                pnl = 0.0
                
                # Check stop loss
                if signal_type.upper() == "LONG":
                    stop_loss_price = position['entry_price'] * (1 - self.stop_loss_pct)
                    if low <= stop_loss_price:
                        exit_reason = "STOP_LOSS"
                        exit_price = stop_loss_price
                        pnl = (exit_price - position['entry_price']) / position['entry_price']
                    else:
                        # Check take profit
                        take_profit_price = position['entry_price'] * (1 + self.take_profit_pct)
                        if high >= take_profit_price:
                            exit_reason = "TAKE_PROFIT"
                            exit_price = take_profit_price
                            pnl = (exit_price - position['entry_price']) / position['entry_price']
                        # Check trailing stop
                        # FIXED: Trailing stop only activates after price moves favorably
                        # Previously, trailing stop was initialized immediately on entry, causing premature exits
                        elif current_price > position['highest_price']:
                            position['highest_price'] = current_price
                            position['trailing_stop'] = current_price * (1 - self.trailing_stop_pct)
                        elif position['trailing_stop'] is not None and current_price <= position['trailing_stop']:
                            exit_reason = "TRAILING_STOP"
                            exit_price = position['trailing_stop']
                            pnl = (exit_price - position['entry_price']) / position['entry_price']
                        # Check max hold period
                        elif (i - position['entry_index']) >= self.max_hold_periods:
                            exit_reason = "MAX_HOLD"
                            exit_price = current_price
                            pnl = (exit_price - position['entry_price']) / position['entry_price']
                else:  # SHORT
                    stop_loss_price = position['entry_price'] * (1 + self.stop_loss_pct)
                    if high >= stop_loss_price:
                        exit_reason = "STOP_LOSS"
                        exit_price = stop_loss_price
                        pnl = (position['entry_price'] - exit_price) / position['entry_price']
                    else:
                        # Check take profit
                        take_profit_price = position['entry_price'] * (1 - self.take_profit_pct)
                        if low <= take_profit_price:
                            exit_reason = "TAKE_PROFIT"
                            exit_price = take_profit_price
                            pnl = (position['entry_price'] - exit_price) / position['entry_price']
                        # Check trailing stop
                        # FIXED: Trailing stop only activates after price moves favorably
                        # Previously, trailing stop was initialized immediately on entry, causing premature exits
                        elif current_price < position['lowest_price']:
                            position['lowest_price'] = current_price
                            position['trailing_stop'] = current_price * (1 + self.trailing_stop_pct)
                        elif position['trailing_stop'] is not None and current_price >= position['trailing_stop']:
                            exit_reason = "TRAILING_STOP"
                            exit_price = position['trailing_stop']
                            pnl = (position['entry_price'] - exit_price) / position['entry_price']
                        # Check max hold period
                        elif (i - position['entry_index']) >= self.max_hold_periods:
                            exit_reason = "MAX_HOLD"
                            exit_price = current_price
                            pnl = (position['entry_price'] - exit_price) / position['entry_price']
                
                if exit_reason:
                    # Close position
                    trade = {
                        'entry_index': position['entry_index'],
                        'exit_index': i,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'entry_time': df.index[position['entry_index']],
                        'exit_time': df.index[i],
                        'signal_type': signal_type,
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'pnl_pct': pnl * 100,
                        'hold_periods': i - position['entry_index'],
                    }
                    trades.append(trade)
                    position = None
            
            # Check if we should enter a new position
            if position is None and signals.iloc[i] != 0:
                if (signal_type.upper() == "LONG" and signals.iloc[i] > 0) or \
                   (signal_type.upper() == "SHORT" and signals.iloc[i] < 0):
                    # FIXED: Trailing stop initialization
                    # Previously: trailing_stop was initialized immediately on entry, causing premature exits
                    # Now: trailing_stop is None initially, only set after price moves favorably
                    # This prevents trailing stop from triggering immediately if price moves against position
                    position = {
                        'entry_index': i,
                        'entry_price': current_price,
                        'entry_time': df.index[i],
                        'highest_price': current_price if signal_type.upper() == "LONG" else None,
                        'lowest_price': current_price if signal_type.upper() == "SHORT" else None,
                        'trailing_stop': None,  # Will be set after favorable price movement
                    }
        
        # Close any remaining position at the end
        if position is not None:
            final_price = df.iloc[-1]['close']
            pnl = (final_price - position['entry_price']) / position['entry_price'] if signal_type.upper() == "LONG" else (position['entry_price'] - final_price) / position['entry_price']
            trade = {
                'entry_index': position['entry_index'],
                'exit_index': len(df) - 1,
                'entry_price': position['entry_price'],
                'exit_price': final_price,
                'entry_time': df.index[position['entry_index']],
                'exit_time': df.index[-1],
                'signal_type': signal_type,
                'exit_reason': "END_OF_DATA",
                'pnl': pnl,
                'pnl_pct': pnl * 100,
                'hold_periods': len(df) - 1 - position['entry_index'],
            }
            trades.append(trade)
        
        return trades
    
    def _calculate_equity_curve(
        self,
        trades: List[Dict],
        initial_capital: float,
        num_periods: int,
    ) -> pd.Series:
        """
        Calculate equity curve from trades.
        
        Returns:
            Series with cumulative equity values
        """
        equity = [initial_capital]
        current_capital = initial_capital
        
        for trade in trades:
            # Assume we risk a fixed percentage per trade (e.g., 1%)
            # Formula: trade_pnl = current_capital * risk_per_trade * trade['pnl']
            # Where trade['pnl'] is the percentage return of the trade
            risk_per_trade = 0.01
            trade_pnl = current_capital * risk_per_trade * trade['pnl']
            current_capital += trade_pnl
            equity.append(current_capital)
        
        # Pad to num_periods if needed
        while len(equity) < num_periods:
            equity.append(current_capital)
        
        return pd.Series(equity[:num_periods])
    
    def _calculate_metrics(
        self,
        trades: List[Dict],
        equity_curve: pd.Series,
    ) -> Dict:
        """
        Calculate performance metrics from trades and equity curve.
        
        Returns:
            Dictionary with performance metrics
        """
        if not trades:
            return {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0,
                'profit_factor': 0.0,
            }
        
        # Calculate win rate
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        
        # Calculate average win/loss
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0.0
        
        # Calculate total return
        initial_capital = equity_curve.iloc[0] if len(equity_curve) > 0 else 10000.0
        final_capital = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
        total_return = (final_capital - initial_capital) / initial_capital if initial_capital > 0 else 0.0
        
        # Calculate Sharpe ratio
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = calculate_sharpe_ratio(returns, periods_per_year=365*24) or 0.0
        
        # Calculate max drawdown
        max_drawdown = calculate_max_drawdown(equity_curve) or 0.0
        
        # Calculate profit factor
        total_profit = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0.0
        total_loss = abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else 0.0)
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'profit_factor': profit_factor,
        }
    
    def _empty_backtest_result(self) -> Dict:
        """Return empty backtest result structure."""
        return {
            'trades': [],
            'equity_curve': pd.Series([10000.0]),
            'metrics': {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0,
                'profit_factor': 0.0,
            },
        }


def _calculate_signal_batch_worker(
    start_idx: int,
    end_idx: int,
    df_bytes: bytes,
    symbol: str,
    timeframe: str,
    limit: int,
    signal_type: str,
    osc_length: int,
    osc_mult: float,
    osc_strategies: List[int],
    spc_params: Optional[Dict],
    enabled_indicators: List[str],
    use_confidence_weighting: bool,
    min_indicators_agreement: int,
) -> Dict[int, int]:
    """
    Worker function for parallel signal calculation.
    
    This function is called by multiprocessing Pool to calculate signals
    for a batch of periods.
    
    Args:
        start_idx: Start index of the batch
        end_idx: End index of the batch (exclusive)
        df_bytes: Pickled DataFrame bytes
        Other args: Configuration for signal calculation
        
    Returns:
        Dictionary mapping period index to signal value
    """
    import pickle
    from modules.position_sizing.core.hybrid_signal_calculator import HybridSignalCalculator
    from modules.common.core.exchange_manager import ExchangeManager
    from modules.common.core.data_fetcher import DataFetcher
    from modules.common.utils import log_warn
    
    # Unpickle DataFrame
    df = pickle.loads(df_bytes)
    
    # Create new DataFetcher instance for this worker
    # (DataFetcher contains connection objects that can't be pickled)
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)
    
    # Create a new HybridSignalCalculator instance for this worker
    hybrid_calc = HybridSignalCalculator(
        data_fetcher=data_fetcher,
        enabled_indicators=enabled_indicators,
        use_confidence_weighting=use_confidence_weighting,
        min_indicators_agreement=min_indicators_agreement,
    )
    
    batch_signals = {}
    
    # Calculate signals for each period in this batch
    for i in range(start_idx, end_idx):
        try:
            signal, confidence = hybrid_calc.calculate_hybrid_signal(
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                period_index=i,
                signal_type=signal_type,
                osc_length=osc_length,
                osc_mult=osc_mult,
                osc_strategies=osc_strategies,
                spc_params=spc_params,
            )
            batch_signals[i] = signal
        except Exception as e:
            log_warn(f"Error calculating signal for period {i} in batch [{start_idx}:{end_idx}]: {e}")
            batch_signals[i] = 0  # Default to no signal on error
    
    return batch_signals

