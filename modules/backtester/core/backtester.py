"""
Full Backtester for trading strategy simulation.

This module simulates trading with entry/exit rules based on signals
from indicators (ATC, Oscillator, SPC, etc.) and calculates performance metrics.
"""

from typing import Dict, List, Optional
import pandas as pd
import traceback
import time

from modules.common.core.data_fetcher import DataFetcher
from config.position_sizing import (
    BACKTEST_STOP_LOSS_PCT,
    BACKTEST_TAKE_PROFIT_PCT,
    BACKTEST_TRAILING_STOP_PCT,
    BACKTEST_MAX_HOLD_PERIODS,
    BACKTEST_RISK_PER_TRADE,
    ENABLED_INDICATORS,
    USE_CONFIDENCE_WEIGHTING,
    MIN_INDICATORS_AGREEMENT,
    ENABLE_PARALLEL_PROCESSING,
    ENABLE_PERFORMANCE_PROFILING,
    LOG_PERFORMANCE_METRICS,
    CLEAR_CACHE_ON_COMPLETE,
    SIGNAL_CALCULATION_MODE,
)
from modules.common.utils import (
    log_error,
    log_warn,
    log_progress,
)

# Import modules
from .signal_calculator import (
    calculate_signals,
    calculate_signals_parallel,
    calculate_single_signals,
    calculate_single_signals_parallel,
    calculate_signals_incremental,
    calculate_single_signals_incremental,
)
from .trade_simulator import simulate_trades
from .equity_curve import calculate_equity_curve
from .metrics import calculate_metrics, empty_backtest_result


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
        risk_per_trade: float = BACKTEST_RISK_PER_TRADE,
        enabled_indicators: Optional[List[str]] = None,
        use_confidence_weighting: bool = USE_CONFIDENCE_WEIGHTING,
        min_indicators_agreement: int = MIN_INDICATORS_AGREEMENT,
        signal_mode: str = 'majority_vote',
        signal_calculation_mode: str = SIGNAL_CALCULATION_MODE,
    ):
        """
        Initialize Full Backtester.
        
        Args:
            data_fetcher: DataFetcher instance for fetching OHLCV data
            stop_loss_pct: Stop loss percentage (default: 2%)
            take_profit_pct: Take profit percentage (default: 4%)
            trailing_stop_pct: Trailing stop percentage (default: 1.5%)
            max_hold_periods: Maximum periods to hold a position (default: 100)
            risk_per_trade: Risk percentage per trade for equity curve (default: 1%)
            enabled_indicators: List of enabled indicators (default: from config)
            use_confidence_weighting: Whether to weight votes by confidence (default: True)
            min_indicators_agreement: Minimum indicators that must agree (default: 3)
            signal_mode: Signal calculation mode - 'majority_vote' (default) or 'single_signal'
            signal_calculation_mode: Signal calculation approach - 'precomputed' (default) or 'incremental'
        """
        if signal_mode not in ('majority_vote', 'single_signal'):
            raise ValueError(f"Invalid signal_mode: {signal_mode}. Must be 'majority_vote' or 'single_signal'")
        if signal_calculation_mode not in ('precomputed', 'incremental'):
            raise ValueError(f"Invalid signal_calculation_mode: {signal_calculation_mode}. Must be 'precomputed' or 'incremental'")
        
        self.data_fetcher = data_fetcher
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_hold_periods = max_hold_periods
        self.risk_per_trade = risk_per_trade
        self.signal_mode = signal_mode
        self.signal_calculation_mode = signal_calculation_mode
        
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
        df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Run full backtest simulation for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            lookback: Number of candles to look back (should be converted from days using days_to_candles)
            signal_type: "LONG" or "SHORT"
            initial_capital: Initial capital for backtesting (default: 10000)
            df: Optional DataFrame to use instead of fetching from API
            
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
                },
                'total_time': float,  # Total backtest execution time in seconds
            }
        """
        try:
            # Validate input parameters
            if signal_type.upper() not in ("LONG", "SHORT"):
                raise ValueError(f"Invalid signal_type: {signal_type}. Must be 'LONG' or 'SHORT'")
            
            if initial_capital <= 0:
                raise ValueError(f"initial_capital must be > 0, got {initial_capital}")
            
            if self.stop_loss_pct <= 0:
                raise ValueError(f"stop_loss_pct must be > 0, got {self.stop_loss_pct}")
            
            if self.take_profit_pct <= 0:
                raise ValueError(f"take_profit_pct must be > 0, got {self.take_profit_pct}")
            
            if self.trailing_stop_pct <= 0:
                raise ValueError(f"trailing_stop_pct must be > 0, got {self.trailing_stop_pct}")
            
            if self.max_hold_periods <= 0:
                raise ValueError(f"max_hold_periods must be > 0, got {self.max_hold_periods}")
            
            if lookback <= 0:
                raise ValueError(f"lookback must be > 0, got {lookback}")
            
            # Start timing the entire backtest
            backtest_start_time = time.time()
            # Use provided DataFrame if available, otherwise fetch from API
            if df is None:
                df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
                    symbol,
                    limit=lookback,
                    timeframe=timeframe,
                    check_freshness=False,
                )
            
            if df is None or df.empty:
                log_warn(f"No data available for {symbol}")
                return empty_backtest_result()
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                log_error(f"Missing required columns for {symbol}: {missing_columns}")
                return empty_backtest_result()
            
            # Calculate signals for each period
            start_time = time.time() if (ENABLE_PERFORMANCE_PROFILING or LOG_PERFORMANCE_METRICS) else None
            
            # Check if using incremental mode (combines signal calculation and trade simulation)
            if self.signal_calculation_mode == 'incremental':
                # Incremental mode: combine signal calculation and trade simulation
                if self.signal_mode == 'single_signal':
                    signals, trades = calculate_single_signals_incremental(
                        df=df,
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=lookback,
                        hybrid_signal_calculator=self.hybrid_signal_calculator,
                        stop_loss_pct=self.stop_loss_pct,
                        take_profit_pct=self.take_profit_pct,
                        trailing_stop_pct=self.trailing_stop_pct,
                        max_hold_periods=self.max_hold_periods,
                        initial_capital=initial_capital,
                    )
                else:
                    signals, trades = calculate_signals_incremental(
                        df=df,
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=lookback,
                        signal_type=signal_type,
                        hybrid_signal_calculator=self.hybrid_signal_calculator,
                        stop_loss_pct=self.stop_loss_pct,
                        take_profit_pct=self.take_profit_pct,
                        trailing_stop_pct=self.trailing_stop_pct,
                        max_hold_periods=self.max_hold_periods,
                        initial_capital=initial_capital,
                    )
            else:
                # Precomputed mode: calculate all signals first, then simulate trades
                if self.signal_mode == 'single_signal':
                    # Use single signal (highest confidence) mode
                    if ENABLE_PARALLEL_PROCESSING and len(df) > 100:  # Only use parallel for large datasets
                        signals = calculate_single_signals_parallel(
                            df=df,
                            symbol=symbol,
                            timeframe=timeframe,
                            limit=lookback,
                            hybrid_signal_calculator=self.hybrid_signal_calculator,
                            fallback_calculate_single_signals=lambda df, symbol, timeframe, limit, hybrid_signal_calculator: calculate_single_signals(
                                df=df,
                                symbol=symbol,
                                timeframe=timeframe,
                                limit=limit,
                                hybrid_signal_calculator=hybrid_signal_calculator,
                            ),
                        )
                    else:
                        signals = calculate_single_signals(
                            df=df,
                            symbol=symbol,
                            timeframe=timeframe,
                            limit=lookback,
                            hybrid_signal_calculator=self.hybrid_signal_calculator,
                        )
                else:
                    # Use majority vote mode (default)
                    if ENABLE_PARALLEL_PROCESSING and len(df) > 100:  # Only use parallel for large datasets
                        signals = calculate_signals_parallel(
                            df=df,
                            symbol=symbol,
                            timeframe=timeframe,
                            limit=lookback,
                            signal_type=signal_type,
                            hybrid_signal_calculator=self.hybrid_signal_calculator,
                            fallback_calculate_signals=lambda df, symbol, timeframe, limit, signal_type: calculate_signals(
                                df=df,
                                symbol=symbol,
                                timeframe=timeframe,
                                limit=limit,
                                signal_type=signal_type,
                                hybrid_signal_calculator=self.hybrid_signal_calculator,
                            ),
                        )
                    else:
                        signals = calculate_signals(
                            df=df,
                            symbol=symbol,
                            timeframe=timeframe,
                            limit=lookback,
                            signal_type=signal_type,
                            hybrid_signal_calculator=self.hybrid_signal_calculator,
                        )
                
                if start_time and LOG_PERFORMANCE_METRICS:
                    elapsed = time.time() - start_time
                    log_progress(f"  Signal calculation took {elapsed:.2f} seconds")
                
                # Simulate trades
                trades = simulate_trades(
                    df=df,
                    signals=signals,
                    signal_type=signal_type,
                    initial_capital=initial_capital,
                    stop_loss_pct=self.stop_loss_pct,
                    take_profit_pct=self.take_profit_pct,
                    trailing_stop_pct=self.trailing_stop_pct,
                    max_hold_periods=self.max_hold_periods,
                )
            
            if not trades:
                log_warn(f"No trades generated for {symbol}")
                return empty_backtest_result()
            
            # Calculate equity curve
            equity_curve = calculate_equity_curve(
                trades=trades,
                initial_capital=initial_capital,
                num_periods=len(df),
                risk_per_trade=self.risk_per_trade,
            )
            
            # Calculate performance metrics
            metrics_start_time = time.time() if LOG_PERFORMANCE_METRICS else None
            metrics = calculate_metrics(trades=trades, equity_curve=equity_curve)
            if metrics_start_time and LOG_PERFORMANCE_METRICS:
                metrics_elapsed = time.time() - metrics_start_time
                log_progress(f"  Metrics calculation took {metrics_elapsed:.2f} seconds")
            
            # Log cache statistics
            if LOG_PERFORMANCE_METRICS:
                cache_stats = self.hybrid_signal_calculator.get_cache_stats()
                log_progress(f"  Cache stats: {cache_stats['signal_cache_size']}/{cache_stats['signal_cache_max_size']} signals, "
                           f"hit rate: {cache_stats['cache_hit_rate']*100:.1f}%")
            
            # Clear caches if configured to save memory
            if CLEAR_CACHE_ON_COMPLETE:
                self.hybrid_signal_calculator.clear_cache()
            
            # Calculate total backtest time
            total_time = time.time() - backtest_start_time
            
            return {
                'trades': trades,
                'equity_curve': equity_curve,
                'metrics': metrics,
                'total_time': total_time,
            }
            
        except Exception as e:
            log_error(f"Error backtesting {symbol}: {e}")
            log_error(f"Traceback: {traceback.format_exc()}")
            return empty_backtest_result()
