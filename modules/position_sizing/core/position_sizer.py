"""
Position Sizer - Main orchestrator for position sizing calculation.

This module combines backtesting and Kelly calculation
to determine optimal position sizes for trading symbols.
"""
from typing import Any, Dict, List
from pandas.core.frame import DataFrame
import pandas as pd
import traceback
import math


from modules.common.core.data_fetcher import DataFetcher
from modules.backtester import FullBacktester
from modules.position_sizing.core.kelly_calculator import BayesianKellyCalculator
from config.position_sizing import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_TIMEFRAME,
    DEFAULT_MAX_POSITION_SIZE,
    DEFAULT_MIN_POSITION_SIZE,
    DEFAULT_MAX_PORTFOLIO_EXPOSURE,
    SIGNAL_CALCULATION_MODE,
)
from modules.common.utils import (
    log_error,
    log_warn,
    log_progress,
    days_to_candles,
)
from modules.common.ui.progress_bar import ProgressBar


class PositionSizer:
    """
    Main orchestrator for position sizing calculation.
    
    Combines:
    - Backtesting (performance metrics)
    - Kelly Criterion (optimal position size)
    """
    
    def __init__(
        self,
        data_fetcher: DataFetcher,
        timeframe: str = DEFAULT_TIMEFRAME,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        max_position_size: float = DEFAULT_MAX_POSITION_SIZE,
        min_position_size: float = DEFAULT_MIN_POSITION_SIZE,
        max_portfolio_exposure: float = DEFAULT_MAX_PORTFOLIO_EXPOSURE,
        signal_mode: str = 'single_signal',
        signal_calculation_mode: str = SIGNAL_CALCULATION_MODE,
    ):
        """
        Initialize Position Sizer.
        
        Args:
            data_fetcher: DataFetcher instance for fetching OHLCV data
            timeframe: Timeframe for backtesting (default: "1h")
            lookback_days: Number of days to look back for backtesting (default: 90)
            max_position_size: Maximum position size as fraction of account (default: 0.1 = 10%)
            min_position_size: Minimum position size as fraction of account (default: 0.01 = 1%)
            max_portfolio_exposure: Maximum total portfolio exposure (default: 0.5 = 50%)
            signal_mode: Signal calculation mode - 'majority_vote' or 'single_signal' (default: 'single_signal')
            signal_calculation_mode: Signal calculation approach - 'precomputed' or 'incremental' (default: from config)
        """
        self.data_fetcher = data_fetcher
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.max_portfolio_exposure = max_portfolio_exposure
        self.signal_mode = signal_mode
        self.signal_calculation_mode = signal_calculation_mode
        
        # Initialize components
        self.backtester = FullBacktester(
            data_fetcher=data_fetcher,
            signal_mode=signal_mode,
            signal_calculation_mode=signal_calculation_mode,
        )
        self.kelly_calculator = BayesianKellyCalculator()
    
    def calculate_position_size(
        self,
        symbol: str,
        account_balance: float,
        signal_type: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size for a symbol.
        
        Workflow:
        1. Run backtest to get performance metrics
        2. Calculate Kelly fraction from metrics
        3. Calculate position size in USDT
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            account_balance: Total account balance in USDT
            signal_type: "LONG" or "SHORT"
            **kwargs: Additional parameters (timeframe, lookback, etc.)
            
        Returns:
            Dictionary with position sizing results:
            {
                'symbol': str,  # Trading pair symbol (e.g., "BTC/USDT")
                'signal_type': str,  # "LONG" or "SHORT"
                'position_size_usdt': float,  # Position size in USDT
                'position_size_pct': float,  # Position size as percentage of account
                'kelly_fraction': float,  # Raw Kelly fraction calculated from backtest metrics
                'adjusted_kelly_fraction': float,  # Kelly fraction adjusted by cumulative performance multiplier and bounds
                'cumulative_performance_multiplier': float,  # Multiplier based on cumulative equity curve performance (0.5-1.5x)
                'metrics': dict,  # Backtest performance metrics (win_rate, avg_win, avg_loss, etc.)
                'backtest_result': dict,  # Full backtest result including trades and equity curve
            }
        """
        try:
            # Override defaults if provided
            timeframe = kwargs.get('timeframe', self.timeframe)
            lookback_days = kwargs.get('lookback', self.lookback_days)
            
            # Convert days to number of candles based on timeframe
            lookback_candles = days_to_candles(lookback_days, timeframe)
            
            log_progress(f"Calculating position size for {symbol} ({signal_type})...")
            log_progress(f"  Lookback: {lookback_days} days = {lookback_candles} candles ({timeframe})")
            
            # Fetch data once to share between regime detection and backtest
            log_progress(f"  Fetching data for {symbol}...")
            df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=lookback_candles,
                timeframe=timeframe,
                check_freshness=False,
            )
            
            if df is None or df.empty:
                log_warn(f"No data available for {symbol}")
                return self._empty_result(symbol, signal_type)
            
            # Step 1: Run backtest
            log_progress(f"  Step 1: Running backtest for {symbol}...")
            backtest_result = self.backtester.backtest(
                symbol=symbol,
                timeframe=timeframe,
                lookback=lookback_candles,
                signal_type=signal_type,
                df=df,
            )
            
            metrics = backtest_result.get('metrics', {})
            equity_curve = backtest_result.get('equity_curve')
            
            # Step 2: Calculate Kelly fraction
            log_progress(f"  Step 2: Calculating Kelly fraction for {symbol}...")
            # Log metrics for debugging
            log_progress(f"    Metrics: win_rate={metrics.get('win_rate', 0.0):.2%}, "
                        f"num_trades={metrics.get('num_trades', 0)}, "
                        f"avg_win={metrics.get('avg_win', 0.0):.4f}, "
                        f"avg_loss={metrics.get('avg_loss', 0.0):.4f}")
            kelly_fraction = self.kelly_calculator.calculate_kelly_from_metrics(metrics)
            log_progress(f"    Kelly fraction calculated: {kelly_fraction:.4f}")
            
            # Step 2.5: Adjust Kelly fraction based on cumulative performance (if equity curve available)
            # NOTE: Overfitting Risk Mitigation
            # The cumulative performance multiplier can induce overfitting by basing position-size adjustments
            # solely on past equity changes, especially when the sample size (number of trades) is small.
            # To mitigate this, we apply a confidence factor based on trade count:
            # - Low trade count (< threshold) = low confidence = smaller multiplier adjustments
            # - High trade count (>= threshold) = high confidence = full multiplier adjustments
            # This prevents overfitting to small-sample performance fluctuations.
            cumulative_performance_multiplier = 1.0
            if equity_curve is not None and len(equity_curve) > 0:
                try:
                    initial_capital = equity_curve.iloc[0] if hasattr(equity_curve, 'iloc') else equity_curve[0]
                    final_equity = equity_curve.iloc[-1] if hasattr(equity_curve, 'iloc') else equity_curve[-1]
                    
                    if initial_capital > 0:
                        cumulative_performance = (final_equity - initial_capital) / initial_capital
                        
                        # Get trade count for confidence calculation
                        # Derive from metrics if available, otherwise try to get from backtest_result
                        trades_count = metrics.get('num_trades', 0)
                        if trades_count == 0 and backtest_result:
                            trades = backtest_result.get('trades', [])
                            if trades is not None:
                                trades_count = len(trades) if isinstance(trades, (list, pd.Series)) else 0
                        
                        # Calculate confidence factor based on trade count [0, 1]
                        # Maps number_of_trades to confidence weight: more trades = higher confidence
                        confidence = self._calculate_trade_count_confidence(trades_count)
                        
                        # Adjust multiplier based on cumulative performance
                        # Positive performance: increase position size (up to 1.5x)
                        # Negative performance: decrease position size (down to 0.5x)
                        if cumulative_performance > 0:
                            # Scale from 0 to 0.5 -> multiplier from 1.0 to 1.5
                            raw_multiplier = 1.0 + min(0.5, cumulative_performance)
                        elif cumulative_performance < 0:
                            # Scale from 0 to -0.5 -> multiplier from 1.0 to 0.5
                            raw_multiplier = 1.0 + max(-0.5, cumulative_performance)
                        else:
                            raw_multiplier = 1.0
                        
                        # Compute capped_multiplier (conservative multiplier with ±0.25 cap) before branching
                        # This ensures it exists for all confidence values and enables smooth blending
                        max_adjustment = 0.25
                        raw_range = 0.5  # Raw adjustment range for cumulative performance scaling
                        raw_scale = min(raw_range, abs(cumulative_performance))
                        scaled_adjustment = raw_scale * (max_adjustment / raw_range)
                        if cumulative_performance > 0:
                            capped_multiplier = 1.0 + scaled_adjustment
                        elif cumulative_performance < 0:
                            capped_multiplier = 1.0 - scaled_adjustment
                        else:
                            capped_multiplier = 1.0
                        
                        # Apply confidence factor: blend between raw (aggressive) and capped (conservative) multipliers
                        # Single continuous formula ensures smooth transition at confidence=0.5
                        # Higher confidence = more weight on raw_multiplier (aggressive)
                        # Lower confidence = more weight on capped_multiplier (conservative)
                        cumulative_performance_multiplier = confidence * raw_multiplier + (1.0 - confidence) * capped_multiplier
                        
                        log_progress(f"  Cumulative performance: {cumulative_performance*100:.2f}%, "
                                   f"trades: {trades_count}, confidence: {confidence:.3f}, "
                                   f"multiplier: {cumulative_performance_multiplier:.3f}")
                except Exception as e:
                    log_warn(f"Error calculating cumulative performance: {e}")
                    cumulative_performance_multiplier = 1.0
            
            # Step 3: Adjust for cumulative performance
            adjusted_kelly_fraction = kelly_fraction * cumulative_performance_multiplier
            
            # Apply bounds
            # FIX: Don't clamp to min_position_size if kelly_fraction is 0.0 (no trades or invalid)
            # If Kelly is 0, it means strategy is not working, so position size should be 0
            adjusted_before = adjusted_kelly_fraction
            if kelly_fraction == 0.0:
                # If Kelly is 0 (no trades or invalid), keep it at 0, don't clamp to minimum
                adjusted_kelly_fraction = 0.0
            else:
                # Only apply min_position_size if Kelly is positive
                adjusted_kelly_fraction = max(
                    self.min_position_size,
                    min(self.max_position_size, adjusted_kelly_fraction)
                )
            
            # Step 4: Calculate position size in USDT
            position_size_usdt = account_balance * adjusted_kelly_fraction
            position_size_pct = adjusted_kelly_fraction * 100
            
            result = {
                'symbol': symbol,
                'signal_type': signal_type.upper(),
                'position_size_usdt': position_size_usdt,
                'position_size_pct': position_size_pct,
                'kelly_fraction': kelly_fraction,
                'adjusted_kelly_fraction': adjusted_kelly_fraction,
                'cumulative_performance_multiplier': cumulative_performance_multiplier,
                'metrics': metrics,
                'backtest_result': backtest_result,
            }
            
            log_progress(f"  Completed: {symbol} - Position size: {position_size_usdt:.2f} USDT ({position_size_pct:.2f}%)")
            
            return result
            
        except Exception as e:
            log_error(f"Error calculating position size for {symbol}: {e}")
            log_error(f"Traceback: {traceback.format_exc()}")
            return self._empty_result(symbol, signal_type)
    
    def calculate_portfolio_allocation(
        self,
        symbols: List[Dict[str, Any]],
        account_balance: float,
        **kwargs,
    ) -> DataFrame:
        """
        Calculate position sizing for multiple symbols (portfolio allocation).
        
        Args:
            symbols: List of symbol dictionaries with 'symbol' and 'signal' keys
            account_balance: Total account balance in USDT
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with position sizing results for all symbols
        """
        results = []
        total_exposure = 0.0
        
        log_progress(f"\nCalculating portfolio allocation for {len(symbols)} symbols...")
        log_progress(f"Account balance: {account_balance:.2f} USDT")
        log_progress(f"Max portfolio exposure: {self.max_portfolio_exposure * 100:.1f}%")
        print()  # Newline before progress bar
        
        # Initialize progress bar
        progress = ProgressBar(total=len(symbols), label="Processing symbols")
        
        try:
            # Calculate position size for each symbol
            for idx, symbol_dict in enumerate(symbols, 1):
                symbol = symbol_dict.get('symbol')
                signal = symbol_dict.get('signal', 0)
                
                if not symbol:
                    log_warn(f"Skipping invalid symbol: {symbol_dict}")
                    progress.update()  # Update progress even for skipped symbols
                    continue
                
                # Update progress bar label to show current symbol
                progress.set_label(f"Processing {symbol} ({idx}/{len(symbols)})")
                
                # Determine signal type
                if isinstance(signal, str):
                    signal_type = "LONG" if signal.upper() in ["LONG", "BUY", "1"] else "SHORT"
                elif isinstance(signal, (int, float)):
                    signal_type = "LONG" if signal > 0 else "SHORT"
                else:
                    signal_type = "LONG"  # Default
                
                # Calculate position size
                result = self.calculate_position_size(
                    symbol=symbol,
                    account_balance=account_balance,
                    signal_type=signal_type,
                    **kwargs,
                )
                
                results.append(result)
                total_exposure += result.get('adjusted_kelly_fraction', 0.0)
                
                # Update progress bar
                progress.update()
        finally:
            # Ensure progress bar is finished even if there's an error
            progress.finish()
        
        # Normalize if total exposure exceeds maximum
        if total_exposure > self.max_portfolio_exposure:
            log_warn(f"Total exposure ({total_exposure*100:.1f}%) exceeds maximum ({self.max_portfolio_exposure*100:.1f}%). Normalizing...")
            
            # Check for division by zero
            if total_exposure > 0:
                normalization_factor = self.max_portfolio_exposure / total_exposure
            else:
                log_warn("Total exposure is 0, cannot normalize. Skipping normalization.")
                normalization_factor = 1.0
            
            for result in results:
                result['adjusted_kelly_fraction'] *= normalization_factor
                result['position_size_usdt'] = account_balance * result['adjusted_kelly_fraction']
                result['position_size_pct'] = result['adjusted_kelly_fraction'] * 100
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by position size (descending)
        if not df.empty:
            df = df.sort_values('position_size_usdt', ascending=False).reset_index(drop=True)
        
        # Add summary row
        if not df.empty:
            total_position_size = df['position_size_usdt'].sum()
            total_exposure_pct = (total_position_size / account_balance) * 100
            
            log_progress(f"\nPortfolio Summary:")
            log_progress(f"  Total position size: {total_position_size:.2f} USDT")
            log_progress(f"  Total exposure: {total_exposure_pct:.2f}%")
            log_progress(f"  Number of positions: {len(df)}")
        
        return df
    
    def _calculate_trade_count_confidence(self, trades_count: int, min_trades_threshold: int = 30) -> float:
        """
        Calculate confidence factor based on trade count to mitigate overfitting.
        
        Maps number_of_trades to a [0, 1] confidence weight:
        - Few trades (< threshold) = low confidence (closer to 0)
        - Many trades (>= threshold) = high confidence (closer to 1)
        
        This prevents overfitting by reducing position-size adjustments when
        the sample size is too small to reliably estimate performance.
        
        Args:
            trades_count: Number of trades executed in the backtest
            min_trades_threshold: Minimum number of trades for high confidence (default: 30)
            
        Returns:
            Confidence factor in range [0, 1] where:
            - 0.0 = very low confidence (few trades)
            - 1.0 = high confidence (many trades)
        """
        if trades_count <= 0:
            return 0.0
        
        # Use a sigmoid-like function to map trade count to confidence
        # At threshold trades, confidence should be around 0.5
        # As trades increase beyond threshold, confidence approaches 1.0
        # Below threshold, confidence approaches 0.0
        normalized = trades_count / min_trades_threshold
        # Smooth sigmoid-like curve: confidence = 1 / (1 + exp(-k*(x - 1)))
        # Using k=2 for a reasonable transition
        confidence = 1.0 / (1.0 + math.exp(-2.0 * (normalized - 1.0)))
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, confidence))
    
    def _empty_result(self, symbol: str, signal_type: str) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'symbol': symbol,
            'signal_type': signal_type.upper(),
            'position_size_usdt': 0.0,
            'position_size_pct': 0.0,
            'kelly_fraction': 0.0,
            'adjusted_kelly_fraction': 0.0,
            'cumulative_performance_multiplier': 1.0,
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
            'backtest_result': {
                'trades': [],
                'equity_curve': pd.Series([10000.0]),
                'metrics': {},
            },
        }
