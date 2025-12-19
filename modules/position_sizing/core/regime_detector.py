"""
Regime Detector using HMM module.

This module uses HMM (Hidden Markov Model) to detect current market regime:
BULLISH, NEUTRAL, or BEARISH.
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import logging

from modules.common.core.data_fetcher import DataFetcher
from modules.hmm.core.swings import (
    hmm_swings,
    BULLISH,
    NEUTRAL,
    BEARISH,
)
from modules.common.utils import (
    log_error,
    log_warn,
    log_progress,
)

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects market regime using HMM (Hidden Markov Model).
    
    Uses HMM-Swings module to identify current market state:
    - BULLISH: Upward trending market
    - NEUTRAL: Sideways/consolidating market
    - BEARISH: Downward trending market
    """
    
    def __init__(
        self,
        data_fetcher: DataFetcher,
        train_ratio: float = 0.8,
        orders_argrelextrema: Optional[int] = None,
        strict_mode: Optional[bool] = None,
    ):
        """
        Initialize Regime Detector.
        
        Args:
            data_fetcher: DataFetcher instance for fetching OHLCV data
            train_ratio: Ratio of data to use for HMM training (default: 0.8)
            orders_argrelextrema: Order parameter for swing detection (optional)
            strict_mode: Use strict mode for swing-to-state conversion (optional)
        """
        self.data_fetcher = data_fetcher
        self.train_ratio = train_ratio
        self.orders_argrelextrema = orders_argrelextrema
        self.strict_mode = strict_mode
    
    def detect_regime(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1500,
    ) -> str:
        """
        Detect current market regime for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for data (e.g., "1h", "4h")
            limit: Number of candles to fetch (default: 1500)
            
        Returns:
            Regime string: "BULLISH", "NEUTRAL", or "BEARISH"
        """
        try:
            # Fetch OHLCV data
            df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=limit,
                timeframe=timeframe,
                check_freshness=False,
            )
            
            if df is None or df.empty:
                log_warn(f"No data available for {symbol}. Returning NEUTRAL regime.")
                return "NEUTRAL"
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                log_error(f"Missing required columns for {symbol}: {missing_columns}")
                return "NEUTRAL"
            
            # Run HMM analysis
            result = hmm_swings(
                df=df,
                train_ratio=self.train_ratio,
                eval_mode=False,  # Don't evaluate, just predict
                orders_argrelextrema=self.orders_argrelextrema,
                strict_mode=self.strict_mode,
            )
            
            # Convert HMM state to regime string
            next_state = result.next_state_with_high_order_hmm
            
            # #region agent log
            import json
            import os
            log_path = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-\.cursor\debug.log"
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id": f"log_hmm_result_{os.getpid()}", "timestamp": pd.Timestamp.now().timestamp() * 1000, "location": "regime_detector.py:106", "message": "HMM result received", "data": {"symbol": symbol, "next_state": next_state, "next_state_type": str(type(next_state)), "result_type": str(type(result)), "result_has_next_state": hasattr(result, 'next_state_with_high_order_hmm')}, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "C"}) + "\n")
            except: pass
            # #endregion
            
            if next_state == BULLISH:
                return "BULLISH"
            elif next_state == BEARISH:
                return "BEARISH"
            else:  # NEUTRAL or 0
                return "NEUTRAL"
                
        except Exception as e:
            log_error(f"Error detecting regime for {symbol}: {e}")
            logger.exception(f"Regime detection error for {symbol}")
            return "NEUTRAL"
    
    def get_regime_probabilities(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1500,
    ) -> Dict[str, float]:
        """
        Get probability distribution of regimes.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            limit: Number of candles to fetch
            
        Returns:
            Dictionary with regime probabilities:
            {
                "BULLISH": 0.6,
                "NEUTRAL": 0.2,
                "BEARISH": 0.2
            }
        """
        try:
            # Fetch OHLCV data
            df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=limit,
                timeframe=timeframe,
                check_freshness=False,
            )
            
            if df is None or df.empty:
                return {"BULLISH": 0.33, "NEUTRAL": 0.34, "BEARISH": 0.33}
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return {"BULLISH": 0.33, "NEUTRAL": 0.34, "BEARISH": 0.33}
            
            # Run HMM analysis
            result = hmm_swings(
                df=df,
                train_ratio=self.train_ratio,
                eval_mode=False,
                orders_argrelextrema=self.orders_argrelextrema,
                strict_mode=self.strict_mode,
            )
            
            # Extract probability from result
            next_state_prob = result.next_state_probability
            
            # Map state to regime probabilities
            # Since HMM gives us the most likely state, we distribute probabilities
            # around that state with some uncertainty
            next_state = result.next_state_with_high_order_hmm
            
            if next_state == BULLISH:
                return {
                    "BULLISH": min(0.95, next_state_prob + 0.2),
                    "NEUTRAL": (1 - min(0.95, next_state_prob + 0.2)) * 0.5,
                    "BEARISH": (1 - min(0.95, next_state_prob + 0.2)) * 0.5,
                }
            elif next_state == BEARISH:
                return {
                    "BULLISH": (1 - min(0.95, next_state_prob + 0.2)) * 0.5,
                    "NEUTRAL": (1 - min(0.95, next_state_prob + 0.2)) * 0.5,
                    "BEARISH": min(0.95, next_state_prob + 0.2),
                }
            else:  # NEUTRAL
                return {
                    "BULLISH": (1 - next_state_prob) * 0.4,
                    "NEUTRAL": next_state_prob,
                    "BEARISH": (1 - next_state_prob) * 0.4,
                }
                
        except Exception as e:
            log_error(f"Error getting regime probabilities for {symbol}: {e}")
            logger.exception(f"Regime probability error for {symbol}")
            return {"BULLISH": 0.33, "NEUTRAL": 0.34, "BEARISH": 0.33}
    
    def detect_regime_batch(
        self,
        symbols: list,
        timeframe: str,
        limit: int = 1500,
    ) -> Dict[str, str]:
        """
        Detect regimes for multiple symbols.
        
        Args:
            symbols: List of trading pair symbols
            timeframe: Timeframe for data
            limit: Number of candles to fetch
            
        Returns:
            Dictionary mapping symbol to regime:
            {
                "BTC/USDT": "BULLISH",
                "ETH/USDT": "NEUTRAL",
                ...
            }
        """
        regimes = {}
        
        for symbol in symbols:
            try:
                regime = self.detect_regime(symbol, timeframe, limit)
                regimes[symbol] = regime
            except Exception as e:
                log_warn(f"Failed to detect regime for {symbol}: {e}")
                regimes[symbol] = "NEUTRAL"
        
        return regimes

