"""
Bayesian Kelly Criterion Calculator.

This module implements Bayesian Kelly Criterion for position sizing,
combining historical performance with confidence intervals.
"""

from typing import Optional
import numpy as np
from scipy import stats
import logging
import pandas as pd

from config.position_sizing import (
    DEFAULT_FRACTIONAL_KELLY,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_MIN_WIN_RATE,
    DEFAULT_MIN_TRADES,
    KELLY_PRIOR_ALPHA,
    KELLY_PRIOR_BETA,
    KELLY_MIN_FRACTION,
    KELLY_MAX_FRACTION,
)
from modules.common.utils import (
    log_error,
    log_warn,
)

logger = logging.getLogger(__name__)


class BayesianKellyCalculator:
    """
    Calculates optimal position size using Bayesian Kelly Criterion.
    
    Uses Beta distribution as prior for win rate, updates with historical data,
    and calculates Kelly fraction with confidence intervals.
    """
    
    def __init__(
        self,
        fractional_kelly: float = DEFAULT_FRACTIONAL_KELLY,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        min_win_rate: float = DEFAULT_MIN_WIN_RATE,
        min_trades: int = DEFAULT_MIN_TRADES,
        prior_alpha: float = KELLY_PRIOR_ALPHA,
        prior_beta: float = KELLY_PRIOR_BETA,
    ):
        """
        Initialize Bayesian Kelly Calculator.
        
        Args:
            fractional_kelly: Fraction of full Kelly to use (default: 0.25 = 25%)
            confidence_level: Confidence level for Bayesian estimation (default: 0.95)
            min_win_rate: Minimum win rate to consider (default: 0.4)
            min_trades: Minimum number of trades required (default: 10)
            prior_alpha: Prior alpha parameter for Beta distribution (default: 2.0)
            prior_beta: Prior beta parameter for Beta distribution (default: 2.0)
        """
        self.fractional_kelly = fractional_kelly
        self.confidence_level = confidence_level
        self.min_win_rate = min_win_rate
        self.min_trades = min_trades
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
    
    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        num_trades: int,
        confidence: Optional[float] = None,
    ) -> float:
        """
        Calculate Kelly fraction using Bayesian approach.
        
        Args:
            win_rate: Historical win rate (0.0 to 1.0)
            avg_win: Average winning trade return (positive value)
            avg_loss: Average losing trade return (absolute value, positive)
            num_trades: Number of trades in historical data
            confidence: Optional confidence level (overrides default)
            
        Returns:
            Kelly fraction (0.0 to 1.0) representing optimal position size
        """
        if confidence is None:
            confidence = self.confidence_level
        
        # Validate inputs
        if num_trades < self.min_trades:
            log_warn(f"Insufficient trades ({num_trades} < {self.min_trades}). Returning 0.0")
            return 0.0
        
        if win_rate < self.min_win_rate:
            log_warn(f"Win rate too low ({win_rate:.2%} < {self.min_win_rate:.2%}). Returning 0.0")
            return 0.0
        
        if avg_win <= 0 or avg_loss <= 0:
            log_warn(f"Invalid avg_win ({avg_win:.6f}) or avg_loss ({avg_loss:.6f}). Returning 0.0")
            return 0.0
        
        try:
            # Calculate number of wins and losses
            num_wins = int(win_rate * num_trades)
            num_losses = num_trades - num_wins
            
            # Update Beta distribution with observed data
            # Posterior: Beta(alpha + num_wins, beta + num_losses)
            posterior_alpha = self.prior_alpha + num_wins
            posterior_beta = self.prior_beta + num_losses
            
            # Calculate posterior mean (expected win rate)
            posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
            
            # Calculate confidence interval for win rate
            lower_bound, upper_bound = stats.beta.interval(
                confidence,
                posterior_alpha,
                posterior_beta,
            )
            
            # Use conservative estimate (lower bound) for Kelly calculation
            # However, if lower bound is too low (negative or very close to 0), use posterior mean instead
            # This prevents overly conservative estimates when sample size is small
            if lower_bound < 0.1 or (num_trades < 20 and lower_bound < posterior_mean * 0.7):
                # For small samples or very low lower bounds, use posterior mean with a small discount
                conservative_win_rate = posterior_mean * 0.9  # 10% discount from mean for safety
            else:
                conservative_win_rate = lower_bound
            
            # Calculate Kelly fraction: f* = (p * b - q) / b
            # where p = win_rate, q = 1 - p, b = avg_win / avg_loss
            b = avg_win / avg_loss  # Win/loss ratio
            p = conservative_win_rate
            q = 1 - p
            
            # Full Kelly formula
            full_kelly = (p * b - q) / b if b > 0 else 0.0
            
            # Apply fractional Kelly to reduce risk
            kelly_fraction = full_kelly * self.fractional_kelly
            
            # Apply bounds
            kelly_fraction_before_bounds = kelly_fraction
            kelly_fraction = max(KELLY_MIN_FRACTION, min(KELLY_MAX_FRACTION, kelly_fraction))
            
            # Additional safety check: if Kelly is negative, return 0
            if kelly_fraction < 0:
                log_warn(f"Negative Kelly fraction calculated ({kelly_fraction:.4f}). Strategy is not profitable (p*b-q={p*b-q:.4f} < 0). Returning 0.0")
                return 0.0
            
            return kelly_fraction
            
        except Exception as e:
            log_error(f"Error calculating Kelly fraction: {e}")
            logger.exception("Kelly calculation error")
            return 0.0
    
    def adjust_for_confidence(
        self,
        kelly_fraction: float,
        confidence: Optional[float] = None,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ) -> float:
        """
        Adjust Kelly fraction based on confidence intervals.
        
        Args:
            kelly_fraction: Base Kelly fraction
            confidence: Confidence level (overrides default)
            lower_bound: Lower bound for Kelly fraction (default: KELLY_MIN_FRACTION)
            upper_bound: Upper bound for Kelly fraction (default: KELLY_MAX_FRACTION)
            
        Returns:
            Adjusted Kelly fraction
        """
        if lower_bound is None:
            lower_bound = KELLY_MIN_FRACTION
        if upper_bound is None:
            upper_bound = KELLY_MAX_FRACTION
        
        # Apply bounds
        adjusted = max(lower_bound, min(upper_bound, kelly_fraction))
        
        # Further reduce if confidence is lower
        if confidence is None:
            confidence = self.confidence_level
        
        if confidence < 0.9:
            # Reduce by confidence factor
            adjusted = adjusted * (confidence / 0.95)
        
        return adjusted
    
    def calculate_kelly_from_metrics(
        self,
        metrics: dict,
        confidence: Optional[float] = None,
    ) -> float:
        """
        Calculate Kelly fraction directly from backtest metrics.
        
        Args:
            metrics: Dictionary with keys: win_rate, avg_win, avg_loss, num_trades
            confidence: Optional confidence level
            
        Returns:
            Kelly fraction
        """
        win_rate = metrics.get('win_rate', 0.0)
        avg_win = metrics.get('avg_win', 0.0)
        avg_loss = metrics.get('avg_loss', 0.0)
        num_trades = metrics.get('num_trades', 0)
        
        # Convert avg_win and avg_loss to absolute values if needed
        if avg_win < 0:
            avg_win = abs(avg_win)
        if avg_loss < 0:
            avg_loss = abs(avg_loss)
        
        result = self.calculate_kelly_fraction(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            num_trades=num_trades,
            confidence=confidence,
        )
        
        return result
    
    def get_posterior_distribution(
        self,
        num_wins: int,
        num_losses: int,
    ) -> dict:
        """
        Get posterior Beta distribution parameters.
        
        Args:
            num_wins: Number of winning trades
            num_losses: Number of losing trades
            
        Returns:
            Dictionary with posterior parameters and statistics
        """
        posterior_alpha = self.prior_alpha + num_wins
        posterior_beta = self.prior_beta + num_losses
        
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        posterior_mode = (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2) if (posterior_alpha + posterior_beta) > 2 else posterior_mean
        
        lower_bound, upper_bound = stats.beta.interval(
            self.confidence_level,
            posterior_alpha,
            posterior_beta,
        )
        
        return {
            'alpha': posterior_alpha,
            'beta': posterior_beta,
            'mean': posterior_mean,
            'mode': posterior_mode,
            'confidence_interval': (lower_bound, upper_bound),
            'confidence_level': self.confidence_level,
        }

