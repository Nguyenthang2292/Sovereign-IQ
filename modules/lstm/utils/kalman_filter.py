"""
Kalman Filter preprocessing utilities for LSTM models.

This module provides Kalman Filter implementation to smooth OHLC price data
before generating technical indicators, helping reduce noise and overfitting.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from copy import deepcopy

from modules.common.ui.logging import log_model, log_error, log_warn


class KalmanFilterOHLC:
    """
    Kalman Filter for smoothing OHLC price data.
    
    Uses univariate Kalman Filter for each OHLC series independently.
    The filter assumes a random walk model for the true price state.
    
    State model: x_t = x_{t-1} + w_t (random walk)
    Observation: y_t = x_t + v_t
    
    Where:
    - x_t: true price at time t
    - y_t: observed price at time t
    - w_t ~ N(0, Q): process noise
    - v_t ~ N(0, R): observation noise
    """
    
    def __init__(
        self,
        process_variance: float = 1e-5,
        observation_variance: float = 1.0,
        initial_state: Optional[float] = None,
        initial_uncertainty: float = 1.0
    ):
        """
        Initialize Kalman Filter.
        
        Args:
            process_variance: Process noise covariance (Q). Smaller values mean 
                            more trust in state model (smoother output).
            observation_variance: Observation noise covariance (R). Smaller values 
                                 mean more trust in observations (less smoothing).
            initial_state: Initial state estimate. If None, uses first observation.
            initial_uncertainty: Initial uncertainty (covariance) of state estimate.
        """
        self.process_variance = max(1e-10, process_variance)  # Prevent division by zero
        self.observation_variance = max(1e-10, observation_variance)
        self.initial_state = initial_state
        self.initial_uncertainty = max(1e-10, initial_uncertainty)
        
        # State tracking for each series
        self.state_history: Dict[str, list] = {}
        self.fitted = False
        
    def _kalman_step(
        self, 
        observation: float, 
        prev_state: float, 
        prev_uncertainty: float
    ) -> tuple[float, float]:
        """
        Perform one Kalman Filter step.
        
        Args:
            observation: Current observation (price)
            prev_state: Previous state estimate
            prev_uncertainty: Previous state uncertainty
            
        Returns:
            Tuple of (updated_state, updated_uncertainty)
        """
        # Skip if observation is invalid
        if not np.isfinite(observation):
            return prev_state, prev_uncertainty
        
        # Prediction step
        predicted_state = prev_state
        predicted_uncertainty = prev_uncertainty + self.process_variance
        
        # Update step
        kalman_gain = predicted_uncertainty / (predicted_uncertainty + self.observation_variance)
        updated_state = predicted_state + kalman_gain * (observation - predicted_state)
        updated_uncertainty = (1 - kalman_gain) * predicted_uncertainty
        
        return updated_state, updated_uncertainty
    
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit Kalman Filter to OHLC data and return smoothed data.
        
        Args:
            df: DataFrame with OHLC columns (open, high, low, close)
            
        Returns:
            DataFrame with smoothed OHLC values
        """
        if df.empty:
            log_error("Empty DataFrame provided to Kalman Filter")
            return df.copy()
        
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Check for required OHLC columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            log_error(f"Missing required columns for Kalman Filter: {missing_cols}")
            return df
        
        # Initialize state history
        self.state_history = {}
        smoothed_df = pd.DataFrame(index=df.index, columns=required_cols)
        
        # Apply Kalman Filter to each OHLC series
        for col in required_cols:
            series = df[col].values
            smoothed_series = np.zeros_like(series, dtype=float)
            
            # Initialize state with first valid observation
            initial_idx = None
            for i, val in enumerate(series):
                if np.isfinite(val):
                    initial_idx = i
                    break
            
            if initial_idx is None:
                log_warn(f"No valid values found for {col}, skipping Kalman Filter")
                smoothed_series = series.copy()
                smoothed_df[col] = smoothed_series
                continue
            
            # Initialize state
            initial_state = self.initial_state if self.initial_state is not None else series[initial_idx]
            current_state = initial_state
            current_uncertainty = self.initial_uncertainty
            
            # Store initial state
            self.state_history[col] = [current_state]
            
            # Apply Kalman Filter forward pass
            smoothed_series[:initial_idx] = np.nan  # Mark invalid leading values
            
            for i in range(initial_idx, len(series)):
                obs = series[i]
                current_state, current_uncertainty = self._kalman_step(
                    obs, current_state, current_uncertainty
                )
                smoothed_series[i] = current_state
                self.state_history[col].append(current_state)
            
            smoothed_df[col] = smoothed_series
        
        # Preserve other columns (e.g., volume, timestamp)
        for col in df.columns:
            if col not in required_cols:
                smoothed_df[col] = df[col]
        
        self.fitted = True
        log_model(f"Kalman Filter applied to OHLC data: process_var={self.process_variance:.2e}, "
                 f"obs_var={self.observation_variance:.2e}")
        
        return smoothed_df
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted Kalman Filter to new data.
        
        For online/incremental prediction, continue from last fitted state.
        For batch prediction of new data, reinitialize with first observation.
        
        Args:
            df: DataFrame with OHLC columns
            
        Returns:
            DataFrame with smoothed OHLC values
        """
        if not self.fitted:
            log_warn("Kalman Filter not fitted, calling fit() instead")
            return self.fit(df)
        
        # For simplicity, re-fit on new data
        # In production, could implement incremental updates from last state
        return self.fit(df)
    
    def get_params(self) -> Dict[str, Any]:
        """Get Kalman Filter parameters."""
        return {
            'process_variance': self.process_variance,
            'observation_variance': self.observation_variance,
            'initial_state': self.initial_state,
            'initial_uncertainty': self.initial_uncertainty,
            'fitted': self.fitted
        }
    
    def set_params(self, **params):
        """Set Kalman Filter parameters."""
        if 'process_variance' in params:
            self.process_variance = max(1e-10, params['process_variance'])
        if 'observation_variance' in params:
            self.observation_variance = max(1e-10, params['observation_variance'])
        if 'initial_state' in params:
            self.initial_state = params['initial_state']
        if 'initial_uncertainty' in params:
            self.initial_uncertainty = max(1e-10, params['initial_uncertainty'])


def apply_kalman_to_ohlc(
    df: pd.DataFrame,
    process_variance: float = 1e-5,
    observation_variance: float = 1.0,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to apply Kalman Filter to OHLC data.
    
    Args:
        df: DataFrame with OHLC columns
        process_variance: Process noise covariance (Q)
        observation_variance: Observation noise covariance (R)
        **kwargs: Additional parameters passed to KalmanFilterOHLC
        
    Returns:
        DataFrame with smoothed OHLC values
    """
    if df.empty:
        return df.copy()
    
    # Check minimum data length
    min_length = 5
    if len(df) < min_length:
        log_warn(f"Data too short for Kalman Filter ({len(df)} < {min_length}), returning original data")
        return df.copy()
    
    try:
        kf = KalmanFilterOHLC(
            process_variance=process_variance,
            observation_variance=observation_variance,
            **kwargs
        )
        return kf.fit(df)
    except Exception as e:
        log_error(f"Error applying Kalman Filter: {e}")
        log_warn("Returning original data without Kalman Filter")
        return df.copy()


def validate_kalman_params(params: Optional[Dict[str, Any]]) -> bool:
    """
    Validate Kalman Filter parameters.
    
    Args:
        params: Dictionary of parameters to validate
        
    Returns:
        True if valid, False otherwise
    """
    if params is None:
        return True
    
    if not isinstance(params, dict):
        return False
    
    # Check process_variance
    if 'process_variance' in params:
        pv = params['process_variance']
        if not isinstance(pv, (int, float)) or pv <= 0:
            return False
    
    # Check observation_variance
    if 'observation_variance' in params:
        ov = params['observation_variance']
        if not isinstance(ov, (int, float)) or ov <= 0:
            return False
    
    # Check initial_uncertainty
    if 'initial_uncertainty' in params:
        iu = params['initial_uncertainty']
        if not isinstance(iu, (int, float)) or iu <= 0:
            return False
    
    return True

