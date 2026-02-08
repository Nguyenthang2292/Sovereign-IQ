"""
Risk Manager Module

Handles risk management, position sizing, and pre-flight checks.
Fetches account balance and calculates safe position sizes.
"""

from typing import Optional

from modules.common.core.data_fetcher import DataFetcher
from modules.common.ui.logging import log_error, log_info, log_warn


class RiskManager:
    """
    Risk Manager for position sizing and safety checks.

    Example:
        >>> risk_mgr = RiskManager(data_fetcher, balance_pct=0.95, default_leverage=2)
        >>> position_size = risk_mgr.calculate_position_size(api_key, api_secret)
    """

    def __init__(
        self,
        data_fetcher: DataFetcher,
        balance_percentage: float = 0.95,
        default_leverage: int = 2,
        max_leverage: int = 125,
        min_position_size: float = 10.0,
        max_position_size: Optional[float] = None,
        emergency_stop_enabled: bool = True,
    ):
        """
        Initialize RiskManager.

        Args:
            data_fetcher: DataFetcher instance for balance fetching
            balance_percentage: Percentage of balance to use (default: 0.95 = 95%)
            default_leverage: Default leverage (default: 2x)
            max_leverage: Maximum allowed leverage (default: 125x)
            min_position_size: Minimum position size in USDT (default: 10.0)
            max_position_size: Optional maximum position size in USDT
            emergency_stop_enabled: Enable emergency stop mechanism
        """
        if not (0 < balance_percentage <= 1.0):
            raise ValueError(f"balance_percentage must be between 0 and 1, got {balance_percentage}")
        if default_leverage < 1 or default_leverage > max_leverage:
            raise ValueError(f"default_leverage must be between 1 and {max_leverage}, got {default_leverage}")

        self.data_fetcher = data_fetcher
        self.balance_percentage = balance_percentage
        self.default_leverage = default_leverage
        self.max_leverage = max_leverage
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.emergency_stop_enabled = emergency_stop_enabled
        self._emergency_stop_triggered = False

    def fetch_account_balance(self, api_key: str, api_secret: str, testnet: bool = False) -> Optional[float]:
        """
        Fetch USDT balance from Binance Futures.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet if True

        Returns:
            Available USDT balance or None if error
        """
        try:
            balance: Optional[float] = self.data_fetcher.fetch_binance_account_balance(
                api_key=api_key, api_secret=api_secret, testnet=testnet, currency="USDT"
            )

            if balance is None:
                log_error("Failed to fetch account balance from Binance")
                return None

            log_info(f"Fetched account balance: ${balance:,.2f} USDT")
            return balance

        except Exception as e:
            log_error(f"Error fetching account balance: {e}", exc_info=True)
            return None

    def calculate_position_size(self, api_key: str, api_secret: str, testnet: bool = False) -> Optional[float]:
        """
        Calculate position size based on account balance and risk percentage.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet if True

        Returns:
            Position size in USDT or None if error

        Formula:
            position_size = available_balance × balance_percentage
        """
        # Check emergency stop
        if self._emergency_stop_triggered:
            log_error("Emergency stop is active. Trading is disabled.")
            return None

        # Fetch balance
        balance: Optional[float] = self.fetch_account_balance(api_key, api_secret, testnet)
        if balance is None or balance <= 0:
            log_error(f"Invalid balance: {balance}")
            return None

        # Calculate position size
        position_size: float = balance * self.balance_percentage

        # Validate minimum
        if position_size < self.min_position_size:
            log_warn(f"Position size ${position_size:.2f} is below minimum ${self.min_position_size:.2f}")
            return None

        # Validate maximum
        if self.max_position_size and position_size > self.max_position_size:
            log_warn(
                f"Position size ${position_size:.2f} exceeds maximum ${self.max_position_size:.2f}, capping at maximum"
            )
            position_size = self.max_position_size

        log_info(
            f"Calculated position size: ${position_size:,.2f} USDT ({self.balance_percentage * 100:.1f}% of balance)"
        )
        return position_size

    def validate_leverage(self, symbol: str, leverage: int, market_info: Optional[dict] = None) -> bool:
        """
        Validate leverage for a given symbol.

        Args:
            symbol: Trading symbol
            leverage: Requested leverage
            market_info: Optional market info containing leverage limits

        Returns:
            True if leverage is valid, False otherwise
        """
        if leverage < 1:
            log_error(f"Leverage must be >= 1, got {leverage}")
            return False

        if leverage > self.max_leverage:
            log_error(f"Leverage {leverage}x exceeds maximum {self.max_leverage}x for {symbol}")
            return False

        # TODO: Check symbol-specific leverage limits from market_info
        # if market_info:
        #     max_leverage_symbol = market_info.get('limits', {}).get('leverage', {}).get('max')
        #     if max_leverage_symbol and leverage > max_leverage_symbol:
        #         log_error(f"Leverage {leverage}x exceeds symbol limit {max_leverage_symbol}x for {symbol}")
        #         return False

        return True

    def validate_sufficient_margin(
        self,
        position_size: float,
        leverage: int,
        entry_price: float,
        balance: float,
    ) -> bool:
        """
        Validate sufficient margin for a position.

        Args:
            position_size: Position size in USDT
            leverage: Leverage multiplier
            entry_price: Entry price
            balance: Available balance in USDT

        Returns:
            True if sufficient margin, False otherwise

        Formula:
            required_margin = (position_size × entry_price) / leverage
        """
        if leverage <= 0 or entry_price <= 0:
            log_error(f"Invalid leverage={leverage} or entry_price={entry_price}")
            return False

        # Calculate required margin
        position_value = position_size  # Already in USDT
        required_margin = position_value / leverage

        if required_margin > balance:
            log_error(f"Insufficient margin: required ${required_margin:.2f}, available ${balance:.2f}")
            return False

        log_info(f"Margin check passed: required ${required_margin:.2f}, available ${balance:.2f}")
        return True

    def trigger_emergency_stop(self, reason: str):
        """
        Trigger emergency stop to prevent further trading.

        Args:
            reason: Reason for emergency stop
        """
        if self.emergency_stop_enabled:
            self._emergency_stop_triggered = True
            log_error(f"🛑 EMERGENCY STOP TRIGGERED: {reason}")
        else:
            log_warn(f"Emergency stop would be triggered (disabled): {reason}")

    def reset_emergency_stop(self, reason: str = "Manual reset"):
        """
        Reset emergency stop.

        Args:
            reason: Reason for reset
        """
        self._emergency_stop_triggered = False
        log_info(f"✅ Emergency stop reset: {reason}")

    @property
    def is_emergency_stop_active(self) -> bool:
        """Check if emergency stop is active."""
        return self._emergency_stop_triggered

    def pre_flight_checks(self, symbol: str, leverage: int, balance: float) -> bool:
        """
        Run pre-flight checks before order execution.

        Args:
            symbol: Trading symbol
            leverage: Requested leverage
            balance: Available balance

        Returns:
            True if all checks pass, False otherwise
        """
        # Check emergency stop
        if self._emergency_stop_triggered:
            log_error("Pre-flight failed: Emergency stop is active")
            return False

        # Check leverage
        if not self.validate_leverage(symbol, leverage):
            log_error(f"Pre-flight failed: Invalid leverage {leverage}x for {symbol}")
            return False

        # Check balance
        if balance <= 0:
            log_error(f"Pre-flight failed: Invalid balance ${balance:.2f}")
            return False

        log_info(f"✅ Pre-flight checks passed for {symbol}")
        return True
