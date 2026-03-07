"""Risk management and limit checking."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .main_window import AutoTradeDashboard


class RiskManager:
    """Manages risk limits and trading constraints."""

    def __init__(self, parent: "AutoTradeDashboard"):
        self.parent = parent

    def check_limits(
        self,
        symbol: Optional[str] = None,
        position_size: Optional[float] = None,
        leverage: Optional[int] = None,
    ) -> bool:
        """
        Check if trading within risk limits:
        - Max open positions
        - Max daily loss
        - Max position size
        - Total exposure
        - Per-symbol position limits
        - Leverage limits
        - Account balance
        """
        if not self.parent.settings_manager.get("risk.limits_enabled", True):
            return True
        try:
            positions = self.parent.data_service.get_positions()
            if positions is None:
                print("Warning: Could not fetch positions for risk check")
                return False

            if not isinstance(positions, (list, tuple)):
                print(f"Error: Invalid positions type: {type(positions)}")
                return False

            account_data = self.parent.data_service.get_account_data()
            if not account_data:
                print("Warning: Could not fetch account data for risk check")
                return False

            balance = account_data.get("balance", 0)
            if balance <= 0:
                print("Error: Invalid account balance")
                return False

            if not self._check_max_positions(list(positions)):
                return False

            if not self._check_daily_loss(account_data):
                return False

            if not self._check_exposure(list(positions), balance, position_size, leverage):
                return False

            if position_size is not None and not self._check_position_size(balance, position_size):
                return False

            if symbol is not None and not self._check_symbol_limit(list(positions), symbol):
                return False

            if leverage is not None and not self._check_leverage(leverage):
                return False

            if not self._check_min_balance(balance):
                return False

            return True

        except Exception as e:
            print(f"Error checking risk limits: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _check_max_positions(self, positions: list) -> bool:
        """Check max open positions limit."""
        max_positions = self.parent.settings_manager.get("risk.max_open_positions", 3)

        if not isinstance(max_positions, int) or max_positions <= 0:
            print(f"Error: Invalid max_positions setting: {max_positions}, using default 3")
            max_positions = 3

        if len(positions) >= max_positions:
            print(f"Risk limit exceeded: Max positions reached ({len(positions)}/{max_positions})")
            return False
        return True

    def _check_daily_loss(self, account_data: dict) -> bool:
        """Check daily loss limit."""
        max_daily_loss_pct = self.parent.settings_manager.get("risk.max_daily_loss_pct", 5.0)

        if not isinstance(max_daily_loss_pct, (int, float)) or max_daily_loss_pct <= 0:
            print(f"Error: Invalid max_daily_loss_pct setting: {max_daily_loss_pct}, using default 5.0")
            max_daily_loss_pct = 5.0

        daily_pnl_pct = account_data.get("daily_pnl_pct", 0)
        if daily_pnl_pct <= -max_daily_loss_pct:
            print(f"Risk limit exceeded: Daily loss limit hit ({daily_pnl_pct:.2f}% / -{max_daily_loss_pct}%)")
            return False
        return True

    def _check_exposure(
        self, positions: list, balance: float, position_size: Optional[float], leverage: Optional[int]
    ) -> bool:
        """Check total exposure limit."""
        max_exposure_pct = self.parent.settings_manager.get("risk.max_exposure_pct", 30.0)

        if not isinstance(max_exposure_pct, (int, float)) or max_exposure_pct <= 0:
            print(f"Error: Invalid max_exposure_pct setting: {max_exposure_pct}, using default 30.0")
            max_exposure_pct = 30.0

        total_exposure = sum(abs(float(p.get("notional", 0))) for p in positions)

        if position_size is not None and leverage is not None:
            total_exposure += position_size * leverage

        exposure_pct = (total_exposure / balance) * 100
        if exposure_pct >= max_exposure_pct:
            print(f"Risk limit exceeded: Max exposure reached ({exposure_pct:.1f}% / {max_exposure_pct}%)")
            return False
        return True

    def _check_position_size(self, balance: float, position_size: float) -> bool:
        """Check max position size limit."""
        max_position_size_pct = self.parent.settings_manager.get("risk.max_position_size_pct", 10.0)

        if not isinstance(max_position_size_pct, (int, float)) or max_position_size_pct <= 0:
            print(f"Error: Invalid max_position_size_pct: {max_position_size_pct}, using default 10.0")
            max_position_size_pct = 10.0

        max_position_size = balance * (max_position_size_pct / 100)
        if position_size > max_position_size:
            print(
                f"Risk limit exceeded: Position size too large "
                f"({position_size:.2f} USDT > {max_position_size:.2f} USDT / "
                f"{max_position_size_pct}% of balance)"
            )
            return False
        return True

    def _check_symbol_limit(self, positions: list, symbol: str) -> bool:
        """Check per-symbol position limit."""
        from modules.common.domain.symbol_codec import SymbolCodec

        _symbol_codec = SymbolCodec()
        max_per_symbol = self.parent.settings_manager.get("risk.max_positions_per_symbol", 1)

        if not isinstance(max_per_symbol, int) or max_per_symbol <= 0:
            print(f"Error: Invalid max_positions_per_symbol: {max_per_symbol}, using default 1")
            max_per_symbol = 1

        # Normalize both sides for comparison (BTCUSDT and BTC/USDT should match)
        normalized_input = _symbol_codec.to_db(symbol)
        symbol_positions = [p for p in positions if _symbol_codec.to_db(p.get("symbol", "")) == normalized_input]

        if len(symbol_positions) >= max_per_symbol:
            print(f"Risk limit exceeded: Max positions for {symbol} reached ({len(symbol_positions)}/{max_per_symbol})")
            return False
        return True

    def _check_leverage(self, leverage: int) -> bool:
        """Check leverage limit."""
        max_leverage = self.parent.settings_manager.get("risk.max_leverage", 5)

        if not isinstance(max_leverage, int) or max_leverage <= 0:
            print(f"Error: Invalid max_leverage setting: {max_leverage}, using default 5")
            max_leverage = 5

        if leverage > max_leverage:
            print(f"Risk limit exceeded: Leverage too high ({leverage}x > {max_leverage}x)")
            return False
        return True

    def _check_min_balance(self, balance: float) -> bool:
        """Check minimum account balance."""
        min_balance = self.parent.settings_manager.get("risk.min_account_balance", 10.0)

        if not isinstance(min_balance, (int, float)) or min_balance < 0:
            print(f"Error: Invalid min_account_balance: {min_balance}, using default 10.0")
            min_balance = 10.0

        if balance < min_balance:
            print(f"Risk limit exceeded: Account balance too low ({balance:.2f} USDT < {min_balance:.2f} USDT minimum)")
            return False
        return True
