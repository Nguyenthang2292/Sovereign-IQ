"""
Auto-Trade Backtester Adapter.

Adapts the existing backtester module to work with auto-trade signal pipeline
and position monitoring strategies.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from modules.backtester.core.backtester import FullBacktester
from modules.common.core.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)


class AutoTradeBacktester:
    """
    Adapter for backtesting auto-trade strategies.

    This class wraps the existing FullBacktester and provides auto-trade-specific
    functionality like Martingale strategy simulation and break-even testing.
    """

    def __init__(
        self,
        data_fetcher: DataFetcher,
        stop_loss_pct: float = 0.50,  # 50% stop loss as per auto_trade spec
        take_profit_pct: float = 0.05,  # 5% take profit as per auto_trade spec
        trailing_stop_pct: float = 0.015,  # 1.5% trailing stop
        max_hold_periods: int = 100,
        risk_per_trade: float = 0.95,  # 95% account balance per trade
        leverage: int = 2,  # 2x leverage default
        enable_breakeven: bool = True,  # Enable break-even at 30% drawdown
        breakeven_drawdown_pct: float = 0.30,  # 30% drawdown threshold for BE
        enable_martingale: bool = False,  # Martingale strategy disabled by default for safety
        martingale_max_steps: int = 4,  # Max 4 Martingale steps
        martingale_max_leverage: int = 16,  # Max 16x leverage
    ):
        """
        Initialize Auto-Trade Backtester.

        Args:
            data_fetcher: DataFetcher instance
            stop_loss_pct: Stop loss percentage (default: 50%)
            take_profit_pct: Take profit percentage (default: 5%)
            trailing_stop_pct: Trailing stop percentage (default: 1.5%)
            max_hold_periods: Maximum periods to hold position
            risk_per_trade: Risk percentage per trade (default: 95%)
            leverage: Initial leverage (default: 2x)
            enable_breakeven: Enable break-even protection (default: True)
            breakeven_drawdown_pct: Drawdown threshold for BE move (default: 30%)
            enable_martingale: Enable Martingale strategy (default: False for safety)
            martingale_max_steps: Maximum Martingale steps (default: 4)
            martingale_max_leverage: Maximum leverage in Martingale (default: 16x)
        """
        self.data_fetcher = data_fetcher
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_hold_periods = max_hold_periods
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.enable_breakeven = enable_breakeven
        self.breakeven_drawdown_pct = breakeven_drawdown_pct
        self.enable_martingale = enable_martingale
        self.martingale_max_steps = martingale_max_steps
        self.martingale_max_leverage = martingale_max_leverage

        # Initialize base backtester with auto-trade parameters
        self.backtester = FullBacktester(
            data_fetcher=data_fetcher,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trailing_stop_pct=trailing_stop_pct,
            max_hold_periods=max_hold_periods,
            risk_per_trade=risk_per_trade,
            signal_mode="single_signal",  # Auto-trade uses single best signal
        )

        logger.info(
            f"AutoTradeBacktester initialized: SL={stop_loss_pct}, TP={take_profit_pct}, "
            f"Leverage={leverage}, BE={enable_breakeven}, Martingale={enable_martingale}"
        )

    def backtest_strategy(
        self,
        symbol: str,
        timeframe: str,
        lookback: int,
        initial_capital: float = 10000.0,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Run backtest with auto-trade strategy.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            lookback: Number of candles to look back
            initial_capital: Initial capital (default: 10000)
            df: Optional DataFrame to use instead of fetching

        Returns:
            Dictionary with backtest results including auto-trade metrics
        """
        try:
            # Run base backtest
            result = self.backtester.backtest(
                symbol=symbol,
                timeframe=timeframe,
                lookback=lookback,
                signal_type="LONG",  # Will be overridden by single_signal mode
                initial_capital=initial_capital,
                df=df,
            )

            if not result or not result.get("trades"):
                logger.warning(f"No trades generated for {symbol}")
                return result

            # Apply auto-trade specific post-processing
            trades = result["trades"]

            # Simulate break-even moves
            if self.enable_breakeven:
                trades = self._apply_breakeven_simulation(trades, initial_capital)

            # Simulate Martingale if enabled
            if self.enable_martingale:
                trades = self._apply_martingale_simulation(trades, initial_capital)

            # Recalculate metrics with updated trades
            from modules.backtester.core.equity_curve import calculate_equity_curve
            from modules.backtester.core.metrics import calculate_metrics

            equity_curve = calculate_equity_curve(
                trades=trades,
                initial_capital=initial_capital,
                num_periods=len(df) if df is not None else lookback,
                risk_per_trade=self.risk_per_trade,
            )

            metrics = calculate_metrics(trades=trades, equity_curve=equity_curve, timeframe=timeframe)

            # Add auto-trade specific metrics
            metrics["breakeven_moves"] = sum(1 for t in trades if t.get("be_moved", False))
            metrics["martingale_trades"] = sum(1 for t in trades if t.get("martingale_step", 0) > 0)
            metrics["max_martingale_step"] = max((t.get("martingale_step", 0) for t in trades), default=0)
            metrics["leverage_used"] = self.leverage

            result.update(
                {
                    "trades": trades,
                    "equity_curve": equity_curve,
                    "metrics": metrics,
                }
            )

            return result

        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}", exc_info=True)
            from modules.backtester.core.metrics import empty_backtest_result

            return empty_backtest_result()

    def _apply_breakeven_simulation(self, trades: List[Dict], initial_capital: float) -> List[Dict]:
        """
        Simulate break-even moves when drawdown reaches threshold.

        Args:
            trades: List of trade dictionaries
            initial_capital: Initial capital

        Returns:
            List of trades with break-even simulation applied
        """
        updated_trades = []

        for trade in trades:
            trade_copy = trade.copy()

            # Calculate drawdown from entry
            entry_price = trade.get("entry_price", 0)
            if entry_price == 0:
                updated_trades.append(trade_copy)
                continue

            # Check if position reached BE threshold during hold
            # For simplicity, check if worst drawdown reached threshold
            side = trade.get("side", "LONG")
            exit_price = trade.get("exit_price", entry_price)

            # Calculate max drawdown during position
            if side == "LONG":
                # For LONG, drawdown is when price goes below entry
                max_drawdown_pct = (entry_price - exit_price) / entry_price
            else:
                # For SHORT, drawdown is when price goes above entry
                max_drawdown_pct = (exit_price - entry_price) / entry_price

            # If drawdown exceeded threshold, simulate BE move
            if max_drawdown_pct >= self.breakeven_drawdown_pct:
                trade_copy["be_moved"] = True
                # Simulate that TP was moved to break-even (0% profit)
                # This means if position would have closed with loss,
                # it now closes at break-even
                if trade.get("pnl", 0) < 0:
                    trade_copy["pnl"] = 0
                    trade_copy["pnl_pct"] = 0
                    trade_copy["exit_reason"] = "BREAKEVEN_PROTECTION"
                    logger.debug(f"BE protection applied to trade at {trade.get('entry_time', 'unknown')}")
            else:
                trade_copy["be_moved"] = False

            updated_trades.append(trade_copy)

        return updated_trades

    def _apply_martingale_simulation(self, trades: List[Dict], initial_capital: float) -> List[Dict]:
        """
        Simulate Martingale strategy for consecutive losses.

        Args:
            trades: List of trade dictionaries
            initial_capital: Initial capital

        Returns:
            List of trades with Martingale simulation applied
        """
        if not trades:
            return trades

        updated_trades = []
        martingale_step = 0
        current_leverage = self.leverage
        total_loss_to_recover = 0.0

        for i, trade in enumerate(trades):
            trade_copy = trade.copy()
            trade_copy["martingale_step"] = martingale_step
            trade_copy["leverage_used"] = current_leverage

            # Check if previous trade was loss
            if i > 0:
                prev_pnl = trades[i - 1].get("pnl", 0)
                if prev_pnl < 0:
                    # Loss detected, increment Martingale
                    if martingale_step < self.martingale_max_steps:
                        martingale_step += 1
                        current_leverage = min(current_leverage * 2, self.martingale_max_leverage)
                        total_loss_to_recover += abs(prev_pnl)
                        logger.debug(
                            f"Martingale step {martingale_step}: "
                            f"Leverage {current_leverage}x, "
                            f"Loss to recover: ${total_loss_to_recover:.2f}"
                        )
                    else:
                        logger.warning(f"Max Martingale steps reached ({self.martingale_max_steps})")
                else:
                    # Profit detected, reset Martingale
                    if martingale_step > 0:
                        logger.info(f"Martingale chain recovered at step {martingale_step}")
                    martingale_step = 0
                    current_leverage = self.leverage
                    total_loss_to_recover = 0.0

            # Adjust PnL based on leverage
            # Higher leverage = higher profit/loss magnitude
            if martingale_step > 0:
                leverage_multiplier = current_leverage / self.leverage
                trade_copy["pnl"] = trade.get("pnl", 0) * leverage_multiplier
                trade_copy["pnl_pct"] = trade.get("pnl_pct", 0) * leverage_multiplier

            updated_trades.append(trade_copy)

        return updated_trades

    def validate_martingale_safety(self, trades: List[Dict]) -> Dict[str, any]:
        """
        Validate Martingale strategy safety metrics.

        Args:
            trades: List of trades with Martingale applied

        Returns:
            Dictionary with safety metrics
        """
        if not trades:
            return {"safe": True, "max_consecutive_losses": 0, "max_leverage_used": self.leverage}

        consecutive_losses = 0
        max_consecutive_losses = 0
        max_leverage_used = self.leverage

        for trade in trades:
            if trade.get("pnl", 0) < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

            leverage_used = trade.get("leverage_used", self.leverage)
            max_leverage_used = max(max_leverage_used, leverage_used)

        # Check if Martingale would have exceeded limits
        would_exceed_steps = max_consecutive_losses > self.martingale_max_steps
        would_exceed_leverage = max_leverage_used > self.martingale_max_leverage

        return {
            "safe": not (would_exceed_steps or would_exceed_leverage),
            "max_consecutive_losses": max_consecutive_losses,
            "max_leverage_used": max_leverage_used,
            "exceeded_max_steps": would_exceed_steps,
            "exceeded_max_leverage": would_exceed_leverage,
        }
