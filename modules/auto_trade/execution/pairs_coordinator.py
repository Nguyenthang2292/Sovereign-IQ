"""
Pairs Coordinator Module

Orchestrates pairs trading logic including regime detection,
position sizing, and atomic execution.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Tuple

from modules.auto_trade.execution.correlation_scanner import CorrelationScanner, HedgeCandidate
from modules.auto_trade.execution.order_builder import OrderBuilder, OrderTicket
from modules.auto_trade.execution.order_executor import OrderExecutor
from modules.auto_trade.execution.risk_manager import RiskManager
from modules.common.ui.logging import log_error, log_info, log_warn

Regime = Literal["STAT_ARB", "MOMENTUM", "BLENDED"]
HedgeDirection = Literal["opposite", "same", "correlation_based"]


@dataclass
class PairsSettings:
    """Pairs trading settings."""

    enabled: bool = False
    min_correlation: float = 0.65
    lookback: int = 100
    timeframe: str = "1h"
    refresh_interval: int = 7200
    adx_low: int = 20
    adx_high: int = 30
    stat_arb_direction: HedgeDirection = "opposite"
    momentum_direction: HedgeDirection = "opposite"
    blended_direction: HedgeDirection = "correlation_based"
    drift_threshold: float = 0.15
    hedge_leverage_min: int = 1
    hedge_leverage_max: int = 5


@dataclass
class PairExecutionResult:
    """Result of pair execution."""

    success: bool
    pair_id: str
    signal_ticket: Optional[OrderTicket] = None
    hedge_ticket: Optional[OrderTicket] = None
    error: Optional[str] = None
    rollback_performed: bool = False


@dataclass
class PairsState:
    """Current state of pairs trading."""

    active_pairs: Dict[str, Dict] = field(default_factory=dict)
    enabled: bool = False
    last_scan_time: Optional[datetime] = None


class PairsCoordinator:
    """
    Orchestrates pairs trading logic.

    Key responsibilities:
    - Determine if pairs trading should be activated
    - Find hedge symbol using CorrelationScanner
    - Determine trading regime (STAT_ARB, MOMENTUM, BLENDED)
    - Calculate position sizes based on regime
    - Execute pairs atomically with rollback support
    """

    def __init__(
        self,
        correlation_scanner: Optional[CorrelationScanner] = None,
        order_executor: Optional[OrderExecutor] = None,
        order_builder: Optional[OrderBuilder] = None,
        risk_manager: Optional[RiskManager] = None,
        settings: Optional[PairsSettings] = None,
    ):
        """
        Initialize PairsCoordinator.

        Args:
            correlation_scanner: CorrelationScanner instance
            order_executor: OrderExecutor instance
            order_builder: OrderBuilder instance
            risk_manager: RiskManager instance
            settings: Pairs trading settings
        """
        self._correlation_scanner = correlation_scanner
        self._order_executor = order_executor
        self._order_builder = order_builder
        self._risk_manager = risk_manager
        self._settings = settings or PairsSettings()
        self._state = PairsState()

    @property
    def correlation_scanner(self) -> CorrelationScanner:
        """Get or create CorrelationScanner."""
        if self._correlation_scanner is None:
            self._correlation_scanner = CorrelationScanner(
                min_correlation=self._settings.min_correlation,
                lookback=self._settings.lookback,
                timeframe=self._settings.timeframe,
                refresh_interval=self._settings.refresh_interval,
            )
        return self._correlation_scanner

    @property
    def order_builder(self) -> OrderBuilder:
        """Get or create OrderBuilder."""
        if self._order_builder is None:
            self._order_builder = OrderBuilder()
        return self._order_builder

    def update_settings(self, settings: PairsSettings) -> None:
        """Update pairs trading settings."""
        self._settings = settings
        if self._correlation_scanner:
            self._correlation_scanner.min_correlation = settings.min_correlation
            self._correlation_scanner.lookback = settings.lookback
            self._correlation_scanner.timeframe = settings.timeframe
            self._correlation_scanner.refresh_interval = settings.refresh_interval

    def should_activate_pairs(self, signal_symbol: str, settings: Optional[PairsSettings] = None) -> bool:
        """
        Check if pairs trading should be activated for a signal.

        Args:
            signal_symbol: The signal symbol
            settings: Optional settings override

        Returns:
            True if pairs should be activated
        """
        settings = settings or self._settings

        if not settings.enabled:
            return False

        return True

    def find_hedge_symbol(
        self,
        signal_symbol: str,
        candidate_symbols: Optional[List[str]] = None,
    ) -> Optional[HedgeCandidate]:
        """
        Find the best hedge symbol for a signal.

        Args:
            signal_symbol: The signal symbol to hedge
            candidate_symbols: Optional list of candidates

        Returns:
            Best hedge candidate or None if no suitable hedge found
        """
        candidates = self.correlation_scanner.scan_hedge_candidates(
            signal_symbol=signal_symbol,
            candidate_symbols=candidate_symbols,
            max_candidates=5,
        )

        if not candidates:
            log_warn(f"[PairsCoordinator] No suitable hedge found for {signal_symbol}")
            return None

        best = candidates[0]
        log_info(
            f"[PairsCoordinator] Found hedge {best.symbol} for {signal_symbol} " f"(correlation={best.correlation:.3f})"
        )
        return best

    def determine_regime(
        self,
        signal_symbol: str,
        hedge_symbol: str,
        settings: Optional[PairsSettings] = None,
    ) -> Regime:
        """
        Determine trading regime based on ADX.

        Args:
            signal_symbol: Signal symbol
            hedge_symbol: Hedge symbol
            settings: Optional settings override

        Returns:
            Trading regime: STAT_ARB, MOMENTUM, or BLENDED
        """
        settings = settings or self._settings

        regime = self.correlation_scanner.calculate_adx_for_regime(
            signal_symbol,
            hedge_symbol,
            adx_low=settings.adx_low,
            adx_high=settings.adx_high,
        )

        if regime is None:
            regime = "BLENDED"

        log_info(f"[PairsCoordinator] Regime for {signal_symbol}/{hedge_symbol}: {regime}")
        return regime

    def determine_hedge_direction(
        self,
        signal_side: Literal["BUY", "SELL"],
        regime: Regime,
        settings: Optional[PairsSettings] = None,
    ) -> Literal["BUY", "SELL"]:
        """
        Determine hedge direction based on regime config.

        Args:
            signal_side: Signal side (BUY=LONG, SELL=SHORT)
            regime: Trading regime
            settings: Optional settings override

        Returns:
            Hedge side
        """
        settings = settings or self._settings

        direction_config: HedgeDirection
        if regime == "STAT_ARB":
            direction_config = settings.stat_arb_direction
        elif regime == "MOMENTUM":
            direction_config = settings.momentum_direction
        else:
            direction_config = settings.blended_direction

        if direction_config == "same":
            return signal_side
        elif direction_config == "opposite":
            return "SELL" if signal_side == "BUY" else "BUY"
        else:
            return "SELL" if signal_side == "BUY" else "BUY"

    def calculate_position_sizes(
        self,
        regime: Regime,
        signal_symbol: str,
        hedge_symbol: str,
        signal_side: Literal["BUY", "SELL"],
        total_position_size: float,
        hedge_ratio: float,
        hedge_correlation: float,
        signal_leverage: int = 2,
        settings: Optional[PairsSettings] = None,
    ) -> Tuple[float, float, int, int]:
        """
        Calculate position sizes for both legs based on regime.

        Args:
            regime: Trading regime
            signal_symbol: Signal symbol
            hedge_symbol: Hedge symbol
            signal_side: Signal side (BUY/LONG or SELL/SHORT)
            total_position_size: Total position size in USDT
            hedge_ratio: Hedge ratio from correlation
            hedge_correlation: Correlation coefficient
            signal_leverage: Leverage for signal leg
            settings: Optional settings override

        Returns:
            Tuple of (signal_size, hedge_size, signal_leverage, hedge_leverage)
        """
        settings = settings or self._settings

        signal_size: float
        hedge_size: float
        hedge_leverage: int

        if regime == "STAT_ARB":
            signal_size, hedge_size = self._calculate_stat_arb_sizes(total_position_size, hedge_ratio)
            hedge_leverage = min(
                max(int(abs(hedge_ratio) * signal_leverage), settings.hedge_leverage_min),
                settings.hedge_leverage_max,
            )

        elif regime == "MOMENTUM":
            signal_size, hedge_size = self._calculate_momentum_sizes(total_position_size, hedge_correlation)
            hedge_leverage = signal_leverage

        else:
            signal_size, hedge_size = self._calculate_blended_sizes(total_position_size, hedge_ratio, hedge_correlation)
            hedge_leverage = min(
                max(int(abs(hedge_ratio) * signal_leverage), settings.hedge_leverage_min),
                settings.hedge_leverage_max,
            )

        signal_size = round(signal_size, 2)
        hedge_size = round(hedge_size, 2)

        return signal_size, hedge_size, signal_leverage, hedge_leverage

    def _calculate_stat_arb_sizes(self, total_size: float, hedge_ratio: float) -> Tuple[float, float]:
        """Calculate sizes for statistical arbitrage mode using hedge ratio."""
        signal_size = total_size / (1 + abs(hedge_ratio))
        hedge_size = signal_size * abs(hedge_ratio)
        return signal_size, hedge_size

    def _calculate_momentum_sizes(self, total_size: float, correlation: float) -> Tuple[float, float]:
        """Calculate sizes for momentum mode using risk-parity approach."""
        weight = (1 + correlation) / 2
        signal_size = total_size * weight
        hedge_size = total_size * (1 - weight)
        return signal_size, hedge_size

    def _calculate_blended_sizes(
        self, total_size: float, hedge_ratio: float, correlation: float
    ) -> Tuple[float, float]:
        """Calculate sizes for blended mode (50/50 of both methods)."""
        stat_arb_signal, stat_arb_hedge = self._calculate_stat_arb_sizes(total_size * 0.5, hedge_ratio)
        momentum_signal, momentum_hedge = self._calculate_momentum_sizes(total_size * 0.5, correlation)
        return stat_arb_signal + momentum_signal, stat_arb_hedge + momentum_hedge

    async def execute_pair_atomically(
        self,
        signal_ticket: OrderTicket,
        hedge_ticket: OrderTicket,
        order_executor: OrderExecutor,
    ) -> PairExecutionResult:
        """
        Execute both legs of a pair atomically with rollback support.

        Args:
            signal_ticket: Signal leg order ticket
            hedge_ticket: Hedge leg order ticket
            order_executor: OrderExecutor instance

        Returns:
            PairExecutionResult with success status and details
        """
        pair_id = str(uuid.uuid4())
        signal_ticket.client_order_id = f"{pair_id}-SIGNAL"
        hedge_ticket.client_order_id = f"{pair_id}-HEDGE"

        try:
            log_info(f"[PairsCoordinator] Executing pair {pair_id}")

            result_signal = order_executor.place_order(
                symbol=signal_ticket.symbol,
                side=signal_ticket.side,
                amount=signal_ticket.amount,
                leverage=signal_ticket.leverage,
            )

            if not result_signal.get("success", False):
                error_msg = result_signal.get("error", "Signal leg execution failed")
                log_error(f"[PairsCoordinator] Signal leg failed: {error_msg}")
                return PairExecutionResult(
                    success=False,
                    pair_id=pair_id,
                    signal_ticket=signal_ticket,
                    error=error_msg,
                )

            result_hedge = order_executor.place_order(
                symbol=hedge_ticket.symbol,
                side=hedge_ticket.side,
                amount=hedge_ticket.amount,
                leverage=hedge_ticket.leverage,
            )

            if not result_hedge.get("success", False):
                log_warn(f"[PairsCoordinator] Hedge leg failed, initiating rollback: {result_hedge.get('error')}")

                rollback_result = order_executor.place_order(
                    symbol=signal_ticket.symbol,
                    side="SELL" if signal_ticket.side == "BUY" else "BUY",
                    amount=signal_ticket.amount,
                    leverage=signal_ticket.leverage,
                )

                rollback_performed = rollback_result.get("success", False)
                return PairExecutionResult(
                    success=False,
                    pair_id=pair_id,
                    signal_ticket=signal_ticket,
                    hedge_ticket=hedge_ticket,
                    error=f"Hedge failed, rollback {'succeeded' if rollback_performed else 'failed'}",
                    rollback_performed=rollback_performed,
                )

            self._state.active_pairs[pair_id] = {
                "signal_symbol": signal_ticket.symbol,
                "hedge_symbol": hedge_ticket.symbol,
                "created_at": datetime.now(timezone.utc),
            }

            log_info(f"[PairsCoordinator] Pair {pair_id} executed successfully")

            return PairExecutionResult(
                success=True,
                pair_id=pair_id,
                signal_ticket=signal_ticket,
                hedge_ticket=hedge_ticket,
            )

        except Exception as e:
            log_error(f"[PairsCoordinator] Unexpected error in atomic execution: {e}")
            return PairExecutionResult(
                success=False,
                pair_id=pair_id,
                signal_ticket=signal_ticket,
                error=str(e),
            )

    def build_pair_tickets(
        self,
        signal_symbol: str,
        signal_side: Literal["BUY", "SELL"],
        total_position_size: float,
        hedge_candidate: HedgeCandidate,
        regime: Regime,
        tp_pct: float = 5.0,
        sl_pct: float = 50.0,
        signal_leverage: int = 2,
        settings: Optional[PairsSettings] = None,
    ) -> Tuple[OrderTicket, OrderTicket]:
        """
        Build order tickets for both legs of a pair.

        Args:
            signal_symbol: Signal symbol
            signal_side: Signal side (BUY/LONG or SELL/SHORT)
            total_position_size: Total position size
            hedge_candidate: Selected hedge candidate
            regime: Trading regime
            tp_pct: Take profit percentage
            sl_pct: Stop loss percentage
            signal_leverage: Signal leg leverage
            settings: Optional settings override

        Returns:
            Tuple of (signal_ticket, hedge_ticket)
        """
        settings = settings or self._settings

        signal_size, hedge_size, signal_lev, hedge_lev = self.calculate_position_sizes(
            regime=regime,
            signal_symbol=signal_symbol,
            hedge_symbol=hedge_candidate.symbol,
            signal_side=signal_side,
            total_position_size=total_position_size,
            hedge_ratio=hedge_candidate.hedge_ratio,
            hedge_correlation=hedge_candidate.correlation,
            signal_leverage=signal_leverage,
            settings=settings,
        )

        hedge_side = self.determine_hedge_direction(signal_side, regime, settings)

        signal_ticket = OrderTicket(
            symbol=signal_symbol,
            side=signal_side,
            amount=signal_size,
            leverage=signal_lev,
            take_profit_percentage=tp_pct,
            stop_loss_percentage=sl_pct,
        )

        hedge_ticket = OrderTicket(
            symbol=hedge_candidate.symbol,
            side=hedge_side,
            amount=hedge_size,
            leverage=hedge_lev,
            take_profit_percentage=tp_pct,
            stop_loss_percentage=sl_pct,
        )

        return signal_ticket, hedge_ticket

    def get_active_pairs_count(self) -> int:
        """Get number of active pairs."""
        return len(self._state.active_pairs)

    def enable(self) -> None:
        """Enable pairs trading."""
        self._state.enabled = True
        self._settings.enabled = True

    def disable(self) -> None:
        """Disable pairs trading."""
        self._state.enabled = False
        self._settings.enabled = False

    def is_enabled(self) -> bool:
        """Check if pairs trading is enabled."""
        return self._state.enabled
