"""
Risk Calculator Module

Calculates trade risk metrics including position sizing, margin requirements,
profit/loss potential, and liquidation prices.
"""

from typing import Dict, Optional

from modules.common.ui.logging import log_error


class RiskCalculator:
    """
    Calculate comprehensive trade risk metrics.

    Provides calculations for:
    - Contract size and margin requirements
    - Maximum profit and loss potential
    - Take profit and stop loss prices
    - Liquidation price estimation
    - Risk/reward ratios
    """

    @staticmethod
    def calculate(
        symbol: str,
        side: str,
        amount_usdt: float,
        leverage: int,
        current_price: float,
        tp_percent: float,
        sl_percent: float,
    ) -> Optional[Dict[str, float]]:
        """
        Calculate all risk metrics for a trade.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            side: Trade side ("LONG" or "SHORT")
            amount_usdt: Position size in USDT
            leverage: Leverage multiplier
            current_price: Current market price
            tp_percent: Take profit ROI% on capital (e.g. 5.0 = 5% ROI)
            sl_percent: Stop loss ROI% on capital (e.g. 2.5 = 2.5% ROI)

        Returns:
            Dictionary containing:
            - contract_size: Size in base asset
            - margin_required: Required margin in USDT
            - max_profit: Maximum profit potential in USDT
            - max_loss: Maximum loss potential in USDT
            - tp_price: Take profit price
            - sl_price: Stop loss price
            - liquidation_price: Estimated liquidation price
            - risk_reward_ratio: Risk/reward ratio

            Returns None if calculation fails
        """
        try:
            if current_price <= 0 or leverage <= 0:
                return None

            # Contract size (in base asset)
            contract_size: float = amount_usdt / current_price

            # Margin required (with leverage)
            margin_required: float = amount_usdt / leverage

            # tp/sl_percent are ROI% on capital → convert to price-move%
            tp_price_pct: float = tp_percent / max(leverage, 1)
            sl_price_pct: float = sl_percent / max(leverage, 1)

            # TP/SL prices
            tp_price: float
            sl_price: float
            liquidation_price: float
            if side == "LONG":
                tp_price = current_price * (1 + tp_price_pct / 100)
                sl_price = current_price * (1 - sl_price_pct / 100)
                # Liquidation (simplified)
                liquidation_price = current_price * (1 - (1 / leverage))
            else:  # SHORT
                tp_price = current_price * (1 - tp_price_pct / 100)
                sl_price = current_price * (1 + sl_price_pct / 100)
                liquidation_price = current_price * (1 + (1 / leverage))

            # Profit/Loss calculations (with leverage)
            max_profit: float = amount_usdt * (tp_percent / 100) * leverage
            max_loss: float = amount_usdt * (sl_percent / 100) * leverage

            # Risk/Reward ratio
            risk_reward_ratio: float = max_profit / max_loss if max_loss > 0 else 0.0

            return {
                "contract_size": contract_size,
                "margin_required": margin_required,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "liquidation_price": liquidation_price,
                "risk_reward_ratio": risk_reward_ratio,
            }
        except Exception as e:
            log_error("Error calculating risk: %s", e)
            return None
