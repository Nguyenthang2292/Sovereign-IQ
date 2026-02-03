from typing import Dict, Optional


class RiskCalculator:
    """
    Calculate trade risk metrics:
    - Contract size
    - Margin required
    - Potential profit
    - Potential loss
    - Liquidation price
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
    ) -> Dict:
        """
        Calculate all risk metrics

        Returns:
            {
                'contract_size': float,  # BTC amount
                'margin_required': float,  # USDT
                'max_profit': float,  # USDT
                'max_loss': float,  # USDT
                'tp_price': float,
                'sl_price': float,
                'liquidation_price': float,
                'risk_reward_ratio': float
            }
        """
        try:
            # Contract size (in base asset)
            contract_size = amount_usdt / current_price

            # Margin required (with leverage)
            margin_required = amount_usdt / leverage

            # TP/SL prices
            if side == "LONG":
                tp_price = current_price * (1 + tp_percent / 100)
                sl_price = current_price * (1 - sl_percent / 100)

                # Liquidation (simplified)
                # Real formula more complex, includes fees
                liquidation_price = current_price * (1 - (1 / leverage))
            else:  # SHORT
                tp_price = current_price * (1 - tp_percent / 100)
                sl_price = current_price * (1 + sl_percent / 100)
                liquidation_price = current_price * (1 + (1 / leverage))

            # Profit/Loss calculations (with leverage)
            max_profit = amount_usdt * (tp_percent / 100) * leverage
            max_loss = amount_usdt * (sl_percent / 100) * leverage

            # Risk/Reward ratio
            risk_reward_ratio = max_profit / max_loss if max_loss > 0 else 0

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
            print(f"Error calculating risk: {e}")
            return None
