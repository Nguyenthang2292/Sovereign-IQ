from typing import Dict, List, Optional


class DataService:
    def __init__(self):
        self.exchange_manager = None
        self.database_manager = None
        try:
            from modules.auto_trade.exchange_manager import ExchangeManager

            self.exchange_manager = ExchangeManager()
        except Exception as e:
            print(f"Warning: Could not initialize ExchangeManager: {e}")

        try:
            from modules.auto_trade.database.database_manager import DatabaseManager

            self.database_manager = DatabaseManager()
        except Exception as e:
            print(f"Warning: Could not initialize DatabaseManager: {e}")

    def get_account_data(self) -> Optional[Dict]:
        try:
            if self.exchange_manager:
                balance = self.exchange_manager.get_balance()
                positions = self.exchange_manager.get_positions()

                margin_used = sum(float(p.get("position_amt", 0)) * float(p.get("entry_price", 0)) for p in positions)

                unrealized_pnl = sum(float(p.get("un_realized_profit", 0)) for p in positions)

                return {
                    "balance": float(balance.get("total_wallet_balance", 0)) if balance else 0.0,
                    "available": float(balance.get("available_balance", 0)) if balance else 0.0,
                    "margin_used": margin_used,
                    "unrealized_pnl": unrealized_pnl,
                    "daily_pnl": 0.0,
                    "daily_pnl_percent": 0.0,
                }
            return self._get_demo_account_data()
        except Exception as e:
            print(f"Error fetching account data: {e}")
            return self._get_demo_account_data()

    def _get_demo_account_data(self) -> Dict:
        return {
            "balance": 1000.0,
            "available": 1000.0,
            "margin_used": 0.0,
            "unrealized_pnl": 0.0,
            "daily_pnl": 0.0,
            "daily_pnl_percent": 0.0,
        }

    def get_quick_stats(self) -> Optional[Dict]:
        try:
            open_positions = 0
            if self.exchange_manager:
                positions = self.exchange_manager.get_positions()
                open_positions = len([p for p in positions if float(p.get("position_amt", 0)) != 0])

            today_trades = 0
            win_rate = 0.0

            if self.database_manager:
                try:
                    today_trades = (
                        self.database_manager.get_trades_count_today()
                        if hasattr(self.database_manager, "get_trades_count_today")
                        else 0
                    )
                    win_rate = (
                        self.database_manager.calculate_win_rate()
                        if hasattr(self.database_manager, "calculate_win_rate")
                        else 0.0
                    )
                except:
                    pass

            return {
                "open_positions": open_positions,
                "today_trades": today_trades,
                "win_rate": win_rate,
                "mode": "DEMO",
            }
        except Exception as e:
            print(f"Error fetching stats: {e}")
            return {"open_positions": 0, "today_trades": 0, "win_rate": 0.0, "mode": "DEMO"}

    def get_signals(self, min_score: float = 0.7, signal_types: Optional[List[str]] = None) -> List[Dict]:
        try:
            if self.database_manager and hasattr(self.database_manager, "query_recent_signals"):
                signals = self.database_manager.query_recent_signals(limit=100)

                filtered = []
                for s in signals:
                    score = float(s.get("score", 0))
                    if score >= min_score:
                        signal_type = s.get("signal", "").upper()
                        if signal_types is None or signal_type in signal_types:
                            filtered.append(
                                {
                                    "symbol": s.get("symbol", ""),
                                    "signal": signal_type,
                                    "score": score,
                                    "time": s.get("created_at", ""),
                                }
                            )

                return filtered
            return self._get_demo_signals()
        except Exception as e:
            print(f"Error fetching signals: {e}")
            return self._get_demo_signals()

    def _get_demo_signals(self) -> List[Dict]:
        return [
            {"symbol": "BTCUSDT", "signal": "LONG", "score": 0.85, "time": "2024-01-15 10:30"},
            {"symbol": "ETHUSDT", "signal": "SHORT", "score": 0.72, "time": "2024-01-15 10:25"},
            {"symbol": "SOLUSDT", "signal": "NEUTRAL", "score": 0.45, "time": "2024-01-15 10:20"},
        ]

    def get_positions(self) -> List[Dict]:
        try:
            if self.exchange_manager:
                positions = self.exchange_manager.get_positions()

                filtered_positions = []
                for pos in positions:
                    size = float(pos.get("position_amt", 0))
                    if size == 0:
                        continue

                    side = "LONG" if size > 0 else "SHORT"
                    entry_price = float(pos.get("entry_price", 0))

                    current_price = entry_price
                    try:
                        if self.exchange_manager:
                            ticker = self.exchange_manager.get_ticker(pos.get("symbol", ""))
                            if ticker:
                                current_price = float(ticker.get("last_price", entry_price))
                    except:
                        pass

                    pnl = 0.0
                    unrealized_profit = float(pos.get("un_realized_profit", 0))
                    if unrealized_profit:
                        pnl = unrealized_profit
                    elif side == "LONG":
                        pnl = (current_price - entry_price) * abs(size)
                    else:
                        pnl = (entry_price - current_price) * abs(size)

                    filtered_positions.append(
                        {
                            "symbol": pos.get("symbol", ""),
                            "side": side,
                            "size": abs(size),
                            "entry_price": entry_price,
                            "current_price": current_price,
                            "pnl": pnl,
                        }
                    )

                return filtered_positions
            return []
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []
