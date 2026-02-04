import sqlite3
from typing import List, Dict, Optional
from pathlib import Path


class DryRunDB:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "dry_run_positions.db"

        self.db_path = db_path
        self._ensure_db_directory()
        self._create_tables()

    def _ensure_db_directory(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dry_run_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    size REAL NOT NULL,
                    leverage INTEGER NOT NULL,
                    take_profit REAL,
                    stop_loss REAL,
                    unrealized_pnl REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'OPEN',
                    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    close_time TIMESTAMP
                )
            """)
            conn.commit()

    def insert_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        size: float,
        leverage: int,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO dry_run_positions 
                (symbol, side, entry_price, current_price, size, leverage, take_profit, stop_loss, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """,
                (symbol, side, entry_price, current_price, size, leverage, take_profit, stop_loss),
            )
            conn.commit()
            return cursor.lastrowid

    def get_open_positions(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM dry_run_positions WHERE status = 'OPEN'
            """)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_open_positions_by_symbol(self, symbol: str, side: Optional[str] = None) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if side:
                cursor.execute(
                    """
                    SELECT * FROM dry_run_positions 
                    WHERE symbol = ? AND side = ? AND status = 'OPEN'
                """,
                    (symbol, side),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM dry_run_positions 
                    WHERE symbol = ? AND status = 'OPEN'
                """,
                    (symbol,),
                )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def update_position(
        self,
        position_id: int,
        current_price: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        status: Optional[str] = None,
    ):
        updates = []
        values = []

        if current_price is not None:
            updates.append("current_price = ?")
            values.append(current_price)
        if unrealized_pnl is not None:
            updates.append("unrealized_pnl = ?")
            values.append(unrealized_pnl)
        if take_profit is not None:
            updates.append("take_profit = ?")
            values.append(take_profit)
        if stop_loss is not None:
            updates.append("stop_loss = ?")
            values.append(stop_loss)
        if status is not None:
            if status == "CLOSED":
                updates.append("status = ?")
                values.append(status)
                updates.append("close_time = CURRENT_TIMESTAMP")
            else:
                updates.append("status = ?")
                values.append(status)

        if updates:
            values.append(position_id)
            query = f"UPDATE dry_run_positions SET {', '.join(updates)} WHERE id = ?"

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, values)
                conn.commit()

    def close_position(self, position_id: int, close_price: float, realized_pnl: float):
        self.update_position(
            position_id=position_id, current_price=close_price, unrealized_pnl=realized_pnl, status="CLOSED"
        )

    def clear_all_positions(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dry_run_positions")
            conn.commit()
