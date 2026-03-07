"""Dry-run position storage backed by SQLite."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DryRunDB:
    """SQLite database wrapper for simulated positions."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        if db_path is None:
            base_dir = Path(__file__).parent.parent.parent
            self.db_path = base_dir / "data" / "dry_run_positions.db"
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        return dict(row)

    def _create_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dry_run_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    size REAL NOT NULL,
                    leverage INTEGER NOT NULL,
                    unrealized_pnl REAL NOT NULL DEFAULT 0,
                    take_profit REAL,
                    stop_loss REAL,
                    break_even REAL,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    close_price REAL,
                    realized_pnl REAL,
                    close_time TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

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
        if side.upper() == "LONG":
            unrealized_pnl = (current_price - entry_price) * size
        else:
            unrealized_pnl = (entry_price - current_price) * size

        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO dry_run_positions (
                    symbol, side, entry_price, current_price,
                    size, leverage, unrealized_pnl,
                    take_profit, stop_loss, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
                """,
                (
                    symbol,
                    side,
                    entry_price,
                    current_price,
                    size,
                    leverage,
                    unrealized_pnl,
                    take_profit,
                    stop_loss,
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dry_run_positions WHERE status = 'OPEN' ORDER BY id ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_open_positions_by_symbol(self, symbol: str, side: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM dry_run_positions WHERE status = 'OPEN' AND symbol = ?"
        params: List[Any] = [symbol]
        if side:
            query += " AND side = ?"
            params.append(side)
        query += " ORDER BY id ASC"

        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def update_position(
        self,
        position_id: int,
        current_price: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        status: Optional[str] = None,
    ) -> bool:
        updates: List[str] = []
        params: List[Any] = []

        if current_price is not None:
            updates.append("current_price = ?")
            params.append(current_price)
        if unrealized_pnl is not None:
            updates.append("unrealized_pnl = ?")
            params.append(unrealized_pnl)
        if take_profit is not None:
            updates.append("take_profit = ?")
            params.append(take_profit)
        if stop_loss is not None:
            updates.append("stop_loss = ?")
            params.append(stop_loss)
        if status is not None:
            updates.append("status = ?")
            params.append(status)

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(position_id)

        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE dry_run_positions SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )
            return cursor.rowcount > 0

    def close_position(self, position_id: int, close_price: float, realized_pnl: float) -> bool:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE dry_run_positions
                SET
                    status = 'CLOSED',
                    close_price = ?,
                    current_price = ?,
                    realized_pnl = ?,
                    close_time = ?,
                    updated_at = ?
                WHERE id = ? AND status = 'OPEN'
                """,
                (close_price, close_price, realized_pnl, now, now, position_id),
            )
            return cursor.rowcount > 0

    def get_position_by_id(self, position_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM dry_run_positions WHERE id = ?", (position_id,)).fetchone()
        return self._row_to_dict(row)

    def clear_all_positions(self) -> bool:
        with self._connect() as conn:
            conn.execute("DELETE FROM dry_run_positions")
        return True

    def get_closed_positions(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dry_run_positions WHERE status = 'CLOSED' ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_total_pnl(self) -> float:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0.0) AS total_pnl FROM dry_run_positions WHERE status = 'CLOSED'"
            ).fetchone()
        if row is None:
            return 0.0
        value = row["total_pnl"]
        return float(value) if value is not None else 0.0
