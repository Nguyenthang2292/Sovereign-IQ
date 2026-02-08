import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DryRunDB:
    """
    SQLite database for managing dry-run (simulated) positions.

    Used when the application is in DRY_RUN mode to simulate trading
    without actually executing trades on an exchange.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """
        Initialize DryRunDB.

        Args:
            db_path: Path to the SQLite database file. If None, uses default
                     location at modules/auto_trade/data/dry_run_positions.db
        """
        if db_path is None:
            # Default path: modules/auto_trade/data/dry_run_positions.db
            self.db_path = Path(__file__).parent.parent.parent / "data" / "dry_run_positions.db"
        else:
            # Convert string to Path if needed
            self.db_path = Path(db_path) if isinstance(db_path, str) else db_path

        self._ensure_db_directory()
        self._create_tables()

    def _ensure_db_directory(self) -> None:
        """Ensure the database directory exists."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Warning: Could not create database directory: {e}")

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        try:
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
        except sqlite3.Error as e:
            print(f"Error creating database tables: {e}")

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
    ) -> Optional[int]:
        """
        Insert a new position into the database.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            side: Position side ("LONG" or "SHORT")
            entry_price: Entry price
            current_price: Current market price
            size: Position size
            leverage: Leverage used
            take_profit: Take profit price (optional)
            stop_loss: Stop loss price (optional)

        Returns:
            Position ID if successful, None if failed
        """
        try:
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
        except sqlite3.Error as e:
            print(f"Error inserting position: {e}")
            return None

    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions.

        Returns:
            List of position dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM dry_run_positions WHERE status = 'OPEN'
                """)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"Error fetching open positions: {e}")
            return []

    def get_open_positions_by_symbol(self, symbol: str, side: Optional[str] = None) -> List[Dict]:
        """
        Get open positions for a specific symbol.

        Args:
            symbol: Trading symbol to filter by
            side: Optional side filter ("LONG" or "SHORT")

        Returns:
            List of position dictionaries
        """
        try:
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
        except sqlite3.Error as e:
            print(f"Error fetching positions by symbol: {e}")
            return []

    def update_position(
        self,
        position_id: int,
        current_price: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        status: Optional[str] = None,
    ) -> bool:
        """
        Update an existing position.

        Args:
            position_id: ID of the position to update
            current_price: New current price (optional)
            unrealized_pnl: New unrealized PnL (optional)
            take_profit: New take profit price (optional)
            stop_loss: New stop loss price (optional)
            status: New status (optional, e.g., "CLOSED")

        Returns:
            True if update was successful, False otherwise
        """
        updates: List[str] = []
        values: List[Any] = []

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
            updates.append("status = ?")
            values.append(status)
            if status == "CLOSED":
                updates.append("close_time = CURRENT_TIMESTAMP")

        if not updates:
            return False

        try:
            values.append(position_id)
            query = f"UPDATE dry_run_positions SET {', '.join(updates)} WHERE id = ?"

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Error updating position {position_id}: {e}")
            return False

    def close_position(self, position_id: int, close_price: float, realized_pnl: float) -> bool:
        """
        Close a position.

        Args:
            position_id: ID of the position to close
            close_price: Price at which the position was closed
            realized_pnl: Realized profit/loss

        Returns:
            True if successful, False otherwise
        """
        return self.update_position(
            position_id=position_id,
            current_price=close_price,
            unrealized_pnl=realized_pnl,
            status="CLOSED"
        )

    def get_position_by_id(self, position_id: int) -> Optional[Dict]:
        """
        Get a specific position by ID.

        Args:
            position_id: ID of the position to fetch

        Returns:
            Position dictionary if found, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM dry_run_positions WHERE id = ?",
                    (position_id,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            print(f"Error fetching position {position_id}: {e}")
            return None

    def clear_all_positions(self) -> bool:
        """
        Clear all positions from the database.

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM dry_run_positions")
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error clearing positions: {e}")
            return False

    def get_closed_positions(self, limit: int = 100) -> List[Dict]:
        """
        Get closed positions.

        Args:
            limit: Maximum number of positions to return

        Returns:
            List of closed position dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM dry_run_positions
                    WHERE status = 'CLOSED'
                    ORDER BY close_time DESC
                    LIMIT ?
                    """,
                    (limit,)
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"Error fetching closed positions: {e}")
            return []

    def get_total_pnl(self) -> float:
        """
        Get total realized PnL from all closed positions.

        Returns:
            Total PnL value
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COALESCE(SUM(unrealized_pnl), 0.0)
                    FROM dry_run_positions
                    WHERE status = 'CLOSED'
                    """
                )
                result = cursor.fetchone()
                return float(result[0]) if result else 0.0
        except sqlite3.Error as e:
            print(f"Error calculating total PnL: {e}")
            return 0.0
