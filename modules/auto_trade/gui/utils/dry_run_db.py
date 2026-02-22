"""
Dry Run Database Module

Simple JSON-backed database for managing dry-run (simulated) positions.
Used when the application is in DRY_RUN mode to simulate trading.
Replaces the legacy SQLite implementation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DryRunDB:
    """JSON-based database for managing dry-run positions."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize DryRunDB.

        Args:
            db_path: Path to the JSON file. Defaults to dry_run_positions.json
        """
        if db_path is None:
            # Default to data directory relative to project root
            base_dir = Path(__file__).parent.parent.parent
            self.db_path = base_dir / "data" / "dry_run_positions.json"
        else:
            self.db_path = Path(db_path)

        self._ensure_db_directory()
        self._create_tables()

    def _ensure_db_directory(self) -> None:
        """Ensure the directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_tables(self) -> None:
        """Create JSON file if it doesn't exist."""
        if not self.db_path.exists():
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump({"positions": []}, f)

    def _load_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load data from JSON."""
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"positions": []}

    def _save_data(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Save data to JSON."""
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

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
    ) -> bool:
        """Insert a new position."""
        data = self._load_data()

        # Calculate unrealized PNL
        if side == "LONG":
            unrealized_pnl = (current_price - entry_price) * size
        else:
            unrealized_pnl = (entry_price - current_price) * size

        new_id = 1
        if data["positions"]:
            new_id = max(p.get("id", 0) for p in data["positions"]) + 1

        position = {
            "id": new_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "current_price": current_price,
            "size": size,
            "leverage": leverage,
            "unrealized_pnl": unrealized_pnl,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "status": "OPEN",
            "close_price": None,
            "realized_pnl": None,
        }

        data["positions"].append(position)
        self._save_data(data)
        return True

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        data = self._load_data()
        return [p for p in data["positions"] if p.get("status") == "OPEN"]

    def get_open_positions_by_symbol(self, symbol: str, side: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open positions for a specific symbol."""
        open_positions = self.get_open_positions()
        filtered = [p for p in open_positions if p.get("symbol") == symbol]
        if side:
            filtered = [p for p in filtered if p.get("side") == side]
        return filtered

    def update_position(
        self,
        position_id: int,
        current_price: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        status: Optional[str] = None,
    ) -> bool:
        """Update an existing position."""
        data = self._load_data()
        updated = False

        for p in data["positions"]:
            if p.get("id") == position_id:
                if current_price is not None:
                    p["current_price"] = current_price
                if unrealized_pnl is not None:
                    p["unrealized_pnl"] = unrealized_pnl
                if take_profit is not None:
                    p["take_profit"] = take_profit
                if stop_loss is not None:
                    p["stop_loss"] = stop_loss
                if status is not None:
                    p["status"] = status
                updated = True
                break

        if updated:
            self._save_data(data)

        return updated

    def close_position(self, position_id: int, close_price: float, realized_pnl: float) -> bool:
        """Close a position."""
        data = self._load_data()
        updated = False

        for p in data["positions"]:
            if p.get("id") == position_id and p.get("status") == "OPEN":
                p["status"] = "CLOSED"
                p["close_price"] = close_price
                p["realized_pnl"] = realized_pnl
                updated = True
                break

        if updated:
            self._save_data(data)

        return updated

    def get_position_by_id(self, position_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific position by ID."""
        data = self._load_data()
        for p in data["positions"]:
            if p.get("id") == position_id:
                return p
        return None

    def clear_all_positions(self) -> bool:
        """Clear all positions."""
        self._save_data({"positions": []})
        return True

    def get_closed_positions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get closed positions."""
        data = self._load_data()
        closed = [p for p in data["positions"] if p.get("status") == "CLOSED"]
        # Sort by id descending
        closed.sort(key=lambda x: x.get("id", 0), reverse=True)
        return closed[:limit]

    def get_total_pnl(self) -> float:
        """Get total realized PnL from all closed positions."""
        closed = self.get_closed_positions(limit=999999)
        return sum(p.get("realized_pnl", 0.0) for p in closed if p.get("realized_pnl") is not None)
