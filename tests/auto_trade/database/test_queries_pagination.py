"""
Tests for Database Query Pagination and Cursor-based Fetching
==============================================================

Tests pagination functionality for orders, signals, and audit logs.

Run: pytest tests/auto_trade/database/test_queries_pagination.py -v
"""

import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import modules.auto_trade.database as db_module
from modules.auto_trade.database import (
    create_order,
    get_all_programmatic_orders,
    get_db_manager,
    get_orders_cursor,
    initialize_database,
    session_scope,
)


@pytest.mark.usefixtures("test_db")
class TestGetAllProgrammaticOrdersPagination:
    """Test pagination with offset and limit."""

    def test_pagination_first_page(self, test_db, sample_order_data):
        """Test retrieving first page of orders."""
        uid = uuid.uuid4().hex[:8]
        with session_scope() as session:
            for i in range(25):
                data = sample_order_data(
                    order_id=f"ORDER_P1_{uid}_{i:03d}",
                    client_order_id=f"AT_P1_{uid}_{i:03d}_BTCUSDT_abc123",
                    pnl=float(i * 10)
                )
                create_order(session, data)

        with session_scope() as session:
            # First page - 10 orders
            page1 = get_all_programmatic_orders(session, limit=10, offset=0)

            assert len(page1) == 10, f"Expected 10 orders on first page, got {len(page1)}"

    def test_pagination_second_page(self, test_db, sample_order_data):
        """Test retrieving second page of orders."""
        uid = uuid.uuid4().hex[:8]
        with session_scope() as session:
            for i in range(25):
                data = sample_order_data(
                    order_id=f"ORDER_P2_{uid}_{i:03d}",
                    client_order_id=f"AT_P2_{uid}_{i:03d}_BTCUSDT_abc123",
                    pnl=float(i * 10)
                )
                create_order(session, data)

        with session_scope() as session:
            # Second page - next 10 orders
            page2 = get_all_programmatic_orders(session, limit=10, offset=10)

            assert len(page2) == 10, f"Expected 10 orders on second page, got {len(page2)}"

    def test_pagination_no_overlap_between_pages(self, test_db, sample_order_data):
        """Test that pages don't overlap - each order appears only once."""
        uid = uuid.uuid4().hex[:8]
        with session_scope() as session:
            for i in range(25):
                data = sample_order_data(
                    order_id=f"ORDER_P3_{uid}_{i:03d}",
                    client_order_id=f"AT_P3_{uid}_{i:03d}_BTCUSDT_abc123",
                    pnl=float(i * 10)
                )
                create_order(session, data)

        with session_scope() as session:
            page1 = get_all_programmatic_orders(session, limit=10, offset=0)
            page2 = get_all_programmatic_orders(session, limit=10, offset=10)

            page1_ids = {o.order_id for o in page1}
            page2_ids = {o.order_id for o in page2}

            assert page1_ids.isdisjoint(
                page2_ids
            ), f"Pages should not overlap. Overlapping IDs: {page1_ids & page2_ids}"

    def test_pagination_last_partial_page(self, test_db, sample_order_data):
        """Test retrieving last page with fewer items than limit."""
        uid = uuid.uuid4().hex[:8]
        symbol = f"P4PARTIAL_{uid}"
        with session_scope() as session:
            for i in range(25):
                data = sample_order_data(
                    order_id=f"ORDER_P4_{uid}_{i:03d}",
                    client_order_id=f"AT_P4_{uid}_{i:03d}_{symbol}_abc123",
                    symbol=symbol,
                    pnl=float(i * 10)
                )
                create_order(session, data)

        with session_scope() as session:
            # Last page - 5 remaining (indices 20-24) when limit=10, offset=20
            last_page = get_all_programmatic_orders(
                session, limit=10, offset=20, symbol=symbol
            )

            assert len(last_page) == 5, f"Expected 5 orders on last page, got {len(last_page)}"

    def test_pagination_empty_database(self, tmp_path):
        """Test pagination on empty database returns empty list."""
        # Use a dedicated empty DB so we are not affected by shared singleton/ordering
        db_module._db_manager_instance = None
        empty_db_path = tmp_path / "empty_pagination.db"
        initialize_database(str(empty_db_path))
        get_db_manager(str(empty_db_path))

        with session_scope() as session:
            result = get_all_programmatic_orders(session, limit=10, offset=0)

            assert result == [], f"Expected empty list for empty database, got {result}"

    def test_pagination_offset_beyond_total(self, test_db, sample_order_data):
        """Test pagination with offset beyond total count."""
        uid = uuid.uuid4().hex[:8]
        symbol = f"P5OFFSET_{uid}"
        with session_scope() as session:
            for i in range(5):
                data = sample_order_data(
                    order_id=f"ORDER_P5_{uid}_{i:03d}",
                    client_order_id=f"AT_P5_{uid}_{i:03d}_{symbol}_abc123",
                    symbol=symbol,
                )
                create_order(session, data)

        with session_scope() as session:
            # Offset beyond total
            result = get_all_programmatic_orders(session, limit=10, offset=100, symbol=symbol)

            assert result == [], f"Expected empty list when offset exceeds total, got {len(result)} items"


@pytest.mark.usefixtures("test_db")
class TestGetAllProgrammaticOrdersSymbolFilter:
    """Test symbol filtering in pagination."""

    def test_filter_by_symbol(self, test_db, sample_order_data):
        """Test filtering orders by symbol."""
        uid = uuid.uuid4().hex[:8]
        symbols = [f"BTCUSDT_{uid}", f"ETHUSDT_{uid}", f"SOLUSDT_{uid}"]
        with session_scope() as session:
            for symbol in symbols:
                for i in range(3):
                    data = sample_order_data(
                        order_id=f"SYM_{uid}_{symbol}_{i}",
                        client_order_id=f"AT_SYM_{uid}_{i}_{symbol}_abc123",
                        symbol=symbol
                    )
                    create_order(session, data)

        with session_scope() as session:
            btc_symbol = symbols[0]
            btc_orders = get_all_programmatic_orders(session, symbol=btc_symbol)

            assert len(btc_orders) == 3, f"Expected 3 BTC orders, got {len(btc_orders)}"
            assert all(
                o.symbol == btc_symbol for o in btc_orders
            ), f"All orders should be {btc_symbol}, got symbols: {[o.symbol for o in btc_orders]}"

    def test_filter_by_symbol_and_status(self, test_db, sample_order_data):
        """Test filtering orders by symbol and status."""
        uid = uuid.uuid4().hex[:8]
        symbol = f"BTCUSDT_{uid}"
        with session_scope() as session:
            for i in range(3):
                data = sample_order_data(
                    order_id=f"SYMST_{uid}_BTC_OPEN_{i}",
                    client_order_id=f"AT_SYMST_{uid}_{i}_BTCUSDT_open",
                    symbol=symbol,
                    status="OPEN"
                )
                create_order(session, data)

            for i in range(2):
                data = sample_order_data(
                    order_id=f"SYMST_{uid}_BTC_CLOSED_{i}",
                    client_order_id=f"AT_SYMST_{uid}_{i}_BTCUSDT_closed",
                    symbol=symbol,
                    status="CLOSED",
                    pnl=100.0
                )
                create_order(session, data)

        with session_scope() as session:
            open_btc = get_all_programmatic_orders(session, symbol=symbol, status="OPEN")

            assert len(open_btc) == 3, f"Expected 3 open BTC orders, got {len(open_btc)}"
            assert all(
                o.status == "OPEN" for o in open_btc
            ), f"All orders should be OPEN, got statuses: {[o.status for o in open_btc]}"

    def test_filter_nonexistent_symbol(self, test_db, sample_order_data):
        """Test filtering by symbol that doesn't exist."""
        uid = uuid.uuid4().hex[:8]
        with session_scope() as session:
            for i in range(5):
                data = sample_order_data(
                    order_id=f"ORDER_NEX_{uid}_{i}",
                    client_order_id=f"AT_NEX_{uid}_{i:03d}_BTCUSDT_abc123"
                )
                create_order(session, data)

        with session_scope() as session:
            result = get_all_programmatic_orders(session, symbol="NONEXISTENT")

            assert result == [], f"Expected empty list for nonexistent symbol, got {len(result)} items"


@pytest.mark.usefixtures("test_db")
class TestGetOrdersCursor:
    """Test cursor-based pagination."""

    def test_cursor_pagination_first_page(self, test_db, sample_order_data):
        """Test cursor pagination - first page without cursor."""
        uid = uuid.uuid4().hex[:8]
        with session_scope() as session:
            for i in range(20):
                data = sample_order_data(
                    order_id=f"ORDER_C1_{uid}_{i:03d}",
                    client_order_id=f"AT_C1_{uid}_{i:03d}_BTCUSDT_abc123"
                )
                create_order(session, data)

        with session_scope() as session:
            page1 = get_orders_cursor(session, last_id=None, limit=10)

            assert len(page1) == 10, f"Expected 10 orders on first page, got {len(page1)}"

    def test_cursor_pagination_second_page(self, test_db, sample_order_data):
        """Test cursor pagination - second page using last_id."""
        uid = uuid.uuid4().hex[:8]
        with session_scope() as session:
            for i in range(20):
                data = sample_order_data(
                    order_id=f"ORDER_C2_{uid}_{i:03d}",
                    client_order_id=f"AT_C2_{uid}_{i:03d}_BTCUSDT_abc123"
                )
                create_order(session, data)

        with session_scope() as session:
            page1 = get_orders_cursor(session, last_id=None, limit=10)
            last_id = int(page1[-1].id) if page1[-1].id is not None else None

            page2 = get_orders_cursor(session, last_id=last_id, limit=10)

            assert len(page2) == 10, f"Expected 10 orders on second page, got {len(page2)}"

    def test_cursor_pagination_no_overlap(self, test_db, sample_order_data):
        """Test that cursor pagination doesn't return duplicate orders."""
        uid = uuid.uuid4().hex[:8]
        with session_scope() as session:
            for i in range(20):
                data = sample_order_data(
                    order_id=f"ORDER_C3_{uid}_{i:03d}",
                    client_order_id=f"AT_C3_{uid}_{i:03d}_BTCUSDT_abc123"
                )
                create_order(session, data)

        with session_scope() as session:
            page1 = get_orders_cursor(session, last_id=None, limit=10)
            last_id = int(page1[-1].id) if page1[-1].id is not None else None

            page2 = get_orders_cursor(session, last_id=last_id, limit=10)

            page1_ids = {o.id for o in page1}
            page2_ids = {o.id for o in page2}

            assert page1_ids.isdisjoint(
                page2_ids
            ), f"Cursor pages should not overlap. Overlapping IDs: {page1_ids & page2_ids}"

    def test_cursor_pagination_with_symbol_filter(self, test_db, sample_order_data):
        """Test cursor pagination with symbol filter."""
        uid = uuid.uuid4().hex[:8]
        sym_btc = f"BTCUSDT_{uid}"
        sym_eth = f"ETHUSDT_{uid}"
        with session_scope() as session:
            for i in range(10):
                data = sample_order_data(
                    order_id=f"CSYM_{uid}_BTC_{i:03d}",
                    client_order_id=f"AT_CSYM_{uid}_BTC_{i:03d}_{sym_btc}_abc123",
                    symbol=sym_btc
                )
                create_order(session, data)

            for i in range(10):
                data = sample_order_data(
                    order_id=f"CSYM_{uid}_ETH_{i:03d}",
                    client_order_id=f"AT_CSYM_{uid}_ETH_{i:03d}_{sym_eth}_abc123",
                    symbol=sym_eth
                )
                create_order(session, data)

        with session_scope() as session:
            btc_page1 = get_orders_cursor(session, last_id=None, limit=5, symbol=sym_btc)

            assert len(btc_page1) == 5, f"Expected 5 BTC orders, got {len(btc_page1)}"
            assert all(
                o.symbol == sym_btc for o in btc_page1
            ), f"All orders should be {sym_btc}, got: {[o.symbol for o in btc_page1]}"

    def test_cursor_pagination_with_status_filter(self, test_db, sample_order_data):
        """Test cursor pagination with status filter."""
        uid = uuid.uuid4().hex[:8]
        symbol = f"CSTATUS_{uid}"
        with session_scope() as session:
            for i in range(10):
                data = sample_order_data(
                    order_id=f"CST_{uid}_OPEN_{i:03d}",
                    client_order_id=f"AT_CST_{uid}_OPEN_{i:03d}_{symbol}_abc123",
                    symbol=symbol,
                    status="OPEN"
                )
                create_order(session, data)

            for i in range(10):
                data = sample_order_data(
                    order_id=f"CST_{uid}_CLOSED_{i:03d}",
                    client_order_id=f"AT_CST_{uid}_CLOSED_{i:03d}_{symbol}_abc123",
                    symbol=symbol,
                    status="CLOSED",
                    pnl=100.0
                )
                create_order(session, data)

        with session_scope() as session:
            open_orders = get_orders_cursor(session, last_id=None, limit=5, status="OPEN", symbol=symbol)

            assert len(open_orders) == 5, f"Expected 5 open orders, got {len(open_orders)}"
            assert all(
                o.status == "OPEN" for o in open_orders
            ), f"All orders should be OPEN, got: {[o.status for o in open_orders]}"

    def test_cursor_pagination_empty_database(self, test_db):
        """Test cursor pagination on empty database (no matching orders)."""
        uid = uuid.uuid4().hex[:8]
        with session_scope() as session:
            result = get_orders_cursor(session, last_id=None, limit=50, symbol=f"NOEXIST_{uid}")

            assert result == [], f"Expected empty list for non-existent symbol, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
