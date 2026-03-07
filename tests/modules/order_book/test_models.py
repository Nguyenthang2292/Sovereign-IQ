import pytest

from modules.order_book.models import (
    AggTrade,
    CombinedResult,
    OBIDecision,
    OrderBookSnapshot,
)


class TestOrderBookSnapshot:
    def test_creation(self):
        bids = [(50000.0, 1.5), (49999.0, 2.0)]
        asks = [(50001.0, 1.0), (50002.0, 2.5)]
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=1234567890.0,
        )
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.bids == bids
        assert snapshot.asks == asks
        assert snapshot.timestamp == 1234567890.0

    def test_empty_book(self):
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[],
            asks=[],
            timestamp=1234567890.0,
        )
        assert snapshot.bids == []
        assert snapshot.asks == []


class TestAggTrade:
    def test_creation(self):
        trade = AggTrade(
            price=50000.0,
            quantity=0.5,
            timestamp=1234567890.0,
            is_buyer_maker=True,
        )
        assert trade.price == 50000.0
        assert trade.quantity == 0.5
        assert trade.timestamp == 1234567890.0
        assert trade.is_buyer_maker is True

    def test_buyer_not_maker(self):
        trade = AggTrade(
            price=50000.0,
            quantity=0.5,
            timestamp=1234567890.0,
            is_buyer_maker=False,
        )
        assert trade.is_buyer_maker is False


class TestCombinedResult:
    def test_creation(self):
        result = CombinedResult(
            obi_score=0.5,
            delta_score=0.8,
            combined_score=0.68,
            obi_raw=1000.0,
            delta_raw=500.0,
            weighted_mid=50000.5,
        )
        assert result.obi_score == 0.5
        assert result.delta_score == 0.8
        assert result.combined_score == 0.68
        assert result.obi_raw == 1000.0
        assert result.delta_raw == 500.0
        assert result.weighted_mid == 50000.5


class TestOBIDecision:
    def test_enum_values(self):
        assert OBIDecision.PASS == "PASS"
        assert OBIDecision.SKIP == "SKIP"

    def test_enum_members(self):
        assert len(OBIDecision) == 2
        assert "PASS" in OBIDecision.__members__
        assert "SKIP" in OBIDecision.__members__

    def test_string_inheritance(self):
        assert isinstance(OBIDecision.PASS, str)
        assert isinstance(OBIDecision.SKIP, str)
        assert OBIDecision.PASS + "something" == "PASSsomething"
