import math

import pytest

from modules.order_book.models import AggTrade, OrderBookSnapshot
from modules.order_book.order_book_imbalance_calculator import (
    _calculate_aggregated_obi,
    _calculate_cumulative_delta_normalized,
    calculate_combined_score,
)


class TestCalculateCombinedScore:
    def test_obi_all_bids_positive(self):
        bids = [(50000.0, 1000.0), (49999.0, 1000.0)]
        asks = [(50001.0, 1.0), (50002.0, 1.0)]
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=1234567890.0,
        )
        result = calculate_combined_score(snapshot, [])
        assert result.obi_score == pytest.approx(1.0, abs=0.01)

    def test_obi_all_asks_negative(self):
        bids = [(50000.0, 1.0), (49999.0, 1.0)]
        asks = [(50001.0, 1000.0), (50002.0, 1000.0)]
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=1234567890.0,
        )
        result = calculate_combined_score(snapshot, [])
        assert result.obi_score == pytest.approx(-1.0, abs=0.01)

    def test_obi_balanced(self):
        bids = [(50000.0, 50.0), (49999.0, 50.0)]
        asks = [(50001.0, 50.0), (50002.0, 50.0)]
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=1234567890.0,
        )
        result = calculate_combined_score(snapshot, [])
        assert result.obi_score == pytest.approx(0.0, abs=0.01)

    def test_cumulative_delta_all_buy_aggressive(self):
        mid_price = 50000.0
        trades = [
            AggTrade(price=50000.0, quantity=10.0, timestamp=1.0, is_buyer_maker=False),
            AggTrade(price=50001.0, quantity=5.0, timestamp=2.0, is_buyer_maker=False),
            AggTrade(price=49999.0, quantity=8.0, timestamp=3.0, is_buyer_maker=False),
        ]
        delta_raw, delta_score, v_avg = _calculate_cumulative_delta_normalized(trades, mid_price)
        assert delta_raw == pytest.approx(23.0)
        assert delta_score == pytest.approx(math.tanh(23.0 / 7.666), abs=0.01)
        assert v_avg == pytest.approx(7.666, abs=0.01)

    def test_cumulative_delta_all_sell_aggressive(self):
        mid_price = 50000.0
        trades = [
            AggTrade(price=50000.0, quantity=10.0, timestamp=1.0, is_buyer_maker=True),
            AggTrade(price=50001.0, quantity=5.0, timestamp=2.0, is_buyer_maker=True),
            AggTrade(price=49999.0, quantity=8.0, timestamp=3.0, is_buyer_maker=True),
        ]
        delta_raw, delta_score, v_avg = _calculate_cumulative_delta_normalized(trades, mid_price)
        assert delta_raw == pytest.approx(-23.0)
        assert delta_score == pytest.approx(math.tanh(-23.0 / 7.666), abs=0.01)

    def test_cumulative_delta_balanced(self):
        mid_price = 50000.0
        trades = [
            AggTrade(price=50000.0, quantity=10.0, timestamp=1.0, is_buyer_maker=False),
            AggTrade(price=50001.0, quantity=10.0, timestamp=2.0, is_buyer_maker=True),
        ]
        delta_raw, delta_score, v_avg = _calculate_cumulative_delta_normalized(trades, mid_price)
        assert delta_raw == pytest.approx(0.0, abs=0.01)
        assert delta_score == pytest.approx(0.0, abs=0.01)

    def test_combined_score_40_60_ratio(self):
        bids = [(50000.0, 100.0)]
        asks = [(50001.0, 100.0)]
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=1234567890.0,
        )
        trades = [
            AggTrade(price=50000.0, quantity=100.0, timestamp=1.0, is_buyer_maker=False),
        ]
        result = calculate_combined_score(snapshot, trades)
        expected = 0.4 * result.obi_score + 0.6 * result.delta_score
        assert result.combined_score == pytest.approx(expected, abs=0.001)

    def test_combined_score_calculation_example(self):
        bids = [(50000.0, 10.0), (49990.0, 10.0)]
        asks = [(50010.0, 10.0), (50020.0, 10.0)]
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=1234567890.0,
        )
        trades = [
            AggTrade(price=50000.0, quantity=10.0, timestamp=1.0, is_buyer_maker=False),
            AggTrade(price=50000.0, quantity=5.0, timestamp=2.0, is_buyer_maker=False),
            AggTrade(price=50000.0, quantity=2.0, timestamp=3.0, is_buyer_maker=False),
        ]
        result = calculate_combined_score(snapshot, trades)
        obi = result.obi_score
        delta = result.delta_score
        expected_combined = 0.4 * obi + 0.6 * delta
        assert result.combined_score == pytest.approx(expected_combined, abs=0.001)

    def test_empty_orderbook_returns_zeros(self):
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[],
            asks=[],
            timestamp=1234567890.0,
        )
        result = calculate_combined_score(snapshot, [])
        assert result.obi_score == 0.0
        assert result.delta_score == 0.0
        assert result.combined_score == 0.0

    def test_empty_bids_returns_zeros(self):
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[],
            asks=[(50001.0, 10.0)],
            timestamp=1234567890.0,
        )
        result = calculate_combined_score(snapshot, [])
        assert result.obi_score == 0.0
        assert result.delta_score == 0.0

    def test_empty_asks_returns_zeros(self):
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[(50000.0, 10.0)],
            asks=[],
            timestamp=1234567890.0,
        )
        result = calculate_combined_score(snapshot, [])
        assert result.obi_score == 0.0
        assert result.delta_score == 0.0


class TestAggregatedOBI:
    def test_calculate_aggregated_obi_bid_heavy(self):
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[(50000.0, 100.0), (49990.0, 50.0)],
            asks=[(50010.0, 5.0), (50020.0, 5.0)],
            timestamp=1234567890.0,
        )
        mid_price = 50005.0
        obi_raw, obi_score = _calculate_aggregated_obi(snapshot, mid_price, 0.001)
        assert obi_score > 0.8

    def test_calculate_aggregated_obi_ask_heavy(self):
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[(50000.0, 5.0), (49990.0, 5.0)],
            asks=[(50010.0, 100.0), (50020.0, 50.0)],
            timestamp=1234567890.0,
        )
        mid_price = 50005.0
        obi_raw, obi_score = _calculate_aggregated_obi(snapshot, mid_price, 0.001)
        assert obi_score < -0.8


class TestCumulativeDelta:
    def test_delta_normalization_tanh(self):
        trades = [
            AggTrade(price=50000.0, quantity=100.0, timestamp=1.0, is_buyer_maker=False),
        ]
        mid_price = 50000.0
        delta_raw, delta_score, _ = _calculate_cumulative_delta_normalized(trades, mid_price)
        assert delta_raw == pytest.approx(100.0)
        assert delta_score == pytest.approx(math.tanh(100.0 / 100.0), abs=0.01)

    def test_delta_empty_trades(self):
        trades: list[AggTrade] = []
        mid_price = 50000.0
        delta_raw, delta_score, v_avg = _calculate_cumulative_delta_normalized(trades, mid_price)
        assert delta_raw == 0.0
        assert delta_score == 0.0
        assert v_avg == 0.0
