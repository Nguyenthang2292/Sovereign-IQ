import pytest
from unittest.mock import Mock, patch

from modules.order_book.models import (
    AggTrade,
    CombinedResult,
    OBIDecision,
    OrderBookSnapshot,
)
from modules.order_book.order_book_imbalance_gate import OrderBookImbalanceGate


def _create_snapshot(bid_qty: float, ask_qty: float) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol="BTCUSDT",
        bids=[(50000.0, bid_qty)],
        asks=[(50001.0, ask_qty)],
        timestamp=1234567890.0,
    )


def _create_trades(buy_qty: float, sell_qty: float) -> list[AggTrade]:
    return [
        AggTrade(price=50000.0, quantity=buy_qty, timestamp=1.0, is_buyer_maker=False),
        AggTrade(price=50001.0, quantity=sell_qty, timestamp=2.0, is_buyer_maker=True),
    ]


def _make_combined_result(score: float) -> CombinedResult:
    return CombinedResult(
        obi_score=score,
        delta_score=score,
        combined_score=score,
        obi_raw=score * 100,
        delta_raw=score * 50,
        weighted_mid=50000.5,
    )


class TestOrderBookImbalanceGate:
    def test_long_signal_positive_score_passes_immediately(self):
        gate = OrderBookImbalanceGate(
            threshold=0.15,
            retry_wait_seconds=0,
            max_retries=2,
            testnet=True,
        )
        snapshot = _create_snapshot(bid_qty=100.0, ask_qty=10.0)
        trades = _create_trades(buy_qty=10.0, sell_qty=1.0)

        with patch("modules.order_book.order_book_imbalance_gate.fetch_depth", return_value=snapshot):
            with patch("modules.order_book.order_book_imbalance_gate.fetch_agg_trades", return_value=trades):
                decision, result = gate.check("BTCUSDT", "LONG")

        assert decision == OBIDecision.PASS
        assert result is not None
        assert result.combined_score > 0

    def test_long_signal_neutral_score_passes(self):
        gate = OrderBookImbalanceGate(
            threshold=0.15,
            retry_wait_seconds=0,
            max_retries=2,
            testnet=True,
        )
        snapshot = _create_snapshot(bid_qty=55.0, ask_qty=45.0)
        trades = _create_trades(buy_qty=10.0, sell_qty=10.0)

        with patch("modules.order_book.order_book_imbalance_gate.fetch_depth", return_value=snapshot):
            with patch("modules.order_book.order_book_imbalance_gate.fetch_agg_trades", return_value=trades):
                decision, result = gate.check("BTCUSDT", "LONG")

        assert decision == OBIDecision.PASS

    def test_long_signal_negative_then_positive_after_retry(self):
        gate = OrderBookImbalanceGate(
            threshold=0.15,
            retry_wait_seconds=0,
            max_retries=2,
            testnet=True,
        )
        negative_snapshot = _create_snapshot(bid_qty=10.0, ask_qty=100.0)
        positive_snapshot = _create_snapshot(bid_qty=100.0, ask_qty=10.0)
        trades = _create_trades(buy_qty=10.0, sell_qty=1.0)

        with patch(
            "modules.order_book.order_book_imbalance_gate.fetch_depth",
            side_effect=[negative_snapshot, positive_snapshot],
        ):
            with patch("modules.order_book.order_book_imbalance_gate.fetch_agg_trades", return_value=trades):
                decision, result = gate.check("BTCUSDT", "LONG")

        assert decision == OBIDecision.PASS
        assert result is not None

    def test_long_signal_negative_all_retries_skips(self):
        gate = OrderBookImbalanceGate(
            threshold=0.15,
            retry_wait_seconds=0,
            max_retries=2,
            testnet=True,
        )
        negative_snapshot = _create_snapshot(bid_qty=10.0, ask_qty=100.0)
        trades = _create_trades(buy_qty=1.0, sell_qty=10.0)

        with patch("modules.order_book.order_book_imbalance_gate.fetch_depth", return_value=negative_snapshot):
            with patch("modules.order_book.order_book_imbalance_gate.fetch_agg_trades", return_value=trades):
                decision, result = gate.check("BTCUSDT", "LONG")

        assert decision == OBIDecision.SKIP

    def test_short_signal_negative_score_passes(self):
        gate = OrderBookImbalanceGate(
            threshold=0.15,
            retry_wait_seconds=0,
            max_retries=2,
            testnet=True,
        )
        snapshot = _create_snapshot(bid_qty=10.0, ask_qty=100.0)
        trades = _create_trades(buy_qty=1.0, sell_qty=10.0)

        with patch("modules.order_book.order_book_imbalance_gate.fetch_depth", return_value=snapshot):
            with patch("modules.order_book.order_book_imbalance_gate.fetch_agg_trades", return_value=trades):
                decision, result = gate.check("BTCUSDT", "SHORT")

        assert decision == OBIDecision.PASS

    def test_rest_api_error_returns_pass(self):
        gate = OrderBookImbalanceGate(
            threshold=0.15,
            retry_wait_seconds=0,
            max_retries=2,
            testnet=True,
        )

        with patch("modules.order_book.order_book_imbalance_gate.fetch_depth", return_value=None):
            with patch("modules.order_book.order_book_imbalance_gate.fetch_agg_trades", return_value=None):
                decision, result = gate.check("BTCUSDT", "LONG")

        assert decision == OBIDecision.PASS

    def test_enabled_false_passes_immediately(self):
        gate = OrderBookImbalanceGate(
            threshold=0.15,
            retry_wait_seconds=0,
            max_retries=2,
            testnet=True,
            enabled=False,
        )

        decision, result = gate.check("BTCUSDT", "LONG")

        assert decision == OBIDecision.PASS
        assert result is None

    def test_invalid_signal_defaults_to_long(self):
        gate = OrderBookImbalanceGate(
            threshold=0.15,
            retry_wait_seconds=0,
            max_retries=2,
            testnet=True,
        )
        snapshot = _create_snapshot(bid_qty=100.0, ask_qty=10.0)
        trades = _create_trades(buy_qty=10.0, sell_qty=1.0)

        with patch("modules.order_book.order_book_imbalance_gate.fetch_depth", return_value=snapshot):
            with patch("modules.order_book.order_book_imbalance_gate.fetch_agg_trades", return_value=trades):
                decision, result = gate.check("BTCUSDT", "INVALID")

        assert decision == OBIDecision.PASS

    def test_calculation_error_returns_pass(self):
        gate = OrderBookImbalanceGate(
            threshold=0.15,
            retry_wait_seconds=0,
            max_retries=2,
            testnet=True,
        )
        snapshot = _create_snapshot(bid_qty=100.0, ask_qty=10.0)
        trades = _create_trades(buy_qty=10.0, sell_qty=1.0)

        with patch("modules.order_book.order_book_imbalance_gate.fetch_depth", return_value=snapshot):
            with patch("modules.order_book.order_book_imbalance_gate.fetch_agg_trades", return_value=trades):
                with patch(
                    "modules.order_book.order_book_imbalance_gate.calculate_combined_score",
                    side_effect=Exception("Calculation error"),
                ):
                    decision, result = gate.check("BTCUSDT", "LONG")

        assert decision == OBIDecision.PASS

    def test_threshold_parameter_respected(self):
        gate = OrderBookImbalanceGate(
            threshold=0.5,
            retry_wait_seconds=0,
            max_retries=2,
            testnet=True,
        )
        snapshot = _create_snapshot(bid_qty=60.0, ask_qty=40.0)
        trades = _create_trades(buy_qty=6.0, sell_qty=4.0)

        with patch("modules.order_book.order_book_imbalance_gate.fetch_depth", return_value=snapshot):
            with patch("modules.order_book.order_book_imbalance_gate.fetch_agg_trades", return_value=trades):
                decision, result = gate.check("BTCUSDT", "LONG")

        assert decision == OBIDecision.PASS
        assert result is not None
        assert result.combined_score < 0.5

    def test_max_retries_configuration(self):
        gate = OrderBookImbalanceGate(
            threshold=0.15,
            retry_wait_seconds=0,
            max_retries=1,
            testnet=True,
        )
        negative_snapshot = _create_snapshot(bid_qty=10.0, ask_qty=100.0)
        trades = _create_trades(buy_qty=1.0, sell_qty=10.0)

        with patch("modules.order_book.order_book_imbalance_gate.fetch_depth", return_value=negative_snapshot):
            with patch("modules.order_book.order_book_imbalance_gate.fetch_agg_trades", return_value=trades):
                decision, result = gate.check("BTCUSDT", "LONG")

        assert decision == OBIDecision.SKIP

    def test_depth_limit_configuration_is_used_for_depth_fetch(self):
        gate = OrderBookImbalanceGate(
            threshold=0.15,
            retry_wait_seconds=0,
            max_retries=0,
            depth_limit=42,
            testnet=True,
        )
        snapshot = _create_snapshot(bid_qty=100.0, ask_qty=10.0)
        trades = _create_trades(buy_qty=10.0, sell_qty=1.0)

        with patch("modules.order_book.order_book_imbalance_gate.fetch_depth", return_value=snapshot) as mock_depth:
            with patch("modules.order_book.order_book_imbalance_gate.fetch_agg_trades", return_value=trades):
                decision, result = gate.check("BTCUSDT", "LONG")

        assert decision == OBIDecision.PASS
        assert result is not None
        mock_depth.assert_called_once_with(symbol="BTCUSDT", limit=42, testnet=True)
