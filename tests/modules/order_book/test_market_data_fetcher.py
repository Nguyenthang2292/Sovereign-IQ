import pytest
from unittest.mock import Mock, patch

from modules.order_book.market_data_fetcher import fetch_depth, fetch_agg_trades
from modules.order_book.models import OrderBookSnapshot, AggTrade


class TestFetchDepth:
    def test_valid_response_parses_correctly(self):
        mock_response = Mock()
        mock_response.json.return_value = {
            "lastUpdateId": 160,
            "bids": [["50000.00", "1.5"], ["49999.00", "2.0"]],
            "asks": [["50001.00", "1.0"], ["50002.00", "2.5"]],
            "E": 1234567890000,
        }
        mock_response.raise_for_status = Mock()

        with patch("modules.order_book.market_data_fetcher.requests.get", return_value=mock_response):
            result = fetch_depth("BTCUSDT", limit=100, testnet=False)

        assert result is not None
        assert isinstance(result, OrderBookSnapshot)
        assert result.symbol == "BTCUSDT"
        assert len(result.bids) == 2
        assert len(result.asks) == 2
        assert result.bids[0] == (50000.0, 1.5)
        assert result.asks[0] == (50001.0, 1.0)

    def test_invalid_payload_returns_none(self):
        mock_response = Mock()
        mock_response.json.return_value = {"invalid": "structure"}
        mock_response.raise_for_status = Mock()

        with patch("modules.order_book.market_data_fetcher.requests.get", return_value=mock_response):
            result = fetch_depth("BTCUSDT")

        assert result is None

    def test_timeout_returns_none(self):
        with patch(
            "modules.order_book.market_data_fetcher.requests.get",
            side_effect=pytest.importorskip("requests").exceptions.Timeout(),
        ):
            result = fetch_depth("BTCUSDT")

        assert result is None

    def test_connection_error_returns_none(self):
        with patch(
            "modules.order_book.market_data_fetcher.requests.get",
            side_effect=pytest.importorskip("requests").exceptions.ConnectionError(),
        ):
            result = fetch_depth("BTCUSDT")

        assert result is None

    def test_http_4xx_error_returns_none(self):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = pytest.importorskip("requests").exceptions.HTTPError(
            "400 Bad Request"
        )

        with patch("modules.order_book.market_data_fetcher.requests.get", return_value=mock_response):
            result = fetch_depth("BTCUSDT")

        assert result is None

    def test_http_5xx_error_returns_none(self):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = pytest.importorskip("requests").exceptions.HTTPError(
            "500 Server Error"
        )

        with patch("modules.order_book.market_data_fetcher.requests.get", return_value=mock_response):
            result = fetch_depth("BTCUSDT")

        assert result is None


class TestFetchAggTrades:
    def test_valid_response_parses_correctly(self):
        mock_response = Mock()
        mock_response.json.return_value = [
            {"p": "50000.0", "q": "1.5", "T": 1234567890000, "m": False},
            {"p": "50001.0", "q": "2.0", "T": 1234567891000, "m": True},
            {"p": "50002.0", "q": "0.5", "T": 1234567892000, "m": False},
        ]
        mock_response.raise_for_status = Mock()

        with patch("modules.order_book.market_data_fetcher.requests.get", return_value=mock_response):
            result = fetch_agg_trades("BTCUSDT", window_minutes=5, testnet=False)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(t, AggTrade) for t in result)
        assert result[0].price == 50000.0
        assert result[0].quantity == 1.5
        assert result[0].is_buyer_maker is False
        assert result[1].is_buyer_maker is True

    def test_non_list_response_returns_none(self):
        mock_response = Mock()
        mock_response.json.return_value = {"error": "not a list"}
        mock_response.raise_for_status = Mock()

        with patch("modules.order_book.market_data_fetcher.requests.get", return_value=mock_response):
            result = fetch_agg_trades("BTCUSDT")

        assert result is None

    def test_empty_list_returns_empty_list(self):
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = Mock()

        with patch("modules.order_book.market_data_fetcher.requests.get", return_value=mock_response):
            result = fetch_agg_trades("BTCUSDT")

        assert result == []

    def test_timeout_returns_none(self):
        with patch(
            "modules.order_book.market_data_fetcher.requests.get",
            side_effect=pytest.importorskip("requests").exceptions.Timeout(),
        ):
            result = fetch_agg_trades("BTCUSDT")

        assert result is None

    def test_connection_error_returns_none(self):
        with patch(
            "modules.order_book.market_data_fetcher.requests.get",
            side_effect=pytest.importorskip("requests").exceptions.ConnectionError(),
        ):
            result = fetch_agg_trades("BTCUSDT")

        assert result is None

    def test_http_error_returns_none(self):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = pytest.importorskip("requests").exceptions.HTTPError(
            "500 Server Error"
        )

        with patch("modules.order_book.market_data_fetcher.requests.get", return_value=mock_response):
            result = fetch_agg_trades("BTCUSDT")

        assert result is None

    def test_invalid_trade_items_skipped(self):
        mock_response = Mock()
        mock_response.json.return_value = [
            {"p": "50000.0", "q": "1.5", "T": 1234567890000, "m": False},
            {"invalid": "item"},
            {"p": "50001.0", "q": "2.0", "T": 1234567891000, "m": True},
        ]
        mock_response.raise_for_status = Mock()

        with patch("modules.order_book.market_data_fetcher.requests.get", return_value=mock_response):
            result = fetch_agg_trades("BTCUSDT")

        assert result is not None
        assert len(result) == 2
