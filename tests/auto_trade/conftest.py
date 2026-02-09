"""Shared pytest fixtures for auto_trade tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modules.auto_trade.core.atc_scanner import SignalResult
from modules.auto_trade.core.gemini_integration import GeminiSignal


@pytest.fixture
def sample_signal_result():
    """Factory fixture for creating SignalResult objects."""

    def _make_signal(
        symbol: str = "BTC/USDT", score: float = 0.9, signal_type: str = "LONG", xgboost_conf: float = 0.8, **kwargs
    ):
        """Create a SignalResult with default values."""
        details = kwargs.get("details", {"xgboost_conf": xgboost_conf})
        details.update(kwargs.get("extra_details", {}))

        strengths = kwargs.get("strengths", {"15m": 0.8, "1h": 0.7, "4h": 0.9})
        strengths.update(kwargs.get("extra_strengths", {}))

        return SignalResult(symbol, score, signal_type, details, strengths)

    return _make_signal


@pytest.fixture
def sample_gemini_signal():
    """Factory fixture for creating GeminiSignal objects."""

    def _make_signal(symbol: str = "BTC/USDT", **kwargs):
        """Create a GeminiSignal with default values."""
        return GeminiSignal(
            trend=kwargs.get("trend", "UP"),
            signal=kwargs.get("signal", "LONG"),
            confidence=kwargs.get("confidence", 0.9),
            entry=kwargs.get("entry", 50000),
            stop_loss=kwargs.get("stop_loss", 49000),
            take_profit=kwargs.get("take_profit", 52000),
            reasoning=kwargs.get("reasoning", ""),
        )

    return _make_signal


@pytest.fixture
def mock_pipeline_components():
    """Factory fixture for creating mocked pipeline components."""

    def _make_components():
        """Create a dictionary of mocked pipeline components."""
        return {
            "symbol_manager": MagicMock(),
            "atc_scanner": MagicMock(),
            "xgboost_filter": MagicMock(),
            "gemini_integration": MagicMock(),
            "signal_selector": MagicMock(),
        }

    return _make_components


@pytest.fixture
def mock_components():
    """Fixture providing mocked pipeline components."""
    return {
        "symbol_manager": MagicMock(),
        "atc_scanner": MagicMock(),
        "xgboost_filter": MagicMock(),
        "gemini_integration": MagicMock(),
        "signal_selector": MagicMock(),
    }


@pytest.fixture
def pipeline(mock_components):
    """Fixture providing a SignalPipeline instance."""
    from modules.auto_trade.core.signal_pipeline import SignalPipeline

    return SignalPipeline(
        symbol_manager=mock_components["symbol_manager"],
        atc_scanner=mock_components["atc_scanner"],
        xgboost_filter=mock_components["xgboost_filter"],
        gemini_integration=mock_components["gemini_integration"],
        signal_selector=mock_components["signal_selector"],
        config={"max_symbols_to_scan": 10, "pipeline_timeout": 5},
    )


@pytest.fixture
def selector():
    """Fixture providing a SignalSelector instance."""
    from modules.auto_trade.core.signal_selector import SignalSelector

    return SignalSelector(
        config={"weight_xgboost": 0.4, "weight_gemini": 0.6, "min_confidence_threshold": 0.5}
    )


@pytest.fixture
def mock_data_fetcher():
    """Fixture providing a mocked DataFetcher."""
    return MagicMock()


@pytest.fixture
def mock_scan_all_symbols():
    """Fixture patching scan_all_symbols in atc_scanner."""
    with patch("modules.auto_trade.core.atc_scanner.scan_all_symbols") as mock:
        yield mock



@pytest.fixture
def sample_order_data():
    """Factory fixture for creating sample order data dictionaries."""

    def _make_data(order_id: str = "TEST_ORDER_001", symbol: str = "BTCUSDT", **kwargs):
        """Create sample order data with default values."""
        return {
            "order_id": order_id,
            "client_order_id": kwargs.get("client_order_id", f"AT_{order_id}_CLIENT"),
            "symbol": symbol,
            "side": kwargs.get("side", "LONG"),
            "entry_price": kwargs.get("entry_price", 50000.0),
            "amount": kwargs.get("amount", 0.01),
            "leverage": kwargs.get("leverage", 2),
            "stop_loss": kwargs.get("stop_loss", 45000.0),
            "take_profit": kwargs.get("take_profit", 52500.0),
            "status": kwargs.get("status", "CLOSED"),
            "order_source": kwargs.get("order_source", "PROGRAMMATIC"),
            "execution_mode": kwargs.get("execution_mode", "AUTO"),
            "pnl": kwargs.get("pnl", 0.0),
            "commission": kwargs.get("commission", 0.0),
        }

    return _make_data


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database and yield the path.

    Initializes the global database manager.
    """
    import modules.auto_trade.database as db_module
    from modules.auto_trade.database import get_db_manager, initialize_database

    # Reset singleton
    db_module._db_manager_instance = None

    db_path = tmp_path / "test_auto_trade.db"
    initialize_database(str(db_path))
    get_db_manager(str(db_path))

    yield str(db_path)

    # Cleanup
    db_module._db_manager_instance = None


@pytest.fixture
def test_db_session(tmp_path_factory):
    """Create a temporary test database and yield a session scope."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    import modules.auto_trade.database as db_module
    from modules.auto_trade.database import get_db_manager, initialize_database, session_scope

    db_path = tmp_path_factory.mktemp("test_db") / "test_auto_trade.db"
    db_module._db_manager_instance = None
    initialize_database(str(db_path))
    get_db_manager(str(db_path))

    yield session_scope

    # Cleanup
    db_module._db_manager_instance = None


@pytest.fixture
def mock_session_scope():
    """Fixture providing a mocked session_scope for unit tests."""
    with patch("modules.auto_trade.database.utils.session_scope") as mock:
        mock_session = MagicMock()
        mock.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock.return_value.__exit__ = MagicMock(return_value=None)
        yield mock, mock_session


@pytest.fixture
def mock_filedialog():
    """Fixture providing a mocked filedialog for GUI tests."""
    with patch("tkinter.filedialog.asksaveasfilename") as mock:
        yield mock


@pytest.fixture
def mock_messagebox():
    """Fixture providing a mocked messagebox for GUI tests."""
    with patch("tkinter.messagebox") as mock:
        yield mock
