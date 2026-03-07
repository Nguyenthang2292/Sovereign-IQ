from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


@dataclass
class OrderBookSnapshot:
    """Raw orderbook snapshot from Binance."""

    symbol: str
    bids: List[Tuple[float, float]]  # [(price, qty), ...] sorted desc
    asks: List[Tuple[float, float]]  # [(price, qty), ...] sorted asc
    timestamp: float


@dataclass
class AggTrade:
    """Aggregated trade snapshot."""

    price: float
    quantity: float
    timestamp: float
    is_buyer_maker: bool  # True = Sell aggressive, False = Buy aggressive


@dataclass
class CombinedResult:
    """Combined calculation result from OBI and Delta."""

    obi_score: float  # Normalized OBI from aggregated bins: -1.0 → +1.0
    delta_score: float  # Normalized Cumulative Delta: -1.0 → +1.0
    combined_score: float  # Score combined (40% OBI + 60% Delta): -1.0 → +1.0
    obi_raw: float  # Raw OBI value
    delta_raw: float  # Raw Cumulative Delta value
    weighted_mid: float  # Weighted mid-price by volume


class OBIDecision(str, Enum):
    """Decision result from OBI Gate."""

    PASS = "PASS"  # Score confirms signal direction → execute order
    SKIP = "SKIP"  # After retry still conflict → skip this signal
