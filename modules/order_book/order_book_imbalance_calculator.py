import math
from typing import List

from .models import AggTrade, CombinedResult, OrderBookSnapshot


def calculate_combined_score(
    snapshot: OrderBookSnapshot,
    trades: List[AggTrade],
    bin_step_pct: float = 0.001,
) -> CombinedResult:
    """
    Calculate Combined Score (40% Aggregated OBI + 60% normalized Cumulative Delta).

    Args:
        snapshot: Order book snapshot with bids and asks
        trades: List of aggregated trades
        bin_step_pct: Bin step percentage (default 0.1% = 0.001)

    Returns:
        CombinedResult with scores
    """
    if not snapshot.bids or not snapshot.asks:
        return CombinedResult(
            obi_score=0.0,
            delta_score=0.0,
            combined_score=0.0,
            obi_raw=0.0,
            delta_raw=0.0,
            weighted_mid=0.0,
        )

    mid_price = (snapshot.bids[0][0] + snapshot.asks[0][0]) / 2.0

    obi_raw, obi_score = _calculate_aggregated_obi(snapshot, mid_price, bin_step_pct)

    delta_raw, delta_score, v_avg_5m = _calculate_cumulative_delta_normalized(trades, mid_price)

    combined_score = 0.4 * obi_score + 0.6 * delta_score

    weighted_mid = _calculate_weighted_mid(snapshot)

    return CombinedResult(
        obi_score=obi_score,
        delta_score=delta_score,
        combined_score=combined_score,
        obi_raw=obi_raw,
        delta_raw=delta_raw,
        weighted_mid=weighted_mid,
    )


def _calculate_aggregated_obi(
    snapshot: OrderBookSnapshot,
    mid_price: float,
    bin_step_pct: float,
) -> tuple[float, float]:
    """
    Calculate Aggregated OBI using bins.

    Formula: I_bins = (sum Q_bid_bins - sum Q_ask_bins) / (sum Q_bid_bins + sum Q_ask_bins)
    """
    bid_volume_by_bin: dict[int, float] = {}
    ask_volume_by_bin: dict[int, float] = {}

    for price, qty in snapshot.bids:
        bin_idx = math.floor((price - mid_price) / (mid_price * bin_step_pct))
        bid_volume_by_bin[bin_idx] = bid_volume_by_bin.get(bin_idx, 0.0) + qty

    for price, qty in snapshot.asks:
        bin_idx = math.floor((price - mid_price) / (mid_price * bin_step_pct))
        ask_volume_by_bin[bin_idx] = ask_volume_by_bin.get(bin_idx, 0.0) + qty

    total_bid_qty = sum(bid_volume_by_bin.values())
    total_ask_qty = sum(ask_volume_by_bin.values())

    if total_bid_qty + total_ask_qty == 0:
        return 0.0, 0.0

    obi_raw = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)

    obi_score = max(-1.0, min(1.0, obi_raw))

    return obi_raw, obi_score


def _calculate_cumulative_delta_normalized(
    trades: List[AggTrade],
    mid_price: float,
) -> tuple[float, float, float]:
    """
    Calculate Cumulative Delta and normalize using tanh.

    Formula:
    - Delta_cum = sum(Q_Buy_Market) - sum(Q_Sell_Market)
    - is_buyer_maker=False means Buy aggressive
    - is_buyer_maker=True means Sell aggressive
    - Normalized: tanh(Delta_cum / V_avg_5m)
    """
    buy_aggressive_qty = 0.0
    sell_aggressive_qty = 0.0

    for trade in trades:
        if trade.is_buyer_maker:
            sell_aggressive_qty += trade.quantity
        else:
            buy_aggressive_qty += trade.quantity

    delta_raw = buy_aggressive_qty - sell_aggressive_qty

    if not trades:
        return 0.0, 0.0, 0.0

    v_avg_5m = sum(t.quantity for t in trades) / len(trades)

    if v_avg_5m == 0:
        return delta_raw, 0.0, 0.0

    delta_score = math.tanh(delta_raw / v_avg_5m)

    return delta_raw, delta_score, v_avg_5m


def _calculate_weighted_mid(snapshot: OrderBookSnapshot) -> float:
    """
    Calculate weighted mid-price by volume.
    """
    if not snapshot.bids or not snapshot.asks:
        return 0.0

    bid_price, bid_qty = snapshot.bids[0]
    ask_price, ask_qty = snapshot.asks[0]

    total_qty = bid_qty + ask_qty
    if total_qty == 0:
        return (bid_price + ask_price) / 2

    weighted_mid = (bid_price * ask_qty + ask_price * bid_qty) / total_qty

    return weighted_mid
