import pytest

from modules.common.domain.order_type_codec import BinanceOrderType


def make_order(ccxt_type: str, info_type: str = None, stop_price: float = None) -> dict:
    o = {"type": ccxt_type, "info": {}}
    if info_type:
        o["info"]["type"] = info_type
    if stop_price is not None:
        o["info"]["stopPrice"] = str(stop_price)
        o["stopPrice"] = str(stop_price)
    return o


@pytest.mark.parametrize(
    "order, expected_type",
    [
        (make_order("market", "STOP_MARKET"), "STOP_MARKET"),
        (make_order("market", "TAKE_PROFIT_MARKET"), "TAKE_PROFIT_MARKET"),
        (make_order("limit", "STOP"), "STOP"),
        (make_order("limit", "TAKE_PROFIT"), "TAKE_PROFIT"),
        (make_order("market"), "MARKET"),
        (make_order("limit"), "LIMIT"),
    ],
)
def test_resolve(order, expected_type):
    assert BinanceOrderType.resolve(order) == expected_type


@pytest.mark.parametrize(
    "order, expected_conditional",
    [
        (make_order("market", "STOP_MARKET", 100.0), True),
        (make_order("market", "TAKE_PROFIT_MARKET", 200.0), True),
        (make_order("limit", "STOP"), True),
        (make_order("limit", "TAKE_PROFIT"), True),
        (make_order("market"), False),
        (make_order("limit"), False),
        (make_order("market", stop_price=150.0), True),  # stopPrice fallback
    ],
)
def test_is_conditional(order, expected_conditional):
    assert BinanceOrderType.is_conditional(order) == expected_conditional


@pytest.mark.parametrize(
    "order, entry, side, expected_kind",
    [
        (make_order("market", "TAKE_PROFIT_MARKET", 110.0), 100.0, "long", "tp"),
        (make_order("market", "STOP_MARKET", 90.0), 100.0, "long", "sl"),
        (make_order("market", "TAKE_PROFIT_MARKET", 90.0), 100.0, "short", "tp"),
        (make_order("market", "STOP_MARKET", 110.0), 100.0, "short", "sl"),
        (make_order("market", stop_price=110.0), 100.0, "long", "tp"),  # price fallback
        (make_order("market", stop_price=90.0), 100.0, "long", "sl"),  # price fallback
    ],
)
def test_classify(order, entry, side, expected_kind):
    assert BinanceOrderType.classify(order, entry, side) == expected_kind


@pytest.mark.parametrize(
    "order, expected_params",
    [
        (make_order("market", "STOP_MARKET"), {"stop": True}),
        (make_order("market"), {}),
        (make_order("limit"), {}),
    ],
)
def test_cancel_params(order, expected_params):
    assert BinanceOrderType.cancel_params(order) == expected_params
