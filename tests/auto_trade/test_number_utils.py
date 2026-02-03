import pytest

from modules.auto_trade.number_utils import coerce_float


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (None, 0.0, 0.0),
        (None, 1.5, 1.5),
        (0, 9.9, 0.0),
        (3, 0.0, 3.0),
        (2.25, 0.0, 2.25),
        ("4.5", 0.0, 4.5),
        ("not-a-number", 7.0, 7.0),
        ({}, 8.0, 8.0),
    ],
)
def test_coerce_float(value, default, expected):
    assert coerce_float(value, default=default) == expected
