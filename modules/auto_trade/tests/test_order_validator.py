from modules.auto_trade.execution.order_builder import OrderTicket
from modules.auto_trade.execution.order_validator import OrderValidator


def test_order_validator_rejections():
    validator = OrderValidator(min_position_size=10.0, max_leverage=50)

    ticket = OrderTicket(symbol="BTC/USDT", side="BUY", amount=5.0, leverage=10)
    # Reject: amount < 10.0
    assert validator.validate_pre_order(ticket, balance=100.0, current_price=10000.0) is False

    ticket.amount = 50.0
    ticket.leverage = 100
    # Reject: leverage > 50
    assert validator.validate_pre_order(ticket, balance=1000.0, current_price=10000.0) is False

    ticket.leverage = 10
    # Reject: required margin (50/10=5) > balance (4)
    assert validator.validate_pre_order(ticket, balance=4.0, current_price=10000.0) is False


def test_order_validator_success():
    validator = OrderValidator(min_position_size=10.0, max_leverage=50)
    ticket = OrderTicket(symbol="BTC/USDT", side="BUY", amount=50.0, leverage=10)
    assert validator.validate_pre_order(ticket, balance=100.0, current_price=10000.0) is True
