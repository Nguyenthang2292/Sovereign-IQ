from modules.gemini_chart_analyzer.core.analyzers.components.response_parser import parse_trading_signal

response = "SHORT position. Entry: 1.2000, SL: 1.2500, TP2: 1.3000"

signal = parse_trading_signal(response)
print(f"Signal: {signal}")
print(f"Direction: {signal.direction}")
print(f"Entry price: {signal.entry_price}")
print(f"Stop loss: {signal.stop_loss}")
print(f"Take profit 2: {signal.take_profit_2}")
