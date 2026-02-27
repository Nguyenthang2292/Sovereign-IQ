"""
ATC Visualizer Sub-module

Renders candlestick charts with ATC signal markers (LONG/SHORT/NEUTRAL)
and all 6 MA lines used by the ATC engine (EMA, WMA, DEMA, LSMA, HMA, KAMA).

Usage:
    from modules.adaptive_trend_LTS_serverless.visualizer import ATCChartRenderer
    from modules.adaptive_trend_LTS_serverless.lambda_client import ATCLambdaClient

    client = ATCLambdaClient()
    renderer = ATCChartRenderer(client)
    path = renderer.render("BTCUSDT", "1h", output_path="chart.png")

CLI:
    python -m modules.adaptive_trend_LTS_serverless.visualizer --symbol BTCUSDT --tf 1h
"""

from .chart import ATCChartRenderer

__all__ = ["ATCChartRenderer"]
