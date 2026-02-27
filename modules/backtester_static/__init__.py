"""
backtester_static – generic signal-consumer static backtester.

Generic: works with any signal source (ATC, XGBoost, LSTM, etc.).

Quick start::

    from modules.backtester_static import StaticBacktester, BacktestConfig

    cfg = BacktestConfig(mode="pct", tp=2.0, sl=1.0, trailing_stop=0.5)
    bt  = StaticBacktester(config=cfg)
    result = bt.run(df=df_ohlcv, signals=signal_series)

    from modules.backtester_static.report import generate_report
    csv_path, table_png = generate_report(result, "BTCUSDT", "1h", cfg, output_dir)
"""

from .config import BacktestConfig
from .engine import BacktestResult, StaticBacktester, TpSlLevel, TradeRecord

__all__ = [
    "BacktestConfig",
    "StaticBacktester",
    "BacktestResult",
    "TradeRecord",
    "TpSlLevel",
]
