from modules.common.data.fetchers import fetch_ohlcv_data_dict
from modules.common.ui.logging import log_error

from .analyzer import SMCAnalyzer
from .charts.renderer import SMCChartRenderer

_DEFAULT_SYMBOL = "BTCUSDT"
_DEFAULT_TIMEFRAME = "4h"
_DEFAULT_LIMIT = 500


def main():
    symbol = (
        input(f"Nhập symbol để phân tích SMC (ví dụ: BTCUSDT, ETHUSDT) [{_DEFAULT_SYMBOL}]: ").strip()
        or _DEFAULT_SYMBOL
    )

    timeframe = input(f"Nhập timeframe (ví dụ: 1h, 4h, 1d) [{_DEFAULT_TIMEFRAME}]: ").strip() or _DEFAULT_TIMEFRAME

    print(f"Đang tải dữ liệu cho {symbol} / {timeframe}...")

    data = fetch_ohlcv_data_dict(
        symbols=[symbol],
        timeframes=[timeframe],
        limit=_DEFAULT_LIMIT,
    )

    if not data or symbol not in data or timeframe not in data[symbol]:
        log_error(f"Không tải được dữ liệu cho {symbol} {timeframe}.")
        print(f"Lỗi: Không tải được dữ liệu cho {symbol} {timeframe}.")
        return

    df = data[symbol][timeframe]

    if df.empty:
        print(f"Không tìm thấy dữ liệu cho {symbol} {timeframe}.")
        return

    analyzer = SMCAnalyzer()
    state = analyzer.run(df)

    renderer = SMCChartRenderer()
    fig = renderer.render(state, f"{symbol} {timeframe}")

    fig.show()


if __name__ == "__main__":
    main()
