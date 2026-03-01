from pathlib import Path
import sys

import pandas as pd

try:
    from .analyzer import SMCAnalyzer
    from .charts.renderer import SMCChartRenderer
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from modules.smart_money_concept.analyzer import SMCAnalyzer
    from modules.smart_money_concept.charts.renderer import SMCChartRenderer

def main():
    try:
        import yfinance as yf
    except ImportError:
        print("Thiếu dependency 'yfinance'. Cài bằng: pip install yfinance")
        return

    ticker = input("Nhập Ticker để phân tích SMC (ví dụ: AAPL, BTC-USD): ").strip()
    if not ticker:
        ticker = "AAPL"
        
    print(f"Đang tải dữ liệu cho {ticker}...")
    try:
        df = yf.download(ticker, start="2023-01-01", end="2024-01-01", auto_adjust=True)
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return

    if df.empty:
        print(f"Không tìm thấy dữ liệu cho {ticker}.")
        return

    # Normalize DataFrame
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df.reset_index(inplace=True)

    analyzer = SMCAnalyzer()
    state = analyzer.run(df)

    renderer = SMCChartRenderer()
    fig = renderer.render(state, ticker)
    
    fig.show()

if __name__ == "__main__":
    main()
