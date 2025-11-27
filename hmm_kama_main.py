"""
CLI tiện ích chạy quy trình HMM-KAMA trên dữ liệu OHLCV mới nhất.

Lấy dữ liệu qua `ExchangeManager` + `DataFetcher`, sau đó gọi `modules.hmm.hmm_kama`.
Tham khảo cách tổ chức của `xgboost_prediction_main.py` và `pairs_trading_main_v2.py`.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Dict, Optional

from colorama import Fore, Style, init as colorama_init

from modules.config import (
    DEFAULT_EXCHANGE_STRING,
    DEFAULT_EXCHANGES,
    DEFAULT_LIMIT,
    DEFAULT_QUOTE,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    HMM_FAST_KAMA_DEFAULT,
    HMM_SLOW_KAMA_DEFAULT,
    HMM_WINDOW_KAMA_DEFAULT,
    HMM_WINDOW_SIZE_DEFAULT,
)
from modules.common.ExchangeManager import ExchangeManager
from modules.common.DataFetcher import DataFetcher
from modules.common.utils import color_text, normalize_symbol
from modules.hmm.hmm_kama import HMM_KAMA, hmm_kama

warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


STATE_TEXT = {
    -1: ("N/A", Fore.WHITE),
    0: ("bearish strong", Fore.RED),
    1: ("bearish weak", Fore.LIGHTRED_EX),
    2: ("bullish weak", Fore.YELLOW),
    3: ("bullish strong", Fore.GREEN),
}


def _state_to_text(value: int) -> str:
    label, color = STATE_TEXT.get(value, (f"{value}", Fore.WHITE))
    style = Style.BRIGHT if value in (0, 3) else Style.NORMAL
    return color_text(label, color, style)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HMM-KAMA state analyzer.")
    parser.add_argument("--symbol", type=str, help="Cặp giao dịch, ví dụ BTC/USDT")
    parser.add_argument(
        "--quote",
        type=str,
        default=DEFAULT_QUOTE,
        help=f"Quote mặc định (default: {DEFAULT_QUOTE})",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=DEFAULT_TIMEFRAME,
        help=f"Khung thời gian (default: {DEFAULT_TIMEFRAME})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Số lượng nến (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--exchanges",
        type=str,
        default=DEFAULT_EXCHANGE_STRING,
        help="Danh sách sàn fallback, cách nhau bởi dấu phẩy.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=HMM_WINDOW_SIZE_DEFAULT,
        help=f"Cửa sổ rolling tối thiểu (default: {HMM_WINDOW_SIZE_DEFAULT})",
    )
    parser.add_argument(
        "--window-kama",
        type=int,
        default=HMM_WINDOW_KAMA_DEFAULT,
        help=f"Cửa sổ KAMA (default: {HMM_WINDOW_KAMA_DEFAULT})",
    )
    parser.add_argument(
        "--fast-kama",
        type=int,
        default=HMM_FAST_KAMA_DEFAULT,
        help=f"Tham số fast KAMA (default: {HMM_FAST_KAMA_DEFAULT})",
    )
    parser.add_argument(
        "--slow-kama",
        type=int,
        default=HMM_SLOW_KAMA_DEFAULT,
        help=f"Tham số slow KAMA (default: {HMM_SLOW_KAMA_DEFAULT})",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Chạy một lần rồi thoát (tắt chế độ loop nhập liên tục).",
    )
    return parser.parse_args()


def _build_param_overrides(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "window_size": args.window_size,
        "window_kama": args.window_kama,
        "fast_kama": args.fast_kama,
        "slow_kama": args.slow_kama,
    }


def _print_summary(symbol: str, exchange_id: Optional[str], result: HMM_KAMA) -> None:
    header = f"HMM-KAMA ANALYSIS | {symbol}"
    if exchange_id:
        header += f" @ {exchange_id.upper()}"
    print(color_text("\n" + "=" * 60, Fore.CYAN, Style.BRIGHT))
    print(color_text(header, Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 60, Fore.CYAN, Style.BRIGHT))

    print(
        f"Next state (HMM): {_state_to_text(result.next_state_with_hmm_kama)}",
    )
    print(
        f"Duration STD state: "
        f"{_state_to_text(result.current_state_of_state_using_std)}",
    )
    print(
        f"Duration HMM state: "
        f"{_state_to_text(result.current_state_of_state_using_hmm)}",
    )
    print(
        "Association Rule (Apriori / FP-Growth): "
        f"{_state_to_text(result.state_high_probabilities_using_arm_apriori)}"
        f" / {_state_to_text(result.state_high_probabilities_using_arm_fpgrowth)}",
    )
    print(
        f"KMeans duration cluster: "
        f"{_state_to_text(result.current_state_of_state_using_kmeans)}",
    )
    print(color_text("=" * 60 + "\n", Fore.CYAN, Style.BRIGHT))


def main() -> None:
    args = parse_args()
    allow_prompt = not args.no_prompt
    quote = args.quote.upper() if args.quote else DEFAULT_QUOTE
    timeframe = args.timeframe.lower()
    limit = max(50, args.limit or DEFAULT_LIMIT)
    exchanges = [
        ex.strip() for ex in args.exchanges.split(",") if ex.strip()
    ] or DEFAULT_EXCHANGES

    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)
    params_override = _build_param_overrides(args)

    def run_once(raw_symbol: str) -> None:
        symbol = normalize_symbol(raw_symbol, quote)
        df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=limit,
            timeframe=timeframe,
            check_freshness=True,
            exchanges=exchanges if exchanges != DEFAULT_EXCHANGES else None,
        )

        if df is None or df.empty:
            print(
                color_text(
                    f"Không thể tải dữ liệu cho {symbol}. Vui lòng thử lại.",
                    Fore.RED,
                    Style.BRIGHT,
                )
            )
            return

        try:
            result = hmm_kama(df, optimizing_params=params_override)
        except Exception as exc:
            print(
                color_text(
                    f"Lỗi khi chạy HMM-KAMA: {exc}",
                    Fore.RED,
                    Style.BRIGHT,
                )
            )
            return

        _print_summary(symbol, exchange_id, result)

    try:
        while True:
            input_symbol = args.symbol or DEFAULT_SYMBOL
            if allow_prompt:
                prompt = f"Nhập symbol (default {input_symbol}): "
                user_val = input(prompt).strip()
                if user_val:
                    input_symbol = user_val
            run_once(input_symbol)
            if not allow_prompt:
                break
            print(color_text("Nhấn Ctrl+C để thoát.\n", Fore.YELLOW))
            args.symbol = None
    except KeyboardInterrupt:
        print(color_text("\nThoát chương trình theo yêu cầu người dùng.", Fore.YELLOW))


if __name__ == "__main__":
    main()

