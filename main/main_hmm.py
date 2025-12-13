"""
CLI tiện ích chạy quy trình HMM Signal Combiner trên dữ liệu OHLCV mới nhất.

Lấy dữ liệu qua `ExchangeManager` + `DataFetcher`, sau đó gọi `modules.hmm.signal_combiner.hmm_signals`.
Kết hợp High-Order HMM và HMM-KAMA để tạo trading signals.
Tham khảo cách tổ chức của `xgboost_prediction_main.py` và `pairs_trading_main_v2.py`.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Dict, Optional

from colorama import Fore, Style, init as colorama_init

from config import (
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
    HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
    HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
)
from modules.common.ExchangeManager import ExchangeManager
from modules.common.DataFetcher import DataFetcher
from modules.common.utils import color_text, normalize_symbol
from modules.hmm.signal_combiner import hmm_signals, Signal
from modules.hmm.signal_resolution import LONG, HOLD, SHORT

warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


SIGNAL_TEXT = {
    LONG: ("LONG", Fore.GREEN),
    HOLD: ("HOLD", Fore.YELLOW),
    SHORT: ("SHORT", Fore.RED),
}


def _signal_to_text(signal: Signal) -> str:
    """Convert signal to colored text."""
    label, color = SIGNAL_TEXT.get(signal, (f"{signal}", Fore.WHITE))
    style = Style.BRIGHT if signal != HOLD else Style.NORMAL
    return color_text(label, color, style)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HMM Signal Combiner (High-Order HMM + HMM-KAMA).")
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
        "--orders-argrelextrema",
        type=int,
        default=HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
        help=f"Order cho swing detection (default: {HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT})",
    )
    parser.add_argument(
        "--strict-mode",
        action="store_true",
        help=f"Sử dụng strict mode cho swing-to-state conversion (default: {HMM_HIGH_ORDER_STRICT_MODE_DEFAULT})",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Chạy một lần rồi thoát (tắt chế độ loop nhập liên tục).",
    )
    return parser.parse_args()


def _build_param_overrides(args: argparse.Namespace) -> Dict[str, Optional[int | bool]]:
    return {
        "window_size": args.window_size,
        "window_kama": args.window_kama,
        "fast_kama": args.fast_kama,
        "slow_kama": args.slow_kama,
        "orders_argrelextrema": args.orders_argrelextrema,
        "strict_mode": args.strict_mode if args.strict_mode else None,  # None = use config default
    }


def _compute_std_targets(df, window: int = 50) -> Optional[Dict[str, float]]:
    """Compute bullish and bearish price targets using 1-3σ offsets from rolling mean."""
    if df is None or "close" not in df.columns:
        return None

    closes = df["close"].tail(max(window, 5)).dropna()
    if len(closes) < 5:
        return None

    mean_price = closes.mean()
    std_price = closes.std(ddof=0)
    if std_price == 0 or not (mean_price and std_price):
        return None

    return {
        "window": len(closes),
        "basis": mean_price,
        "std": std_price,
        # Bearish targets (below mean)
        "bearish_1σ": mean_price - std_price,
        "bearish_2σ": mean_price - 2 * std_price,
        "bearish_3σ": mean_price - 3 * std_price,
        # Bullish targets (above mean)
        "bullish_1σ": mean_price + std_price,
        "bullish_2σ": mean_price + 2 * std_price,
        "bullish_3σ": mean_price + 3 * std_price,
    }


def _print_summary(
    symbol: str,
    exchange_id: Optional[str],
    signal_high_order: Signal,
    signal_kama: Signal,
    std_targets: Optional[Dict[str, float]] = None,
) -> None:
    header = f"HMM SIGNAL ANALYSIS | {symbol}"
    if exchange_id:
        header += f" @ {exchange_id.upper()}"
    print(color_text("\n" + "=" * 60, Fore.CYAN, Style.BRIGHT))
    print(color_text(header, Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 60, Fore.CYAN, Style.BRIGHT))

    print(f"High-Order HMM Signal: {_signal_to_text(signal_high_order)}")
    print(f"HMM-KAMA Signal: {_signal_to_text(signal_kama)}")
    
    # Determine combined signal recommendation
    if signal_high_order == signal_kama and signal_high_order != HOLD:
        combined_signal = signal_high_order
        agreement_status = color_text("✓ AGREEMENT", Fore.GREEN, Style.BRIGHT)
    elif signal_high_order != HOLD and signal_kama != HOLD:
        combined_signal = HOLD  # Conflict - wait
        agreement_status = color_text("⚠ CONFLICT", Fore.YELLOW, Style.BRIGHT)
    else:
        # One is HOLD, use the non-HOLD signal
        combined_signal = signal_high_order if signal_high_order != HOLD else signal_kama
        agreement_status = color_text("○ PARTIAL", Fore.CYAN, Style.NORMAL)
    
    print(f"Combined Recommendation: {_signal_to_text(combined_signal)} {agreement_status}")
    
    if std_targets:
        print(color_text("-" * 60, Fore.MAGENTA, Style.DIM))
        
        # Display targets based on combined signal
        if combined_signal == LONG:
            # Bullish targets (above mean)
            target_color = Fore.GREEN
            target_label = "Bullish Targets"
            print(
                color_text(
                    f"{target_label} (window {std_targets['window']}): "
                    f"Mean {std_targets['basis']:.2f} | "
                    f"+1σ {std_targets['bullish_1σ']:.2f} | "
                    f"+2σ {std_targets['bullish_2σ']:.2f} | "
                    f"+3σ {std_targets['bullish_3σ']:.2f}",
                    target_color,
                    Style.BRIGHT,
                )
            )
        elif combined_signal == SHORT:
            # Bearish targets (below mean)
            target_color = Fore.RED
            target_label = "Bearish Targets"
            print(
                color_text(
                    f"{target_label} (window {std_targets['window']}): "
                    f"Mean {std_targets['basis']:.2f} | "
                    f"-1σ {std_targets['bearish_1σ']:.2f} | "
                    f"-2σ {std_targets['bearish_2σ']:.2f} | "
                    f"-3σ {std_targets['bearish_3σ']:.2f}",
                    target_color,
                    Style.BRIGHT,
                )
            )
        else:
            # HOLD or conflict - show both directions
            print(
                f"Price Targets (window {std_targets['window']}, Mean: {std_targets['basis']:.2f}, Std: {std_targets['std']:.2f}):"
            )
            print(
                color_text(
                    f"  Bullish: +1σ {std_targets['bullish_1σ']:.2f} | "
                    f"+2σ {std_targets['bullish_2σ']:.2f} | "
                    f"+3σ {std_targets['bullish_3σ']:.2f}",
                    Fore.GREEN,
                )
            )
            print(
                color_text(
                    f"  Bearish: -1σ {std_targets['bearish_1σ']:.2f} | "
                    f"-2σ {std_targets['bearish_2σ']:.2f} | "
                    f"-3σ {std_targets['bearish_3σ']:.2f}",
                    Fore.RED,
                )
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

        std_targets = _compute_std_targets(df)

        try:
            signal_high_order, signal_kama = hmm_signals(
                df,
                window_kama=params_override.get("window_kama"),
                fast_kama=params_override.get("fast_kama"),
                slow_kama=params_override.get("slow_kama"),
                window_size=params_override.get("window_size"),
                orders_argrelextrema=params_override.get("orders_argrelextrema"),
                strict_mode=params_override.get("strict_mode"),
            )
        except Exception as exc:
            print(
                color_text(
                    f"Lỗi khi chạy HMM Signal Combiner: {exc}",
                    Fore.RED,
                    Style.BRIGHT,
                )
            )
            return

        _print_summary(symbol, exchange_id, signal_high_order, signal_kama, std_targets)

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

