"""
Portfolio Manager - Refactored version using modular components.
"""

from __future__ import (
    annotations,
)  # all annotations are lazy strings — fixes "Not valid as type" for Optional import vars

import signal
import sys
import threading
from types import ModuleType
from typing import TYPE_CHECKING, Callable, List, Optional

from colorama import Fore, Style
from colorama import init as colorama_init

if TYPE_CHECKING:
    # Only used by type-checkers; never imported at runtime.
    from modules.common.core.data_fetcher import DataFetcher
    from modules.common.core.exchange_manager import ExchangeManager
    from modules.common.models.position import Position
    from modules.portfolio.core.risk_calculator import PortfolioRiskCalculator

# ---------------------------------------------------------------------------
# Optional runtime imports — all default to None so the except branch simply
# does nothing (pass).  `from __future__ import annotations` means that using
# `List[Position]` in a type annotation is fine even though Position is None
# here at import time.
# ---------------------------------------------------------------------------
_Position: type | None = None
_color_text: Callable[..., str] | None = None
_safe_input: Callable[..., str] | None = None
_ExchangeManager: type | None = None
_DataFetcher: type | None = None
_PortfolioRiskCalculator: type | None = None
_correlation_analyzer_mod: ModuleType | None = None
_hedge_finder_mod: ModuleType | None = None

BENCHMARK_SYMBOL = "BTC/USDT"
DEFAULT_VAR_CONFIDENCE = 0.95
DEFAULT_VAR_LOOKBACK_DAYS = 90

try:
    from config import (  # type: ignore[assignment]
        BENCHMARK_SYMBOL,
        DEFAULT_VAR_CONFIDENCE,
        DEFAULT_VAR_LOOKBACK_DAYS,
    )
    from modules.common.core.data_fetcher import DataFetcher as _DataFetcher  # type: ignore[assignment]
    from modules.common.core.exchange_manager import ExchangeManager as _ExchangeManager  # type: ignore[assignment]
    from modules.common.models.position import Position as _Position  # type: ignore[assignment]
    from modules.common.utils import color_text as _color_text  # type: ignore[assignment]
    from modules.common.utils import safe_input as _safe_input
    from modules.portfolio.core import correlation_analyzer as _correlation_analyzer_mod  # type: ignore[assignment]
    from modules.portfolio.core import hedge_finder as _hedge_finder_mod  # type: ignore[assignment]
    from modules.portfolio.core.risk_calculator import (
        PortfolioRiskCalculator as _PortfolioRiskCalculator,  # type: ignore[assignment]
    )
except ImportError:
    pass

colorama_init(autoreset=True)


# ---------------------------------------------------------------------------
# Thin helper so callers don't have to guard every print against None.
# Falls back to plain string when the module isn't available.
# ---------------------------------------------------------------------------
def _ct(text: str, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
    if _color_text is not None:
        return _color_text(text, *args, **kwargs)
    return text


class PortfolioManager:
    """Main portfolio manager orchestrating all components."""

    def _create_correlation_analyzer(self):
        if _correlation_analyzer_mod is None:
            raise RuntimeError("PortfolioCorrelationAnalyzer module not available")
        analyzer_cls = getattr(_correlation_analyzer_mod, "PortfolioCorrelationAnalyzer", None)
        if analyzer_cls is None:
            raise RuntimeError("PortfolioCorrelationAnalyzer class not available")
        return analyzer_cls(self.data_fetcher, self.positions)

    def _create_hedge_finder(self, analyzer):
        if _hedge_finder_mod is None:
            raise RuntimeError("HedgeFinder module not available")
        hedge_cls = getattr(_hedge_finder_mod, "HedgeFinder", None)
        if hedge_cls is None:
            raise RuntimeError("HedgeFinder class not available")
        return hedge_cls(
            self.exchange_manager,
            analyzer,
            self.risk_calculator,
            self.positions,
            self.benchmark_symbol,
            self.shutdown_event,
            self.data_fetcher,
        )

    def __init__(
        self,
        api_key=None,
        api_secret=None,
        testnet=False,
        install_signal_handlers: bool = False,
    ):
        self.positions: List[Position] = []
        self.benchmark_symbol = BENCHMARK_SYMBOL
        self.shutdown_event = threading.Event()
        self._signal_handlers_registered = False
        if install_signal_handlers:
            self.install_signal_handlers()

        # Initialize components — fail fast with a clear message if deps missing
        if _ExchangeManager is None or _DataFetcher is None or _PortfolioRiskCalculator is None:
            raise ImportError(
                "Required modules (ExchangeManager, DataFetcher, PortfolioRiskCalculator) "
                "could not be imported. Check your environment setup."
            )
        self.exchange_manager: ExchangeManager = _ExchangeManager(api_key, api_secret, testnet)
        self.data_fetcher: DataFetcher = _DataFetcher(self.exchange_manager, self.shutdown_event)
        self.risk_calculator: PortfolioRiskCalculator = _PortfolioRiskCalculator(
            self.data_fetcher, self.benchmark_symbol
        )

    def add_position(self, symbol: str, direction: str, entry_price: float, size_usdt: float):
        """Add a position to the portfolio."""
        if _Position is None:
            raise ImportError("Position model not available")
        self.positions.append(_Position(symbol.upper(), direction.upper(), entry_price, size_usdt))

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal."""
        if not self.shutdown_event.is_set():
            print(_ct("\nInterrupt received. Cancelling ongoing tasks...", Fore.YELLOW))
            self.shutdown_event.set()
        sys.exit(0)

    def install_signal_handlers(self):
        """Register OS signal handlers for graceful shutdown when running from CLI."""
        if self._signal_handlers_registered:
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Signal handlers can only be installed from the main thread.")
        signal.signal(signal.SIGINT, self._handle_shutdown)
        try:
            signal.signal(signal.SIGTERM, self._handle_shutdown)
        except AttributeError:
            # SIGTERM may not be available on some platforms (e.g., Windows)
            pass
        self._signal_handlers_registered = True

    def _should_stop(self) -> bool:
        """Check if shutdown was requested."""
        return self.shutdown_event.is_set()

    def load_from_binance(self, api_key=None, api_secret=None, testnet=None, debug=False):
        """Load positions directly from Binance Futures USDT-M."""
        try:
            binance_positions = self.data_fetcher.fetch_binance_futures_positions(
                api_key=api_key or self.exchange_manager.api_key,
                api_secret=api_secret or self.exchange_manager.api_secret,
                testnet=self.exchange_manager.testnet if testnet is None else testnet,
                debug=debug,
            )
        except Exception as exc:
            raise ValueError(f"Error loading positions from Binance: {exc}")

        if not binance_positions:
            print(_ct("No open positions found on Binance.", Fore.YELLOW))
            self.positions = []
            return

        if _Position is None:
            raise ImportError("Position model not available")

        self.positions = [
            _Position(
                symbol=pos["symbol"].upper(),
                direction=pos["direction"].upper(),
                entry_price=pos["entry_price"],
                size_usdt=pos["size_usdt"],
            )
            for pos in binance_positions
        ]

        # Update exchange manager credentials if provided
        credentials_updated = False
        if api_key is not None:
            self.exchange_manager.api_key = api_key
            credentials_updated = True
        if api_secret is not None:
            self.exchange_manager.api_secret = api_secret
            credentials_updated = True
        if testnet is not None:
            self.exchange_manager.testnet = testnet

        if credentials_updated:
            self.exchange_manager.authenticated.update_default_credentials(
                api_key=self.exchange_manager.api_key,
                api_secret=self.exchange_manager.api_secret,
            )

    def fetch_prices(self):
        """Fetches current prices for all symbols from Binance."""
        symbols = list(set([p.symbol for p in self.positions]))
        if symbols:
            self.data_fetcher.fetch_current_prices_from_binance(symbols)

    @property
    def market_prices(self):
        """Get market prices from data fetcher."""
        return self.data_fetcher.market_prices

    def calculate_stats(self):
        """Calculates PnL, simple delta, and beta-weighted delta for the portfolio."""
        return self.risk_calculator.calculate_stats(self.positions, self.market_prices)

    def calculate_beta(self, symbol: str, benchmark_symbol: Optional[str] = None, **kwargs):
        """Calculates beta of a symbol versus a benchmark."""
        return self.risk_calculator.calculate_beta(symbol, benchmark_symbol, **kwargs)

    def calculate_portfolio_var(
        self,
        confidence: float = DEFAULT_VAR_CONFIDENCE,
        lookback_days: int = DEFAULT_VAR_LOOKBACK_DAYS,
    ):
        """Calculates Historical Simulation VaR for the current portfolio."""
        return self.risk_calculator.calculate_portfolio_var(self.positions, confidence, lookback_days)

    @property
    def last_var_value(self):
        """Get last VaR value from risk calculator."""
        return self.risk_calculator.last_var_value

    @property
    def last_var_confidence(self):
        """Get last VaR confidence from risk calculator."""
        return self.risk_calculator.last_var_confidence

    def fetch_ohlcv(self, symbol, limit=1500, timeframe="1h"):
        """Fetches OHLCV data using ccxt with fallback exchanges."""
        df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(symbol, limit, timeframe)
        return df

    def calculate_weighted_correlation(self, new_symbol: str, verbose: bool = True):
        """Calculates weighted correlation with entire portfolio."""
        analyzer = self._create_correlation_analyzer()
        return analyzer.calculate_weighted_correlation_with_new_symbol(new_symbol, verbose)

    def calculate_portfolio_return_correlation(self, new_symbol: str, **kwargs):
        """Calculates correlation between portfolio return and new symbol."""
        analyzer = self._create_correlation_analyzer()
        return analyzer.calculate_portfolio_return_correlation(new_symbol, **kwargs)

    def find_best_hedge_candidate(self, total_delta: float, total_beta_delta: float, **kwargs):
        """Automatically scans Binance futures symbols to find the best hedge candidate."""
        analyzer = self._create_correlation_analyzer()
        hedge_finder = self._create_hedge_finder(analyzer)
        return hedge_finder.find_best_hedge_candidate(total_delta, total_beta_delta, **kwargs)

    def analyze_new_trade(self, new_symbol: str, total_delta: float, total_beta_delta: float, **kwargs):
        """Analyzes a potential new trade and automatically recommends direction for beta-weighted hedging."""
        analyzer = self._create_correlation_analyzer()
        hedge_finder = self._create_hedge_finder(analyzer)
        return hedge_finder.analyze_new_trade(
            new_symbol,
            total_delta,
            total_beta_delta,
            self.last_var_value,
            self.last_var_confidence,
            **kwargs,
        )


def display_portfolio_analysis(pm: PortfolioManager):
    """
    Tính năng 1: Hiển thị Portfolio Correlation và VaR hiện có.
    """
    print("\n" + _ct("=== PORTFOLIO ANALYSIS ===", Fore.CYAN, Style.BRIGHT))

    # Fetch prices and calculate stats
    pm.fetch_prices()
    df, total_pnl, total_delta, total_beta_delta = pm.calculate_stats()

    # Display portfolio status
    print("\n" + _ct("=== PORTFOLIO STATUS ===", Fore.WHITE, Style.BRIGHT))
    print(df.to_string(index=False))
    print("-" * 50)
    print(f"Total PnL: {_ct(f'{total_pnl:.2f} USDT', Fore.GREEN if total_pnl >= 0 else Fore.RED)}")
    print(f"Total Delta: {_ct(f'{total_delta:.2f} USDT', Fore.YELLOW)}")
    print(f"Total Beta Delta (vs {pm.benchmark_symbol}): {_ct(f'{total_beta_delta:.2f} USDT', Fore.YELLOW)}")

    # Calculate and display VaR
    print("\n" + _ct("=== VALUE AT RISK (VaR) ===", Fore.CYAN, Style.BRIGHT))
    var_value = pm.calculate_portfolio_var(confidence=DEFAULT_VAR_CONFIDENCE, lookback_days=DEFAULT_VAR_LOOKBACK_DAYS)
    if var_value is not None:
        conf_pct = int((pm.last_var_confidence or 0) * 100)
        print(
            _ct(
                f"With {conf_pct}% confidence, daily loss should stay within {var_value:.2f} USDT.",
                Fore.WHITE,
            )
        )
    else:
        print(
            _ct(
                "Not enough history for a reliable VaR estimate.",
                Fore.YELLOW,
            )
        )

    # Calculate and display Portfolio Internal Correlation
    print("\n" + _ct("=== PORTFOLIO CORRELATION ===", Fore.CYAN, Style.BRIGHT))
    if len(pm.positions) >= 2:
        try:
            analyzer = pm._create_correlation_analyzer()
        except RuntimeError:
            analyzer = None

        if analyzer is not None:
            internal_corr, pairs = analyzer.calculate_weighted_correlation(verbose=True)

            if internal_corr is not None:
                if abs(internal_corr) > 0.7:
                    status = _ct("HIGH - Consider diversification", Fore.RED)
                elif abs(internal_corr) > 0.4:
                    status = _ct("MODERATE", Fore.YELLOW)
                else:
                    status = _ct("LOW - Good diversification", Fore.GREEN)
                print(f"\nPortfolio Correlation Status: {status}")
        else:
            print(
                _ct(
                    "Correlation analyzer unavailable; skipping correlation section.",
                    Fore.YELLOW,
                )
            )
    else:
        print(
            _ct(
                "Need at least 2 positions to calculate portfolio correlation.",
                Fore.YELLOW,
            )
        )


def display_portfolio_with_hedge_analysis(pm: PortfolioManager):
    """
    Tính năng 2: Hiển thị Portfolio + tự động tìm hedge candidate.
    """
    # Hiển thị tất cả từ tính năng 1
    display_portfolio_analysis(pm)

    # Thêm phần auto hedge
    print("\n" + _ct("=== AUTO HEDGE ANALYSIS ===", Fore.MAGENTA, Style.BRIGHT))

    pm.fetch_prices()
    _, total_pnl, total_delta, total_beta_delta = pm.calculate_stats()

    best_candidate = pm.find_best_hedge_candidate(total_delta, total_beta_delta)
    if best_candidate:
        symbol = best_candidate["symbol"]
        recommended_direction, recommended_size, correlation = pm.analyze_new_trade(
            symbol, total_delta, total_beta_delta
        )
        if recommended_direction and recommended_size is not None:
            print(
                _ct(
                    f"\n✓ Auto-selected hedge: {symbol} | {recommended_direction} {recommended_size:.2f} USDT",
                    Fore.GREEN,
                    Style.BRIGHT,
                )
            )
        else:
            print(
                _ct(
                    f"\n{symbol}: Portfolio already neutral, no trade required.",
                    Fore.WHITE,
                )
            )
    else:
        print(
            _ct(
                "\nCould not determine a suitable hedge candidate automatically.",
                Fore.YELLOW,
            )
        )


def main():
    print(
        _ct(
            "=== Crypto Portfolio Manager (Binance Integration) ===",
            Fore.MAGENTA,
            Style.BRIGHT,
        )
    )

    try:
        pm = PortfolioManager()
        pm.install_signal_handlers()
    except Exception as e:
        print(_ct(f"Error initializing PortfolioManager: {e}", Fore.RED))
        return

    print("\n" + _ct("Loading positions from Binance...", Fore.CYAN))
    try:
        pm.load_from_binance()
    except Exception as e:
        print(_ct(f"Error loading from Binance: {e}", Fore.RED))
        print(_ct("Please check your API credentials and try again.", Fore.YELLOW))
        return

    if not pm.positions:
        print(_ct("No positions available. Exiting.", Fore.YELLOW))
        return

    # Interactive menu
    print("\n" + "=" * 60)
    print(_ct("Select Analysis Mode:", Fore.CYAN, Style.BRIGHT))
    print("=" * 60)
    print("1. Portfolio Analysis (Correlation + VaR)")
    print("2. Portfolio Analysis + Auto Hedge")
    print("3. Exit")
    print("=" * 60)

    while True:
        try:
            choice = _safe_input("\nEnter choice (1-3): ", default="")
            if choice is None:
                choice = ""
            choice = choice.strip()

            if choice == "1":
                display_portfolio_analysis(pm)
                break
            elif choice == "2":
                display_portfolio_with_hedge_analysis(pm)
                break
            elif choice == "3":
                print(_ct("\nExiting...", Fore.YELLOW))
                break
            else:
                print(_ct("Invalid choice. Please enter 1, 2, or 3.", Fore.RED))
        except KeyboardInterrupt:
            print(_ct("\n\nInterrupted by user. Exiting...", Fore.YELLOW))
            break
        except EOFError:
            print(_ct("\n\nExiting...", Fore.YELLOW))
            break


if __name__ == "__main__":
    main()
