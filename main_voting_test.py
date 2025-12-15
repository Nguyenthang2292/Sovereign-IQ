"""
ATC + Range Oscillator + SPC Pure Voting System - TEST MODE (3 symbols only).

Entry point for the Voting Analyzer that combines signals from multiple indicators
using a pure voting system approach without sequential filtering.

TEST MODE: Chỉ fetch và phân tích 3 symbol bất kỳ thay vì toàn bộ thị trường.

Indicators:
    1. ATC (Adaptive Trend Classification): Market trend classification
    2. Range Oscillator: Overbought/oversold zone detection
    3. SPC (Simplified Percentile Clustering): Percentile-based clustering (3 strategies)
    4. XGBoost (optional): Machine learning prediction

Workflow:
    1. Lấy danh sách symbols từ Binance
    2. Chọn ngẫu nhiên 3 symbol đầu tiên
    3. Run ATC scan chỉ cho 3 symbol này
    4. Calculate signals from all indicators in parallel for these 3 symbols
    5. Apply voting system with weighted impact and cumulative vote
    6. Filter symbols based on voting results
    7. Display final results with voting metadata

See Also:
    - core.voting_analyzer.VotingAnalyzer: Main analyzer class
    - main_voting.py: Full market scan version

Example:
    Run from command line:
        $ python main_voting_test.py --timeframe 1h --enable-spc
"""

import warnings
import sys
import pandas as pd

from modules.common.utils import configure_windows_stdio

configure_windows_stdio()

from colorama import Fore, init as colorama_init

from modules.common.utils import (
    color_text,
    log_error,
    log_progress,
    log_success,
    log_warn,
)
from cli.argument_parser import parse_args
from core.voting_analyzer import VotingAnalyzer
from modules.common.utils import initialize_components
from modules.adaptive_trend.utils.config import create_atc_config_from_dict
from modules.common.utils import extract_dict_from_namespace

warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


class VotingAnalyzerTest(VotingAnalyzer):
    """
    Voting Analyzer Test Mode - chỉ scan 3 symbol thay vì toàn bộ thị trường.
    
    Kế thừa từ VotingAnalyzer và override phương thức run_atc_scan()
    để chỉ scan 3 symbol đầu tiên từ danh sách.
    """
    
    def run_atc_scan(self) -> bool:
        """
        Run ATC auto scan chỉ cho 3 symbol đầu tiên.
        
        Returns:
            True if signals found, False otherwise
        """
        log_progress("\nStep 1: Running ATC auto scan (TEST MODE - 3 symbols only)...")
        log_progress("=" * 80)
        
        # Lấy danh sách tất cả symbols
        log_progress("Fetching symbols list from Binance...")
        all_symbols = self.data_fetcher.list_binance_futures_symbols(
            max_candidates=None,
            progress_label="Symbol Discovery",
        )
        
        if not all_symbols:
            log_error("No symbols found from Binance")
            return False
        
        # Chọn 3 symbol đầu tiên
        test_symbols = all_symbols[:3]
        log_success(f"TEST MODE: Selected {len(test_symbols)} symbols: {', '.join(test_symbols)}")
        
        # Get ATC parameters
        atc_param_keys = [
            "limit",
            "ema_len",
            "hma_len",
            "wma_len",
            "dema_len",
            "lsma_len",
            "kama_len",
            "robustness",
            "lambda_param",
            "decay",
            "cutout",
        ]
        atc_params = extract_dict_from_namespace(self.args, atc_param_keys)
        atc_config = create_atc_config_from_dict(atc_params, timeframe=self.selected_timeframe)
        
        # Import hàm _process_symbol từ scanner để xử lý từng symbol
        from modules.adaptive_trend.core.scanner import _process_symbol
        
        results = []
        for symbol in test_symbols:
            try:
                result = _process_symbol(
                    symbol=symbol,
                    data_fetcher=self.data_fetcher,
                    atc_config=atc_config,
                    min_signal=self.args.min_signal,
                )
                if result is not None:
                    results.append(result)
                    log_progress(f"Found signal for {symbol}: {result.get('signal', 0):.2f}")
            except Exception as e:
                log_error(f"Error processing {symbol}: {e}")
        
        if not results:
            log_warn("No ATC signals found for the 3 test symbols.")
            log_warn("Please try:")
            log_warn("  - Different timeframe")
            log_warn("  - Different market conditions")
            log_warn("  - Check ATC configuration parameters")
            self.long_signals_atc = pd.DataFrame()
            self.short_signals_atc = pd.DataFrame()
            return False
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Filter LONG and SHORT signals
        self.long_signals_atc = results_df[results_df["trend"] > 0].copy()
        self.short_signals_atc = results_df[results_df["trend"] < 0].copy()
        
        # Sort by signal strength
        if not self.long_signals_atc.empty:
            self.long_signals_atc = self.long_signals_atc.sort_values(
                "signal", ascending=False
            ).reset_index(drop=True)
        if not self.short_signals_atc.empty:
            self.short_signals_atc = self.short_signals_atc.sort_values(
                "signal", ascending=True
            ).reset_index(drop=True)
        
        original_long_count = len(self.long_signals_atc)
        original_short_count = len(self.short_signals_atc)
        
        log_success(
            f"\nATC Scan Complete (TEST MODE): "
            f"Found {original_long_count} LONG + {original_short_count} SHORT signals "
            f"from {len(test_symbols)} test symbols"
        )
        
        return True


def main() -> None:
    """
    Main entry point for Voting Analyzer TEST MODE workflow.
    
    Chỉ fetch và phân tích 3 symbol đầu tiên từ Binance thay vì toàn bộ thị trường.
    
    Workflow steps:
        1. Parse command-line arguments (voting mode)
        2. Initialize ExchangeManager and DataFetcher
        3. Create VotingAnalyzerTest with parsed args (chỉ scan 3 symbols)
        4. Execute analyzer.run() to perform:
           - ATC scan (chỉ 3 symbols)
           - Parallel calculation of all indicator signals
           - Voting system application
           - Results display with voting metadata
    
    Note:
        Decision Matrix is always enabled in voting mode (it's the core mechanism).
        SPC can be enabled/disabled via command-line arguments.
    
    Raises:
        SystemExit: On KeyboardInterrupt or unhandled exceptions
    """
    args = parse_args(mode="voting", force_enable_spc=False, force_enable_decision_matrix=False)
    exchange_manager, data_fetcher = initialize_components()
    analyzer = VotingAnalyzerTest(args, data_fetcher)
    analyzer.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color_text("\nExiting program by user request.", Fore.YELLOW))
        sys.exit(0)
    except Exception as e:
        log_error(f"Error: {type(e).__name__}: {e}")
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

