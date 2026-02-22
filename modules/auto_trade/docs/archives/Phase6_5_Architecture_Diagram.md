```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AUTO-TRADE BACKTESTING ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────────────────┘

                          User Code / Test Script
                                    │
                                    │ Creates
                                    ▼
                     ┌──────────────────────────────┐
                     │   AutoTradeBacktester        │
                     │   (Adapter Layer)            │
                     │                              │
                     │  • Auto-trade parameters     │
                     │    - 50% SL, 5% TP           │
                     │    - 95% risk per trade      │
                     │    - 2x leverage             │
                     │  • BE protection config      │
                     │  • Martingale config         │
                     └──────────────┬───────────────┘
                                    │
                                    │ Wraps
                                    ▼
           ┌─────────────────────────────────────────────────┐
           │         FullBacktester                          │
           │         (from modules/backtester)               │
           │                                                 │
           │  ┌───────────────────────────────────────────┐  │
           │  │  1. Fetch OHLCV Data                      │  │
           │  │     ├─ DataFetcher                        │  │
           │  │     └─ Historical price data              │  │
           │  └───────────────────────────────────────────┘  │
           │                     │                           │
           │                     ▼                           │
           │  ┌───────────────────────────────────────────┐  │
           │  │  2. Calculate Signals                     │  │
           │  │     ├─ HybridSignalCalculator             │  │
           │  │     ├─ Single signal mode (highest conf)  │  │
           │  │     └─ Returns signal series              │  │
           │  └───────────────────────────────────────────┘  │
           │                     │                           │
           │                     ▼                           │
           │  ┌───────────────────────────────────────────┐  │
           │  │  3. Simulate Trades                       │  │
           │  │     ├─ Entry/Exit logic                   │  │
           │  │     ├─ SL/TP/Trailing stop                │  │
           │  │     ├─ Max hold periods                   │  │
           │  │     └─ Returns trades list                │  │
           │  └───────────────────────────────────────────┘  │
           │                     │                           │
           │                     ▼                           │
           │  ┌───────────────────────────────────────────┐  │
           │  │  4. Calculate Standard Metrics            │  │
           │  │     ├─ Win rate                           │  │
           │  │     ├─ Sharpe ratio                       │  │
           │  │     ├─ Max drawdown                       │  │
           │  │     └─ Profit factor                      │  │
           │  └───────────────────────────────────────────┘  │
           └─────────────────────────────────────────────────┘
                                    │
                                    │ Returns trades + metrics
                                    ▼
           ┌────────────────────────────────────────────────┐
           │   AutoTradeBacktester Post-Processing          │
           │                                                │
           │  ┌───────────────────────────────────────────┐ │
           │  │  5. Break-Even Protection Simulation      │ │
           │  │     ├─ Monitor drawdown per trade         │ │
           │  │     ├─ If drawdown >= 30%                 │ │
           │  │     │   └─ Move TP to break-even          │ │
           │  │     └─ Mark "be_moved" flag               │ │
           │  └───────────────────────────────────────────┘ │
           │                     │                          │
           │                     ▼                          │
           │  ┌───────────────────────────────────────────┐ │
           │  │  6. Martingale Strategy Simulation        │ │
           │  │     ├─ Detect consecutive losses          │ │
           │  │     ├─ Double leverage after loss         │ │
           │  │     │   (2x → 4x → 8x → 16x)              │ │
           │  │     ├─ Reset after profit                 │ │
           │  │     └─ Track martingale_step              │ │
           │  └───────────────────────────────────────────┘ │
           │                     │                          │
           │                     ▼                          │
           │  ┌───────────────────────────────────────────┐ │
           │  │  7. Recalculate Metrics                   │ │
           │  │     ├─ Update with BE/Martingale changes  │ │
           │  │     ├─ Add auto-trade metrics:            │ │
           │  │     │   • breakeven_moves                 │ │
           │  │     │   • martingale_trades               │ │
           │  │     │   • max_martingale_step             │ │
           │  │     │   • leverage_used                   │ │
           │  │     └─ Recalculate equity curve           │ │
           │  └───────────────────────────────────────────┘ │
           │                     │                          │
           │                     ▼                          │
           │  ┌───────────────────────────────────────────┐ │
           │  │  8. Safety Validation                     │ │
           │  │     ├─ Max consecutive losses check       │ │
           │  │     ├─ Max leverage check                 │ │
           │  │     ├─ Max steps check                    │ │
           │  │     └─ Return safety metrics              │ │
           │  └───────────────────────────────────────────┘ │
           └────────────────────────────────────────────────┘
                                    │
                                    │ Returns final result
                                    ▼
                        ┌───────────────────────┐
                        │   Backtest Results    │
                        │                       │
                        │  • Trades list        │
                        │  • Equity curve       │
                        │  • Standard metrics   │
                        │  • Auto-trade metrics │
                        │  • Safety metrics     │
                        └───────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                           KEY INTEGRATION POINTS                            │
└─────────────────────────────────────────────────────────────────────────────┘

1. REUSE EXISTING INFRASTRUCTURE
   • Leverages FullBacktester from modules/backtester
   • No duplication of signal calculation logic
   • Benefits from existing optimizations

2. ADAPTER PATTERN
   • AutoTradeBacktester wraps FullBacktester
   • Injects auto-trade specific parameters
   • Post-processes trades for BE/Martingale

3. SAFE DEFAULTS
   • Break-even protection: ENABLED by default ✅
   • Martingale strategy: DISABLED by default ⚠️
   • Conservative risk management

4. COMPREHENSIVE METRICS
   • All standard backtesting metrics preserved
   • Additional auto-trade specific metrics
   • Safety validation for Martingale

5. EXTENSIBILITY
   • Easy to add new strategies
   • Configurable parameters
   • Modular design for future enhancements


┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXAMPLE USAGE FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    User initiates backtest
            │
            ▼
    AutoTradeBacktester.backtest_strategy()
            │
            ├─► [Step 1-4] FullBacktester runs standard backtest
            │       │
            │       ├─► Fetch data
            │       ├─► Calculate signals
            │       ├─► Simulate trades
            │       └─► Calculate standard metrics
            │
            ├─► [Step 5] Apply break-even protection simulation
            │       └─► Updates trades with "be_moved" flag
            │
            ├─► [Step 6] Apply Martingale strategy simulation (if enabled)
            │       └─► Updates trades with leverage scaling
            │
            ├─► [Step 7] Recalculate metrics
            │       └─► Add auto-trade specific metrics
            │
            └─► [Step 8] Validate safety (optional)
                    └─► Returns safety metrics
            │
            ▼
    Return comprehensive backtest results
```
