# Module BINANCE SEND MARKET - Execution Module

Phase 3 implementation of the auto-trading system.

## 📋 Overview

The execution module handles market order execution on Binance Futures with comprehensive risk management, validation, and error handling.

## 🏗️ Architecture

```
modules/auto_trade/execution/
├── __init__.py              # Module exports
├── order_manager.py         # 3.1 Order execution orchestrator
├── order_builder.py         # 3.2 Order ticket builder
├── risk_manager.py          # 3.3 Risk management & position sizing
├── binance_client.py        # 3.4 CCXT Binance integration
└── order_validator.py       # 3.5 Pre/post-order validation
```

## 🔄 Order Execution Flow

```
Signal → Order Manager
    ↓
1. Check Open Positions (DataFetcher)
    ↓
2. Calculate Position Size (Risk Manager)
    ↓
3. Build Order Ticket (Order Builder)
    ↓
4. Fetch Current Price
    ↓
5. Pre-Order Validation (Order Validator)
    ↓
6. Execute Order (Binance Client)
    ├── Set Leverage
    ├── Create Market Order
    ├── Place TP Order
    └── Place SL Order
    ↓
7. Post-Order Validation
    ↓
8. Return Result
```

## 🚀 Quick Start

### Basic Usage

```python
from modules.auto_trade.execution.order_manager import OrderManager
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager

# Initialize
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager=exchange_manager)

order_manager = OrderManager(
    data_fetcher=data_fetcher,
    api_key="your_api_key",
    api_secret="your_api_secret",
    testnet=True,  # Use testnet for testing
    dry_run=True,  # Simulate without executing
)

# Execute signal
result = order_manager.execute_signal(signal)
```

### Testing

```bash
# Dry run (simulate orders)
python modules/auto_trade/test_execution_phase3.py --dry-run

# Testnet execution
python modules/auto_trade/test_execution_phase3.py --testnet

# With custom leverage
python modules/auto_trade/test_execution_phase3.py --dry-run --leverage 5
```

## 📦 Components

### 1. Order Manager (`order_manager.py`)

Orchestrates the entire order execution flow.

**Features:**
- Checks for open positions before trading
- Integrates all execution components
- Handles errors and provides detailed logging
- Supports emergency stop mechanism

**Methods:**
- `check_open_positions()`: Check for existing positions
- `execute_signal(signal)`: Execute a trading signal
- `emergency_stop(reason)`: Halt all trading

### 2. Order Builder (`order_builder.py`)

Builds order tickets from signals and risk parameters.

**Features:**
- Creates properly formatted order tickets
- Calculates TP/SL prices based on percentages
- Supports custom TP/SL and leverage overrides

**Order Ticket Structure:**
```python
OrderTicket(
    symbol="BTC/USDT",
    side="BUY",  # or "SELL"
    order_type="MARKET",
    amount=100.0,  # USDT
    leverage=2,
    take_profit_price=52500.0,  # TP = entry × (1 + 5%)
    stop_loss_price=25000.0,    # SL = entry × (1 - 50%)
)
```

### 3. Risk Manager (`risk_manager.py`)

Handles position sizing and risk management.

**Features:**
- Fetches account balance from Binance
- Calculates position size (95% of balance by default)
- Validates leverage limits
- Emergency stop mechanism
- Pre-flight safety checks

**Configuration:**
```python
RiskManager(
    data_fetcher=data_fetcher,
    balance_percentage=0.95,  # Use 95% of balance
    default_leverage=2,
    max_leverage=125,
    min_position_size=10.0,  # Minimum $10 USDT
)
```

### 4. Binance Client (`binance_client.py`)

CCXT-based client for Binance Futures API.

**Features:**
- Market order execution with TP/SL
- Automatic leverage setting
- Retry logic with exponential backoff
- Rate limiting protection
- Dry-run mode for testing

**Order Flow:**
1. Set leverage via API
2. Fetch current price
3. Calculate contract amount
4. Execute market order
5. Place take profit order (opposite side, reduce-only)
6. Place stop loss order (opposite side, reduce-only)
7. Return combined result

### 5. Order Validator (`order_validator.py`)

Validates orders before and after execution.

**Pre-Order Checks:**
- Sufficient balance
- Valid leverage (1x - 125x)
- Positive price
- Position size within limits
- Market is open
- Symbol exists

**Post-Order Checks:**
- Order filled successfully
- TP/SL orders placed
- Slippage within acceptable range
- Position opened correctly

## 🎯 Order Parameters

### Default Configuration

```python
{
    "balance_percentage": 0.95,      # Use 95% of available balance
    "default_leverage": 2,           # 2x leverage
    "default_tp_percentage": 5.0,    # 5% take profit
    "default_sl_percentage": 50.0,   # 50% stop loss
    "min_position_size": 10.0,       # Minimum $10 USDT
    "max_slippage_pct": 2.0,        # Max 2% slippage
}
```

### TP/SL Price Calculation

**LONG Position:**
```
TP Price = Entry Price × (1 + TP%)
SL Price = Entry Price  × (1 - SL%)

Example (Entry = $50,000):
TP = $50,000 × 1.05 = $52,500 (+5%)
SL = $50,000 × 0.50 = $25,000 (-50%)
```

**SHORT Position:**
```
TP Price = Entry Price × (1 - TP%)
SL Price = Entry Price × (1 + SL%)

Example (Entry = $50,000):
TP = $50,000 × 0.95 = $47,500 (-5%)
SL = $50,000 × 1.50 = $75,000 (+50%)
```

## 🔒 Safety Features

### 1. Position Check
- Prevents opening multiple positions simultaneously
- Can be overridden with `force_execution=True`

### 2. Emergency Stop
```python
# Trigger emergency stop
order_manager.emergency_stop("Daily loss limit exceeded")

# Check status
is_stopped = order_manager.is_emergency_stop_active

# Reset
order_manager.reset_emergency_stop()
```

### 3. Dry Run Mode
```python
# Test without executing real orders
order_manager = OrderManager(..., dry_run=True)
```

### 4. Pre-flight Checks
- Balance validation
- Leverage validation
- Market status check
- Price sanity check

### 5. Retry Logic
- Exponential backoff for failed requests
- Configurable max retries (default: 3)
- Separate retry for market, TP, and SL orders

## 🔧 Integration with Signal Pipeline

```python
from modules.auto_trade.core.signal_pipeline import SignalPipeline
from modules.auto_trade.execution.order_manager import OrderManager

# Run signal pipeline
final_signal = pipeline.run_pipeline()

if final_signal:
    # Execute the signal
    order_result = order_manager.execute_signal(final_signal)
    
    if order_result:
        print(f"✅ Order executed: {order_result['market_order']['id']}")
    else:
        print("❌ Order execution failed")
```

## 📊 Order Result Structure

```python
{
    "market_order": {
        "id": "12345678",
        "status": "closed",
        "symbol": "BTC/USDT",
        "side": "buy",
        "type": "market",
        "amount": 0.002,  # contracts
        "price": 50000.0,
        "average": 50025.0,  # filled price
        ...
    },
    "entry_price": 50025.0,
    "take_profit_order": {
        "id": "12345679",
        "type": "take_profit_market",
        "stopPrice": 52526.25,
        ...
    },
    "stop_loss_order": {
        "id": "12345680",
        "type": "stop_market",
        "stopPrice": 25012.5,
        ...
    },
    "order_ticket": {
        "symbol": "BTC/USDT",
        "side": "BUY",
        "amount": 100.0,
        "leverage": 2,
        ...
    }
}
```

## ⚠️ Important Notes

### API Credentials
- **NEVER** commit API keys to the repository
- Use environment variables or `.env` file
- For production, use IP whitelisting
- Enable 2FA on your Binance account

### Testnet vs Mainnet
- **Always test on testnet first**
- Testnet funds are not real
- Get testnet API keys from: https://testnet.binancefuture.com/
- Testnet may have different behavior than mainnet

### Rate Limits
- Binance has strict rate limits
- CCXT `enableRateLimit=True` helps but may not be enough
- Implement additional delays for high-frequency trading
- Monitor weight usage via API

### Position Sizing
- Default: 95% of balance with 2x leverage
- Actual position value = balance × 0.95 × 2 = 1.9× balance
- With $1000 balance: position = $1900 worth of contracts
- Be cautious with high leverage (max 125x)

### Slippage
- Market orders execute at market price
- Expect 0.1-0.5% slippage normally
- High volatility can cause larger slippage
- Max slippage check: 2% by default

## 🧪 Testing

### Unit Tests (TODO)
```bash
pytest tests/auto_trade/execution/
```

### Integration Tests
```bash
# Test with mock data
python modules/auto_trade/test_execution_phase3.py --dry-run

# Test on testnet
python modules/auto_trade/test_execution_phase3.py --testnet --force
```

## 📝 Implementation Checklist

Phase 3 Tasks:

**3.1 Order Execution Module** ✅
- [x] Integrate with DataFetcher's `fetch_binance_futures_positions()`
- [x] Check if position is open
- [x] Execute order if no position
- [x] Validate preconditions
- [x] Handle order conflicts
- [x] Track order lifecycle

**3.2 Order Builder** ✅
- [x] Build order ticket (symbol, type, side, amount, TP, SL, leverage)
- [x] Validate order parameters
- [x] Calculate precise TP/SL prices
- [x] Support custom TP/SL percentages

**3.3 Risk Manager** ✅
- [x] Fetch account balance
- [x] Calculate position size (95% balance)
- [x] Set leverage via API
- [x] Validate sufficient margin
- [x] Emergency stop mechanism
- [x] Pre-flight checks

**3.4 CCXT Integration** ✅
- [x] Implement `create_market_order_with_sl_tp()`
- [x] Handle API rate limits
- [x] Error handling & retry logic
- [x] Order confirmation verification
- [x] Support USDT-M futures
- [x] Detailed error messages
- [x] Log all order attempts

**3.5 Order Validation & Safety** ✅
- [x] Pre-order validation (balance, leverage, market, symbol, price)
- [x] Post-order validation (filled, TP/SL placed, slippage)
- [x] Comprehensive tests

## 🚧 Next Steps (Phase 4)

- Implement position monitoring (watch_out module)
- Break-even mechanism (30% drawdown → move TP to BE)
- Martingale strategy integration
- Position lifecycle handling
- Real-time updates via WebSocket

## 📚 References

- Binance Futures API: https://binance-docs.github.io/apidocs/futures/en/
- CCXT Documentation: https://docs.ccxt.com/
- Implementation Plan: `modules/auto_trade/docs/@auto_trade_implementation_plan.md`
