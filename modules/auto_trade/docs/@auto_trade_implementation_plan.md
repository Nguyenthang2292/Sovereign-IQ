# AUTO TRADING SYSTEM - DETAILED IMPLEMENTATION PLAN

## 📋 Project Overview

This document provides a comprehensive implementation plan for the Sovereign-IQ Auto Trading System. The system combines Machine Learning (ATC, XGBoost) with technical analysis (Gemini) to generate signals and execute trades on Binance Futures with risk management.

**Key Features**:

- Multi-stage signal filtering pipeline
- Automated market order execution with risk management
- Position monitoring with Martingale strategy
- Lightweight database for order tracking
- Real-time monitoring and alerting

---

## 📊 DETAILED TO-DO LIST - AUTO TRADING SYSTEM

### **Phase 1: Rust Backend Development** 🦀

**Mục tiêu**: Tăng hiệu suất xử lý cho các module tính toán nặng

#### Tasks

**1.1 Setup Rust Environment**

- [x] Cài đặt Rust toolchain và cargo
- [x] Setup PyO3 để tạo Python bindings
- [x] Tạo project structure: `rust_backend/` với các module tương ứng
- [x] Configure build system (setup.py + build.rs)

**1.2 Port adaptive_trend_LTS_mini to Rust**

- [x] Rewrite moving averages calculations (KAMA, EMA, SMA, WMA) trong Rust
- [x] Implement ATC signal computation với SIMD optimization
- [x] Port equity calculations sang Rust
- [x] Tạo Python bindings cho các functions chính
- [x] Benchmark Rust vs Python (target: 10-50x speedup)
- [x] Create unit tests cho Rust functions

**1.3 Port xgboost_LTS to Rust**

- [x] Port feature engineering sang Rust
- [x] Implement label calculation với compatibility
- [x] Create efficient data structures cho batch processing
- [x] Optimize memory allocation cho large datasets
- [x] Add performance tests

**Status Update (2026-02-01)**:

- Completed `sovereign_prime` crate integrating both `adaptive_trend_LTS_mini` and `xgboost_LTS`.
- Created `sync_rust.ps1` to sync code from Python modules to Rust backend.
- Verified functionality with `verify_rust_port.py`.
- Achieved significant speedup for critical calculations.

**Gợi ý tối ưu**:

- Sử dụng `rayon` crate cho parallel processing
- Dùng `ndarray` crate để tương thích với NumPy arrays
- SIMD intrinsics với `packed_simd` để tăng tốc calculations
- Memory-mapped files với `memmap2` cho large datasets
- Use `pyo3-numpy` cho seamless NumPy integration
- Implement compile-time optimizations (LTO, codegen-units=1)
- Use `criterion` crate cho micro-benchmarking

---

### **Phase 2: Module SIGNAL - Signal Filtering Pipeline** 🎯

**Mục tiêu**: Tạo pipeline lọc signal đa tầng với độ chính xác cao

#### Tasks

**2.1 Symbol Management & Random Sampling**

```
📁 modules/auto_trade/core/symbol_manager.py
```

- [x] Load toàn bộ symbols từ DataFetcher
- [x] Implement configurable random sampling (a% symbols)
- [x] Cache symbol list để giảm API calls
- [x] Filter theo volume/liquidity requirements
- [x] Add symbol whitelist/blacklist support
- [x] Periodic refresh of symbol list (daily)
- [x] Unit tests cho symbol filtering logic

**2.2 ATC Multi-Timeframe Scanner**

```
📁 modules/auto_trade/core/atc_scanner.py
```

- [x] Scan trên 3 timeframes: 5m, 15m, 1h
- [x] Aggregate signals từ 3 timeframes (weighted voting)
- [x] Lọc ra UP/DOWN signals → **Danh sách 1**
- [x] Implement concurrent scanning với ThreadPoolExecutor
- [x] Add signal confidence scoring
- [x] Cache OHLCV data để tránh duplicate fetches
- [x] Error handling cho individual symbol failures
- [x] Logging cho debugging

**2.3 XGBoost Signal Filter**

```
📁 modules/auto_trade/core/xgboost_filter.py
```

- [x] Load pre-trained XGBoost model
- [x] Input: Danh sách 1 từ ATC
- [x] Feature engineering cho từng symbol
- [x] Predict và filter → **Danh sách 2**
- [x] Output: Confidence scores
- [x] Model versioning (track which model used)
- [x] Add model performance metrics tracking
- [x] Handle model loading errors gracefully

**2.4 Gemini Chart Analyzer (Batch Mode)**

```
📁 modules/auto_trade/core/gemini_filter.py
```

- [x] Integrate với `gemini_chart_analyzer` batch mode
- [x] Disable browser report generation
- [x] Process Danh sách 2 in batches
- [x] Extract final signals → **Danh sách 3**
- [x] Handle rate limits (Google Gemini API)
- [x] Add retry logic cho failed analyses
- [x] Cache chart images để avoid reprocessing
- [x] Implement timeout handling

**2.5 Final Signal Selection**

```
📁 modules/auto_trade/core/signal_selector.py
```

- [x] Compare số lượng LONG vs SHORT signals
- [x] Nếu LONG > SHORT: chọn LONG với confidence cao nhất
- [x] Nếu SHORT > LONG: chọn SHORT với confidence cao nhất
- [x] Nếu LONG == SHORT: chọn signal có confidence cao hơn
- [x] Output: Single best signal với metadata
- [x] Add signal quality validation
- [x] Store signal history cho analysis

**2.6 Signal Pipeline Orchestration**

```
📁 modules/auto_trade/core/signal_pipeline.py
```

- [x] Orchestrate all 5 sub-modules
- [x] Handle errors at each stage
- [x] Implement retry logic
- [x] Add timeout mechanisms
- [x] Logging cho entire pipeline
- [x] Unit & integration tests

- [x] Cache ATC results cho 5 phút để tránh recalculation (In-Memory Cache)
- [x] Parallel processing cho multi-symbol analysis (ThreadPoolExecutor)
- [x] Implement signal quality scoring system (0-100)
- [x] Add signal persistence để track accuracy over time
- [ ] ~~Use Redis cache cho intermediate results (optional, replaced by In-Memory)~~
- [x] Implement circuit breaker pattern nếu API failures
- [x] Add health checks cho each stage
- [x] Monitor processing time per stage

---

**2.7 Logging & Monitoring Foundation** 📊

**Mục tiêu**: Build robust logging & monitoring system trước khi execute orders.

```
📁 modules/auto_trade/monitoring/
```

**2.7.1 Structured Logging System**

- [x] Configure Python logging with structured JSON output
- [x] Support multiple log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- [x] Separate log files: `signal.log`, `execution.log`, `position.log`, `error.log`, `audit.log`
- [x] Log rotation (daily or by size)
- [x] Contextual logging (correlation IDs)
- [x] Performance logging (execution time)

**2.7.2 Metrics Collection System**

- [x] In-memory metrics storage (with periodic persistence)
- [x] Counter metrics (signal count, order count, error count)
- [x] Gauge metrics (open positions, account balance)
- [x] Histogram metrics (latency)
- [x] Metrics export API (for future Prometheus integration)

**2.7.3 Audit Trail System**

- [x] Append-only audit log
- [x] Critical event tracking (Signal, Order, Position changes)
- [x] Cryptographic signatures (optional)
- [x] Query interface for audit analysis
- [x] Export to database (Phase 5 integration)

**2.7.4 Event Tracking System**

- [x] Publish-subscribe event system
- [x] Define Event types (`SIGNAL_GENERATED`, `ORDER_PLACED`, etc.)
- [x] Event history buffer
- [x] Event persistence
- [x] integration with logging & metrics

**2.7.5 Alert Management (Basic)**

- [x] Alert condition evaluation
- [x] Alert severity levels
- [x] Basic notification channels (Console, Log, Email)
- [x] Alert throttling
- [x] Define Alert Conditions (Pipeline timeout, API errors, etc.)

**2.7.6 System Health Checks**

- [x] Health check registry
- [x] Periodic health checks (API connectivity, Database, Memory)
- [x] Health status: HEALTHY, DEGRADED, UNHEALTHY
- [x] Health check HTTP endpoint `/health`

---

### **Phase 3: Module BINANCE SEND MARKET** 💹

**Mục tiêu**: Execute market orders với risk management

#### Tasks

**3.1 Order Execution Module**

```
📁 modules/auto_trade/execution/order_manager.py
```

- [ ] Integrate với DataFetcher's `fetch_binance_futures_positions()`
- [ ] Check if có position đang mở
- [ ] Nếu không có position → execute order
- [ ] Validate preconditions trước execution
- [ ] Handle order conflicts
- [ ] Track order lifecycle

**3.2 Order Builder**

```
📁 modules/auto_trade/execution/order_builder.py
```

- [ ] Build order ticket:
  - Symbol từ Module SIGNAL
  - Type: MARKET
  - Side: LONG/SHORT từ signal
  - Amount: 95% account balance
  - Take Profit: 5% (price calculation)
  - Stop Loss: 50% (price calculation)
  - Leverage: 2x
- [ ] Validate order parameters
- [ ] Calculate precise TP/SL prices
  - TP Price = Entry Price × (1 + 5%)
  - SL Price = Entry Price × (1 - 50%)
- [ ] Add order builder unit tests
- [ ] Support custom TP/SL percentages

**3.3 Risk Manager**

```
📁 modules/auto_trade/execution/risk_manager.py
```

- [ ] Fetch account balance trước khi order
- [ ] Calculate position size based on 95% balance
- [ ] Set leverage = 2x via API
- [ ] Validate sufficient margin
- [ ] Emergency stop mechanism
- [ ] Check max position size limits
- [ ] Validate leverage limits per symbol
- [ ] Pre-flight checks: market open, price valid, etc.

**3.4 CCXT Integration**

```
📁 modules/auto_trade/execution/binance_client.py
```

- [ ] Extend DataFetcher với order creation
- [ ] Implement `create_market_order_with_sl_tp()`
- [ ] Handle API rate limits with backoff
- [ ] Error handling & retry logic (exponential backoff)
- [ ] Order confirmation verification
- [ ] Support both USDT-M futures
- [ ] Add detailed error messages
- [ ] Log all order attempts (success/failure)

**3.5 Order Validation & Safety**

```
📁 modules/auto_trade/execution/order_validator.py
```

- [ ] Pre-order validation:
  - Sufficient balance
  - Valid leverage
  - Market is open
  - Symbol exists
  - Price sanity check
- [ ] Post-order validation:
  - Confirm order placement
  - Verify SL/TP placement
  - Check position opened
- [ ] Add comprehensive validation tests

**Gợi ý tối ưu**:

- [ ] Use atomic transactions cho order + SL/TP placement
- [ ] Implement pre-flight checks (margin, balance, market status)
- [ ] Add circuit breaker pattern để prevent rapid losses
- [ ] Log all orders với full metadata (timestamp, price, balance, etc.)
- [ ] Dry-run mode cho testing (simulate order without sending)
- [ ] Implement slippage protection (max acceptable slippage)
- [ ] Support batch order creation nếu multiple signals
- [ ] Add order deduplication logic
- [ ] Use WebSocket API cho real-time order status

---

### **Phase 4: Module BINANCE WATCH_OUT** 👁️

**Mục tiêu**: Monitor positions và implement Martingale strategy

#### Tasks

**4.1 Position Monitor**

```
📁 modules/auto_trade/monitoring/position_monitor.py
```

- [ ] Poll positions mỗi 5 giây (configurable)
- [ ] Check open positions count (ensure max 1)
- [ ] Calculate real-time P&L và drawdown
- [ ] Track position lifecycle
- [ ] Handle multiple timeframe updates
- [ ] Add position update callbacks
- [ ] Implement WebSocket listener (optional, for faster updates)

**4.2 Break-Even Manager**

```
📁 modules/auto_trade/monitoring/breakeven_manager.py
```

- [ ] Monitor drawdown của position
- [ ] Khi drawdown = 30% account → move TP to break-even
- [ ] Update database flag: `be_moved = True`
- [ ] Avoid duplicate BE moves (check flag before API call)
- [ ] Add configurable drawdown percentage
- [ ] Log BE move events
- [ ] Track BE move success/failure

**4.3 Market Scanner Scheduler**

```
📁 modules/auto_trade/monitoring/scanner_scheduler.py
```

- [ ] Nếu không có position nào → trigger Module SIGNAL mỗi 5 phút
- [ ] Nếu có signal mới → trigger Module BINANCE SEND MARKET
- [ ] Implement scheduler với APScheduler hoặc asyncio
- [ ] Support configurable scan intervals
- [ ] Add scheduler health checks
- [ ] Handle scheduler errors gracefully
- [ ] Log all scheduled events

**4.4 Martingale Strategy**

```
📁 modules/auto_trade/strategies/martingale.py
```

- [ ] Detect nếu position trước đó LOSS
- [ ] Khi đóng lệnh loss → ghi nhận lệnh số n1
- [ ] Lệnh tiếp theo: leverage = 2x lệnh trước
- [ ] Memory mechanism để track:
  - Số bước Martingale hiện tại
  - Tổng loss cần recover
  - Điều kiện dừng Martingale (max steps, max loss)
- [ ] Implement Martingale chain validation
- [ ] Add Martingale recovery calculator
- [ ] Unit tests cho Martingale logic

**4.5 Position Lifecycle Handler**

```
📁 modules/auto_trade/monitoring/lifecycle_handler.py
```

- [ ] Handle closed positions (profit/loss)
- [ ] Nếu profit: reset Martingale counter
- [ ] Nếu loss: increment Martingale và prepare next order
- [ ] Update database với trade results
- [ ] Calculate realized PnL
- [ ] Track win rate / loss rate
- [ ] Add lifecycle event callbacks

**4.6 Event System & Callbacks**

```
📁 modules/auto_trade/monitoring/event_system.py
```

- [ ] Position opened event
- [ ] Position closed event (profit/loss)
- [ ] BE moved event
- [ ] Martingale triggered event
- [ ] Error events
- [ ] Allow subscribers để listen to events

**Gợi ý tối ưu**:

- Use WebSocket API thay vì polling cho real-time updates
- Implement Exponential Martingale (1x → 2x → 4x → 8x...)
- Add max Martingale steps (e.g., 3-4 steps max)
- Safety mechanism: pause trading nếu daily loss > threshold
- Add notification system (Telegram bot) cho important events
- Circuit breaker: stop auto-trade nếu consecutive losses
- Cache position data để reduce API calls
- Implement exponential backoff cho failed API calls
- Add position update queue (async processing)

---

### **Phase 5: Module DATABASE** 🗄️

**Mục tiêu**: Lightweight, fast database cho order tracking

#### Tasks

**5.1 Database Selection & Setup**

- [ ] **Recommend: SQLite** (gọn nhẹ, zero-config, đủ cho single-instance bot)
  - Alternative: PostgreSQL nếu cần multi-instance scaling
- [ ] Setup database file: `data/auto_trade.db`
- [ ] Create database client wrapper
- [ ] Implement connection pooling (SQLAlchemy)
- [ ] Add WAL mode cho SQLite (better concurrent access)

**5.2 Schema Design**

```sql
-- modules/auto_trade/database/schema.sql

-- Bảng Orders: lưu tất cả orders
CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- 'LONG' or 'SHORT'
    entry_price REAL NOT NULL,
    amount REAL NOT NULL,
    leverage INTEGER NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    status TEXT NOT NULL,  -- 'OPEN', 'CLOSED', 'CANCELLED'
    pnl REAL DEFAULT 0,
    pnl_percentage REAL DEFAULT 0,
    be_moved BOOLEAN DEFAULT 0,  -- Break-even flag
    martingale_step INTEGER DEFAULT 0,
    parent_order_id TEXT,  -- Link to previous order in Martingale chain
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP
);

-- Bảng Signals: lưu signals từ pipeline
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,  -- 'LONG' or 'SHORT'
    confidence REAL NOT NULL,
    atc_score REAL,
    xgboost_score REAL,
    gemini_score REAL,
    executed BOOLEAN DEFAULT 0,  -- Whether signal was executed as order
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bảng Martingale_Chain: track Martingale sequence
CREATE TABLE martingale_chain (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chain_id TEXT NOT NULL,
    original_loss REAL NOT NULL,
    current_step INTEGER NOT NULL,
    total_loss REAL NOT NULL,
    recovered BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    recovered_at TIMESTAMP
);

-- Indexes
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_created ON orders(created_at);
CREATE INDEX idx_signals_created ON signals(created_at);
CREATE INDEX idx_signals_executed ON signals(executed);
CREATE INDEX idx_martingale_chain ON martingale_chain(chain_id);
```

**5.3 ORM/Query Layer**

```python
# modules/auto_trade/database/models.py
# modules/auto_trade/database/queries.py
```

- [ ] Implement Order model với SQLAlchemy
- [ ] Implement Signal model
- [ ] Implement MartingaleChain model
- [ ] CRUD operations cho orders
- [ ] Query methods:
  - `get_open_positions()`
  - `get_last_closed_order()`
  - `get_martingale_state(symbol)`
  - `update_order_status(order_id, status, pnl)`
  - `mark_be_moved(order_id)`
  - `save_signal(symbol, signal_type, confidence)`
  - `find_or_create_martingale_chain(chain_id)`
- [ ] Add database transaction support
- [ ] Implement query logging

**5.4 Migration & Backup**

```python
# modules/auto_trade/database/migrations.py
# modules/auto_trade/database/backup.py
```

- [ ] Auto-migration on startup (Alembic)
- [ ] Daily backup mechanism (automated)
- [ ] Database compaction/cleanup cho old records
- [ ] Implement database recovery procedures
- [ ] Add database integrity checks
- [ ] Version tracking cho schema

**5.5 Database Utilities**

```python
# modules/auto_trade/database/utils.py
```

- [ ] Database connection manager
- [ ] Transaction context manager
- [ ] Bulk insert operations
- [ ] Database statistics (size, record count, etc.)
- [ ] Data export functionality (CSV, JSON)
- [ ] Database reset/cleanup utilities (for testing)

**Gợi ý tối ưu**:

- Use connection pooling (SQLAlchemy)
- Implement write-ahead logging (WAL mode) cho SQLite
- Add database health checks
- Periodic archival của old trades (>30 days)
- Backup trước khi Martingale steps
- Add database metrics (query time, size)
- Use parameterized queries để prevent SQL injection
- Add database query optimization (indexes)
- Implement batch inserts cho performance
- Add data validation in ORM models

---

### **Phase 6: Integration & Testing** 🔗

**Mục tiêu**: Kết nối tất cả modules và test end-to-end

#### Tasks

**6.1 Main Auto-Trade Loop**

```python
# modules/auto_trade/main.py
```

- [ ] Initialize tất cả modules
- [ ] Create main event loop:
  1. Check open positions (Module WATCH_OUT)
  2. Nếu không có position → scan market (Module SIGNAL)
  3. Nếu có signal → execute order (Module SEND MARKET)
  4. Monitor positions → handle BE và Martingale
- [ ] Graceful shutdown handling
- [ ] Error recovery mechanisms
- [ ] Main loop logging
- [ ] Add health check endpoint

**6.2 Configuration Management**

```python
# modules/auto_trade/config.py
```

- [ ] Centralize all configs:
  - Scanning interval (default: 5 min)
  - Symbol sample percentage
  - Risk parameters (leverage, SL, TP)
  - Martingale settings (max steps, multiplier)
  - API credentials
- [ ] Support .env file và CLI arguments
- [ ] Config validation on startup
- [ ] Support config reloading (hot reload)
- [ ] Add config examples

**6.3 Unit Tests**

```
# tests/auto_trade/
```

- [ ] Test ATC scanner với mock data
- [ ] Test XGBoost filter
- [ ] Test order builder với mock balance
- [ ] Test Martingale calculation logic
- [ ] Test database operations
- [ ] Test signal selector logic
- [ ] Test position monitor
- [ ] Test BE move logic
- [ ] Target: >80% code coverage

**6.4 Integration Tests**

```
# tests/auto_trade/integration/
```

- [ ] Test full signal pipeline end-to-end
- [ ] Test order execution flow (với Binance testnet)
- [ ] Test position monitoring
- [ ] Test Martingale chain
- [ ] Test database operations in context
- [ ] Test module communication
- [ ] Stress test với concurrent signals
- [ ] Test error scenarios (API failures, network issues)

**6.5 Backtesting Module**

```python
# modules/auto_trade/backtest/simulator.py
```

- [ ] Historical data simulator
- [ ] Test strategy với historical signals
- [ ] Calculate metrics: win rate, Sharpe ratio, max drawdown
- [ ] Validate Martingale recovery rate
- [ ] Support multiple test scenarios
- [ ] Generate backtest reports
- [ ] Compare different configurations

**6.6 Testing Infrastructure**

- [ ] Setup pytest fixtures cho reusable test data
- [ ] Mock external APIs (Binance, Gemini)
- [ ] Use testnet API credentials for integration tests
- [ ] Implement dry-run mode for production testing
- [ ] Add performance benchmarks for signal pipeline
- [ ] Test error scenarios comprehensively
- [ ] Create test data generators

**Gợi ý tối ưu**:

- Use pytest fixtures cho reusable test data
- Mock external APIs (Binance, Gemini) trong unit tests
- Implement dry-run mode cho production testing
- Add performance benchmarks cho signal pipeline
- Test error scenarios (API failures, network issues)
- Use testnet extensively trước khi production (2-3 weeks minimum)
- Implement continuous integration (GitHub Actions)
- Add code coverage tracking
- Use property-based testing (hypothesis) cho complex logic

---

### **Phase 7: Deployment & Monitoring** 🚀

**Mục tiêu**: Deploy an toàn và monitor hiệu suất

#### Tasks

**7.1 Deployment Setup**

- [ ] Containerize với Docker:

  ```dockerfile
  # Dockerfile
  FROM python:3.12-slim
  # Install dependencies
  # Copy code
  # Run bot
  ```

- [ ] Setup docker-compose cho full stack
- [ ] Environment-specific configs (dev/staging/prod)
- [ ] Secrets management (API keys via env vars)
- [ ] Add deployment documentation
- [ ] Create deployment scripts

**7.2 Monitoring & Logging**

```python
# modules/auto_trade/monitoring/metrics.py
```

- [ ] Structured logging với Python logging module
  - Alternative: ELK stack hoặc CloudWatch for advanced setups
- [ ] Metrics tracking:
  - Signal generation rate
  - Order execution latency
  - P&L tracking (daily, weekly, monthly)
  - Martingale statistics
  - System health (CPU, memory, API rate limits)
  - Signal accuracy (over time)
  - Win rate / Loss rate
- [ ] Integrate với Prometheus + Grafana dashboard
- [ ] Add performance logging
- [ ] Add trade entry/exit logging

**7.3 Alerting System**

```python
# modules/auto_trade/alerts/notifier.py
```

- [ ] Telegram bot cho notifications:
  - New order executed
  - Position closed (profit/loss)
  - Break-even moved
  - Martingale triggered
  - Critical errors
- [ ] Email alerts cho critical failures
- [ ] PagerDuty/Opsgenie cho production incidents
- [ ] Alert throttling (prevent spam)
- [ ] Alert severity levels
- [ ] Alert history tracking

**7.4 Safety Mechanisms**

- [ ] Kill switch: manual stop trading (API endpoint or file flag)
- [ ] Daily loss limit: pause nếu vượt threshold
- [ ] API key rotation mechanism
- [ ] Rate limit monitoring
- [ ] Health checks endpoint (`/health`, `/metrics`)
- [ ] Circuit breaker pattern
- [ ] Graceful degradation
- [ ] Emergency position closing capability

**7.5 Documentation**

```markdown
# docs/auto_trade/
- SETUP.md
- CONFIGURATION.md
- TROUBLESHOOTING.md
- API.md
- ARCHITECTURE.md
- OPERATIONS.md
```

- [ ] Setup guide với examples
- [ ] Configuration reference
- [ ] Troubleshooting common issues
- [ ] Architecture diagrams
- [ ] API documentation
- [ ] Operations manual
- [ ] Runbooks for common tasks

**7.6 Performance & Reliability**

- [ ] Use systemd hoặc supervisor để auto-restart
- [ ] Implement graceful shutdown (close positions trước khi stop)
- [ ] Blue-green deployment cho zero-downtime updates
- [ ] Automated backups của database + logs
- [ ] Performance monitoring với monitoring tools
- [ ] Load testing
- [ ] Failure scenario testing

**Gợi ý tối ưu**:

- Use systemd hoặc supervisor để auto-restart
- Implement graceful shutdown (close positions trước khi stop)
- Blue-green deployment cho zero-downtime updates
- Automated backups của database + logs
- Performance monitoring với NewRelic hoặc Datadog (optional)
- A/B testing framework cho strategy variations
- Implement feature flags cho gradual rollouts
- Add detailed runbooks for operations team
- Setup monitoring dashboard (Grafana)
- Implement health check system (uptime monitoring)
- Add automated alerting for anomalies

---

## 🎯 Priority & Timeline Suggestion

### **Sprint 1 (Week 1-2)**: Phase 1 + Phase 2

**Focus**: Rust backend + Signal pipeline hoàn chỉnh

- Week 1: Setup Rust environment, port ATC to Rust
- Week 2: Port XGBoost, implement signal pipeline

**Deliverables**:

- Working signal pipeline
- Performance improvements from Rust
- Unit tests passing

### **Sprint 2 (Week 3)**: Phase 3 + Phase 5

**Focus**: Order execution + Database

- Setup database schema
- Implement order execution
- Database CRUD operations

**Deliverables**:

- Order execution on testnet
- Database operations working
- Integration tests passing

### **Sprint 3 (Week 4)**: Phase 4

**Focus**: Position monitoring + Martingale

- Implement position monitor
- Implement Martingale logic
- Break-even management

**Deliverables**:

- Position monitoring working
- Martingale strategy tested
- Watchout module functional

### **Sprint 4 (Week 5)**: Phase 6

**Focus**: Integration + Testing trên testnet

- Integration testing
- Backtest old signals
- Full end-to-end testing on testnet

**Deliverables**:

- Complete system tested on testnet
- >80% code coverage
- All integration tests passing

### **Sprint 5 (Week 6)**: Phase 7

**Focus**: Production deployment + Monitoring

- Setup Docker containers
- Setup monitoring/logging
- Deploy to production with small capital

**Deliverables**:

- System running in production
- Monitoring/alerts working
- Documentation complete

---

## ⚠️ Risk Warnings & Best Practices

### **Critical Safety Rules**

1. **Test Extensively trên Testnet**: Ít nhất 1-2 tuần trước khi live trading
2. **Start với Small Capital**: Test production với capital rất nhỏ (0.1% of target)
3. **Max Martingale Steps**: Không nên vượt quá 3-4 steps (risk exponential growth)
4. **Daily Loss Limit**: Pause trading nếu loss > 10-20% daily
5. **Manual Override**: Luôn có cách manual stop system (kill switch)
6. **API Key Security**: Never commit keys, use secrets manager / environment variables
7. **Monitor Continuously**: Đặc biệt trong 1-2 tuần đầu production
8. **Database Backups**: Daily backups before any risky operations

### **Testing Checklist**

- [ ] Unit test coverage >80%
- [ ] Integration tests on testnet: 1-2 weeks
- [ ] Backtest on historical data: all scenarios
- [ ] Small capital live trading: 1-2 weeks
- [ ] Gradual capital increase as system proves reliable

### **Operational Checklist**

- [ ] Real-time monitoring dashboard setup
- [ ] Alert system working (Telegram/Email)
- [ ] Manual kill switch tested
- [ ] Database backup system working
- [ ] Log aggregation setup
- [ ] On-call rotation for monitoring
- [ ] Incident response plan documented

### **Risk Parameters** (Recommended Defaults)

```python
# Risk settings
MAX_LEVERAGE = 2  # Never exceed 2x
MAX_POSITION_SIZE = 0.95  # 95% of account balance
STOP_LOSS_PERCENTAGE = 0.50  # 50% maximum loss per trade
TAKE_PROFIT_PERCENTAGE = 0.05  # 5% target profit
MAX_MARTINGALE_STEPS = 3  # Stop Martingale after 3 steps
DAILY_LOSS_LIMIT = 0.20  # Stop trading if daily loss > 20%
DRAWDOWN_THRESHOLD = 0.30  # Move SL to BE when drawdown > 30%
```

---

## 📁 Project Structure

```
modules/auto_trade/
├── __init__.py
├── main.py                          # Main entry point
├── config.py                        # Configuration management
├── core/
│   ├── __init__.py
│   ├── symbol_manager.py           # Symbol loading & sampling
│   ├── atc_scanner.py              # ATC signal generation
│   ├── xgboost_filter.py           # XGBoost signal filtering
│   ├── gemini_filter.py            # Gemini chart analysis
│   ├── signal_selector.py          # Final signal selection
│   └── signal_pipeline.py          # Pipeline orchestration
├── execution/
│   ├── __init__.py
│   ├── order_manager.py            # Order management
│   ├── order_builder.py            # Order ticket building
│   ├── risk_manager.py             # Risk validation
│   ├── binance_client.py           # Binance integration
│   └── order_validator.py          # Order validation
├── monitoring/
│   ├── __init__.py
│   ├── position_monitor.py         # Position monitoring
│   ├── breakeven_manager.py        # Break-even management
│   ├── scanner_scheduler.py        # Schedule scanning
│   ├── lifecycle_handler.py        # Position lifecycle
│   ├── event_system.py             # Event system
│   └── metrics.py                  # Metrics tracking
├── strategies/
│   ├── __init__.py
│   └── martingale.py               # Martingale strategy
├── database/
│   ├── __init__.py
│   ├── models.py                   # SQLAlchemy models
│   ├── queries.py                  # Query layer
│   ├── schema.sql                  # Database schema
│   ├── migrations.py               # Database migrations
│   ├── backup.py                   # Backup utilities
│   └── utils.py                    # Database utilities
├── alerts/
│   ├── __init__.py
│   └── notifier.py                 # Alert notifications
├── backtest/
│   ├── __init__.py
│   └── simulator.py                # Backtest simulator
├── utils/
│   ├── __init__.py
│   ├── logger.py                   # Logging setup
│   └── validators.py               # Input validators
├── tests/
│   ├── __init__.py
│   ├── test_atc_scanner.py
│   ├── test_xgboost_filter.py
│   ├── test_order_builder.py
│   ├── test_martingale.py
│   ├── test_database.py
│   ├── integration/
│   │   ├── test_signal_pipeline.py
│   │   └── test_full_system.py
│   └── conftest.py                 # Pytest fixtures
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## 🔧 Technology Stack

### **Backend**

- Python 3.12+
- FastAPI (for monitoring API)
- SQLAlchemy 2.0 (ORM)
- SQLite (primary database)
- CCXT (exchange integration)
- pandas/numpy (data processing)
- XGBoost (model training)
- PyO3/Rust (performance critical code)

### **Monitoring**

- Python logging
- Prometheus (metrics collection)
- Grafana (visualization)
- Telegram API (alerts)

### **Deployment**

- Docker & Docker Compose
- Systemd/Supervisor (process management)
- GitHub Actions (CI/CD)

### **Testing**

- pytest (unit tests)
- pytest-mock (mocking)
- hypothesis (property-based testing)
- pytest-cov (coverage)

---

## 📞 Support & Troubleshooting

### **Common Issues**

**Issue**: Signal pipeline timeout

- **Solution**: Increase timeout config, optimize filter performance, use Rust version

**Issue**: Order execution failures

- **Solution**: Check API credentials, verify margin availability, check market status

**Issue**: Database locks

- **Solution**: Enable WAL mode, reduce transaction duration, increase timeout

**Issue**: High memory usage

- **Solution**: Reduce batch size, implement pagination, use streaming

### **Debugging**

- Enable DEBUG logging: `LOG_LEVEL=DEBUG`
- Check logs: `tail -f logs/auto_trade.log`
- Monitor system resources: `htop`, `free -h`
- Check Binance API status
- Run backtest to reproduce issues

---

## 📚 References

- [Binance Futures API](https://binance-docs.github.io/apidocs/futures/en/)
- [CCXT Documentation](https://docs.ccxt.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

---

---

## 🔍 Critical Issues & Improvements Found

**Review Date**: 2026-02-01
**Reviewer**: System Architect

### **Critical Issues Found**

#### 1. **Race Condition in Position Checking (Phase 3 & 4)**

**Issue**: Phase 3 checks if position exists before placing order, but Phase 4 polls every 5 seconds. There's a race condition where:
- Signal pipeline generates signal (Phase 2)
- Order execution checks "no position" (Phase 3)
- Before order is placed, scanner_scheduler triggers again
- System might place duplicate orders

**Solution**:
```python
# Add to Phase 3.1 (Order Execution Module)
- [ ] **CRITICAL**: Implement distributed lock mechanism before order placement
  - Use file-based lock (simple) or Redis lock (distributed)
  - Lock key: "order_placement_lock"
  - Lock timeout: 30 seconds
  - Prevent concurrent order placement from multiple threads/processes

# Add to Phase 4.3 (Scanner Scheduler)
- [ ] **CRITICAL**: Check lock status before triggering signal pipeline
  - Skip scan if order placement is in progress
  - Log skip events for debugging
```

#### 2. **Insufficient Error Recovery in Signal Pipeline (Phase 2)**

**Issue**: Signal pipeline has retry logic but no circuit breaker. If Gemini API is down, system will keep retrying and waste API credits.

**Solution**:
```python
# Add to Phase 2.6 (Signal Pipeline Orchestration)
- [ ] **HIGH PRIORITY**: Implement circuit breaker for external APIs
  - Track failure rate per API (Binance, Gemini)
  - Open circuit after 5 consecutive failures
  - Half-open state after 5 minutes to test recovery
  - Log circuit state changes

- [ ] Add fallback strategy when Gemini is unavailable
  - Option 1: Skip Gemini filter, use ATC + XGBoost only
  - Option 2: Use cached Gemini results (if < 1 hour old)
  - Make fallback behavior configurable
```

#### 3. **Missing Signal Deduplication (Phase 2.5)**

**Issue**: Signal selector might generate the same signal multiple times if pipeline is triggered rapidly.

**Solution**:
```python
# Add to Phase 2.5 (Final Signal Selection)
- [ ] **MEDIUM PRIORITY**: Implement signal deduplication
  - Cache last N signals (symbol + signal_type + timestamp)
  - Check if identical signal was generated in last 15 minutes
  - If duplicate: log and skip execution
  - Store in memory or Redis for distributed setup
```

#### 4. **Dangerous Break-Even Logic (Phase 4.2)**

**Issue**: Current logic "when drawdown = 30% account → move TP to break-even" has a critical flaw:
- Drawdown of 30% means position is at -30% loss
- Moving TP to break-even when already at -30% loss locks in 30% loss
- This is NOT break-even, it's a guaranteed 30% loss!

**Correct Logic**:
```python
# Fix Phase 4.2 (Break-Even Manager)
- [ ] **CRITICAL FIX**: Correct break-even logic
  - Current (WRONG): "when drawdown = 30%, move TP to break-even"
  - Correct: "when position profit = +30%, move SL to entry (break-even)"

  # Correct implementation:
  if position_pnl_percentage >= 0.30:  # Position is at +30% profit
      new_sl = entry_price  # Move stop loss to entry price
      update_stop_loss(new_sl)
      mark_be_moved = True
```

#### 5. **Martingale Position Size Calculation Error (Phase 4.4)**

**Issue**: "Lệnh tiếp theo: leverage = 2x lệnh trước" is ambiguous and potentially dangerous.

**Clarification Needed**:
```python
# Add to Phase 4.4 (Martingale Strategy)
- [ ] **CRITICAL**: Clarify Martingale position sizing

  # Current ambiguity: Does "leverage = 2x" mean:
  # Option A: Double the leverage? (2x → 4x → 8x → 16x) - VERY DANGEROUS
  # Option B: Double the position size? (Keep leverage at 2x, increase amount)

  # Recommended: Option B with MAX limits
  - [ ] Martingale doubles POSITION SIZE, not leverage
  - [ ] Keep leverage constant at 2x (safer)
  - [ ] Position progression: 100% → 200% → 400% → 800% (of initial)
  - [ ] Hard limit: Max 800% of initial position (3 steps max)
  - [ ] Require sufficient account balance before Martingale step
  - [ ] Calculate: new_position_size = initial_size * (2 ^ martingale_step)
```

#### 6. **Missing Slippage Protection (Phase 3)**

**Issue**: Market orders can execute at significantly different prices in volatile markets.

**Solution**:
```python
# Add to Phase 3.2 (Order Builder)
- [ ] **HIGH PRIORITY**: Add slippage protection
  - Calculate acceptable price range: ±1% from current price
  - Use LIMIT order with price limit instead of pure MARKET order
  - Timeout after 10 seconds if order not filled
  - Cancel and retry with updated price if timeout
  - Log slippage for every order
```

#### 7. **Incomplete Position Synchronization (Phase 4.1)**

**Issue**: Polling every 5 seconds might miss rapid position changes (liquidation, stop-loss hit).

**Solution**:
```python
# Add to Phase 4.1 (Position Monitor)
- [ ] **MEDIUM PRIORITY**: Add WebSocket position updates
  - Subscribe to Binance User Data Stream
  - Receive real-time position updates, order fills, stop-loss triggers
  - Keep polling as fallback (every 30 seconds instead of 5)
  - Reconcile WebSocket data with polling data
  - Alert on discrepancies
```

#### 8. **No Position Reconciliation (Critical for Safety)**

**Issue**: System state (database) might diverge from exchange state (Binance).

**Solution**:
```python
# Add NEW task to Phase 4
**4.7 Position Reconciliation**

- [ ] **CRITICAL**: Implement position reconciliation
  - Every 1 minute: fetch positions from Binance
  - Compare with database records
  - Detect discrepancies:
    - Position exists on Binance but not in DB
    - Position exists in DB but not on Binance
    - Position parameters differ (size, leverage, SL, TP)
  - Alert on ANY discrepancy
  - Auto-sync: update DB from Binance (Binance is source of truth)
  - Log all reconciliation events
```

#### 9. **Missing Market Hours Check (Phase 3.5)**

**Issue**: Some symbols have trading restrictions or maintenance windows.

**Solution**:
```python
# Add to Phase 3.5 (Order Validation)
- [ ] **MEDIUM PRIORITY**: Add market status validation
  - Check if symbol is trading (not halted)
  - Check if futures market is in maintenance
  - Validate trading hours for symbol
  - Cache market status (1 minute TTL)
  - Skip order if market not available
```

#### 10. **Database Schema Missing Critical Fields**

**Issue**: Orders table missing important fields for auditing and analysis.

**Solution**:
```sql
# Add to Phase 5.2 (Schema Design) - Update orders table:

CREATE TABLE orders (
    -- Existing fields...

    -- ADD THESE CRITICAL FIELDS:
    signal_correlation_id TEXT,  -- Link to signal that triggered order
    expected_entry_price REAL,   -- Expected price vs actual
    actual_fill_price REAL,      -- Actual execution price
    slippage_percentage REAL,    -- Slippage tracker
    commission REAL,             -- Trading fees
    commission_asset TEXT,       -- Fee currency (BNB, USDT)
    execution_latency_ms INTEGER, -- Time from signal to execution
    market_conditions TEXT,      -- JSON: volatility, volume, spread
    rejection_reason TEXT,       -- If order was rejected, why?
    retry_count INTEGER DEFAULT 0, -- How many retries before success
    risk_score REAL,            -- Risk assessment at order time

    -- Original fields continue...
);

# Add indexes for new fields
CREATE INDEX idx_orders_correlation ON orders(signal_correlation_id);
CREATE INDEX idx_orders_slippage ON orders(slippage_percentage);
CREATE INDEX idx_orders_rejected ON orders(rejection_reason);
```

#### 11. **Missing Dry-Run Mode Implementation Detail**

**Issue**: "Dry-run mode" mentioned but not specified how to implement.

**Solution**:
```python
# Add NEW task to Phase 6.2 (Configuration Management)

**6.2.1 Dry-Run Mode Implementation**

- [ ] **HIGH PRIORITY**: Implement comprehensive dry-run mode
  - Environment variable: DRY_RUN=true
  - In dry-run mode:
    - Signal pipeline runs normally
    - Order building runs normally
    - Order execution: SIMULATE only, don't send to Binance
    - Simulate order fills at current market price
    - Simulate position in memory
    - Simulate P&L changes
    - Simulate stop-loss and take-profit triggers
    - Log all simulated actions
  - Use dry-run for:
    - Development testing
    - Production smoke testing (before real trades)
    - Strategy validation
  - Clear visual indicator in logs when in dry-run mode
```

#### 12. **No Capital Allocation Strategy for Martingale**

**Issue**: Using 95% balance per trade doesn't work with Martingale (need reserves for doubling).

**Solution**:
```python
# Add to Phase 3.3 (Risk Manager)
- [ ] **CRITICAL**: Implement Martingale-aware capital allocation

  # Current problem:
  # - Trade 1: Use 95% balance (e.g., $950 of $1000)
  # - Trade 1 loses: Balance = $475 (50% loss)
  # - Trade 2 needs: 2x of $950 = $1900 (NOT AVAILABLE!)

  # Solution: Reserve capital for potential Martingale steps
  - [ ] Calculate max position size considering Martingale steps:
    # Example with MAX_MARTINGALE_STEPS = 3:
    # Step 0: 1x position
    # Step 1: 2x position
    # Step 2: 4x position
    # Step 3: 8x position
    # Total needed: 1 + 2 + 4 + 8 = 15x initial position

    # If balance = $1000, max initial position:
    # initial_position = balance / 15 = $1000 / 15 = $66.67
    # This ensures we can complete full Martingale sequence

  - [ ] Make this configurable:
    - MARTINGALE_ENABLED: true/false
    - If enabled: use conservative position sizing
    - If disabled: can use 95% balance

  - [ ] Add safety check before each Martingale step:
    - Verify sufficient balance for next step
    - If insufficient: STOP Martingale, alert, wait for recovery
```

#### 13. **Timezone Issues Not Addressed**

**Issue**: System will run 24/7 across timezones, but no timezone handling specified.

**Solution**:
```python
# Add to Phase 6.2 (Configuration Management)
- [ ] **MEDIUM PRIORITY**: Implement timezone handling
  - Use UTC for all timestamps internally
  - Database timestamps: UTC only
  - Logs: UTC timestamps with ISO8601 format
  - User-facing displays: configurable timezone
  - Binance API: already uses UTC, ensure consistency
  - Add timezone conversion utilities
  - Never use local timezone in calculations
```

#### 14. **No API Rate Limit Tracking**

**Issue**: System might hit Binance API rate limits and get banned.

**Solution**:
```python
# Add to Phase 2.7.2 (Metrics Collection) and Phase 7.2
- [ ] **HIGH PRIORITY**: Implement API rate limit tracking
  - Track API calls per endpoint
  - Track rate limit headers from Binance responses
  - Alert when approaching limit (>80% used)
  - Implement exponential backoff when near limit
  - Add rate limit metrics to monitoring dashboard
  - Log rate limit violations
  - Automatic throttling when limits approached
```

#### 15. **Missing Correlation Between Signal Quality and Outcome**

**Issue**: No feedback loop to learn which signals are actually profitable.

**Solution**:
```python
# Add NEW task to Phase 5.3 (Query Methods)

**5.3.1 Signal Performance Tracking**

- [ ] **HIGH PRIORITY**: Link signals to outcomes for analysis
  - Add query: `get_signal_performance(symbol, timeframe, days=30)`
  - Calculate metrics per signal source:
    - ATC-only signals: win rate, avg profit
    - XGBoost-only signals: win rate, avg profit
    - Gemini-only signals: win rate, avg profit
    - Combined signals: win rate, avg profit
  - Track performance by confidence level
  - Generate weekly signal quality report
  - Use insights to adjust signal weights over time
  - Potential future: ML model to predict signal success
```

---

### **Additional Improvements & Recommendations**

#### 16. **Add Pre-Trade Risk Assessment**

```python
# Add NEW task to Phase 3.3 (Risk Manager)

**3.3.1 Pre-Trade Risk Assessment**

- [ ] **MEDIUM PRIORITY**: Calculate risk score before each trade
  - Factors to consider:
    - Current market volatility (VIX equivalent for crypto)
    - Symbol 24h volume (low volume = higher risk)
    - Recent price action (choppy = risky)
    - Time since last trade (avoid overtrading)
    - Current portfolio heat (total risk exposure)
    - Martingale step (higher step = higher risk)
  - Risk score: 0-100 (0=safest, 100=extreme risk)
  - Configurable risk threshold: reject trades > threshold
  - Store risk score in database for analysis
  - Alert on high-risk trades
```

#### 17. **Add Position Sizing Based on Signal Confidence**

```python
# Add to Phase 3.3 (Risk Manager)
- [ ] **ENHANCEMENT**: Variable position sizing by confidence
  - Current: Always use 95% balance (or Martingale-adjusted)
  - Improvement: Scale position by signal confidence
    - High confidence (>0.9): Use full calculated size
    - Medium confidence (0.7-0.9): Use 75% of calculated size
    - Low confidence (0.5-0.7): Use 50% of calculated size
    - Below 0.5: Reject (should not reach this stage)
  - Reduces risk on uncertain signals
  - Configurable: CONFIDENCE_SCALING_ENABLED
```

#### 18. **Add Trade Journal / Notebook**

```python
# Add NEW section to Phase 5

**5.6 Trade Journal**

- [ ] **ENHANCEMENT**: Implement automated trade journal
  - For each trade, capture:
    - Pre-trade: Setup, reasoning, market conditions
    - During trade: Position updates, emotions/notes
    - Post-trade: Outcome, lessons learned
  - Generate daily/weekly trading reports
  - Include charts, statistics, insights
  - Export to PDF or HTML
  - Useful for performance review and improvement
```

#### 19. **Add Multi-Symbol Position Support (Future)**

```python
# Add to "Gợi ý tối ưu" section in Phase 4

**Future Enhancement: Multi-Symbol Positions**

- [ ] Currently: Max 1 position at a time (any symbol)
- [ ] Future enhancement: Max N positions simultaneously
  - Requires portfolio-level risk management
  - Correlation analysis between positions
  - Diversification rules
  - Total exposure limits
  - More complex but better capital utilization
  - Recommended: Start with 1, expand after 3 months success
```

#### 20. **Add Paper Trading Mode**

```python
# Add to Phase 6.6 (Testing Infrastructure)

- [ ] **HIGH PRIORITY**: Implement paper trading mode
  - Different from dry-run: uses LIVE market data but simulated execution
  - Connects to Binance for real-time prices
  - Simulates order fills with realistic slippage
  - Simulates commission costs
  - Tracks paper portfolio separately
  - Perfect for testing in production environment without risk
  - Run paper trading parallel to live for comparison
  - Configuration: PAPER_TRADING=true
```

---

### **Implementation Priority Matrix**

| Priority | Tasks | Reason |
|----------|-------|--------|
| **P0 - Blocker** | Issues #1, #4, #5, #12 | Prevent loss of funds |
| **P1 - Critical** | Issues #2, #6, #8, #10, #14 | Major risk reduction |
| **P2 - High** | Issues #3, #7, #11, #15, #16, #20 | Quality & reliability |
| **P3 - Medium** | Issues #9, #13, #17 | Nice to have |
| **P4 - Enhancement** | Issues #18, #19 | Future improvements |

---

### **Pre-Production Checklist**

Before deploying to live trading, verify:

- [ ] All P0 issues resolved
- [ ] All P1 issues resolved
- [ ] Dry-run mode tested thoroughly
- [ ] Paper trading mode runs successfully for 2+ weeks
- [ ] Testnet trading runs successfully for 2+ weeks
- [ ] Position reconciliation working correctly
- [ ] All alerts tested and working
- [ ] Kill switch tested multiple times
- [ ] Database backups automated and tested
- [ ] Monitoring dashboard operational
- [ ] Rate limit tracking in place
- [ ] Break-even logic verified (issue #4 fix confirmed)
- [ ] Martingale capital allocation tested (issue #12 fix confirmed)
- [ ] Race condition protection tested (issue #1 fix confirmed)
- [ ] All critical logs reviewed for any anomalies

---

**Last Updated**: 2026-02-01
**Status**: Ready for Implementation (WITH CRITICAL FIXES REQUIRED)
**Next Review**: After P0 and P1 issues addressed
