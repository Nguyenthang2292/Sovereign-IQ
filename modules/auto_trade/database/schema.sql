-- AUTO TRADING SYSTEM - DATABASE SCHEMA
-- SQLite Database Schema for Order Tracking, Signal Management, and Martingale Strategy
-- Created: 2026-02-03

-- ============================================================================
-- TABLE: orders
-- Purpose: Store all orders created by the auto_trade system
-- Critical: Only PROGRAMMATIC orders (created by system) are stored here
-- ============================================================================

CREATE TABLE IF NOT EXISTS orders (
    -- Primary Key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Order Identification
    order_id TEXT UNIQUE NOT NULL,                    -- Binance order ID
    client_order_id TEXT UNIQUE,                      -- Client-side order ID (AT_ prefix for programmatic)
    symbol TEXT NOT NULL,                              -- Trading pair (e.g., 'BTCUSDT')
    
    -- Order Type and Direction
    side TEXT NOT NULL CHECK(side IN ('LONG', 'SHORT')),  -- Position direction
    order_type TEXT NOT NULL DEFAULT 'MARKET',         -- Order type: MARKET, LIMIT, etc.
    
    -- Order Source Tracking (CRITICAL for programmatic vs manual distinction)
    order_source TEXT NOT NULL DEFAULT 'PROGRAMMATIC' CHECK(order_source IN ('PROGRAMMATIC', 'MANUAL', 'EXTERNAL')),
    execution_mode TEXT NOT NULL DEFAULT 'AUTO' CHECK(execution_mode IN ('AUTO', 'MANUAL', 'EXTERNAL')),
    
    -- Pricing
    entry_price REAL NOT NULL,                         -- Actual fill price
    expected_entry_price REAL,                         -- Expected price (for slippage calculation)
    amount REAL NOT NULL,                              -- Position size (in base currency)
    notional_value REAL,                               -- Total value (amount * price)
    
    -- Risk Management
    leverage INTEGER NOT NULL DEFAULT 1,               -- Position leverage (1-125)
    stop_loss REAL,                                    -- Stop loss price
    take_profit REAL,                                  -- Take profit price
    stop_loss_percentage REAL,                         -- SL as % of entry
    take_profit_percentage REAL,                       -- TP as % of entry
    
    -- Order Status
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'OPEN', 'CLOSED', 'CANCELLED', 'FAILED')),
    
    -- P&L Tracking
    pnl REAL DEFAULT 0,                                -- Realized profit/loss (USDT)
    pnl_percentage REAL DEFAULT 0,                     -- P&L as percentage
    unrealized_pnl REAL DEFAULT 0,                     -- Current unrealized P&L
    commission REAL DEFAULT 0,                         -- Trading fees paid
    commission_asset TEXT DEFAULT 'USDT',              -- Fee currency
    
    -- Break-Even Management
    be_moved BOOLEAN DEFAULT 0,                        -- Flag: SL moved to break-even
    be_moved_at TIMESTAMP,                             -- When BE was triggered
    original_stop_loss REAL,                           -- Original SL before BE move

    -- Trailing Stop Step Management
    trailing_step_index INTEGER DEFAULT 0,             -- Current trailing stop step index
    
    -- Martingale Chain Tracking
    martingale_step INTEGER DEFAULT 0,                 -- Current Martingale step (0 = initial order)
    parent_order_id TEXT,                              -- Link to previous order in Martingale chain
    martingale_chain_id TEXT,                          -- Chain identifier for related orders
    is_martingale_recovery BOOLEAN DEFAULT 0,          -- Flag: is this a recovery order?
    
    -- Signal Correlation
    signal_correlation_id TEXT,                        -- Link to signal that triggered this order
    signal_confidence REAL,                            -- Confidence score of triggering signal
    
    -- Execution Metrics
    execution_latency_ms INTEGER,                      -- Time from signal to execution
    slippage_percentage REAL,                          -- Actual vs expected price difference
    retry_count INTEGER DEFAULT 0,                     -- Number of retries before success
    
    -- Market Conditions (JSON)
    market_conditions TEXT,                            -- JSON: {volatility, volume, spread, etc.}
    
    -- Risk Assessment
    risk_score REAL,                                   -- Pre-trade risk assessment (0-100)
    
    -- Failure Tracking
    rejection_reason TEXT,                             -- If order failed, why?
    error_message TEXT,                                -- Detailed error message
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,    -- When order was created in DB
    opened_at TIMESTAMP,                               -- When position opened on exchange
    closed_at TIMESTAMP,                               -- When position closed
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,    -- Last update time
    
    -- Foreign Keys
    FOREIGN KEY (parent_order_id) REFERENCES orders(order_id) ON DELETE SET NULL
);

-- ============================================================================
-- TABLE: signals
-- Purpose: Store all signals generated by the signal pipeline
-- ============================================================================

CREATE TABLE IF NOT EXISTS signals (
    -- Primary Key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Signal Identification
    correlation_id TEXT UNIQUE NOT NULL,               -- Unique ID for this signal
    symbol TEXT NOT NULL,                              -- Trading pair
    
    -- Signal Details
    signal_type TEXT NOT NULL CHECK(signal_type IN ('LONG', 'SHORT', 'NEUTRAL')),
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    
    -- Component Scores
    atc_score REAL CHECK(atc_score >= -1 AND atc_score <= 1),     -- ATC signal (-1 to 1)
    xgboost_score REAL CHECK(xgboost_score >= 0 AND xgboost_score <= 1),  -- XGBoost confidence
    gemini_score REAL CHECK(gemini_score >= 0 AND gemini_score <= 1),     -- Gemini confidence
    final_score REAL,                                  -- Weighted final score
    
    -- Signal Quality
    signal_quality TEXT CHECK(signal_quality IN ('HIGH', 'MEDIUM', 'LOW')),
    quality_score REAL,                                -- 0-100 quality metric
    
    -- Timeframe Analysis
    timeframe_5m_signal TEXT,                          -- 5-minute timeframe signal
    timeframe_15m_signal TEXT,                         -- 15-minute timeframe signal
    timeframe_1h_signal TEXT,                          -- 1-hour timeframe signal
    timeframe_consensus TEXT,                          -- Consensus across timeframes
    
    -- Market Context (JSON)
    market_context TEXT,                               -- JSON: price, volume, volatility
    
    -- Execution Status
    executed BOOLEAN DEFAULT 0,                        -- Was this signal executed as order?
    execution_order_id TEXT,                           -- Link to orders table
    rejected BOOLEAN DEFAULT 0,                        -- Was signal rejected?
    rejection_reason TEXT,                             -- Why was signal rejected?
    
    -- Outcome Tracking (for learning)
    outcome TEXT CHECK(outcome IN ('WIN', 'LOSS', 'BREAKEVEN', 'PENDING', NULL)),
    outcome_pnl REAL,                                  -- Final P&L if executed
    outcome_duration_minutes INTEGER,                  -- How long until outcome determined
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP,
    outcome_at TIMESTAMP,
    
    -- Foreign Keys
    FOREIGN KEY (execution_order_id) REFERENCES orders(order_id) ON DELETE SET NULL
);

-- ============================================================================
-- TABLE: martingale_chain
-- Purpose: Track Martingale recovery sequences
-- ============================================================================

CREATE TABLE IF NOT EXISTS martingale_chain (
    -- Primary Key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Chain Identification
    chain_id TEXT UNIQUE NOT NULL,                     -- Unique chain identifier
    symbol TEXT NOT NULL,                              -- Trading pair for this chain
    
    -- Chain Status
    status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE', 'RECOVERED', 'FAILED', 'CANCELLED')),
    
    -- Loss Tracking
    original_loss REAL NOT NULL,                       -- Initial loss to recover
    current_step INTEGER NOT NULL DEFAULT 0,           -- Current Martingale step (0-based)
    max_step_reached INTEGER DEFAULT 0,                -- Highest step reached
    total_loss REAL NOT NULL DEFAULT 0,                -- Cumulative loss in chain
    total_recovery REAL DEFAULT 0,                     -- Total recovered so far
    
    -- Recovery Status
    recovered BOOLEAN DEFAULT 0,                       -- Has chain recovered?
    recovery_pnl REAL DEFAULT 0,                       -- Final recovery P&L
    
    -- Safety Limits
    max_allowed_steps INTEGER DEFAULT 4,               -- Maximum steps allowed
    max_allowed_loss REAL,                             -- Maximum total loss allowed
    
    -- Order References
    initial_order_id TEXT,                             -- First order in chain
    latest_order_id TEXT,                              -- Most recent order
    recovery_order_id TEXT,                            -- Order that achieved recovery
    
    -- Chain Metadata
    leverage_progression TEXT,                         -- JSON: [2, 4, 8, 16]
    position_size_progression TEXT,                    -- JSON: position sizes per step
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    recovered_at TIMESTAMP,
    failed_at TIMESTAMP,
    
    -- Foreign Keys
    FOREIGN KEY (initial_order_id) REFERENCES orders(order_id) ON DELETE SET NULL,
    FOREIGN KEY (latest_order_id) REFERENCES orders(order_id) ON DELETE SET NULL,
    FOREIGN KEY (recovery_order_id) REFERENCES orders(order_id) ON DELETE SET NULL
);

-- ============================================================================
-- TABLE: system_state
-- Purpose: Store system-wide state and configuration
-- ============================================================================

CREATE TABLE IF NOT EXISTS system_state (
    -- Primary Key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- State Key-Value
    key TEXT UNIQUE NOT NULL,                          -- State key (e.g., 'last_scan_time')
    value TEXT NOT NULL,                               -- State value (JSON for complex types)
    value_type TEXT DEFAULT 'string' CHECK(value_type IN ('string', 'integer', 'float', 'boolean', 'json')),
    
    -- Metadata
    description TEXT,                                  -- What this state represents
    category TEXT,                                     -- State category for grouping
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TABLE: audit_log
-- Purpose: Comprehensive audit trail for all system actions
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit_log (
    -- Primary Key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Event Classification
    event_type TEXT NOT NULL,                          -- Type of event (ORDER_CREATED, SIGNAL_GENERATED, etc.)
    event_category TEXT NOT NULL,                      -- Category: TRADING, SYSTEM, RISK, etc.
    severity TEXT NOT NULL CHECK(severity IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    
    -- Event Details
    event_summary TEXT NOT NULL,                       -- Human-readable summary
    event_data TEXT,                                   -- JSON: full event data
    
    -- Correlation
    correlation_id TEXT,                               -- Link to related entities
    order_id TEXT,                                     -- Related order if applicable
    signal_id TEXT,                                    -- Related signal if applicable
    
    -- Source
    source_module TEXT,                                -- Which module generated event
    source_function TEXT,                              -- Which function
    
    -- Result
    success BOOLEAN DEFAULT 1,                         -- Was action successful?
    error_message TEXT,                                -- Error if failed
    
    -- Timestamps
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES for Performance Optimization
-- ============================================================================

-- Orders Table Indexes
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_source ON orders(order_source);           -- Filter programmatic vs manual
CREATE INDEX IF NOT EXISTS idx_orders_execution_mode ON orders(execution_mode); -- Track execution method
CREATE INDEX IF NOT EXISTS idx_orders_client_id ON orders(client_order_id);     -- Fast lookup by client ID
CREATE INDEX IF NOT EXISTS idx_orders_martingale_chain ON orders(martingale_chain_id);
CREATE INDEX IF NOT EXISTS idx_orders_signal_correlation ON orders(signal_correlation_id);
CREATE INDEX IF NOT EXISTS idx_orders_parent ON orders(parent_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_status ON orders(symbol, status);  -- Composite for common queries
CREATE INDEX IF NOT EXISTS idx_orders_trailing_step ON orders(trailing_step_index);  -- Trailing stop step index

-- Signals Table Indexes
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_executed ON signals(executed);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_correlation ON signals(correlation_id);
CREATE INDEX IF NOT EXISTS idx_signals_execution_order ON signals(execution_order_id);
CREATE INDEX IF NOT EXISTS idx_signals_outcome ON signals(outcome);

-- Martingale Chain Indexes
CREATE INDEX IF NOT EXISTS idx_martingale_chain_id ON martingale_chain(chain_id);
CREATE INDEX IF NOT EXISTS idx_martingale_status ON martingale_chain(status);
CREATE INDEX IF NOT EXISTS idx_martingale_symbol ON martingale_chain(symbol);
CREATE INDEX IF NOT EXISTS idx_martingale_created ON martingale_chain(created_at DESC);

-- System State Indexes
CREATE INDEX IF NOT EXISTS idx_system_state_key ON system_state(key);
CREATE INDEX IF NOT EXISTS idx_system_state_category ON system_state(category);

-- Audit Log Indexes
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_correlation ON audit_log(correlation_id);
CREATE INDEX IF NOT EXISTS idx_audit_order_id ON audit_log(order_id);
CREATE INDEX IF NOT EXISTS idx_audit_severity ON audit_log(severity);

-- ============================================================================
-- TRIGGERS for Automatic Timestamp Updates
-- ============================================================================

-- NOTE: Timestamp triggers have been removed for performance reasons.
-- SQLAlchemy handles timestamp updates via onupdate parameter in models.py:
--   updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
-- This approach is more efficient as it avoids an additional UPDATE statement per operation.

-- ============================================================================
-- VIEWS for Common Queries
-- ============================================================================

-- Active programmatic orders only
CREATE VIEW IF NOT EXISTS v_active_programmatic_orders AS
SELECT * FROM orders
WHERE 
    order_source = 'PROGRAMMATIC' 
    AND status IN ('OPEN', 'PENDING')
ORDER BY created_at DESC;

-- Recent signals with outcomes
CREATE VIEW IF NOT EXISTS v_signal_performance AS
SELECT 
    s.correlation_id,
    s.symbol,
    s.signal_type,
    s.confidence,
    s.atc_score,
    s.xgboost_score,
    s.gemini_score,
    s.executed,
    s.outcome,
    s.outcome_pnl,
    s.created_at,
    o.order_id,
    o.pnl as actual_pnl
FROM signals s
LEFT JOIN orders o ON s.execution_order_id = o.order_id
WHERE s.executed = 1
ORDER BY s.created_at DESC;

-- Active Martingale chains
CREATE VIEW IF NOT EXISTS v_active_martingale_chains AS
SELECT * FROM martingale_chain
WHERE status = 'ACTIVE'
ORDER BY created_at DESC;

-- Daily statistics
CREATE VIEW IF NOT EXISTS v_daily_stats AS
SELECT 
    DATE(created_at) AS trade_date,
    COUNT(*) AS total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS winning_trades,
    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS losing_trades,
    SUM(pnl) AS total_pnl,
    AVG(pnl) AS avg_pnl,
    MAX(pnl) AS best_trade,
    MIN(pnl) AS worst_trade,
    SUM(commission) AS total_fees
FROM orders
WHERE 
    order_source = 'PROGRAMMATIC' 
    AND status = 'CLOSED'
GROUP BY DATE(created_at)
ORDER BY trade_date DESC;

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert default system state entries
INSERT OR IGNORE INTO system_state (key, value, value_type, description, category) VALUES
    ('schema_version', '1.0.0', 'string', 'Database schema version', 'SYSTEM'),
    ('last_migration', 'initial', 'string', 'Last applied migration', 'SYSTEM'),
    ('trading_enabled', 'false', 'boolean', 'Is automated trading enabled?', 'TRADING'),
    ('last_scan_time', '0', 'integer', 'Last market scan timestamp', 'TRADING'),
    ('total_trades', '0', 'integer', 'Total number of trades', 'STATS'),
    ('total_pnl', '0.0', 'float', 'Total realized P&L', 'STATS'),
    ('last_backup_time', '0', 'integer', 'Last database backup timestamp', 'SYSTEM');

-- Insert initial audit log entry
INSERT INTO audit_log (event_type, event_category, severity, event_summary, source_module) VALUES
    ('SCHEMA_INITIALIZED', 'SYSTEM', 'INFO', 'Database schema initialized successfully', 'schema.sql');

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
