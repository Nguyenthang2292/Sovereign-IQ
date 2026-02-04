"""
Database Schema for Gradual Recovery Table
============================================

SQL schema for tracking gradual recovery sequences.

This migration creates the `gradual_recovery` table for tracking
loss recovery progress using controlled scaling approach.
"""

-- ============================================================================
-- GRADUAL RECOVERY TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS gradual_recovery (
    -- Primary Key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Recovery Identification
    recovery_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    
    -- Recovery Status
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    
    -- Loss Tracking
    initial_loss REAL NOT NULL,
    remaining_loss REAL NOT NULL,
    total_profit_accumulated REAL DEFAULT 0.0,
    recovery_percentage REAL DEFAULT 0.0,
    
    -- Trade Tracking
    trades_count INTEGER DEFAULT 0,
    win_streak INTEGER DEFAULT 0,
    estimated_trades_remaining INTEGER DEFAULT 0,
    
    -- Configuration (JSON)
    config_data TEXT,  -- Stores RecoveryConfig as JSON
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    failed_at TIMESTAMP,
    
    -- Constraints
    CHECK (status IN ('ACTIVE', 'COMPLETE', 'FAILED', 'CANCELLED'))
);

-- ============================================================================
-- INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_gradual_recovery_recovery_id ON gradual_recovery(recovery_id);
CREATE INDEX IF NOT EXISTS idx_gradual_recovery_symbol ON gradual_recovery(symbol);
CREATE INDEX IF NOT EXISTS idx_gradual_recovery_status ON gradual_recovery(status);
CREATE INDEX IF NOT EXISTS idx_gradual_recovery_created_at ON gradual_recovery(created_at);
