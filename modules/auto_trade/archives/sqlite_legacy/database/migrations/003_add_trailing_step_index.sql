-- ============================================================================
-- Database Migration: Add trailing_step_index to orders table
-- ============================================================================
--
-- Migration: 003_add_trailing_step_index.sql
-- Purpose: Add column to track trailing stop step index for step-based
--          trailing stop implementation (BE → +step% → +2*step% …)
--
-- Created: 2026-02-06
-- ============================================================================

-- ============================================================================
-- ADD COLUMN: trailing_step_index
-- ============================================================================

ALTER TABLE orders ADD COLUMN trailing_step_index INTEGER DEFAULT 0;

-- ============================================================================
-- INDEX for performance
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_orders_trailing_step ON orders(trailing_step_index);

-- ============================================================================
-- UPDATE existing orders to have default value
-- ============================================================================

UPDATE orders SET trailing_step_index = 0 WHERE trailing_step_index IS NULL;

-- ============================================================================
-- MIGRATION LOG
-- ============================================================================

INSERT OR REPLACE INTO system_state (key, value, value_type, description, category) VALUES
    ('last_migration', '003_add_trailing_step_index', 'string', 'Last applied migration', 'SYSTEM'),
    ('schema_version', '1.0.1', 'string', 'Database schema version', 'SYSTEM');

INSERT INTO audit_log (event_type, event_category, severity, event_summary, source_module) VALUES
    ('MIGRATION_APPLIED', 'SYSTEM', 'INFO', 'Migration 003_add_trailing_step_index applied successfully', '003_add_trailing_step_index.sql');
