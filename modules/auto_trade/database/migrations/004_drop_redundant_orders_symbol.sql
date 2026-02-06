-- Migration: 004_drop_redundant_orders_symbol
-- Description: Drop redundant index idx_orders_symbol (covered by idx_orders_symbol_status and idx_orders_symbol_closed).
-- Ref: REFACTORING_RECOMMENDATIONS.md § Remove Redundant Index

DROP INDEX IF EXISTS idx_orders_symbol;
