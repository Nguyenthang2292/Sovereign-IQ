# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **binance_client**: Refactored into modular sub-package architecture (2026-02-11)
  - Split monolithic `binance_client.py` (793 lines) into focused modules:
    - `binance/exchange_setup.py` - CCXT exchange initialization
    - `binance/order_execution.py` - Market orders with TP/SL placement
    - `binance/position_management.py` - Position operations
    - `binance/order_management.py` - TP/SL modification and cancellation
    - `binance/client.py` - Main orchestrator with backward compatibility
  - Maintained 100% backward compatibility via legacy import layer
  - All 35 critical tests passing (trailing stop, fresh signal, order executor)
  - Benefits: Better separation of concerns, easier testing, improved maintainability
  - Added comprehensive README documenting new architecture

### Added
- **auto_trade**: Integration tests (Day 3)
  - End-to-end workflow tests (`tests/auto_trade/integration/test_e2e_workflows.py`)
    - Database init/migrate/insert/query full workflow
    - Signal pipeline with mocked components
    - Reconcile workflow with mocked Binance exchange
    - Backup create and verify workflow
  - Performance benchmarks (`tests/auto_trade/integration/test_performance_benchmarks.py`)
    - get_overall_stats with 10k+ orders (< 5s)
    - get_orders_cursor first page (< 1s)
    - Backup creation (< 10s for 10k-order DB)
    - Reconcile with mocked exchange (< 2s)
  - Stress tests (`tests/auto_trade/integration/test_stress.py`)
    - High-volume stats (5k+ orders)
    - Cursor pagination through large dataset
    - Concurrent stats reads
    - Concurrent reconcile calls (serialized via lock)

### Changed
- **auto_trade**: Week 5 Quality & Polish completion
  - Day 3 Integration Testing completed
  - Day 4 Final Review tasks documented

## [3.0.0] - Previous

- See project history for earlier releases.
