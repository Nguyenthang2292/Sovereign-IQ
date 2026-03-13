# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added raw/execution signal contract documentation: `docs/RAW_EXEC_SIGNAL_CONTRACT.md`.
- Added execution-shift adapter helpers in `core/compute_atc_signals/execution_shift.py`:
  - `apply_execution_shift_series`
  - `apply_execution_shift_value`
- Added regression tests in `tests/test_execution_shift_contract.py` to lock:
  - raw signal remains unshifted in strategy mode
  - execution signal is exactly single-shift (`shift(1).fillna(0.0)`)
  - no double-shift in incremental flow

### Changed

- Refactored execution shift out of core signal engine:
  - `Average_Signal` is now always raw/causal output from core
  - shifted execution view is exposed as optional `Average_Signal_Exec`
- Updated batch/incremental compute flow to preserve raw parity and apply strategy shift only at adapter/output layer.
- Updated docs (`README.md`, `docs/API_REFERENCE.md`, `docs/setting_guides.md`, `docs/optimization_flow_diagram.md`) to reflect the raw vs exec contract.

### Fixed

- Fixed strategy-mode double-shift risk by separating raw signal generation from execution-shift presentation.
