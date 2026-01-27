# Add Approximate MAs to Batch Scanner

## Goal
Add Approximate MAs configuration to `standard_batch_scan_config.yaml` and implement it in the batch scanner workflow to enable 2-3x faster scanning for large symbol sets.

## Tasks

- [ ] Task 1: Add approximate MA fields to `ATCConfig` class in `modules/adaptive_trend_LTS/utils/config.py` → Verify: Class has `use_approximate`, `use_adaptive_approximate`, `approximate_volatility_window`, `approximate_volatility_factor` fields
- [ ] Task 2: Update `create_atc_config_from_dict()` to read approximate MA params from dict → Verify: Function extracts `use_approximate`, `use_adaptive_approximate`, etc. from params dict
- [ ] Task 3: Add approximate MA config section to `standard_batch_scan_config.yaml` under `atc_performance` → Verify: YAML has `use_approximate`, `use_adaptive_approximate`, `approximate_volatility_window`, `approximate_volatility_factor` keys
- [ ] Task 4: Update `workflow.py` to pass approximate MA params from `atc_performance` to `args` namespace → Verify: `args.use_approximate`, `args.use_adaptive_approximate`, etc. are set from `atc_performance` dict
- [ ] Task 5: Find where ATCConfig is created in VotingAnalyzer/ATCAnalyzer and ensure approximate MA params are passed → Verify: ATCConfig includes approximate MA params when created from args
- [ ] Task 6: Test with `use_approximate=True` in config → Verify: Scanner uses approximate MAs and runs 2-3x faster for large symbol sets

## Done When

- [ ] `standard_batch_scan_config.yaml` has approximate MA configuration options
- [ ] Batch scanner workflow passes approximate MA params from config to ATC computation
- [ ] Approximate MAs are used when enabled in config (verified by performance improvement)

## Notes

- Approximate MAs provide 2-3x speedup for large-scale scanning (1000+ symbols)
- Default should be `False` (full precision) for backward compatibility
- `use_approximate` and `use_adaptive_approximate` are mutually exclusive (adaptive takes precedence)
