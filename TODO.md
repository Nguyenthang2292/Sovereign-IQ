# Gemini Chart Batch Scanner Improvements

## Completed Tasks

- [x] **Priority 1: Fix User Prompt (HIGH)** - Updated `modules/gemini_chart_analyzer/cli/prompts/pre_filter.py` to accurately describe the 3-stage filtering process in "Fast" mode.
- [x] **Priority 2: Simplify Fast Mode Flag (HIGH)** - Updated `modules/gemini_chart_analyzer/core/prefilter_worker.py` to clarify that ML models are disabled for Stages 1-2 but force-enabled for Stage 3 in fast mode. This avoids ML overhead during initial filtering stages.
- [x] **Priority 3: Implement Score-Based Percentage Filter (MEDIUM)** - Modified `_filter_stage_3_ml_models` to return weighted scores from the voting system. Updated `run_prefilter_worker` to use these scores to sort symbols descending, ensuring the top $N\%$ selected are actually the highest quality signals.

## Next Steps

- [ ] Monitor pre-filter performance with the new 3-stage logic.
- [ ] Verify that the percentage filter correctly selects top-scoring symbols in real-world scans.
