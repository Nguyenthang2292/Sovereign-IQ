# Profiling-Guided Optimizations (Section 6)

## Goal
Introduce lightweight profiling workflows (cProfile + py-spy) to identify hot paths and guide targeted optimizations for `adaptive_trend_LTS`, focusing first on the benchmark comparison pipeline.

## Tasks
- [x] Task 1: Add profiling entrypoints → Create a small helper script or CLI option to run `benchmark_comparison/main.py` under `cProfile` and save `profile.stats` → Verify: running the command produces a non-empty `profile.stats` file.
- [x] Task 2: Add documentation for cProfile usage → Extend `modules/adaptive_trend_LTS/docs/optimization_suggestions.md` or a dedicated profiling doc with exact commands (`python -m cProfile -o profile.stats ...`) and how to inspect results (e.g. `snakeviz`, `pstats`) → Verify: doc reflects the current benchmark entrypoint (`benchmark_comparison/main.py` instead of old paths).
- [x] Task 3: Integrate py-spy flamegraph workflow → Add a short how-to section (or script snippet) for running `py-spy record -o profile.svg -- python ...` on the main benchmark/production-like entrypoints → Verify: `profile.svg` is generated and opens in a browser without errors.
- [x] Task 4: Define a minimal profiling checklist → Document a 3–5 step checklist for profiling new regressions (e.g. “1) run benchmark, 2) run cProfile, 3) run py-spy, 4) log top hot functions”) → Verify: checklist lives in docs and references the correct commands and files.
- [x] Task 5: Wire profiling into developer workflow → Optionally add a `make`/PowerShell task or a short `scripts/profile_benchmarks.py` to run both cProfile and py-spy with sane defaults → Verify: one command/script runs profiling end-to-end and writes outputs into a dedicated `profiles/` folder (gitignored).

## Done When
- [x] There is a repeatable way to generate `profile.stats` and `profile.svg` for the benchmark comparison pipeline.
- [x] Docs clearly show how to run and interpret these profiles.
- [x] Profiling artifacts (profiles/) are excluded from git and do not clutter the repo.

## Notes
- Keep profiling optional and developer-focused; don’t bake it into normal runs.
- Prefer simple text/HTML tooling (e.g. `snakeviz`, `py-spy`) that works cross-platform.
