# CLAUDE.md — Sovereign-IQ (crypto-probability)

## Project Overview
Sovereign-IQ là hệ thống phân tích giao dịch crypto/forex toàn diện, kết hợp Machine Learning, Deep Learning và Quantitative Strategies.

- **Package**: `crypto-probability` v3.0.0
- **Python**: ≥3.12
- **Package manager**: `uv` (preferred) hoặc pip
- **Build backend**: setuptools + maturin (Rust extension)

---

## Architecture

```
crypto-probability/
├── modules/               # Core analysis modules
│   ├── adaptive_trend/    # ATC strategy
│   ├── adaptive_trend_LTS/# ATC-LTS với Rust/CUDA
│   ├── auto_trade/        # Auto-trading engine
│   ├── backtester/        # Backtesting framework
│   ├── deeplearning/      # TFT, LSTM, CNN-LSTM
│   ├── hmm/               # Hidden Markov Models
│   ├── xgboost/           # XGBoost models
│   ├── random_forest/     # Random Forest
│   ├── order_book/        # Order Book Imbalance
│   ├── gemini_chart_analyzer/ # Gemini AI chart analysis
│   ├── common/            # Shared utilities & UI
│   └── ...
├── core/                  # Core infrastructure
├── tests/                 # Test suite
├── web/                   # FastAPI web server
├── cli/                   # CLI interface
├── rust_backend/          # Rust extension (atc_rust)
├── main.py                # Main entry point
└── pyproject.toml         # Project config
```

---

## Commands

### Setup
```bash
uv sync                              # Install all dependencies
uv pip install -e .                  # Editable install
```

### Running
```bash
python main.py                       # Main interactive menu
python main_complex_hybrid.py        # Hybrid analyzer
python main_complex_voting.py        # Voting analyzer
python main_complex_atc_oscillator.py # ATC Oscillator
python run_auto_trade_gui.py         # Auto-trade GUI
```

### Testing
```bash
# Preferred: use scripts (activates .venv automatically)
run_tests.bat                        # Windows
run_tests.ps1                        # PowerShell

# Manual
.venv\Scripts\python -m pytest       # Windows
.venv/bin/python -m pytest           # Unix
python -m pytest tests/ -v           # With verbose
python -m pytest -m unit             # Unit tests only
python -m pytest -m "not slow"       # Skip slow tests
```

### Linting & Type Checking
```bash
ruff check .                         # Linting (line-length=120)
ruff format .                        # Formatting
mypy .                               # Type checking
```

### Rust Extension
```bash
build_rust.bat                       # Build atc_rust (Windows)
build_rust.ps1                       # Build atc_rust (PowerShell)
```

---

## Code Style

- **Line length**: 120 (ruff)
- **Python target**: 3.12+
- **Import style**: Organized via ruff (E, F, W, I rules)
- **Exception**: `E402` ignored (intentional top-level imports on Windows)
- **Per-file exceptions**: xem `pyproject.toml [tool.ruff.lint.per-file-ignores]`

---

## Testing Markers

| Marker | Meaning |
|--------|---------|
| `unit` | Fast unit tests |
| `slow` | Slow tests (skip with `-m "not slow"`) |
| `integration` | Integration tests |
| `gpu` | Requires CUDA/GPU |
| `memory_intensive` | High RAM usage |
| `performance` | Performance benchmarks |

---

## Key Technologies

| Layer | Technology |
|-------|------------|
| Data | pandas, polars, dask |
| ML | XGBoost, scikit-learn, hmmlearn |
| DL | PyTorch, pytorch-lightning, pytorch-forecasting |
| Exchange | ccxt[pro] (8+ exchanges) |
| AI Analysis | Google Gemini API |
| Web | FastAPI + uvicorn |
| GUI | customtkinter |
| Database | SQLAlchemy + Redis |
| Performance | Rust (maturin), CUDA (cupy), numba |

---

## Important Notes

- **Rust stub**: `modules/adaptive_trend_LTS/` (pyright stubPath)
- **Windows quirks**: `--capture=no --color=no` trong pytest để tránh lỗi stdio
- **Coverage target**: 90%+ cho `modules/auto_trade`
- **PYTHONPATH**: phải set `.` khi chạy pytest
- **Env file**: `.env` cho API keys (BINANCE_API_KEY, GEMINI_API_KEY, etc.)
- **Git worktrees**: project sử dụng `.claude/worktrees/` cho isolated development

---

## Module Entry Points

- `main.py` — interactive menu chính
- `main_gemini_chart_web_server.py` — web server phân tích chart
- `headless_bot.py` — headless trading bot
- `main_cal_position_totals.py` — position sizing calculator
- `main_position_sizing.py` — position sizing analysis
