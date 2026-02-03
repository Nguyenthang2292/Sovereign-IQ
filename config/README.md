# Configuration Structure

This directory contains all configuration for the crypto probability trading system.

## 📁 Directory Structure

```
config/
├── __init__.py              # Main entry point (backward compatible)
├── common.py                # ⚙️ Global settings (logging, paths, API endpoints)
├── config_api.py            # 🔑 API keys and secrets
│
├── modules/                 # 📦 Module-specific configurations
│   ├── auto_trade.py        # Auto-trading system config
│   ├── xgboost.py          # XGBoost model parameters
│   ├── gemini_chart_analyzer.py
│   ├── hmm.py              # Hidden Markov Model config
│   ├── lstm.py             # LSTM/CNN-LSTM config
│   ├── deep_learning.py    # Deep learning models
│   ├── random_forest.py    # Random Forest config
│   ├── decision_matrix.py  # Voting system config
│   ├── spc.py              # Simplified Percentile Clustering
│   ├── spc_enhancements.py
│   ├── range_oscillator.py
│   ├── portfolio.py        # Portfolio management
│   ├── pairs_trading.py
│   ├── position_sizing.py
│   └── iching.py           # I Ching config
│
└── shared/                  # 🔗 Shared configurations
    ├── evaluation.py        # Evaluation metrics & thresholds
    ├── model_features.py    # ML feature definitions
    └── forex_pairs.py       # Currency pair definitions

```

## 🔧 Usage

### Backward Compatible Imports (Recommended for existing code)

All existing imports continue to work:

```python
# Import from root config (still works!)
from config import XGBOOST_PARAMS
from config import MODEL_FEATURES
from config import GEMINI_API_KEY
from config import AUTO_TRADE_CONFIG

# Import from specific module (still works!)
from config.xgboost import XGBOOST_USE_PARALLEL_CV
from config.common import ARTIFACTS_DIR
```

### New Structure Imports (Recommended for new code)

```python
# Module-specific configs
from config.modules.xgboost import XGBOOST_PARAMS, XGBOOST_USE_PARALLEL_CV
from config.modules.auto_trade import AUTO_TRADE_CONFIG
from config.modules.gemini_chart_analyzer import GEMINI_CONFIG

# Shared configs
from config.shared.model_features import MODEL_FEATURES
from config.shared.evaluation import CONFIDENCE_THRESHOLDS
from config.shared.forex_pairs import MAJOR_PAIRS

# Global configs (always at root)
from config.common import ARTIFACTS_DIR, MODELS_DIR
from config.config_api import GEMINI_API_KEY
```

## 📋 Configuration Categories

### Global Settings (`common.py`, `config_api.py`)
- File paths and directories
- Logging configuration
- API endpoints
- API keys and secrets

### Module Configs (`modules/`)
- Module-specific parameters
- Model hyperparameters
- Strategy configurations
- Each module is self-contained

### Shared Configs (`shared/`)
- Configurations used by multiple modules
- Feature definitions
- Evaluation metrics
- Asset/pair definitions

## 🔒 Security

**⚠️ NEVER commit `config_api.py` to version control!**

This file should be in `.gitignore` and contain:
- API keys
- Secrets
- Credentials

## 🎯 Design Principles

1. **Backward Compatibility**: All existing `from config import X` statements work
2. **Organization**: Related configs grouped in subpackages
3. **Discoverability**: Clear structure makes configs easy to find
4. **No Duplication**: Shared configs in one place
5. **Type Safety**: Use type hints where possible

## 🔄 Migration Guide

### For New Features
Use the new structure:
```python
from config.modules.my_module import MY_CONFIG
```

### For Existing Code
No changes needed! But you can optionally migrate:
```python
# Old (still works)
from config import XGBOOST_PARAMS

# New (more explicit)
from config.modules.xgboost import XGBOOST_PARAMS
```

## 📝 Adding New Configurations

### For a new module:
1. Create `config/modules/my_module.py`
2. Add your config constants
3. Update `config/modules/__init__.py` to export them
4. Import from `config.modules.my_module`

### For shared config:
1. Determine if it belongs in `shared/evaluation.py`, `model_features.py`, or `forex_pairs.py`
2. If none fit, consider if it should be in `common.py`
3. Update appropriate `__init__.py`

## 🧪 Testing

Verify backward compatibility:
```python
# All these should work
from config import XGBOOST_PARAMS
from config import MODEL_FEATURES
from config import GEMINI_API_KEY
from config import AUTO_TRADE_CONFIG
```

## 🗂️ File Organization Rules

**Place config in `modules/` if:**
- ✅ Used primarily by one module
- ✅ Contains module-specific parameters
- ✅ Can change independently

**Place config in `shared/` if:**
- ✅ Used by 3+ modules
- ✅ Defines interfaces/contracts between modules
- ✅ Contains reference data (pairs, features)

**Place config in root if:**
- ✅ Global system settings
- ✅ API keys/secrets
- ✅ File paths used everywhere

## 📚 Related Documentation

- `REFACTORING_PLAN.md` - Detailed refactoring plan and risk assessment
- `config_api.py.example` - Example API configuration (TODO: Create)
- Individual module docs in `modules/*/docs/`
