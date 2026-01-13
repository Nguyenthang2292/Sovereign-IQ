"""
Configuration package.

This package contains configuration constants and API keys for all components.

The configuration is organized into separate modules:
- common: Common/shared configuration
- evaluation: Shared evaluation configuration
- model_features: Shared model features configuration
- forex_pairs: Forex pairs definitions (major and minor pairs)
- position_sizing: Position sizing and risk management configuration
- range_oscillator: Range Oscillator configuration
- decision_matrix: Decision Matrix voting system configuration
- spc: Simplified Percentile Clustering configuration
- xgboost: XGBoost prediction configuration
- random_forest: Random Forest model configuration
- deep_learning: Deep learning models configuration
- lstm: LSTM/CNN-LSTM models configuration
- hmm: Hidden Markov Model configuration
- portfolio: Portfolio manager configuration
- pairs_trading: Pairs trading configuration
- iching: I Ching configuration
- gemini_chart_analyzer: Gemini chart analyzer configuration
- config_api: API keys and secrets (imported last to avoid circular dependencies)
"""

# Import order matters to avoid circular dependencies
# Import common first as it's used by many modules
from .common import *  # noqa: F403, F401

# Import API configuration last to avoid circular dependencies
from .config_api import *  # noqa: F403, F401
from .decision_matrix import *  # noqa: F403, F401
from .deep_learning import *  # noqa: F403, F401 (imports from xgboost)

# Import shared configurations before component-specific ones
from .evaluation import *  # noqa: F403, F401

# Import data/asset configurations
from .forex_pairs import *  # noqa: F403, F401
from .gemini_chart_analyzer import *  # noqa: F403, F401
from .hmm import *  # noqa: F403, F401

# Import analyzer configurations
from .iching import *  # noqa: F403, F401
from .lstm import *  # noqa: F403, F401 (imports from model_features)
from .model_features import *  # noqa: F403, F401
from .pairs_trading import *  # noqa: F403, F401
from .portfolio import *  # noqa: F403, F401

# Import trading/portfolio configurations
from .position_sizing import *  # noqa: F403, F401
from .random_forest import *  # noqa: F403, F401

# Import component-specific configurations
from .range_oscillator import *  # noqa: F403, F401
from .spc import *  # noqa: F403, F401

# Import ML model configurations (order matters: xgboost before deep_learning)
from .xgboost import *  # noqa: F403, F401
