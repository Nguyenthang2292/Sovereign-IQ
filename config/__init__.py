
from .common import *  # noqa: F403, F401
from .config_api import *  # noqa: F403, F401
from .decision_matrix import *  # noqa: F403, F401
from .deep_learning import *  # noqa: F403, F401 (imports from xgboost)
from .evaluation import *  # noqa: F403, F401
from .gemini_chart_analyzer import *  # noqa: F403, F401
from .hmm import *  # noqa: F403, F401
from .iching import *  # noqa: F403, F401
from .lstm import *  # noqa: F403, F401 (imports from model_features)
from .model_features import *  # noqa: F403, F401
from .pairs_trading import *  # noqa: F403, F401
from .portfolio import *  # noqa: F403, F401
from .random_forest import *  # noqa: F403, F401
from .range_oscillator import *  # noqa: F403, F401
from .spc import *  # noqa: F403, F401
from .xgboost import *  # noqa: F403, F401

"""
Configuration package.

This package contains configuration constants and API keys for all components.

The configuration is organized into separate modules:
- common: Common/shared configuration
- range_oscillator: Range Oscillator configuration
- decision_matrix: Decision Matrix voting system configuration
- spc: Simplified Percentile Clustering configuration
- xgboost: XGBoost prediction configuration
- deep_learning: Deep learning models configuration
- lstm: LSTM/CNN-LSTM models configuration
- hmm: Hidden Markov Model configuration
- portfolio: Portfolio manager configuration
- pairs_trading: Pairs trading configuration
- random_forest: Random Forest model configuration
- iching: I Ching configuration
- gemini_chart_analyzer: Gemini chart analyzer configuration
- config_api: API keys and secrets
"""

# Import order matters to avoid circular dependencies
# Import common first as it's used by many modules

# Import API configuration last
from .config_api import *  # noqa: F403, F401
from .decision_matrix import *  # noqa: F403, F401
from .deep_learning import *  # noqa: F403, F401 (imports from xgboost)

# Import shared evaluation configuration before random_forest (which uses it)
from .evaluation import *  # noqa: F403, F401
from .gemini_chart_analyzer import *  # noqa: F403, F401
from .hmm import *  # noqa: F403, F401
from .iching import *  # noqa: F403, F401
from .lstm import *  # noqa: F403, F401 (imports from model_features)

# Import shared model features before xgboost and random_forest
from .model_features import *  # noqa: F403, F401
from .pairs_trading import *  # noqa: F403, F401
from .portfolio import *  # noqa: F403, F401
from .random_forest import *  # noqa: F403, F401

# Import component-specific configurations
from .range_oscillator import *  # noqa: F403, F401
from .spc import *  # noqa: F403, F401
from .xgboost import *  # noqa: F403, F401
