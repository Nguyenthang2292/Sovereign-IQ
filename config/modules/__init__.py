"""
Module-specific configurations.

This package contains configuration for individual modules/components.
All configs are re-exported from config root for backward compatibility.
"""

# Re-export all module configs for easy import
from .auto_trade import *  # noqa: F403, F401
from .decision_matrix import *  # noqa: F403, F401
from .deep_learning import *  # noqa: F403, F401
from .gemini_chart_analyzer import *  # noqa: F403, F401
from .hmm import *  # noqa: F403, F401
from .iching import *  # noqa: F403, F401
from .lstm import *  # noqa: F403, F401
from .pairs_trading import *  # noqa: F403, F401
from .portfolio import *  # noqa: F403, F401
from .position_sizing import *  # noqa: F403, F401
from .random_forest import *  # noqa: F403, F401
from .range_oscillator import *  # noqa: F403, F401
from .spc import *  # noqa: F403, F401
from .spc_enhancements import *  # noqa: F403, F401
from .xgboost import *  # noqa: F403, F401
