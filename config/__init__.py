"""
Configuration package.

🔄 REFACTORED STRUCTURE:
The configuration is now organized into:
- config/common.py - Global settings
- config/config_api.py - API keys and secrets
- config/modules/ - Module-specific configurations
- config/shared/ - Shared configurations used by multiple modules

See config/README.md for detailed documentation.

⚠️ BACKWARD COMPATIBILITY:
All existing imports continue to work:
  from config import XGBOOST_PARAMS  # Still works!
  from config.xgboost import X       # Still works!

New recommended imports:
  from config.modules.xgboost import XGBOOST_PARAMS
  from config.shared.model_features import MODEL_FEATURES
"""

# Import order matters to avoid circular dependencies

# ============================================================================
# STEP 1: Import common/global configs first (used by many modules)
# ============================================================================
from .common import *  # noqa: F403, F401

# ============================================================================
# STEP 2: Import API configuration (contains secrets)
# ============================================================================
from .config_api import *  # noqa: F403, F401

# ============================================================================
# BACKWARD COMPATIBILITY: Keep old module references
# ============================================================================
# Allow imports like: from config.xgboost import X (old style)
# These are now in modules/ but accessible for backward compatibility
from .modules import (
    auto_trade,  # noqa: F401
    decision_matrix,  # noqa: F401
    deep_learning,  # noqa: F401
    gemini_chart_analyzer,  # noqa: F401
    hmm,  # noqa: F401
    iching,  # noqa: F401
    lstm,  # noqa: F401
    pairs_trading,  # noqa: F401
    portfolio,  # noqa: F401
    position_sizing,  # noqa: F401
    random_forest,  # noqa: F401
    range_oscillator,  # noqa: F401
    spc,  # noqa: F401
    spc_enhancements,  # noqa: F401
    xgboost,  # noqa: F401
)

# ============================================================================
# STEP 4: Import other module configs (in dependency order)
# ============================================================================
# Auto-trade (depends on many modules above)
from .modules.auto_trade import *  # noqa: F403, F401
from .modules.decision_matrix import *  # noqa: F403, F401

# Deep learning modules (depends on xgboost TARGET_HORIZON)
from .modules.deep_learning import *  # noqa: F403, F401

# Analysis/Strategy modules
from .modules.gemini_chart_analyzer import *  # noqa: F403, F401
from .modules.hmm import *  # noqa: F403, F401
from .modules.iching import *  # noqa: F403, F401
from .modules.lstm import *  # noqa: F403, F401
from .modules.pairs_trading import *  # noqa: F403, F401
from .modules.portfolio import *  # noqa: F403, F401

# Trading/Portfolio modules
from .modules.position_sizing import *  # noqa: F403, F401

# ML modules
from .modules.random_forest import *  # noqa: F403, F401
from .modules.range_oscillator import *  # noqa: F403, F401
from .modules.spc import *  # noqa: F403, F401
from .modules.spc_enhancements import *  # noqa: F403, F401

# ============================================================================
# STEP 3: Import XGBoost first (many modules depend on it)
# ============================================================================
# XGBoost first (deep_learning depends on TARGET_HORIZON from xgboost)
from .modules.xgboost import *  # noqa: F403, F401
from .modules.xgboost import (  # noqa: F401, F403
    OPTUNA_N_JOBS,
    OPTUNA_PARALLEL_TRIALS,
    TARGET_HORIZON,
    XGBOOST_USE_FLOAT32,
    XGBOOST_USE_PARALLEL_CV,
    XGBOOST_VOLATILITY_ROLLING_WINDOW,
)

# ============================================================================
# STEP 5: Import from SHARED package (used by multiple modules)
# ============================================================================
# These are now in config/shared/ but still accessible from root for compatibility
# Allow imports like: from config.evaluation import X (old style)
# These are now in shared/ but accessible for backward compatibility
from .shared import (
    evaluation,  # noqa: F401
    forex_pairs,  # noqa: F401
    model_features,  # noqa: F401
)

# Import shared configs (these don't have circular dependencies)
from .shared.evaluation import *  # noqa: F403, F401
from .shared.forex_pairs import *  # noqa: F403, F401
from .shared.model_features import *  # noqa: F403, F401
