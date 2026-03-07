import importlib
import importlib.abc
import importlib.util
import os
import sys
from types import ModuleType

"""
Modules package for crypto prediction system.
Provides compatibility aliases for legacy import paths used in tests.
"""

__all__: list[str] = []

# Backwards-compatible import aliases

_ALIASES = {
    # common subpackage
    "modules.DataFetcher": "modules.common.DataFetcher",
    "modules.ExchangeManager": "modules.common.ExchangeManager",
    "modules.ProgressBar": "modules.common.ProgressBar",
    "modules.Position": "modules.common.Position",
    "modules.utils": "modules.common.utils",
    # deeplearning subpackage
    "modules.deeplearning_data_pipeline": "modules.deeplearning.data_pipeline",
    "modules.deeplearning_dataset": "modules.deeplearning.dataset",
    "modules.deeplearning_environment_setup": "modules.deeplearning.environment_setup",
    "modules.deeplearning_feature_selection": "modules.deeplearning.feature_selection",
    "modules.deeplearning_model": "modules.deeplearning.model",
    "modules.feature_selection": "modules.deeplearning.feature_selection",
    # pairs trading subpackage
    "modules.pairs_trading_cli": "modules.pairs_trading.cli",
    "modules.pairs_trading_hedge_ratio": "modules.common.quantitative_metrics.hedge_ratios.ols_hedge_ratio",
    "modules.pairs_trading_opportunity_scorer": "modules.pairs_trading.core.opportunity_scorer",
    "modules.pairs_trading_pair_metrics_computer": "modules.pairs_trading.core.pair_metrics_computer",
    "modules.pairs_trading_pairs_analyzer": "modules.pairs_trading.core.pairs_analyzer",
    "modules.pairs_trading_performance_analyzer": "modules.pairs_trading.analysis.performance_analyzer",
    "modules.pairs_trading_risk_metrics": "modules.common.quantitative_metrics.risk.max_drawdown",
    "modules.pairs_trading_zscore_metrics": "modules.common.quantitative_metrics.classification.direction_metrics",
    # portfolio subpackage
    "modules.portfolio_correlation_analyzer": "modules.portfolio.core.correlation_analyzer",
    "modules.portfolio_hedge_finder": "modules.portfolio.core.hedge_finder",
    "modules.portfolio_risk_calculator": "modules.portfolio.core.risk_calculator",
    "modules.PortfolioCorrelationAnalyzer": "modules.portfolio.core.correlation_analyzer",
    "modules.HedgeFinder": "modules.portfolio.core.hedge_finder",
    "modules.PortfolioRiskCalculator": "modules.portfolio.core.risk_calculator",
    # xgboost subpackage
    "modules.xgboost_prediction_cli": "modules.xgboost.cli",
    "modules.xgboost_prediction_display": "modules.xgboost.display",
    "modules.xgboost_prediction_labeling": "modules.xgboost.labeling",
    "modules.xgboost_prediction_model": "modules.xgboost.model",
    "modules.xgboost_prediction_utils": "modules.xgboost.utils",
}

_this_package = sys.modules[__name__]

_ALIASES_BY_ATTR = {
    alias.split(".", 1)[1]: target
    for alias, target in _ALIASES.items()
    if alias.startswith("modules.") and "." in alias
}


def _resolve_alias(alias: str) -> ModuleType:
    target = _ALIASES.get(alias)
    if not target:
        raise ModuleNotFoundError(f"No alias mapping for {alias}")

    module = importlib.import_module(target)
    sys.modules[alias] = module

    attr_name = alias.split(".", 1)[1] if "." in alias else None
    if attr_name and not hasattr(_this_package, attr_name):
        setattr(_this_package, attr_name, module)

    return module


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, alias: str):
        self.alias = alias

    def create_module(self, spec):
        return _resolve_alias(self.alias)

    def exec_module(self, module):
        return None


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in _ALIASES:
            return importlib.util.spec_from_loader(fullname, _AliasLoader(fullname))
        return None


if not any(type(f).__name__ == "_AliasFinder" for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())


def __getattr__(name: str):
    target = _ALIASES_BY_ATTR.get(name)
    if target is None:
        raise AttributeError(name)
    return _resolve_alias(f"modules.{name}")


if os.getenv("MODULES_EAGER_ALIASES", "0") == "1":
    for alias in _ALIASES:
        try:
            _resolve_alias(alias)
        except Exception:
            continue
