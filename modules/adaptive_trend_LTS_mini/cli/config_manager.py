"""
Configuration Management for ATC Analysis.

This module handles ATC configuration creation and parameter management.
"""

from argparse import Namespace
from typing import Any, Dict, Optional, cast

from modules.adaptive_trend_LTS_mini.cli.config_utils import get_atc_params
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig, create_atc_config_from_dict

__all__ = ["ConfigManager"]


class ConfigManager:
    """
    Manages ATC configuration and parameters.

    Handles extraction of ATC parameters from arguments and
    creation of ATCConfig instances.
    """

    def __init__(self, args: Namespace):
        """
        Initialize ConfigManager.

        Args:
            args: Command-line arguments namespace
        """
        self.args = args
        self._cached_params: Optional[Dict[str, Any]] = None

    def get_atc_params(self) -> Dict[str, Any]:
        """
        Get ATC parameters from arguments.

        Caches parameters for reuse.

        Returns:
            Dictionary of ATC parameters
        """
        if self._cached_params is None:
            params = get_atc_params(self.args)
            self._cached_params = cast(Dict[str, Any], params)
            return self._cached_params
        return self._cached_params

    def create_config(self, timeframe: str) -> ATCConfig:
        """
        Create ATCConfig instance.

        Args:
            timeframe: Timeframe for analysis

        Returns:
            Configured ATCConfig instance
        """
        params = self.get_atc_params()
        return create_atc_config_from_dict(params, timeframe=timeframe)

    def invalidate_cache(self) -> None:
        """
        Invalidate cached parameters.

        Forces re-extraction of parameters on next get_atc_params() call.
        """
        self._cached_params = None
