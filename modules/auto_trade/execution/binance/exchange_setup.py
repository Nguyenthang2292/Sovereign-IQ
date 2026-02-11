"""
Exchange Setup Module

Handles CCXT Binance exchange initialization and configuration.
"""

from typing import Any, cast

import ccxt

from modules.common.ui.logging import log_info


class ExchangeSetup:
    """
    Handles Binance exchange initialization with CCXT.
    """

    @staticmethod
    def initialize_exchange(
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        enable_rate_limiting: bool = True,
    ) -> ccxt.binance:
        """
        Initialize CCXT Binance exchange instance.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use demo environment if True (uses demo-fapi.binance.com)
            enable_rate_limiting: Enable rate limiting

        Returns:
            CCXT Binance exchange instance
        """
        config = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": enable_rate_limiting,
            "options": {
                "defaultType": "future",  # Use USDT-M futures
                "adjustForTimeDifference": True,
                "recvWindow": 60000,  # 60 seconds tolerance for timestamp difference
            },
        }

        if testnet:
            # Binance Futures Demo Account (NEW - replaces old testnet)
            # REST base URL for demo: https://demo-fapi.binance.com
            # CRITICAL: Must override ALL futures endpoints (fapiPublic, fapiPrivate, fapiPrivateV2, etc.)
            # because the balance/position calls use fapiPrivateV2, not just "private"
            config["urls"] = {
                "api": {
                    # Override ALL futures endpoints for demo
                    "fapiPublic": "https://demo-fapi.binance.com/fapi/v1",
                    "fapiPublicV2": "https://demo-fapi.binance.com/fapi/v2",
                    "fapiPublicV3": "https://demo-fapi.binance.com/fapi/v3",
                    "fapiPrivate": "https://demo-fapi.binance.com/fapi/v1",
                    "fapiPrivateV2": "https://demo-fapi.binance.com/fapi/v2",
                    "fapiPrivateV3": "https://demo-fapi.binance.com/fapi/v3",
                    "fapiData": "https://demo-fapi.binance.com/futures/data",
                }
            }
            log_info("Initialized Binance Demo client (uses demo-fapi.binance.com)")
        else:
            # Production or Demo Account
            # Note: Binance demo accounts use production endpoints (fapi.binance.com)
            # with special demo API keys. No special URL configuration needed.
            log_info("Initialized Binance Live/Demo client (uses production endpoints)")

        exchange = ccxt.binance(cast(Any, config))

        # CRITICAL: Force time synchronization with the server BEFORE any authenticated request
        # This resolves Binance -1021 timestamp errors
        try:
            exchange.load_time_difference()
        except Exception:
            pass  # Ignore errors - adjustForTimeDifference will handle it

        return exchange
