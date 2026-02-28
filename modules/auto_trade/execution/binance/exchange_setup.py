"""
Exchange Setup Module

Handles CCXT Binance exchange initialization and configuration.
Supports EC2 SOCKS5 proxy for fixed outbound IP (Binance IP whitelist).
"""

import os
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
                # Increase recvWindow to absorb proxy latency variability
                # (proxy adds ~100-300ms; default 5000ms is too tight)
                "recvWindow": 50000,
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

        # ── EC2 Proxy (Fixed IP for Binance whitelist) ─────────────────────
        # When EC2_PROXY_ENABLED=true, all API calls route through EC2 Elastic IP
        # so Binance always sees the same static IP regardless of local IP changes
        if os.getenv("EC2_PROXY_ENABLED", "false").lower() == "true":
            proxy_port = os.getenv("EC2_PROXY_PORT", "1080")
            proxy_url = f"socks5h://127.0.0.1:{proxy_port}"
            config["proxies"] = {
                "http": proxy_url,
                "https": proxy_url,
            }
            log_info(f"Binance using EC2 proxy (fixed IP): {proxy_url}")

        exchange = ccxt.binance(cast(Any, config))

        # CRITICAL: Force time synchronization BEFORE any authenticated request
        # When using SOCKS5 proxy, retry a few times — proxy adds latency that
        # can cause -1021 "timestamp ahead" errors on the first attempt
        proxy_enabled = os.getenv("EC2_PROXY_ENABLED", "false").lower() == "true"
        max_sync_attempts = 3 if proxy_enabled else 1
        for attempt in range(max_sync_attempts):
            try:
                exchange.load_time_difference()
                break  # Success
            except Exception:
                if attempt < max_sync_attempts - 1:
                    import time as _time

                    _time.sleep(1)  # Brief pause before retry

        return exchange
