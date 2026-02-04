"""
Exchange connection factory for authenticated exchange connections.

This module contains the ExchangeConnectionFactory class that provides
convenience methods for connecting to various cryptocurrency exchanges
with credentials.
"""

from typing import TYPE_CHECKING, Optional

import ccxt

if TYPE_CHECKING:
    from modules.common.core.exchange_manager import AuthenticatedExchangeManager


class ExchangeConnectionFactory:
    """
    Factory class for creating authenticated exchange connections.

    This class provides convenience methods for connecting to various
    cryptocurrency exchanges using the AuthenticatedExchangeManager.
    Each method is a wrapper around connect_to_exchange_with_credentials
    for a specific exchange.
    """

    def create_authenticated_exchange(
        self,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        contract_type: str = "future",
    ) -> ccxt.Exchange:
        """
        Create authenticated exchange instance with proper Futures API configuration.

        This method ensures that all exchanges use Futures API by default.

        Args:
            exchange_id: Exchange name (e.g., 'binance', 'okx', 'kucoin')
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: Use testnet if True (default: False)
            contract_type: Contract type - 'future' (default), 'spot', or 'margin'

        Returns:
            ccxt.Exchange: Configured exchange instance with Futures API as default

        Raises:
            ValueError: If exchange is not supported by ccxt

        Example:
            >>> factory = ExchangeConnectionFactory()
            >>> exchange = factory.create_authenticated_exchange(
            ...     'binance',
            ...     'your_api_key',
            ...     'your_secret',
            ...     testnet=True,
            ...     contract_type='future'
            ... )
        """
        # Validate exchange is supported
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt")

        exchange_class = getattr(ccxt, exchange_id)

        # Build config with Futures as default
        # CRITICAL: defaultType MUST be set to prevent Spot API usage
        config = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": contract_type,  # ✅ FUTURES by default!
                "adjustForTimeDifference": True,  # Handle time sync issues
                "recvWindow": 60000,  # 60 seconds tolerance for timestamp difference (default is 5000ms)
            },
        }

        # Configure testnet/demo URLs if needed
        if testnet:
            if exchange_id == "binance":
                # Binance Futures Demo Account (NEW - replaces old testnet)
                # REST base URL for demo: https://demo-fapi.binance.com
                config["urls"] = {
                    "api": {
                        "public": "https://demo-fapi.binance.com/fapi/v1",
                        "private": "https://demo-fapi.binance.com/fapi/v1",
                    }
                }
            elif exchange_id == "bybit":
                # Bybit Testnet
                config["urls"] = {
                    "api": {
                        "public": "https://api-testnet.bybit.com",
                        "private": "https://api-testnet.bybit.com",
                    }
                }
            # Add more testnet URLs for other exchanges as needed

        # Create exchange instance
        exchange_instance = exchange_class(config)

        # CRITICAL: Force time synchronization with the server BEFORE any authenticated request
        # This resolves Binance -1021 timestamp errors
        if exchange_id == "binance":
            try:
                exchange_instance.load_time_difference()
            except Exception:
                pass  # Ignore errors - adjustForTimeDifference will handle it

        return exchange_instance

    def _create_exchange_method(exchange_id: str):
        """Factory to generate exchange connection methods dynamically."""

        def connect(
            self,
            manager: "AuthenticatedExchangeManager",
            api_key: Optional[str] = None,
            api_secret: Optional[str] = None,
            testnet: Optional[bool] = None,
            contract_type: Optional[str] = None,
        ) -> ccxt.Exchange:
            """
            Connect to authenticated {exchange_id} exchange instance (REQUIRES credentials).

            Convenience method for connect_to_exchange_with_credentials('{exchange_id}').

            Args:
                manager: AuthenticatedExchangeManager instance to use for connection
                api_key: API key (optional, uses set credentials or default)
                api_secret: API secret (optional, uses set credentials or default)
                testnet: Use testnet if True (optional, uses instance default)
                contract_type: Contract type ('spot', 'margin', 'future') (optional)

            Returns:
                ccxt.Exchange: Authenticated exchange instance

            Raises:
                ValueError: If API key/secret not provided
            """
            return manager.connect_to_exchange_with_credentials(
                exchange_id, api_key, api_secret, testnet, contract_type
            )

        connect.__name__ = f"connect_to_{exchange_id}_with_credentials"
        connect.__doc__ = connect.__doc__.format(exchange_id=exchange_id)
        return connect

    # Dynamic generation of convenience methods
    # This reduces code duplication while maintaining the exact same API
    _SUPPORTED_EXCHANGES = [
        "binance",
        "kraken",
        "kucoin",
        "gate",
        "okx",
        "bybit",
        "mexc",
        "huobi",
    ]

    for _exc in _SUPPORTED_EXCHANGES:
        _method_name = f"connect_to_{_exc}_with_credentials"
        locals()[_method_name] = _create_exchange_method(_exc)
