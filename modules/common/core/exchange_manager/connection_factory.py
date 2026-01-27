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

    def connect_to_binance_with_credentials(
        self, manager: "AuthenticatedExchangeManager"
    ) -> ccxt.Exchange:
        """
        Connect to authenticated Binance exchange instance (REQUIRES credentials).

        Use this for:
        - fetch_ticker() - Get current prices
        - load_markets() - List available symbols
        - fetch_positions() - Get account positions
        - Any authenticated API calls

        Args:
            manager: AuthenticatedExchangeManager instance to use for connection

        Returns:
            ccxt.Exchange: Authenticated Binance exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return manager.connect_to_exchange_with_credentials("binance")

    def connect_to_kraken_with_credentials(
        self,
        manager: "AuthenticatedExchangeManager",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated Kraken exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('kraken').

        Args:
            manager: AuthenticatedExchangeManager instance to use for connection
            api_key: API key for Kraken (optional, uses set credentials or default)
            api_secret: API secret for Kraken (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated Kraken exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return manager.connect_to_exchange_with_credentials("kraken", api_key, api_secret, testnet, contract_type)

    def connect_to_kucoin_with_credentials(
        self,
        manager: "AuthenticatedExchangeManager",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated KuCoin exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('kucoin').

        Args:
            manager: AuthenticatedExchangeManager instance to use for connection
            api_key: API key for KuCoin (optional, uses set credentials or default)
            api_secret: API secret for KuCoin (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated KuCoin exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return manager.connect_to_exchange_with_credentials("kucoin", api_key, api_secret, testnet, contract_type)

    def connect_to_gate_with_credentials(
        self,
        manager: "AuthenticatedExchangeManager",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated Gate.io exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('gate').

        Args:
            manager: AuthenticatedExchangeManager instance to use for connection
            api_key: API key for Gate.io (optional, uses set credentials or default)
            api_secret: API secret for Gate.io (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated Gate.io exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return manager.connect_to_exchange_with_credentials("gate", api_key, api_secret, testnet, contract_type)

    def connect_to_okx_with_credentials(
        self,
        manager: "AuthenticatedExchangeManager",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated OKX exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('okx').

        Args:
            manager: AuthenticatedExchangeManager instance to use for connection
            api_key: API key for OKX (optional, uses set credentials or default)
            api_secret: API secret for OKX (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated OKX exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return manager.connect_to_exchange_with_credentials("okx", api_key, api_secret, testnet, contract_type)

    def connect_to_bybit_with_credentials(
        self,
        manager: "AuthenticatedExchangeManager",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated Bybit exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('bybit').

        Args:
            manager: AuthenticatedExchangeManager instance to use for connection
            api_key: API key for Bybit (optional, uses set credentials or default)
            api_secret: API secret for Bybit (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated Bybit exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return manager.connect_to_exchange_with_credentials("bybit", api_key, api_secret, testnet, contract_type)

    def connect_to_mexc_with_credentials(
        self,
        manager: "AuthenticatedExchangeManager",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated MEXC exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('mexc').

        Args:
            manager: AuthenticatedExchangeManager instance to use for connection
            api_key: API key for MEXC (optional, uses set credentials or default)
            api_secret: API secret for MEXC (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated MEXC exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return manager.connect_to_exchange_with_credentials("mexc", api_key, api_secret, testnet, contract_type)

    def connect_to_huobi_with_credentials(
        self,
        manager: "AuthenticatedExchangeManager",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated Huobi exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('huobi').

        Args:
            manager: AuthenticatedExchangeManager instance to use for connection
            api_key: API key for Huobi (optional, uses set credentials or default)
            api_secret: API secret for Huobi (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated Huobi exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return manager.connect_to_exchange_with_credentials("huobi", api_key, api_secret, testnet, contract_type)
