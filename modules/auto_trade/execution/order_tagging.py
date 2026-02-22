"""
Order Tagging System for Auto Trading
======================================

Generates unique client order IDs and metadata for programmatic orders.
This enables distinction between auto_trade orders and manual Binance trades.

Created: 2026-02-03
"""

import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Order source types
ORDER_SOURCE_PROGRAMMATIC = "PROGRAMMATIC"
ORDER_SOURCE_MANUAL = "MANUAL"
ORDER_SOURCE_EXTERNAL = "EXTERNAL"

# Execution modes
EXECUTION_MODE_AUTO = "AUTO"
EXECUTION_MODE_MANUAL = "MANUAL"
EXECUTION_MODE_EXTERNAL = "EXTERNAL"

# Client order ID prefix for auto_trade system
CLIENT_ORDER_ID_PREFIX = "AT_"


class OrderTagger:
    """
    Generates unique client order IDs and metadata for orders.
    """

    @staticmethod
    def generate_client_order_id(symbol: str, timestamp: Optional[int] = None, use_milliseconds: bool = False) -> str:
        """
        Generate unique client order ID with AT_ prefix.

        Format: AT_{timestamp}_{symbol}_{random_suffix}
        Example: AT_1707043200_BTCUSDT_a1b2c3

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timestamp: Optional timestamp (default: current time)
            use_milliseconds: If True, use milliseconds for higher precision

        Returns:
            Unique client order ID string
        """
        if timestamp is None:
            timestamp = int(time.time() * 1000) if use_milliseconds else int(time.time())

        # Generate random suffix (6 characters, cryptographically secure)
        random_suffix = secrets.token_hex(3)  # 3 bytes = 6 hex chars

        # Build client order ID
        client_order_id = f"{CLIENT_ORDER_ID_PREFIX}{timestamp}_{symbol}_{random_suffix}"

        return client_order_id

    @staticmethod
    def is_programmatic_order_id(client_order_id: str) -> bool:
        """
        Check if a client order ID belongs to auto_trade system.

        Args:
            client_order_id: Client order ID to check

        Returns:
            True if order is from auto_trade system
        """
        if not client_order_id:
            return False

        return client_order_id.startswith(CLIENT_ORDER_ID_PREFIX)

    @staticmethod
    def parse_client_order_id(client_order_id: str) -> Optional[Dict[str, Any]]:
        """
        Parse client order ID to extract components.

        Args:
            client_order_id: Client order ID (e.g., AT_1707043200_BTCUSDT_a1b2c3)

        Returns:
            Dictionary with parsed components or None if invalid
        """
        if not OrderTagger.is_programmatic_order_id(client_order_id):
            return None

        try:
            # Remove prefix
            without_prefix: str = client_order_id[len(CLIENT_ORDER_ID_PREFIX) :]

            # Split by underscore
            parts: list[str] = without_prefix.split("_")

            if len(parts) < 3:
                return None

            timestamp: int = int(parts[0])
            symbol: str = parts[1]
            random_suffix: str = parts[2]

            return {
                "timestamp": timestamp,
                "symbol": symbol,
                "random_suffix": random_suffix,
                "datetime": datetime.fromtimestamp(timestamp if timestamp < 1e12 else timestamp / 1000),
                "is_programmatic": True,
            }

        except (ValueError, IndexError):
            return None

    @staticmethod
    def create_order_metadata(
        client_order_id: str,
        order_source: str = ORDER_SOURCE_PROGRAMMATIC,
        execution_mode: str = EXECUTION_MODE_AUTO,
        signal_correlation_id: Optional[str] = None,
        martingale_chain_id: Optional[str] = None,
        additional_metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Create comprehensive order metadata for database storage.

        Args:
            client_order_id: Unique client order ID
            order_source: Order source type (default: PROGRAMMATIC)
            execution_mode: Execution mode (default: AUTO)
            signal_correlation_id: Optional signal correlation ID
            martingale_chain_id: Optional Martingale chain ID
            additional_metadata: Optional additional metadata dict

        Returns:
            Dictionary with order metadata
        """
        metadata = {
            "client_order_id": client_order_id,
            "order_source": order_source,
            "execution_mode": execution_mode,
            "created_at": datetime.now(timezone.utc),
            "is_programmatic": order_source == ORDER_SOURCE_PROGRAMMATIC,
        }

        # Add optional IDs
        if signal_correlation_id:
            metadata["signal_correlation_id"] = signal_correlation_id

        if martingale_chain_id:
            metadata["martingale_chain_id"] = martingale_chain_id

        # Merge additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)

        return metadata

    @staticmethod
    def generate_martingale_chain_id(symbol: str, initial_order_id: str) -> str:
        """
        Generate unique Martingale chain ID.

        Format: CHAIN_{symbol}_{timestamp}_{order_id_suffix}

        Args:
            symbol: Trading symbol
            initial_order_id: First order ID in chain

        Returns:
            Unique Martingale chain ID
        """
        timestamp = int(time.time())

        # Extract last 8 chars of order ID for uniqueness
        order_suffix = initial_order_id[-8:] if len(initial_order_id) >= 8 else initial_order_id

        chain_id = f"CHAIN_{symbol}_{timestamp}_{order_suffix}"

        return chain_id

    @staticmethod
    def generate_signal_correlation_id(symbol: str, signal_type: str, timestamp: Optional[int] = None) -> str:
        """
        Generate unique signal correlation ID.

        Format: SIGNAL_{symbol}_{type}_{timestamp}_{random}

        Args:
            symbol: Trading symbol
            signal_type: Signal type (LONG/SHORT)
            timestamp: Optional timestamp

        Returns:
            Unique signal correlation ID
        """
        if timestamp is None:
            timestamp = int(time.time())

        random_suffix = secrets.token_hex(4)  # 8 hex chars

        correlation_id = f"SIGNAL_{symbol}_{signal_type}_{timestamp}_{random_suffix}"

        return correlation_id


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def generate_order_id(symbol: str) -> str:
    """
    Quick function to generate client order ID.

    Args:
        symbol: Trading symbol

    Returns:
        Unique client order ID with AT_ prefix
    """
    return OrderTagger.generate_client_order_id(symbol)


def tag_programmatic_order(
    symbol: str, signal_id: Optional[str] = None, martingale_chain_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create complete order tagging for programmatic order.

    This is the main function to use when creating orders.

    Args:
        symbol: Trading symbol
        signal_id: Optional signal correlation ID
        martingale_chain_id: Optional Martingale chain ID

    Returns:
        Dictionary with client_order_id and all metadata
    """
    # Generate unique client order ID
    client_order_id = OrderTagger.generate_client_order_id(symbol)

    # Create metadata
    metadata = OrderTagger.create_order_metadata(
        client_order_id=client_order_id,
        order_source=ORDER_SOURCE_PROGRAMMATIC,
        execution_mode=EXECUTION_MODE_AUTO,
        signal_correlation_id=signal_id,
        martingale_chain_id=martingale_chain_id,
    )

    return metadata


def is_auto_trade_order(client_order_id: str) -> bool:
    """
    Check if order is from auto_trade system.

    Args:
        client_order_id: Client order ID to check

    Returns:
        True if order is programmatic
    """
    return OrderTagger.is_programmatic_order_id(client_order_id)


def extract_order_info(client_order_id: str) -> Optional[Dict[str, Any]]:
    """
    Extract information from client order ID.

    Args:
        client_order_id: Client order ID to parse

    Returns:
        Dictionary with order info or None
    """
    return OrderTagger.parse_client_order_id(client_order_id)


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def validate_order_metadata(metadata: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate order metadata completeness.

    Args:
        metadata: Order metadata dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["client_order_id", "order_source", "execution_mode"]

    # Check required fields
    for field in required_fields:
        if field not in metadata:
            return False, f"Missing required field: {field}"

    # Validate client_order_id format
    if not OrderTagger.is_programmatic_order_id(metadata["client_order_id"]):
        return False, f"Invalid client_order_id format (must start with {CLIENT_ORDER_ID_PREFIX})"

    # Validate order_source
    valid_sources = [ORDER_SOURCE_PROGRAMMATIC, ORDER_SOURCE_MANUAL, ORDER_SOURCE_EXTERNAL]
    if metadata["order_source"] not in valid_sources:
        return False, f"Invalid order_source: {metadata['order_source']}"

    # Validate execution_mode
    valid_modes = [EXECUTION_MODE_AUTO, EXECUTION_MODE_MANUAL, EXECUTION_MODE_EXTERNAL]
    if metadata["execution_mode"] not in valid_modes:
        return False, f"Invalid execution_mode: {metadata['execution_mode']}"

    return True, None


# ============================================================================
# BATCH OPERATIONS
# ============================================================================


def tag_multiple_orders(
    symbols: list[str], signal_id: Optional[str] = None, martingale_chain_id: Optional[str] = None
) -> list[Dict[str, Any]]:
    """
    Tag multiple orders at once (for batch processing).

    Args:
        symbols: List of trading symbols
        signal_id: Optional shared signal ID
        martingale_chain_id: Optional shared Martingale chain ID

    Returns:
        List of metadata dictionaries
    """
    return [tag_programmatic_order(symbol, signal_id, martingale_chain_id) for symbol in symbols]


# ============================================================================
# STATISTICS AND REPORTING
# ============================================================================


def get_order_tag_stats(order_ids: list[str]) -> Dict[str, Any]:
    """
    Get statistics about order IDs.

    Args:
        order_ids: List of client order IDs

    Returns:
        Statistics dictionary
    """
    total = len(order_ids)
    programmatic = sum(1 for oid in order_ids if is_auto_trade_order(oid))
    manual = total - programmatic

    # Extract symbols
    symbols = set()
    for oid in order_ids:
        info = extract_order_info(oid)
        if info:
            symbols.add(info["symbol"])

    return {
        "total_orders": total,
        "programmatic_orders": programmatic,
        "manual_orders": manual,
        "programmatic_percentage": (programmatic / total * 100) if total > 0 else 0,
        "unique_symbols": len(symbols),
        "symbols": list(symbols),
    }
