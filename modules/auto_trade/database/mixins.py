"""
Database Mixins for Auto Trading System
========================================

Reusable mixins for common functionality across models.

Created: 2026-02-03
"""

import json
from typing import Any, Optional, Union

from modules.common.ui.logging import log_error, log_warn


class JSONSerializableMixin:
    """
    Mixin for models with JSON fields.
    Provides common methods for parsing and setting JSON data.
    """

    def get_json_field(self, field_name: str) -> Optional[Union[dict, list, Any]]:
        """
        Parse a JSON field safely.

        Args:
            field_name: Name of the JSON field to parse

        Returns:
            Parsed dictionary or None if parsing fails

        Example:
            >>> order = Order(...)
            >>> conditions = order.get_json_field('market_conditions')
        """
        field_value = getattr(self, field_name, None)

        if field_value:
            try:
                return json.loads(field_value)
            except (json.JSONDecodeError, TypeError) as e:
                preview = str(field_value)[:100]
                log_warn(
                    f"Failed to parse JSON field '{field_name}' in {self.__class__.__name__}: {e} "
                    f"(value_preview={preview!r})"
                )
                return None

        return None

    def set_json_field(self, field_name: str, value: Union[dict, list, Any]) -> None:
        """
        Set a JSON field with proper serialization.

        Args:
            field_name: Name of the JSON field to set
            value: Dictionary to serialize

        Example:
            >>> order = Order(...)
            >>> order.set_json_field('market_conditions', {'volatility': 0.5})
        """
        try:
            setattr(self, field_name, json.dumps(value))
        except (TypeError, ValueError) as e:
            log_error(
                f"Failed to serialize JSON for field '{field_name}' in {self.__class__.__name__}: {e} "
                f"(value_type={type(value).__name__})"
            )
            raise


class TimestampMixin:
    """
    Mixin for models with timestamp fields.
    Provides common timestamp-related properties.
    """

    @property
    def age_seconds(self) -> Optional[float]:
        """
        Calculate age in seconds from created_at.

        Returns:
            Age in seconds or None if created_at is not set
        """
        created_at = getattr(self, "created_at", None)
        if created_at:
            from datetime import datetime

            return (datetime.utcnow() - created_at).total_seconds()
        return None

    @property
    def is_recent(self, threshold_seconds: int = 3600) -> bool:
        """
        Check if record was created recently.

        Args:
            threshold_seconds: Time threshold in seconds (default: 1 hour)

        Returns:
            True if created within threshold
        """
        age = self.age_seconds
        return age is not None and age < threshold_seconds


class StatusMixin:
    """
    Mixin for models with status fields.
    Provides common status-checking methods.
    """

    def is_status(self, *statuses: str) -> bool:
        """
        Check if current status matches any of the provided statuses.

        Args:
            *statuses: One or more status values to check

        Returns:
            True if status matches any of the provided values

        Example:
            >>> order.is_status('OPEN', 'PENDING')
        """
        status = getattr(self, "status", None)
        if status is not None:
            return status in statuses
        return False

    @property
    def status_display(self) -> str:
        """
        Get human-readable status display.

        Returns:
            Formatted status string
        """
        status = getattr(self, "status", None)
        if status is not None:
            return str(status).replace("_", " ").title()
        return "Unknown"


class ValidationMixin:
    """
    Mixin for models requiring validation.
    Provides common validation patterns.
    """

    def validate_required_fields(self, *field_names: str) -> bool:
        """
        Validate that required fields are present and not None.

        Args:
            *field_names: Names of required fields

        Returns:
            True if all required fields are present

        Raises:
            ValueError: If any required field is missing or None
        """
        missing_fields = []

        for field_name in field_names:
            if not hasattr(self, field_name) or getattr(self, field_name) is None:
                missing_fields.append(field_name)

        if missing_fields:
            raise ValueError(
                f"Missing required fields in {self.__class__.__name__}: {', '.join(missing_fields)}"
            )

        return True

    def validate_numeric_range(
        self, field_name: str, min_value: Optional[float] = None, max_value: Optional[float] = None
    ) -> bool:
        """
        Validate that a numeric field is within a specified range.

        Args:
            field_name: Name of the field to validate
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)

        Returns:
            True if value is within range

        Raises:
            ValueError: If value is out of range
        """
        if not hasattr(self, field_name):
            raise ValueError(f"Field '{field_name}' does not exist in {self.__class__.__name__}")

        value = getattr(self, field_name)

        if value is None:
            return True  # Allow None values

        if not isinstance(value, (int, float)):
            raise ValueError(f"Field '{field_name}' must be numeric, got {type(value).__name__}")

        if min_value is not None and value < min_value:
            raise ValueError(f"Field '{field_name}' value {value} is below minimum {min_value}")

        if max_value is not None and value > max_value:
            raise ValueError(f"Field '{field_name}' value {value} is above maximum {max_value}")

        return True
