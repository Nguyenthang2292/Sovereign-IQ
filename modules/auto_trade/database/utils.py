"""
Database Utilities for Auto Trading System
===========================================

Helper functions for database operations.
Created: 2026-02-03
"""

from modules.common.ui.logging import log_info, log_error, log_warn, log_debug, log_success, log_system


# With DynamoDB, session and transactions are managed differently.
# This file is kept for backward compatibility if any generic util is needed.
