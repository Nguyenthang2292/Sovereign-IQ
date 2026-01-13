"""
CLI tools for Random Forest module.

This module provides command-line interface tools for Random Forest operations.
"""

from modules.random_forest.cli.argument_parser import parse_args
from modules.random_forest.cli.main import main

__all__ = [
    "parse_args",
    "main",
]
