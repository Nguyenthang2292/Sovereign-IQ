"""
check_optuna.py

Script to verify that the Optuna package is installed and can be correctly imported.
Prints the file location of the installed Optuna module and confirms successful import.

Usage:
    python check_optuna.py

This can be used as part of diagnostics to confirm Optuna is properly set up in
the current Python environment.
"""

import optuna

print(optuna.__file__)
try:
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
