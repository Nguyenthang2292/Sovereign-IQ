import importlib.util
from pathlib import Path

"""CLI tools for XGBoost module."""


# Import from argument_parser.py (renamed from cli.py) to avoid circular import
cli_file_path = Path(__file__).parent / "argument_parser.py"
spec = importlib.util.spec_from_file_location("xgboost_cli_module", cli_file_path)
cli_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli_module)

# Import main from cli.main
from modules.xgboost.cli.main import main

# Re-export functions from argument_parser.py
prompt_with_default = cli_module.prompt_with_default
resolve_input = cli_module.resolve_input
parse_args = cli_module.parse_args

__all__ = [
    "main",
    "prompt_with_default",
    "resolve_input",
    "parse_args",
]
