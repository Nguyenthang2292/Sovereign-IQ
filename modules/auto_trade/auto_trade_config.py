from __future__ import annotations

"""
Auto Trading System - Configuration Management
===============================================

Centralized configuration for all auto trading parameters.
Supports environment variables, CLI arguments, and config files.

Created: 2026-02-03
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# Load environment variables (optional)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv not installed, will use system env vars only
    pass


@dataclass
class ScanningConfig:
    """Signal scanning configuration."""

    scan_interval: int = 300  # 5 minutes
    symbol_sample_percentage: float = 1.0  # 100% of symbols
    enabled_scanners: list = field(default_factory=lambda: ["ATC", "XGBOOST"])
    min_confidence: float = 0.7


@dataclass
class RiskConfig:
    """Risk management configuration."""

    leverage: int = 2
    position_size_percent: float = 0.95  # 95% of balance
    stop_loss_percent: float = 0.50  # 50% SL
    take_profit_percent: float = 0.05  # 5% TP
    max_open_positions: int = 3
    max_daily_loss: float = 0.10  # 10% daily loss limit


@dataclass
class MartingaleConfig:
    """Martingale recovery configuration."""

    enabled: bool = True
    max_steps: int = 3
    multiplier: float = 2.0
    min_recovery_profit: float = 0.01  # 1% minimum recovery profit


@dataclass
class BreakEvenConfig:
    """Break-even configuration."""

    enabled: bool = True
    trigger_profit_percent: float = 0.02  # Move BE at 2% profit
    offset_percent: float = 0.001  # Offset BE by 0.1%


@dataclass
class DatabaseConfig:
    """Database configuration (DynamoDB)."""

    backend: str = field(default_factory=lambda: os.getenv("DB_BACKEND", "dynamodb"))
    table_name_prefix: str = field(default_factory=lambda: os.getenv("DYNAMODB_TABLE_PREFIX", "auto_trade"))
    region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    endpoint_url: Optional[str] = field(default_factory=lambda: os.getenv("DYNAMODB_ENDPOINT_URL", None))
    backup_enabled: bool = field(default_factory=lambda: os.getenv("DYNAMODB_BACKUP_ENABLED", "True").lower() == "true")
    auto_cleanup_days: int = 90

    @property
    def path(self) -> str:
        """
        Backward-compatible alias used by legacy tests and older SQLite-oriented code.

        For DynamoDB mode, the logical equivalent is the table name prefix.
        """
        return self.table_name_prefix

    @path.setter
    def path(self, value: str) -> None:
        # Keep behavior symmetric for callers that still assign `database.path`.
        self.table_name_prefix = value


@dataclass
class BinanceConfig:
    """Binance API configuration."""

    api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    testnet: bool = field(default_factory=lambda: os.getenv("BINANCE_TESTNET", "False").lower() == "true")
    base_url: Optional[str] = None


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    log_dir: str = "data/logs"
    log_file: str = "auto_trade.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class AutoTradeConfig:
    """Main auto trading configuration."""

    # Sub-configurations
    scanning: ScanningConfig = field(default_factory=ScanningConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    martingale: MartingaleConfig = field(default_factory=MartingaleConfig)
    breakeven: BreakEvenConfig = field(default_factory=BreakEvenConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # System settings
    dry_run: bool = field(default_factory=lambda: os.getenv("DRY_RUN", "False").lower() == "true")
    enable_telegram: bool = False
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self):
        """Validate configuration values."""
        errors = []

        # Validate scanning
        if self.scanning.scan_interval < 60:
            errors.append("Scan interval must be at least 60 seconds")

        if not 0 < self.scanning.symbol_sample_percentage <= 1.0:
            errors.append("Symbol sample percentage must be between 0 and 1")

        # Validate risk
        if self.risk.leverage < 1 or self.risk.leverage > 125:
            errors.append("Leverage must be between 1 and 125")

        if not 0 < self.risk.position_size_percent <= 1.0:
            errors.append("Position size percent must be between 0 and 1")

        if self.risk.max_open_positions < 1:
            errors.append("Max open positions must be at least 1")

        # Validate Martingale
        if self.martingale.max_steps < 1 or self.martingale.max_steps > 10:
            errors.append("Martingale max steps must be between 1 and 10")

        if self.martingale.multiplier < 1.0:
            errors.append("Martingale multiplier must be at least 1.0")

        # Validate Binance credentials (if not dry run)
        if not self.dry_run and not self.binance.testnet:
            if not self.binance.api_key:
                errors.append("Binance API key is required (not in dry run)")
            if not self.binance.api_secret:
                errors.append("Binance API secret is required (not in dry run)")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {err}" for err in errors))

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, path: Optional[str] = None) -> str:
        """
        Export config to JSON.

        Args:
            path: Optional file path to save JSON

        Returns:
            JSON string
        """
        config_dict = self.to_dict()

        # Remove sensitive data
        if "binance" in config_dict:
            config_dict["binance"]["api_key"] = "***REDACTED***" if config_dict["binance"]["api_key"] else ""
            config_dict["binance"]["api_secret"] = "***REDACTED***" if config_dict["binance"]["api_secret"] else ""

        json_str = json.dumps(config_dict, indent=2)

        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_json(cls, path: str) -> AutoTradeConfig:
        """
        Load config from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            AutoTradeConfig instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        # Reconstruct sub-configs
        config = cls(
            scanning=ScanningConfig(**data.get("scanning", {})),
            risk=RiskConfig(**data.get("risk", {})),
            martingale=MartingaleConfig(**data.get("martingale", {})),
            breakeven=BreakEvenConfig(**data.get("breakeven", {})),
            database=DatabaseConfig(**data.get("database", {})),
            binance=BinanceConfig(**data.get("binance", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            dry_run=data.get("dry_run", False),
            enable_telegram=data.get("enable_telegram", False),
            telegram_token=data.get("telegram_token", ""),
            telegram_chat_id=data.get("telegram_chat_id", ""),
        )

        return config

    def get_summary(self) -> str:
        """Get human-readable configuration summary."""
        lines = [
            "=" * 60,
            "AUTO TRADING CONFIGURATION",
            "=" * 60,
            "",
            "SCANNING:",
            f"  Interval: {self.scanning.scan_interval}s",
            f"  Symbol Sample: {self.scanning.symbol_sample_percentage * 100}%",
            f"  Scanners: {', '.join(self.scanning.enabled_scanners)}",
            f"  Min Confidence: {self.scanning.min_confidence}",
            "",
            "RISK MANAGEMENT:",
            f"  Leverage: {self.risk.leverage}x",
            f"  Position Size: {self.risk.position_size_percent * 100}%",
            f"  Stop Loss: {self.risk.stop_loss_percent * 100}%",
            f"  Take Profit: {self.risk.take_profit_percent * 100}%",
            f"  Max Positions: {self.risk.max_open_positions}",
            "",
            "MARTINGALE:",
            f"  Enabled: {self.martingale.enabled}",
            f"  Max Steps: {self.martingale.max_steps}",
            f"  Multiplier: {self.martingale.multiplier}x",
            "",
            "BREAK-EVEN:",
            f"  Enabled: {self.breakeven.enabled}",
            f"  Trigger: {self.breakeven.trigger_profit_percent * 100}%",
            "",
            "DATABASE:",
            f"  Backend: {self.database.backend}",
            f"  Region: {self.database.region}",
            f"  Table Prefix: {self.database.table_name_prefix}",
            "",
            "BINANCE:",
            f"  Testnet: {self.binance.testnet}",
            f"  API Key: {'***SET***' if self.binance.api_key else 'NOT SET'}",
            "",
            "SYSTEM:",
            f"  Dry Run: {self.dry_run}",
            f"  Telegram: {self.enable_telegram}",
            "=" * 60,
        ]

        return "\n".join(lines)


# Global configuration instance
_config: Optional[AutoTradeConfig] = None


def get_config() -> AutoTradeConfig:
    """
    Get global configuration instance.

    Returns:
        AutoTradeConfig instance
    """
    global _config

    if _config is None:
        _config = AutoTradeConfig()

    return _config


def load_config(path: Optional[str] = None) -> AutoTradeConfig:
    """
    Load configuration from file or environment.

    Args:
        path: Optional path to JSON config file

    Returns:
        AutoTradeConfig instance
    """
    global _config

    if path and Path(path).exists():
        _config = AutoTradeConfig.from_json(path)
    else:
        _config = AutoTradeConfig()

    return _config


def save_config(config: AutoTradeConfig, path: str):
    """
    Save configuration to file.

    Args:
        config: Configuration instance
        path: Path to save JSON file
    """
    config.to_json(path)


# Example configurations for different scenarios
def get_conservative_config() -> AutoTradeConfig:
    """Get conservative trading configuration."""
    config = AutoTradeConfig()
    config.risk.leverage = 1
    config.risk.stop_loss_percent = 0.30  # 30% SL
    config.risk.take_profit_percent = 0.10  # 10% TP
    config.risk.max_open_positions = 1
    config.martingale.enabled = False
    return config


def get_aggressive_config() -> AutoTradeConfig:
    """Get aggressive trading configuration."""
    config = AutoTradeConfig()
    config.risk.leverage = 3
    config.risk.stop_loss_percent = 0.70  # 70% SL
    config.risk.take_profit_percent = 0.03  # 3% TP
    config.risk.max_open_positions = 5
    config.martingale.max_steps = 4
    config.martingale.multiplier = 2.5
    return config


def get_testnet_config() -> AutoTradeConfig:
    """Get testnet configuration."""
    config = AutoTradeConfig()
    config.binance.testnet = True
    config.dry_run = True
    config.database.endpoint_url = os.getenv("DYNAMODB_ENDPOINT_URL")
    config.database.table_name_prefix = "auto_trade_test"
    return config
