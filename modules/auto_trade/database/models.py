"""
SQLAlchemy ORM Models for Auto Trading System
==================================================

Defines database models for:
- Orders (programmatic trades only)
- Signals (from signal pipeline)
- Martingale Chains (recovery sequences)
- System State (configuration and state)
- Audit Log (comprehensive tracking)

Created: 2026-02-03
"""

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    event,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Import mixins
from .mixins import JSONSerializableMixin

# Logger
logger = logging.getLogger(__name__)

# Create base class for all models
Base = declarative_base()


# ============================================================================
# ORDER MODEL - Programmatic Orders Only
# ============================================================================


class Order(Base, JSONSerializableMixin):
    """
    Represents a trading order created by the auto_trade system.

    CRITICAL: Only PROGRAMMATIC orders (created by system) are stored.
    Manual trades on Binance are NOT synced to this table.
    """

    __tablename__ = "orders"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Order Identification
    order_id = Column(String(100), unique=True, nullable=False, index=True)
    client_order_id = Column(String(100), unique=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)

    # Order Type and Direction
    side = Column(String(10), nullable=False)  # 'LONG' or 'SHORT'
    order_type = Column(String(20), nullable=False, default="MARKET")

    # Order Source Tracking (CRITICAL)
    order_source = Column(String(20), nullable=False, default="PROGRAMMATIC", index=True)
    execution_mode = Column(String(20), nullable=False, default="AUTO", index=True)

    # Pricing
    entry_price = Column(Float, nullable=False)
    expected_entry_price = Column(Float)
    amount = Column(Float, nullable=False)
    notional_value = Column(Float)

    # Risk Management
    leverage = Column(Integer, nullable=False, default=1)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    stop_loss_percentage = Column(Float)
    take_profit_percentage = Column(Float)

    # Order Status
    status = Column(String(20), nullable=False, default="PENDING", index=True)

    # P&L Tracking
    pnl = Column(Float, default=0.0)
    pnl_percentage = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    commission_asset = Column(String(10), default="USDT")

    # Break-Even Management
    be_moved = Column(Boolean, default=False)
    be_moved_at = Column(DateTime)
    original_stop_loss = Column(Float)

    # Martingale Chain Tracking
    martingale_step = Column(Integer, default=0)
    parent_order_id = Column(String(100), ForeignKey("orders.order_id"), index=True)
    martingale_chain_id = Column(String(100), index=True)
    is_martingale_recovery = Column(Boolean, default=False)

    # Signal Correlation
    signal_correlation_id = Column(String(100), index=True)
    signal_confidence = Column(Float)

    # Execution Metrics
    execution_latency_ms = Column(Integer)
    slippage_percentage = Column(Float)
    retry_count = Column(Integer, default=0)

    # Market Conditions (JSON)
    market_conditions = Column(Text)

    # Risk Assessment
    risk_score = Column(Float)

    # Failure Tracking
    rejection_reason = Column(Text)
    error_message = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    opened_at = Column(DateTime)
    closed_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    parent_order = relationship("Order", remote_side=[order_id], backref="child_orders")

    # Table constraints
    __table_args__ = (
        CheckConstraint("side IN ('LONG', 'SHORT')", name="check_side"),
        CheckConstraint("order_source IN ('PROGRAMMATIC', 'MANUAL', 'EXTERNAL')", name="check_order_source"),
        CheckConstraint("execution_mode IN ('AUTO', 'MANUAL', 'EXTERNAL')", name="check_execution_mode"),
        CheckConstraint("status IN ('PENDING', 'OPEN', 'CLOSED', 'CANCELLED', 'FAILED')", name="check_status"),
        Index("idx_orders_symbol_status", "symbol", "status"),
    )

    def __repr__(self):
        return (
            f"<Order(id={self.id}, order_id='{self.order_id}', "
            f"symbol='{self.symbol}', side='{self.side}', "
            f"status='{self.status}', pnl={self.pnl})>"
        )

    def to_dict(self):
        """Convert order to dictionary."""
        return {
            "id": self.id,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "order_source": self.order_source,
            "execution_mode": self.execution_mode,
            "entry_price": self.entry_price,
            "amount": self.amount,
            "leverage": self.leverage,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status,
            "pnl": self.pnl,
            "pnl_percentage": self.pnl_percentage,
            "be_moved": self.be_moved,
            "martingale_step": self.martingale_step,
            "martingale_chain_id": self.martingale_chain_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
        }

    @property
    def is_programmatic(self) -> bool:
        """Check if this order is programmatic (created by auto_trade)."""
        return self.order_source == "PROGRAMMATIC"

    @property
    def is_open(self) -> bool:
        """Check if order is currently open."""
        return self.status == "OPEN"

    @property
    def is_closed(self) -> bool:
        """Check if order is closed."""
        return self.status == "CLOSED"

    def get_market_conditions(self) -> Optional[dict]:
        """Parse market conditions JSON using mixin."""
        return self.get_json_field("market_conditions")

    def set_market_conditions(self, conditions: dict):
        """Set market conditions using mixin."""
        self.set_json_field("market_conditions", conditions)


# ============================================================================
# SIGNAL MODEL
# ============================================================================


class Signal(Base, JSONSerializableMixin):
    """
    Represents a trading signal generated by the signal pipeline.
    Tracks signal quality, execution status, and outcomes.
    """

    __tablename__ = "signals"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Signal Identification
    correlation_id = Column(String(100), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)

    # Signal Details
    signal_type = Column(String(10), nullable=False)  # 'LONG', 'SHORT', 'NEUTRAL'
    confidence = Column(Float, nullable=False)

    # Component Scores
    atc_score = Column(Float)
    xgboost_score = Column(Float)
    gemini_score = Column(Float)
    final_score = Column(Float)

    # Signal Quality
    signal_quality = Column(String(10))  # 'HIGH', 'MEDIUM', 'LOW'
    quality_score = Column(Float)

    # Timeframe Analysis
    timeframe_5m_signal = Column(String(10))
    timeframe_15m_signal = Column(String(10))
    timeframe_1h_signal = Column(String(10))
    timeframe_consensus = Column(String(10))

    # Market Context (JSON)
    market_context = Column(Text)

    # Execution Status
    executed = Column(Boolean, default=False, index=True)
    execution_order_id = Column(String(100), ForeignKey("orders.order_id"), index=True)
    rejected = Column(Boolean, default=False)
    rejection_reason = Column(Text)

    # Outcome Tracking
    outcome = Column(String(20), index=True)  # 'WIN', 'LOSS', 'BREAKEVEN', 'PENDING'
    outcome_pnl = Column(Float)
    outcome_duration_minutes = Column(Integer)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    executed_at = Column(DateTime)
    outcome_at = Column(DateTime)

    # Relationships
    execution_order = relationship("Order", backref="triggering_signal", foreign_keys=[execution_order_id])

    # Table constraints
    __table_args__ = (
        CheckConstraint("signal_type IN ('LONG', 'SHORT', 'NEUTRAL')", name="check_signal_type"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="check_confidence_range"),
        CheckConstraint(
            "signal_quality IN ('HIGH', 'MEDIUM', 'LOW') OR signal_quality IS NULL", name="check_signal_quality"
        ),
        CheckConstraint("outcome IN ('WIN', 'LOSS', 'BREAKEVEN', 'PENDING') OR outcome IS NULL", name="check_outcome"),
    )

    def __repr__(self):
        return (
            f"<Signal(id={self.id}, symbol='{self.symbol}', "
            f"type='{self.signal_type}', confidence={self.confidence:.2f}, "
            f"executed={self.executed})>"
        )

    def to_dict(self):
        """Convert signal to dictionary."""
        return {
            "id": self.id,
            "correlation_id": self.correlation_id,
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "confidence": self.confidence,
            "atc_score": self.atc_score,
            "xgboost_score": self.xgboost_score,
            "gemini_score": self.gemini_score,
            "final_score": self.final_score,
            "signal_quality": self.signal_quality,
            "executed": self.executed,
            "outcome": self.outcome,
            "outcome_pnl": self.outcome_pnl,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def get_market_context(self) -> Optional[dict]:
        """Parse market context JSON using mixin."""
        return self.get_json_field("market_context")

    def set_market_context(self, context: dict):
        """Set market context using mixin."""
        self.set_json_field("market_context", context)


# ============================================================================
# MARTINGALE CHAIN MODEL
# ============================================================================


class MartingaleChain(Base, JSONSerializableMixin):
    """
    Represents a Martingale recovery sequence.
    Tracks loss recovery progress across multiple orders.
    """

    __tablename__ = "martingale_chain"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Chain Identification
    chain_id = Column(String(100), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)

    # Chain Status
    status = Column(String(20), nullable=False, default="ACTIVE", index=True)

    # Loss Tracking
    original_loss = Column(Float, nullable=False)
    current_step = Column(Integer, nullable=False, default=0)
    max_step_reached = Column(Integer, default=0)
    total_loss = Column(Float, nullable=False, default=0.0)
    total_recovery = Column(Float, default=0.0)

    # Recovery Status
    recovered = Column(Boolean, default=False)
    recovery_pnl = Column(Float, default=0.0)

    # Safety Limits
    max_allowed_steps = Column(Integer, default=4)
    max_allowed_loss = Column(Float)

    # Order References
    initial_order_id = Column(String(100), ForeignKey("orders.order_id"))
    latest_order_id = Column(String(100), ForeignKey("orders.order_id"))
    recovery_order_id = Column(String(100), ForeignKey("orders.order_id"))

    # Chain Metadata (JSON)
    leverage_progression = Column(Text)
    position_size_progression = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    recovered_at = Column(DateTime)
    failed_at = Column(DateTime)

    # Relationships
    initial_order = relationship("Order", foreign_keys=[initial_order_id], backref="martingale_chain_initial")
    latest_order = relationship("Order", foreign_keys=[latest_order_id], backref="martingale_chain_latest")
    recovery_order = relationship("Order", foreign_keys=[recovery_order_id], backref="martingale_chain_recovery")

    # Table constraints
    __table_args__ = (
        CheckConstraint("status IN ('ACTIVE', 'RECOVERED', 'FAILED', 'CANCELLED')", name="check_chain_status"),
    )

    def __repr__(self):
        return (
            f"<MartingaleChain(id={self.id}, chain_id='{self.chain_id}', "
            f"symbol='{self.symbol}', step={self.current_step}, "
            f"status='{self.status}')>"
        )

    def to_dict(self):
        """Convert chain to dictionary."""
        return {
            "id": self.id,
            "chain_id": self.chain_id,
            "symbol": self.symbol,
            "status": self.status,
            "original_loss": self.original_loss,
            "current_step": self.current_step,
            "max_step_reached": self.max_step_reached,
            "total_loss": self.total_loss,
            "total_recovery": self.total_recovery,
            "recovered": self.recovered,
            "recovery_pnl": self.recovery_pnl,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "recovered_at": self.recovered_at.isoformat() if self.recovered_at else None,
        }

    @property
    def is_active(self) -> bool:
        """Check if chain is still active."""
        return self.status == "ACTIVE"

    @property
    def net_pnl(self) -> float:
        """Calculate net P&L (recovery - loss)."""
        return self.total_recovery - abs(self.total_loss)

    def get_leverage_progression(self) -> Optional[list]:
        """Parse leverage progression JSON using mixin."""
        return self.get_json_field("leverage_progression")

    def set_leverage_progression(self, progression: list):
        """Set leverage progression using mixin."""
        self.set_json_field("leverage_progression", progression)

    def get_position_size_progression(self) -> Optional[list]:
        """Parse position size progression JSON using mixin."""
        return self.get_json_field("position_size_progression")

    def set_position_size_progression(self, progression: list):
        """Set position size progression using mixin."""
        self.set_json_field("position_size_progression", progression)


# ============================================================================
# GRADUAL RECOVERY MODEL
# ============================================================================


class GradualRecovery(Base, JSONSerializableMixin):
    """
    Represents a Gradual Recovery sequence.
    Tracks gradual loss recovery progress using controlled scaling.
    """

    __tablename__ = "gradual_recovery"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Recovery Identification
    recovery_id = Column(String(100), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)

    # Recovery Status
    status = Column(String(20), nullable=False, default="ACTIVE", index=True)

    # Loss Tracking
    initial_loss = Column(Float, nullable=False)
    remaining_loss = Column(Float, nullable=False)
    total_profit_accumulated = Column(Float, default=0.0)
    recovery_percentage = Column(Float, default=0.0)

    # Trade Tracking
    trades_count = Column(Integer, default=0)
    win_streak = Column(Integer, default=0)
    estimated_trades_remaining = Column(Integer, default=0)

    # Configuration (JSON)
    config_data = Column(Text)  # Stores RecoveryConfig as JSON

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    completed_at = Column(DateTime)
    failed_at = Column(DateTime)

    # Table constraints
    __table_args__ = (
        CheckConstraint("status IN ('ACTIVE', 'COMPLETE', 'FAILED', 'CANCELLED')", name="check_recovery_status"),
    )

    def __repr__(self):
        return (
            f"<GradualRecovery(id={self.id}, recovery_id='{self.recovery_id}', "
            f"symbol='{self.symbol}', progress={self.recovery_percentage:.1f}%, "
            f"status='{self.status}')>"
        )

    def to_dict(self):
        """Convert recovery to dictionary."""
        return {
            "id": self.id,
            "recovery_id": self.recovery_id,
            "symbol": self.symbol,
            "status": self.status,
            "initial_loss": self.initial_loss,
            "remaining_loss": self.remaining_loss,
            "total_profit_accumulated": self.total_profit_accumulated,
            "recovery_percentage": self.recovery_percentage,
            "trades_count": self.trades_count,
            "win_streak": self.win_streak,
            "estimated_trades_remaining": self.estimated_trades_remaining,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @property
    def is_active(self) -> bool:
        """Check if recovery is still active."""
        return self.status == "ACTIVE"

    @property
    def is_complete(self) -> bool:
        """Check if recovery is complete."""
        return self.status == "COMPLETE"

    def get_config(self) -> Optional[dict]:
        """Parse config JSON using mixin."""
        return self.get_json_field("config_data")

    def set_config(self, config: dict):
        """Set config using mixin."""
        self.set_json_field("config_data", config)


# ============================================================================
# SYSTEM STATE MODEL
# ============================================================================


class SystemState(Base):
    """
    Stores system-wide state and configuration as key-value pairs.
    """

    __tablename__ = "system_state"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # State Key-Value
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    value_type = Column(String(20), default="string")

    # Metadata
    description = Column(Text)
    category = Column(String(50), index=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Table constraints
    __table_args__ = (
        CheckConstraint("value_type IN ('string', 'integer', 'float', 'boolean', 'json')", name="check_value_type"),
    )

    def __repr__(self):
        return f"<SystemState(key='{self.key}', value='{self.value}')>"

    def get_typed_value(self):
        """Get value with correct type."""
        try:
            if self.value_type == "integer":
                return int(self.value)
            elif self.value_type == "float":
                return float(self.value)
            elif self.value_type == "boolean":
                return self.value.lower() in ("true", "1", "yes")
            elif self.value_type == "json":
                try:
                    return json.loads(self.value)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON for key {self.key}: {e}")
                    return self.value
            else:
                return self.value
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to convert value for key {self.key} to type {self.value_type}: {e}")
            return None


# ============================================================================
# AUDIT LOG MODEL
# ============================================================================


class AuditLog(Base, JSONSerializableMixin):
    """
    Comprehensive audit trail for all system actions.
    Immutable log for compliance and debugging.
    """

    __tablename__ = "audit_log"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Event Classification
    event_type = Column(String(50), nullable=False, index=True)
    event_category = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False, index=True)

    # Event Details
    event_summary = Column(Text, nullable=False)
    event_data = Column(Text)  # JSON

    # Correlation
    correlation_id = Column(String(100), index=True)
    order_id = Column(String(100), index=True)
    signal_id = Column(String(100))

    # Source
    source_module = Column(String(100))
    source_function = Column(String(100))

    # Result
    success = Column(Boolean, default=True)
    error_message = Column(Text)

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Table constraints
    __table_args__ = (
        CheckConstraint("severity IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')", name="check_severity"),
    )

    def __repr__(self):
        return (
            f"<AuditLog(id={self.id}, event_type='{self.event_type}', "
            f"severity='{self.severity}', timestamp={self.timestamp})>"
        )

    def to_dict(self):
        """Convert audit log to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "event_category": self.event_category,
            "severity": self.severity,
            "event_summary": self.event_summary,
            "correlation_id": self.correlation_id,
            "order_id": self.order_id,
            "success": self.success,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    def get_event_data(self) -> Optional[dict]:
        """Parse event data JSON using mixin."""
        return self.get_json_field("event_data")

    def set_event_data(self, data: dict):
        """Set event data using mixin."""
        self.set_json_field("event_data", data)


# ============================================================================
# EVENT LISTENERS (for automatic updates)
# ============================================================================


@event.listens_for(Order, "before_update")
def receive_before_update(mapper, connection, target):
    """Update timestamp before any order update."""
    target.updated_at = datetime.utcnow()


@event.listens_for(SystemState, "before_update")
def receive_before_update_state(mapper, connection, target):
    """Update timestamp before any system state update."""
    target.updated_at = datetime.utcnow()
