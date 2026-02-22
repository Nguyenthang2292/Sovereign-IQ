"""
SQLite to DynamoDB Migration Tool
=================================

Migrates data from SQLite to DynamoDB for the AutoTrade database.

Usage:
    python -m modules.auto_trade.database.migration_tool.sqlite_to_dynamodb \
        --sqlite-path data/auto_trade.db \
        --dynamodb-table AutoTrade \
        --region us-east-1

Created: 2026-02-20
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.auto_trade.database.models import (
    AuditLog,
    GradualRecovery,
    MartingaleChain,
    Order,
    Signal,
    SystemState,
)

TABLE_NAME = "AutoTrade"
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-southeast-1"


def get_dynamodb_table(endpoint_url: Optional[str] = None):
    """Get DynamoDB table resource."""
    import boto3

    kwargs: Dict[str, Any] = {"region_name": AWS_REGION}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    dynamodb = boto3.resource("dynamodb", **kwargs)
    return dynamodb.Table(TABLE_NAME)


def clean_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values from dict for DynamoDB."""
    return {key: value for key, value in data.items() if value is not None}


def to_dynamo_value(value: Any) -> Any:
    """Convert Python value to DynamoDB-compatible value."""
    if value is None:
        return None
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, datetime):
        return value.isoformat() if value.tzinfo else value.replace(tzinfo=timezone.utc).isoformat()
    if isinstance(value, dict):
        return {key: to_dynamo_value(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dynamo_value(nested_value) for nested_value in value]
    return value


def export_table(session: Any, model_class: Any) -> List[Any]:
    """Export all records from a SQLite table."""
    return session.query(model_class).all()


def transform_order(order: Order) -> Dict[str, Any]:
    """Transform SQLite Order to DynamoDB item format."""
    created_at = order.created_at or datetime.now(timezone.utc)
    created_iso = created_at.isoformat()
    order_source = order.order_source or "PROGRAMMATIC"

    entity: Dict[str, Any] = {
        "pk": f"ORDER#{order.order_id}",
        "sk": "METADATA",
        "entity_type": "ORDER",
        "order_id": order.order_id,
        "symbol": order.symbol,
        "side": order.side,
        "status": order.status,
        "order_source": order_source,
        "created_at": created_iso,
        "gsi1pk": order.symbol,
        "gsi1sk": f"ORDER#{order.status}#{created_iso}",
        "gsi2pk": "ORDER",
        "gsi2sk": created_iso,
        "gsi3pk": f"{order_source}#{order.status}",
        "gsi3sk": created_iso,
    }

    optional_fields = [
        "entry_price",
        "amount",
        "exit_price",
        "pnl",
        "stop_loss",
        "take_profit",
        "client_order_id",
        "be_moved",
        "closed_at",
    ]

    for field in optional_fields:
        value = getattr(order, field, None)
        if value is not None:
            entity[field] = to_dynamo_value(value)

    return clean_none_values(entity)


def transform_signal(signal: Signal) -> Dict[str, Any]:
    """Transform SQLite Signal to DynamoDB item format."""
    created_at = signal.created_at or datetime.now(timezone.utc)
    created_iso = created_at.isoformat()
    is_executed = bool(getattr(signal, "executed", False))
    execution_status = "EXECUTED" if is_executed else "PENDING"

    entity: Dict[str, Any] = {
        "pk": f"SIGNAL#{signal.correlation_id}",
        "sk": "METADATA",
        "entity_type": "SIGNAL",
        "correlation_id": signal.correlation_id,
        "symbol": signal.symbol,
        "signal_type": signal.signal_type,
        "created_at": created_iso,
        "gsi1pk": signal.symbol,
        "gsi1sk": f"SIGNAL#{execution_status}#{created_iso}",
        "gsi2pk": "SIGNAL",
        "gsi2sk": created_iso,
    }

    optional_fields = [
        "confidence",
        "executed",
        "execution_order_id",
        "executed_at",
        "outcome",
        "outcome_pnl",
        "outcome_at",
    ]

    for field in optional_fields:
        value = getattr(signal, field, None)
        if value is not None:
            entity[field] = to_dynamo_value(value)

    return clean_none_values(entity)


def transform_martingale_chain(chain: MartingaleChain) -> Dict[str, Any]:
    """Transform SQLite MartingaleChain to DynamoDB item format."""
    created_at = chain.created_at or datetime.now(timezone.utc)
    created_iso = created_at.isoformat()

    entity: Dict[str, Any] = {
        "pk": f"CHAIN#{chain.chain_id}",
        "sk": "METADATA",
        "entity_type": "CHAIN",
        "chain_id": chain.chain_id,
        "symbol": chain.symbol,
        "status": chain.status,
        "created_at": created_iso,
        "gsi1pk": chain.symbol,
        "gsi1sk": f"CHAIN#{chain.status}#{created_iso}",
        "gsi2pk": "CHAIN",
        "gsi2sk": created_iso,
    }

    optional_fields = [
        "initial_order_id",
        "initial_loss",
        "current_step",
        "max_steps",
        "total_loss",
        "recovered",
        "latest_order_id",
        "updated_at",
    ]

    for field in optional_fields:
        value = getattr(chain, field, None)
        if value is not None:
            entity[field] = to_dynamo_value(value)

    return clean_none_values(entity)


def transform_gradual_recovery(recovery: GradualRecovery) -> Dict[str, Any]:
    """Transform SQLite GradualRecovery to DynamoDB item format."""
    created_at = recovery.created_at or datetime.now(timezone.utc)
    created_iso = created_at.isoformat()

    entity: Dict[str, Any] = {
        "pk": f"RECOVERY#{recovery.recovery_id}",
        "sk": "METADATA",
        "entity_type": "RECOVERY",
        "recovery_id": recovery.recovery_id,
        "symbol": recovery.symbol,
        "status": recovery.status,
        "created_at": created_iso,
        "gsi1pk": recovery.symbol,
        "gsi1sk": f"RECOVERY#{recovery.status}#{created_iso}",
        "gsi2pk": "RECOVERY",
        "gsi2sk": created_iso,
    }

    optional_fields = [
        "target_pnl",
        "current_pnl",
        "step_count",
        "initial_loss",
        "remaining_loss",
        "total_profit_accumulated",
        "config",
        "updated_at",
        "completed_at",
        "failed_at",
    ]

    for field in optional_fields:
        value = getattr(recovery, field, None)
        if value is not None:
            entity[field] = to_dynamo_value(value)

    return clean_none_values(entity)


def transform_system_state(state: SystemState) -> Dict[str, Any]:
    """Transform SQLite SystemState to DynamoDB item format."""
    key_parts = state.key.split(".", 1)
    category = key_parts[0] if len(key_parts) > 1 else "global"

    value_type = str(state.value_type)
    raw_value = str(state.value)

    if value_type == "json":
        parsed_value = json.loads(raw_value)
    elif value_type in ("int", "integer"):
        parsed_value = int(raw_value)
    elif value_type == "float":
        parsed_value = float(raw_value)
    elif value_type == "boolean":
        parsed_value = raw_value.lower() in ("true", "1", "yes")
    else:
        parsed_value = raw_value

    entity: Dict[str, Any] = {
        "pk": f"STATE#{category}",
        "sk": f"KEY#{state.key}",
        "entity_type": "STATE",
        "key": state.key,
        "value": to_dynamo_value(parsed_value),
        "value_type": value_type,
        "category": category,
    }

    return clean_none_values(entity)


def transform_audit_log(log: AuditLog) -> Optional[Dict[str, Any]]:
    """Transform SQLite AuditLog to DynamoDB item format and skip expired items."""
    # AuditLog model uses 'timestamp' not 'created_at'
    created_at = getattr(log, "timestamp", None) or getattr(log, "created_at", None) or datetime.now(timezone.utc)
    created_at_utc = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)

    if datetime.now(timezone.utc) - created_at_utc > timedelta(days=90):
        return None

    created_iso = created_at_utc.isoformat()
    expire_at = int((created_at_utc + timedelta(days=90)).timestamp())

    entity: Dict[str, Any] = {
        "pk": f"AUDIT#{log.correlation_id or log.id}",
        "sk": f"TS#{created_iso}",
        "entity_type": "AUDIT",
        "correlation_id": log.correlation_id,
        "event_type": log.event_type,
        "event_category": log.event_category,
        "severity": log.severity,
        "event_summary": log.event_summary,
        "event_data": log.event_data,  # model field is event_data, not event_details
        "created_at": created_iso,
        "gsi2pk": "AUDIT",
        "gsi2sk": created_iso,
        "expire_at": expire_at,
    }

    return clean_none_values(entity)


def batch_write_items(table: Any, items: List[Dict[str, Any]], max_retries: int = 3) -> None:
    """Write items to DynamoDB in batches with retries."""
    for attempt in range(max_retries + 1):
        try:
            with table.batch_writer() as batch:
                for item in items:
                    batch.put_item(Item=item)
            return
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(0.5 * (2**attempt))


def count_dynamodb_items(table: Any, pk_prefix: str) -> int:
    """Count DynamoDB items matching a partition key prefix."""
    from boto3.dynamodb.conditions import Attr

    response = table.scan(
        FilterExpression=Attr("pk").begins_with(pk_prefix),
        Select="COUNT",
    )
    total = response.get("Count", 0)

    while "LastEvaluatedKey" in response:
        response = table.scan(
            FilterExpression=Attr("pk").begins_with(pk_prefix),
            ExclusiveStartKey=response["LastEvaluatedKey"],
            Select="COUNT",
        )
        total += response.get("Count", 0)

    return total


def migrate_table(
    session: Any,
    model_class: Any,
    transform_func: Callable[[Any], Optional[Dict[str, Any]]],
    table: Any,
    batch_size: int,
    verbose: bool,
) -> int:
    """Migrate a single entity type."""
    records = export_table(session, model_class)
    migrated = 0
    batch: List[Dict[str, Any]] = []

    if verbose:
        print(f"Migrating {model_class.__name__}: {len(records)} records")

    for record in records:
        item = transform_func(record)
        if item is None:
            continue
        batch.append(item)
        migrated += 1

        if len(batch) >= batch_size:
            batch_write_items(table, batch)
            batch = []

    if batch:
        batch_write_items(table, batch)

    return migrated


def migrate_all(
    sqlite_db_path: str,
    dynamodb_table_name: str,
    region: str,
    endpoint_url: Optional[str] = None,
    batch_size: int = 100,
    verbose: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the full migration and verify counts."""
    global TABLE_NAME, AWS_REGION
    TABLE_NAME = dynamodb_table_name
    AWS_REGION = region

    engine = create_engine(f"sqlite:///{sqlite_db_path}")
    session = sessionmaker(bind=engine)()
    table = get_dynamodb_table(endpoint_url)

    entities: List[Tuple[Any, Callable[[Any], Optional[Dict[str, Any]]], str, str]] = [
        (Signal, transform_signal, "Signals", "SIGNAL#"),
        (Order, transform_order, "Orders", "ORDER#"),
        (MartingaleChain, transform_martingale_chain, "Martingale Chains", "CHAIN#"),
        (GradualRecovery, transform_gradual_recovery, "Gradual Recoveries", "RECOVERY#"),
        (SystemState, transform_system_state, "System States", "STATE#"),
        (AuditLog, transform_audit_log, "Audit Logs", "AUDIT#"),
    ]

    report: Dict[str, Any] = {"dry_run": dry_run, "total_migrated": 0, "entities": {}}

    for model_class, transform_func, label, pk_prefix in entities:
        sqlite_count = session.query(model_class).count()
        if dry_run:
            migrated = sqlite_count
            dynamodb_count = None
            count_match = None
        else:
            migrated = migrate_table(session, model_class, transform_func, table, batch_size, verbose)
            dynamodb_count = count_dynamodb_items(table, pk_prefix)
            count_match = migrated == dynamodb_count

        report["entities"][label] = {
            "sqlite_count": sqlite_count,
            "migrated": migrated,
            "dynamodb_count": dynamodb_count,
            "count_match": count_match,
        }
        report["total_migrated"] += migrated

    session.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate SQLite to DynamoDB")
    parser.add_argument("--sqlite-path", default="data/auto_trade.db", help="Path to SQLite database")
    parser.add_argument("--dynamodb-table", default=TABLE_NAME, help="DynamoDB table name")
    parser.add_argument("--region", default=AWS_REGION, help="AWS region")
    parser.add_argument("--endpoint", help="DynamoDB endpoint URL (for local testing)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for writes")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    report = migrate_all(
        sqlite_db_path=args.sqlite_path,
        dynamodb_table_name=args.dynamodb_table,
        region=args.region,
        endpoint_url=args.endpoint,
        batch_size=args.batch_size,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\n=== DRY RUN ===")
    else:
        print("\n=== Migration Complete ===")

    for name, entity_report in report["entities"].items():
        print(
            f"{name}: sqlite={entity_report['sqlite_count']}, "
            f"migrated={entity_report['migrated']}, "
            f"dynamodb={entity_report['dynamodb_count']}"
        )

    print(f"Total migrated: {report['total_migrated']}")

    if not args.dry_run:
        mismatches = [
            name for name, entity_report in report["entities"].items() if entity_report["count_match"] is False
        ]
        if mismatches:
            print(f"Count mismatches detected: {', '.join(mismatches)}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
