"""
Migration Verification Script
=============================

Verifies that migration from SQLite to DynamoDB was successful.

Usage:
    python -m modules.auto_trade.database.migration_tool.verify_migration \
        --sqlite-path data/auto_trade.db \
        --dynamodb-table AutoTrade \
        --region us-east-1

Created: 2026-02-20
"""

import argparse
import os
import random
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

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


def count_dynamodb_items(table: Any, pk_prefix: str) -> int:
    """Count items in DynamoDB by PK prefix."""
    from boto3.dynamodb.conditions import Attr

    response = table.scan(
        FilterExpression=Attr("pk").begins_with(pk_prefix),
        Select="COUNT",
    )
    count = response.get("Count", 0)

    while "LastEvaluatedKey" in response:
        response = table.scan(
            FilterExpression=Attr("pk").begins_with(pk_prefix),
            ExclusiveStartKey=response["LastEvaluatedKey"],
            Select="COUNT",
        )
        count += response.get("Count", 0)

    return count


def compare_values(sqlite_val: Any, dynamo_val: Any) -> bool:
    """Compare SQLite and DynamoDB values."""
    if sqlite_val is None and dynamo_val is None:
        return True
    if sqlite_val is None or dynamo_val is None:
        return False

    if isinstance(dynamo_val, Decimal):
        try:
            return float(sqlite_val) == float(dynamo_val)
        except (ValueError, TypeError):
            return str(sqlite_val) == str(dynamo_val)

    if hasattr(sqlite_val, "isoformat"):
        return sqlite_val.isoformat() == str(dynamo_val)

    return str(sqlite_val) == str(dynamo_val)


def dynamo_key_for_record(pk_prefix: str, record: Any) -> Tuple[str, str]:
    """Build DynamoDB key from SQLite record for each entity type."""
    if pk_prefix == "ORDER#":
        return f"ORDER#{record.order_id}", "METADATA"
    if pk_prefix == "SIGNAL#":
        return f"SIGNAL#{record.correlation_id}", "METADATA"
    if pk_prefix == "CHAIN#":
        return f"CHAIN#{record.chain_id}", "METADATA"
    if pk_prefix == "RECOVERY#":
        return f"RECOVERY#{record.recovery_id}", "METADATA"
    if pk_prefix == "STATE#":
        key_parts = record.key.split(".", 1)
        category = key_parts[0] if len(key_parts) > 1 else "global"
        return f"STATE#{category}", f"KEY#{record.key}"

    # AuditLog uses .timestamp not .created_at
    # Migration normalizes naive datetimes to UTC (+00:00), so we must match
    from datetime import timezone as _tz

    created_at_val = getattr(record, "timestamp", None) or getattr(record, "created_at", None)
    if created_at_val:
        if created_at_val.tzinfo is None:
            created_at_val = created_at_val.replace(tzinfo=_tz.utc)
        created_at = created_at_val.isoformat()
    else:
        created_at = ""
    audit_pk_suffix = record.correlation_id or record.id
    return f"AUDIT#{audit_pk_suffix}", f"TS#{created_at}"


def verify_entity(
    session: Any,
    dynamodb_table: Any,
    sqlite_model: Any,
    pk_prefix: str,
    sample_size: int,
    field_checks: List[str],
) -> Dict[str, Any]:
    """Verify migration for a specific entity type."""
    sqlite_records = session.query(sqlite_model).all()
    sqlite_count = len(sqlite_records)
    dynamo_count = count_dynamodb_items(dynamodb_table, pk_prefix)

    result: Dict[str, Any] = {
        "sqlite_count": sqlite_count,
        "dynamo_count": dynamo_count,
        "count_match": sqlite_count == dynamo_count,
        "sample_verified": True,
        "sample_errors": [],
    }

    if sqlite_count == 0:
        return result

    sample_count = min(sample_size, sqlite_count)
    sampled = random.sample(sqlite_records, sample_count)

    for sqlite_record in sampled:
        pk, sk = dynamo_key_for_record(pk_prefix, sqlite_record)
        response = dynamodb_table.get_item(Key={"pk": pk, "sk": sk})
        dynamo_item = response.get("Item")

        if not dynamo_item:
            result["sample_errors"].append(f"Missing item: pk={pk}, sk={sk}")
            result["sample_verified"] = False
            continue

        for field in field_checks:
            sqlite_val = getattr(sqlite_record, field, None)
            dynamo_val = dynamo_item.get(field)
            if not compare_values(sqlite_val, dynamo_val):
                result["sample_errors"].append(f"Mismatch {pk}:{field} sqlite={sqlite_val} dynamodb={dynamo_val}")
                result["sample_verified"] = False

    return result


def main() -> int:
    global TABLE_NAME, AWS_REGION

    parser = argparse.ArgumentParser(description="Verify SQLite to DynamoDB migration")
    parser.add_argument("--sqlite-path", default="data/auto_trade.db", help="Path to SQLite database")
    parser.add_argument("--dynamodb-table", default=TABLE_NAME, help="DynamoDB table name")
    parser.add_argument("--region", default=AWS_REGION, help="AWS region")
    parser.add_argument("--endpoint", help="DynamoDB endpoint URL (for local testing)")
    parser.add_argument("--sample-size", type=int, default=10, help="Random samples per entity")
    parser.add_argument("--dry-run", action="store_true", help="Only report SQLite counts")
    args = parser.parse_args()

    engine = create_engine(f"sqlite:///{args.sqlite_path}")
    session = sessionmaker(bind=engine)()

    entity_checks: List[Tuple[Any, str, str, List[str]]] = [
        (Signal, "SIGNAL#", "Signals", ["symbol", "signal_type", "executed"]),
        (Order, "ORDER#", "Orders", ["symbol", "status", "order_source"]),
        (MartingaleChain, "CHAIN#", "Martingale Chains", ["symbol", "status", "current_step"]),
        (GradualRecovery, "RECOVERY#", "Gradual Recoveries", ["symbol", "status"]),
        (SystemState, "STATE#", "System States", ["key", "value_type"]),
        (AuditLog, "AUDIT#", "Audit Logs", ["event_type", "severity"]),
    ]

    if args.dry_run:
        print("=== DRY RUN (SQLite counts only) ===")
        for model, _, name, _ in entity_checks:
            print(f"{name}: {session.query(model).count()}")
        session.close()
        return 0

    TABLE_NAME = args.dynamodb_table
    AWS_REGION = args.region
    dynamodb_table = get_dynamodb_table(args.endpoint)

    all_passed = True
    for sqlite_model, pk_prefix, name, fields in entity_checks:
        print(f"\nVerifying {name}...")
        result = verify_entity(
            session=session,
            dynamodb_table=dynamodb_table,
            sqlite_model=sqlite_model,
            pk_prefix=pk_prefix,
            sample_size=args.sample_size,
            field_checks=fields,
        )

        count_ok = "✓" if result["count_match"] else "✗"
        sample_ok = "✓" if result["sample_verified"] else "✗"

        print(f"  Count: {count_ok} SQLite={result['sqlite_count']}, DynamoDB={result['dynamo_count']}")
        print(f"  Sample: {sample_ok}")

        if result["sample_errors"]:
            print("  Errors:")
            for error in result["sample_errors"][:5]:
                print(f"    - {error}")

        if not result["count_match"] or not result["sample_verified"]:
            all_passed = False

    session.close()

    print("\n" + "=" * 50)
    print("✓ VERIFICATION PASSED" if all_passed else "✗ VERIFICATION FAILED")
    print("=" * 50)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
