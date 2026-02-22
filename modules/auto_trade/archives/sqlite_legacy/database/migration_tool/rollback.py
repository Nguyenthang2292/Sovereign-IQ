"""
Rollback Script
===============

Emergency rollback - deletes all items from DynamoDB table.
USE WITH CAUTION - This will delete all data!

Usage:
    python -m modules.auto_trade.database.migration_tool.rollback \\
        --dynamodb-table AutoTrade \\
        --region us-east-1

Created: 2026-02-20
"""

import argparse
import os
import sys
from typing import Optional

import boto3
from botocore.exceptions import ClientError


TABLE_NAME = "AutoTrade"
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-southeast-1"


def get_dynamodb_table(endpoint_url: Optional[str] = None):
    """Get DynamoDB table resource."""
    kwargs = {"region_name": AWS_REGION}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    dynamodb = boto3.resource("dynamodb", **kwargs)
    return dynamodb.Table(TABLE_NAME)


def delete_all_items(table, batch_size: int = 25):
    """Delete all items from a DynamoDB table."""
    print("Scanning for items to delete...")

    total_deleted = 0
    scanned = 0

    while True:
        # Scan for items
        response = table.scan()
        items = response.get("Items", [])
        scanned += len(items)

        if not items:
            break

        # Delete items in batches
        print(f"Deleting batch of {len(items)} items...")

        with table.batch_writer() as batch:
            for item in items:
                # Get the primary key
                pk = item.get("pk")
                sk = item.get("sk")

                if pk and sk:
                    batch.delete_item(Key={"pk": pk, "sk": sk})
                    total_deleted += 1

        # Check if there are more items
        if "LastEvaluatedKey" not in response:
            break

    print(f"Scanned {scanned} items, deleted {total_deleted} items")
    return total_deleted


def main():
    global TABLE_NAME, AWS_REGION

    parser = argparse.ArgumentParser(description="Rollback DynamoDB - DELETE ALL DATA")
    parser.add_argument("--dynamodb-table", default=TABLE_NAME, help="DynamoDB table name")
    parser.add_argument("--region", default=AWS_REGION, help="AWS region")
    parser.add_argument("--endpoint", help="DynamoDB endpoint URL (for local testing)")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    TABLE_NAME = args.dynamodb_table
    AWS_REGION = args.region

    print("=" * 60)
    print("WARNING: This will DELETE ALL DATA from the DynamoDB table!")
    print("=" * 60)
    print()

    if not args.force:
        response = input(f"Type 'yes' to confirm deletion of all data from '{TABLE_NAME}': ")
        if response.lower() != "yes":
            print("Cancelled.")
            return 1

    print(f"\nConnecting to DynamoDB: {args.region}/{args.dynamodb_table}")
    table = get_dynamodb_table(args.endpoint)

    print(f"Deleting all items from table '{TABLE_NAME}'...")
    delete_all_items(table)

    print("\n✓ Rollback complete - table is now empty")
    return 0


if __name__ == "__main__":
    sys.exit(main())
