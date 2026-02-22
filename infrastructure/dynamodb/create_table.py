"""
DynamoDB Table Creation Script
==============================

Creates the AutoTrade DynamoDB table with the required schema.

Usage:
    python create_table.py --env prod --region us-east-1
    python create_table.py --env local --endpoint http://localhost:8000

Created: 2026-02-20
"""

import argparse
import json
import os
import sys

import boto3
from botocore.exceptions import ClientError


def get_table_definition(env: str = "prod") -> dict:
    """Load and update table definition with environment-specific values."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    definition_path = os.path.join(script_dir, "table_definition.json")

    with open(definition_path, "r") as f:
        definition = json.load(f)

    # Update tags with environment
    for tag in definition.get("Tags", []):
        if tag["Key"] == "Environment":
            tag["Value"] = env

    return definition


def create_table(dynamodb_resource, definition: dict, wait: bool = True):
    """Create the DynamoDB table."""
    table_name = definition["TableName"]

    try:
        # Check if table already exists
        existing_tables = dynamodb_resource.meta.client.list_tables().get("TableNames", [])
        if table_name in existing_tables:
            print(f"Table '{table_name}' already exists.")
            return dynamodb_resource.Table(table_name)

        # Create table
        print(f"Creating table '{table_name}'...")

        # Keep only parameters accepted by create_table
        create_params = {
            "TableName": definition["TableName"],
            "KeySchema": definition["KeySchema"],
            "AttributeDefinitions": definition["AttributeDefinitions"],
            "BillingMode": definition["BillingMode"],
            "GlobalSecondaryIndexes": definition.get("GlobalSecondaryIndexes", []),
            "Tags": definition.get("Tags", []),
        }

        # Add SSE if present
        if "SSESpecification" in definition:
            create_params["SSESpecification"] = definition["SSESpecification"]

        table = dynamodb_resource.create_table(**create_params)

        if wait:
            print("Waiting for table to be created...")
            table.wait_until_exists()
            print(f"Table '{table_name}' created successfully!")

        return table

    except ClientError as e:
        print(f"Error creating table: {e.response['Error']['Message']}")
        raise


def enable_ttl(table):
    """Enable TTL on the table for expire_at attribute."""
    try:
        table.meta.client.update_time_to_live(
            TableName=table.name, TimeToLiveSpecification={"Enabled": True, "AttributeName": "expire_at"}
        )
        print("TTL enabled on 'expire_at' attribute")
    except ClientError as e:
        print(f"Warning: Could not enable TTL: {e.response['Error']['Message']}")


def enable_point_in_time_recovery(table):
    """Enable point-in-time recovery (PITR) for the table."""
    try:
        table.meta.client.update_continuous_backups(
            TableName=table.name,
            PointInTimeRecoverySpecification={"PointInTimeRecoveryEnabled": True},
        )
        print("Point-in-time recovery enabled")
    except ClientError as e:
        print(f"Warning: Could not enable PITR: {e.response['Error']['Message']}")


def main():
    parser = argparse.ArgumentParser(description="Create AutoTrade DynamoDB table")
    parser.add_argument(
        "--env", choices=["prod", "staging", "local"], default="prod", help="Environment (affects tags)"
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-southeast-1",
        help="AWS region",
    )
    parser.add_argument(
        "--endpoint", default=os.getenv("DYNAMODB_ENDPOINT_URL"), help="DynamoDB endpoint URL (for local development)"
    )
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for table creation to complete")
    parser.add_argument("--skip-ttl", action="store_true", help="Skip TTL configuration")
    parser.add_argument("--skip-pitr", action="store_true", help="Skip point-in-time recovery configuration")

    args = parser.parse_args()

    # Create DynamoDB client
    kwargs = {
        "region_name": args.region,
    }

    if args.endpoint:
        kwargs["endpoint_url"] = args.endpoint
        print(f"Using local DynamoDB at {args.endpoint}")
    elif args.env == "local":
        kwargs["endpoint_url"] = "http://localhost:8000"
        kwargs["aws_access_key_id"] = "local"
        kwargs["aws_secret_access_key"] = "local"
        print("Using local DynamoDB at http://localhost:8000")

    dynamodb = boto3.resource("dynamodb", **kwargs)

    # Load and create table
    definition = get_table_definition(args.env)
    table = create_table(dynamodb, definition, wait=not args.no_wait)

    # Enable TTL
    if not args.skip_ttl:
        enable_ttl(table)

    if not args.skip_pitr:
        enable_point_in_time_recovery(table)

    print(f"\nTable '{table.name}' is ready!")
    print(f"  Region: {args.region}")
    if args.endpoint or args.env == "local":
        print(f"  Endpoint: {kwargs.get('endpoint_url', 'N/A')}")


if __name__ == "__main__":
    main()
