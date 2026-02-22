"""
DynamoDB Test Fixtures
======================

Sets up moto mock for DynamoDB.

Created: 2026-02-20
"""

import importlib
import os

import boto3  # type: ignore[import-untyped]
import pytest

# Ensure backend is dynamodb for these tests
os.environ["DB_BACKEND"] = "dynamodb"
os.environ["DYNAMODB_TABLE_NAME"] = "TestAutoTrade"
os.environ["AWS_REGION"] = "us-east-1"


@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"


@pytest.fixture(scope="function")
def dynamodb_client(aws_credentials):
    """DynamoDB mock client."""
    try:
        moto_module = importlib.import_module("moto")
    except ImportError:
        pytest.skip("moto is required for DynamoDB tests")

    mock_aws = getattr(moto_module, "mock_aws", None) or getattr(moto_module, "mock_dynamodb", None)
    if mock_aws is None:
        pytest.skip("moto mock context is unavailable for DynamoDB tests")

    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-east-1")
        yield client


@pytest.fixture(scope="function")
def setup_dynamodb_table(dynamodb_client):
    """Create the exact DynamoDB table schema needed for AutoTrade."""
    table_name = "TestAutoTrade"

    dynamodb_client.create_table(
        TableName=table_name,
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
            {"AttributeName": "gsi1pk", "AttributeType": "S"},
            {"AttributeName": "gsi1sk", "AttributeType": "S"},
            {"AttributeName": "gsi2pk", "AttributeType": "S"},
            {"AttributeName": "gsi2sk", "AttributeType": "S"},
            {"AttributeName": "gsi3pk", "AttributeType": "S"},
            {"AttributeName": "gsi3sk", "AttributeType": "S"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "GSI1",
                "KeySchema": [
                    {"AttributeName": "gsi1pk", "KeyType": "HASH"},
                    {"AttributeName": "gsi1sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "GSI2",
                "KeySchema": [
                    {"AttributeName": "gsi2pk", "KeyType": "HASH"},
                    {"AttributeName": "gsi2sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "GSI3",
                "KeySchema": [
                    {"AttributeName": "gsi3pk", "KeyType": "HASH"},
                    {"AttributeName": "gsi3sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
        BillingMode="PAY_PER_REQUEST",
    )

    # We must reset cached connections in client so it picks up the mocked boto3 session
    from modules.auto_trade.database.repository.dynamodb.client import reset_connections

    reset_connections()

    yield table_name

    reset_connections()
