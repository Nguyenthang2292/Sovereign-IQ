"""Test GSI1 query on DynamoDB after region fix."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
import boto3
from boto3.dynamodb.conditions import Key

region     = os.environ.get("DYNAMODB_REGION", os.environ.get("AWS_REGION", "ap-southeast-1"))
table_name = os.environ.get("DYNAMODB_TABLE_NAME", "AutoTrade")

print(f"Region : {region}")
print(f"Table  : {table_name}")
print()

ddb = boto3.resource(
    "dynamodb",
    region_name=region,
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)
table = ddb.Table(table_name)

try:
    resp = table.query(
        IndexName="GSI1",
        KeyConditionExpression=Key("gsi1pk").eq("BTCUSDT") & Key("gsi1sk").begins_with("RECOVERY#ACTIVE#"),
        Limit=1,
    )
    print(f"GSI1 query OK — items returned: {len(resp['Items'])}")
except Exception as e:
    print(f"ERROR: {e}")
