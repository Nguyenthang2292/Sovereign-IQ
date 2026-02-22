"""
Step 7: Monitor - DynamoDB Table Health Check
"""

import os
import sys

sys.path.insert(0, ".")
os.environ["DB_BACKEND"] = "dynamodb"
os.environ["DYNAMODB_TABLE_NAME"] = "AutoTrade"
os.environ["AWS_REGION"] = "ap-southeast-1"

import boto3

dynamodb = boto3.client("dynamodb", region_name="ap-southeast-1")

resp = dynamodb.describe_table(TableName="AutoTrade")
t = resp["Table"]
print("=== DynamoDB Table Status ===")
print("Table:", t["TableName"])
print("Status:", t["TableStatus"])
print("Item count (approx):", t.get("ItemCount", "N/A"))
print("Size bytes (approx):", t.get("TableSizeBytes", "N/A"))
bms = t.get("BillingModeSummary", {})
print("Billing mode:", bms.get("BillingMode", "N/A"))

gsis = t.get("GlobalSecondaryIndexes", [])
print("GSIs: {} indexes".format(len(gsis)))
for g in gsis:
    print("  {}: {}".format(g["IndexName"], g["IndexStatus"]))

pitr = dynamodb.describe_continuous_backups(TableName="AutoTrade")
pitr_status = pitr["ContinuousBackupsDescription"]["PointInTimeRecoveryDescription"]["PointInTimeRecoveryStatus"]
print("PITR:", pitr_status)

ttl = dynamodb.describe_time_to_live(TableName="AutoTrade")
ttl_status = ttl["TimeToLiveDescription"]["TimeToLiveStatus"]
print("TTL:", ttl_status, "(attribute: expire_at)")

# Live backend smoke check
from modules.auto_trade.database.repository.context import RepositoryContext

ctx = RepositoryContext.from_env()
orders = ctx.orders.get_open_positions()
signals = ctx.signals.get_recent_signals(limit=3)
trading_enabled = ctx.system_state.get_system_state("trading_enabled")

print("")
print("=== Live Backend Check ===")
print("Open positions:", len(orders))
print("Recent signals:", len(signals))
print("trading_enabled:", trading_enabled)

print("")
print("=" * 50)
if (
    t["TableStatus"] == "ACTIVE"
    and pitr_status == "ENABLED"
    and ttl_status in ("ENABLED", "ENABLING")
    and trading_enabled is True
):
    print("STEP 7: ALL SYSTEMS GO - Cutover Successful!")
else:
    print("STEP 7: WARNING - check output above")
print("=" * 50)
