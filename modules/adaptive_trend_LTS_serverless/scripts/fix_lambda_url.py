#!/usr/bin/env python3
"""
Fix Lambda Function URL auth: delete and recreate with AuthType=NONE (public).
Run once after deployment to enable unauthenticated HTTP access.
"""

import json
import time
import boto3

FUNCTION_NAME = "atc-serverless"
REGION = "us-east-1"

client = boto3.client("lambda", region_name=REGION)

# Step 1: Delete existing Function URL
print("Step 1: Deleting existing Function URL...")
try:
    client.delete_function_url_config(FunctionName=FUNCTION_NAME)
    print("  Deleted OK")
except Exception as e:
    print(f"  Skip (not found): {e}")

# Step 2: Remove all existing resource-based permissions
print("Step 2: Removing existing resource permissions...")
try:
    pol_resp = client.get_policy(FunctionName=FUNCTION_NAME)
    policy = json.loads(pol_resp["Policy"])
    for stmt in policy["Statement"]:
        sid = stmt["Sid"]
        client.remove_permission(FunctionName=FUNCTION_NAME, StatementId=sid)
        print(f"  Removed: {sid}")
except client.exceptions.ResourceNotFoundException:
    print("  No policy found, skipping")
except Exception as e:
    print(f"  Error: {e}")

time.sleep(3)

# Step 3: Create new Function URL with AuthType=NONE
print("Step 3: Creating new Function URL with AuthType=NONE...")
result = client.create_function_url_config(
    FunctionName=FUNCTION_NAME,
    AuthType="NONE",
    Cors={
        "AllowOrigins": ["*"],
        "AllowMethods": ["POST", "GET"],
        "AllowHeaders": ["Content-Type"],
        "MaxAge": 86400,
    },
)
new_url = result["FunctionUrl"]
print(f"  Created: {new_url}")
print(f"  AuthType: {result['AuthType']}")

# Step 4: Add public invoke permission
print("Step 4: Adding public invoke permission...")
client.add_permission(
    FunctionName=FUNCTION_NAME,
    StatementId="AllowPublicInvoke",
    Action="lambda:InvokeFunctionUrl",
    Principal="*",
    FunctionUrlAuthType="NONE",
)
print("  Permission added OK")

print()
print("=" * 60)
print("DONE! New Lambda Function URL (public, no auth):")
print(f"  {new_url}")
print()
print("Test with:")
print(f"  python binance_lambda_demo.py --symbols 3 --endpoint {new_url}")
print("=" * 60)
