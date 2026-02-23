"""Check CloudFormation stack events for FAILED resources.

Credentials are loaded from environment variables (or .env file).
Never hardcode AWS keys in source code.

Required env vars:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION     (optional, defaults to us-east-1)
    CFN_STACK_NAME         (optional, defaults to xgboost-lts-serverless-staging)
"""

import os
import sys

# Load .env from repo root (3 levels up from this script)
try:
    from dotenv import load_dotenv

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
    load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)
except ImportError:
    pass  # dotenv not installed — rely on shell environment

import boto3

# ── Config from environment ───────────────────────────────────────────────────
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
STACK_NAME = os.environ.get("CFN_STACK_NAME", "xgboost-lts-serverless-staging")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    print(
        "ERROR: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set as environment variables or in the .env file.",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Output file ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
OUTPUT_FILE = os.path.join(REPO_ROOT, "stack_errors.txt")

# ── CloudFormation client ─────────────────────────────────────────────────────
cfn = boto3.client(
    "cloudformation",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# ── Fetch and filter events ───────────────────────────────────────────────────
try:
    evts = cfn.describe_stack_events(StackName=STACK_NAME)["StackEvents"]
    print(f"Total events: {len(evts)}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for event in evts:
            status = event.get("ResourceStatus", "")
            if "FAILED" in status:
                rid = event["LogicalResourceId"]
                reason = event.get("ResourceStatusReason", "")
                f.write(f"[FAILED] Resource: {rid}\n")
                f.write(f"  Status: {status}\n")
                f.write(f"  Reason: {reason}\n\n")
    print(f"Written to {OUTPUT_FILE}")
except Exception as ex:
    print(f"Error: {ex}")
