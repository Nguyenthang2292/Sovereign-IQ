#!/usr/bin/env python3
"""
Deploy XGBoost Trainer Stack to AWS CloudFormation via boto3.
Bypasses aws/sam CLI pager issues entirely.

Credentials are loaded from environment variables (or .env file).
Never hardcode AWS keys in source code.

Required env vars:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION  (optional, defaults to us-east-1)
    CFN_STACK_NAME      (optional, defaults to xgboost-lts-serverless-staging)
"""

import os
import sys
import time

# Load .env from repo root (3 levels up from this script)
try:
    from dotenv import load_dotenv

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
    load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)
except ImportError:
    pass  # dotenv not installed — rely on shell environment

import boto3
import botocore

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
TEMPLATE_FILE = os.path.join(MODULE_ROOT, "template-trainer-only.yaml")

cfn = boto3.client(
    "cloudformation",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def get_stack_status():
    try:
        resp = cfn.describe_stacks(StackName=STACK_NAME)
        st = resp["Stacks"][0]
        return st["StackStatus"], st.get("StackStatusReason", "")
    except botocore.exceptions.ClientError as e:
        if "does not exist" in str(e):
            return None, None
        raise


def delete_if_review_in_progress():
    status, reason = get_stack_status()
    print(f"Current stack status: {status} ({reason})")
    if status in ("REVIEW_IN_PROGRESS", "ROLLBACK_COMPLETE"):
        print(f"Deleting stack stuck at {status}...")
        cfn.delete_stack(StackName=STACK_NAME)
        while True:
            status, _ = get_stack_status()
            if status is None or status == "DELETE_COMPLETE":
                break
            print(f"  Waiting for delete... {status}")
            time.sleep(5)
        print("Stack deleted.")


def wait_for_stack(timeout=600):
    start = time.time()
    seen_events = set()
    last_status = None
    while time.time() - start < timeout:
        status, reason = get_stack_status()
        if status != last_status:
            print(f"\n[Stack] {status} - {reason}")
            last_status = status

        try:
            events = cfn.describe_stack_events(StackName=STACK_NAME)["StackEvents"]
            for ev in reversed(events):
                eid = ev["EventId"]
                if eid not in seen_events:
                    seen_events.add(eid)
                    state = ev.get("ResourceStatus", "")
                    name = ev.get("LogicalResourceId", "")
                    reason_ev = ev.get("ResourceStatusReason", "")
                    print(f"  {name}: {state}  {reason_ev}")
        except Exception:
            pass

        terminal = {
            "CREATE_COMPLETE",
            "UPDATE_COMPLETE",
            "DELETE_COMPLETE",
            "CREATE_FAILED",
            "UPDATE_FAILED",
            "ROLLBACK_COMPLETE",
            "ROLLBACK_FAILED",
            "UPDATE_ROLLBACK_COMPLETE",
        }
        if status in terminal:
            return status, reason
        time.sleep(5)
    return "TIMEOUT", ""


with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
    template_body = f.read()

print("=== Step 1: Clean up old stacks ===")
delete_if_review_in_progress()

print("\n=== Step 2: Creating CloudFormation stack ===")
common_args = dict(
    Capabilities=["CAPABILITY_IAM", "CAPABILITY_AUTO_EXPAND"],
    Parameters=[],
)

status, _ = get_stack_status()
try:
    if status is None or status == "DELETE_COMPLETE":
        print("Creating new stack...")
        cfn.create_stack(StackName=STACK_NAME, TemplateBody=template_body, **common_args)
    else:
        print(f"Updating existing stack ({status})...")
        cfn.update_stack(StackName=STACK_NAME, TemplateBody=template_body, **common_args)
except botocore.exceptions.ClientError as e:
    if "No updates are to be performed" in str(e):
        print("No changes — stack is already up to date.")
        sys.exit(0)
    raise

print("\n=== Step 3: Waiting for stack to complete ===")
final_status, final_reason = wait_for_stack()

if "COMPLETE" in final_status and "ROLLBACK" not in final_status:
    print(f"\n=== SUCCESS: {final_status} ===")
    resp = cfn.describe_stacks(StackName=STACK_NAME)
    outputs = resp["Stacks"][0].get("Outputs", [])
    print("\nOutputs:")
    for o in outputs:
        print(f"  {o['OutputKey']}: {o['OutputValue']}")
    sys.exit(0)
else:
    print(f"\n=== FAILED: {final_status} - {final_reason} ===")
    sys.exit(1)
