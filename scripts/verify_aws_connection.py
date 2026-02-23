"""
AWS Connection Verification Script
====================================

Kiểm tra kết nối AWS đang hoạt động đúng không.
Chạy từ root dự án:

    python scripts/verify_aws_connection.py

    # Hoặc kiểm tra cả DynamoDB Local:
    python scripts/verify_aws_connection.py --local

Output màu xanh = OK, màu đỏ = Lỗi.
"""

from __future__ import annotations

import argparse
import sys
from typing import NamedTuple

# Thêm root vào sys.path để import config
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3  # noqa: E402
from botocore.exceptions import (  # noqa: E402
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
)

from config.aws_config import get_aws_config  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}[{title}]{RESET}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


class CheckResult(NamedTuple):
    passed: bool
    detail: str


def check_env_vars() -> CheckResult:
    """Kiểm tra biến môi trường cần thiết."""
    cfg = get_aws_config()
    missing = []

    if not cfg.access_key_id:
        missing.append("AWS_ACCESS_KEY_ID")
    if not cfg.secret_access_key:
        missing.append("AWS_SECRET_ACCESS_KEY")

    if missing:
        # Có thể dùng IAM Role / instance profile – không fatal
        return CheckResult(
            passed=False,
            detail=(
                f"Chưa set: {', '.join(missing)}\n"
                "    → Điền vào .env hoặc dùng 'aws configure'.\n"
                "    → Nếu đang chạy trên EC2/Lambda, IAM Role sẽ tự xác thực."
            ),
        )

    return CheckResult(passed=True, detail=f"region={cfg.region}, table={cfg.dynamodb_table}")


def check_sts_identity() -> CheckResult:
    """Gọi STS GetCallerIdentity để xác thực credentials."""
    cfg = get_aws_config()
    try:
        sts = boto3.client("sts", **{k: v for k, v in cfg.boto3_kwargs().items() if k != "endpoint_url"})
        identity = sts.get_caller_identity()
        account = identity["Account"]
        arn = identity["Arn"]
        return CheckResult(passed=True, detail=f"Account={account} | ARN={arn}")
    except NoCredentialsError:
        return CheckResult(passed=False, detail="Không tìm thấy credentials AWS. Điền vào .env hoặc chạy 'aws configure'.")
    except PartialCredentialsError as e:
        return CheckResult(passed=False, detail=f"Credentials không đầy đủ: {e}")
    except ClientError as e:
        return CheckResult(passed=False, detail=f"Lỗi API: {e.response['Error']['Message']}")
    except Exception as e:
        return CheckResult(passed=False, detail=f"Lỗi không xác định: {e}")


def check_dynamodb(local: bool = False) -> CheckResult:
    """Kiểm tra kết nối DynamoDB (cloud hoặc local)."""
    cfg = get_aws_config()
    kwargs = cfg.boto3_kwargs()

    if local and not cfg.dynamodb_endpoint_url:
        kwargs["endpoint_url"] = "http://localhost:8000"
        kwargs.setdefault("aws_access_key_id", "local")
        kwargs.setdefault("aws_secret_access_key", "local")

    try:
        client = boto3.client("dynamodb", **kwargs)
        response = client.list_tables(Limit=5)
        tables = response.get("TableNames", [])
        endpoint = kwargs.get("endpoint_url", f"https://dynamodb.{cfg.region}.amazonaws.com")

        table_info = f"Tables: {tables}" if tables else "Không có bảng nào (hoặc chưa tạo)"
        target_found = cfg.dynamodb_table in tables
        detail = f"Endpoint={endpoint} | {table_info}"

        if not target_found:
            detail += f"\n    ⚠  Bảng '{cfg.dynamodb_table}' chưa tồn tại. Chạy: python infrastructure/dynamodb/create_table.py"

        return CheckResult(passed=True, detail=detail)

    except EndpointConnectionError:
        endpoint = kwargs.get("endpoint_url", "AWS Cloud")
        return CheckResult(
            passed=False,
            detail=(
                f"Không kết nối được đến {endpoint}.\n"
                "    → Kiểm tra internet / Docker nếu dùng local."
            ),
        )
    except (NoCredentialsError, PartialCredentialsError):
        return CheckResult(passed=False, detail="Credentials AWS không hợp lệ.")
    except ClientError as e:
        return CheckResult(passed=False, detail=f"Lỗi API: {e.response['Error']['Message']}")
    except Exception as e:
        return CheckResult(passed=False, detail=f"Lỗi: {e}")


def check_s3() -> CheckResult:
    """Kiểm tra truy cập S3 (nếu S3_BUCKET_NAME hoặc MODEL_BUCKET được cấu hình)."""
    cfg = get_aws_config()
    # MODEL_BUCKET được dùng bởi deploy_lambda.py / xgboost_LTS_serverless
    bucket = cfg.s3_bucket or os.getenv("MODEL_BUCKET") or None
    if not bucket:
        return CheckResult(passed=False, detail="S3_BUCKET_NAME chưa cấu hình trong .env → bỏ qua.")

    kwargs = {k: v for k, v in cfg.boto3_kwargs().items() if k != "endpoint_url"}
    try:
        s3 = boto3.client("s3", **kwargs)
        s3.head_bucket(Bucket=bucket)

        # Đếm objects trong bucket
        resp = s3.list_objects_v2(Bucket=bucket, MaxKeys=5)
        count = resp.get("KeyCount", 0)
        is_truncated = resp.get("IsTruncated", False)
        count_str = f"{count}+" if is_truncated else str(count)

        objects = resp.get("Contents", [])
        sample = ", ".join(o["Key"] for o in objects[:3]) if objects else "(rỗng)"
        return CheckResult(
            passed=True,
            detail=f"s3://{bucket} tồn tại | objects: {count_str} | ví dụ: {sample}",
        )
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "404":
            return CheckResult(passed=False, detail=f"Bucket '{bucket}' không tồn tại.")
        if code == "403":
            return CheckResult(passed=False, detail=f"Không có quyền truy cập bucket '{bucket}' (403).")
        return CheckResult(passed=False, detail=f"Lỗi: {e.response['Error']['Message']}")
    except Exception as e:
        return CheckResult(passed=False, detail=f"Lỗi: {e}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_checks(local: bool = False) -> int:
    """Chạy toàn bộ checks. Trả về số lượng lỗi."""
    print(f"\n{BOLD}===== AWS Connection Verification ====={RESET}")

    cfg = get_aws_config()
    print(f"Config: {cfg}")

    errors = 0

    # 1. Env vars
    section("1. Environment Variables")
    r = check_env_vars()
    if r.passed:
        ok(r.detail)
    else:
        warn(r.detail)
        # Không tính là lỗi cứng – có thể dùng IAM Role

    # 2. STS Identity
    section("2. AWS STS – GetCallerIdentity")
    r = check_sts_identity()
    if r.passed:
        ok(r.detail)
    else:
        fail(r.detail)
        errors += 1

    # 3. DynamoDB
    section(f"3. DynamoDB {'(Local)' if local else '(Cloud)'}")
    r = check_dynamodb(local=local)
    if r.passed:
        ok(r.detail)
    else:
        fail(r.detail)
        errors += 1

    # 4. S3
    section("4. S3 Bucket (tuỳ chọn)")
    r = check_s3()
    if r.passed:
        ok(r.detail)
    else:
        warn(r.detail)  # S3 là optional

    # Summary
    print()
    if errors == 0:
        print(f"{GREEN}{BOLD}✓ Tất cả kết nối AWS hoạt động bình thường!{RESET}")
    else:
        print(f"{RED}{BOLD}✗ {errors} lỗi kết nối – xem hướng dẫn phía trên.{RESET}")
        print(f"\nTips:\n"
              f"  1. Điền AWS_ACCESS_KEY_ID và AWS_SECRET_ACCESS_KEY vào .env\n"
              f"  2. Hoặc chạy: aws configure\n"
              f"  3. Tài liệu: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra kết nối AWS của dự án crypto-probability")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Kiểm tra DynamoDB Local (http://localhost:8000) thay vì AWS Cloud",
    )
    args = parser.parse_args()

    exit_code = run_checks(local=args.local)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
