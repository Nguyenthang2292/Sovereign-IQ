"""
AWS Configuration Module
========================

Cung cấp cấu hình AWS tập trung cho toàn bộ dự án.
Tự động load từ .env (hoặc biến môi trường hệ thống).

Usage:
    from config.aws_config import AWSConfig, get_aws_config

    cfg = get_aws_config()
    print(cfg.region)          # ap-southeast-1
    print(cfg.dynamodb_table)  # AutoTrade

Created: 2026-02-23
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

# Load .env từ root dự án (một lần khi module được import)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"), override=False)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AWSConfig:
    """Immutable snapshot của cấu hình AWS lấy từ environment variables."""

    # --- Credentials (None → boto3 tự tìm trong ~/.aws/credentials) ---------
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    profile: Optional[str] = None  # AWS_PROFILE

    # --- Region ---------------------------------------------------------------
    region: str = "ap-southeast-1"

    # --- DynamoDB -------------------------------------------------------------
    dynamodb_table: str = "AutoTrade"
    dynamodb_endpoint_url: Optional[str] = None  # None = AWS cloud; local dev dùng http://localhost:8000

    # --- S3 -------------------------------------------------------------------
    s3_bucket: Optional[str] = None
    s3_model_prefix: str = "models/xgboost_LTS/"

    # --- Lambda ---------------------------------------------------------------
    lambda_function_name: Optional[str] = None

    # --- Flags ----------------------------------------------------------------
    dry_run: bool = True

    # -------------------------------------------------------------------------

    @property
    def has_explicit_credentials(self) -> bool:
        """True nếu credentials được cung cấp tường minh qua env vars."""
        return bool(self.access_key_id and self.secret_access_key)

    @property
    def is_local(self) -> bool:
        """True nếu đang chạy với DynamoDB Local endpoint."""
        return self.dynamodb_endpoint_url is not None

    def boto3_kwargs(self) -> dict:
        """
        Trả về kwargs để truyền vào boto3.client() / boto3.resource().

        Ví dụ:
            import boto3
            from config.aws_config import get_aws_config

            cfg = get_aws_config()
            dynamodb = boto3.resource("dynamodb", **cfg.boto3_kwargs())
        """
        kwargs: dict = {"region_name": self.region}

        if self.has_explicit_credentials:
            kwargs["aws_access_key_id"] = self.access_key_id
            kwargs["aws_secret_access_key"] = self.secret_access_key

        if self.dynamodb_endpoint_url:
            kwargs["endpoint_url"] = self.dynamodb_endpoint_url

        return kwargs

    def __repr__(self) -> str:  # ẩn secret key khi print
        key_display = f"...{self.access_key_id[-4:]}" if self.access_key_id else "from ~/.aws"
        return (
            f"AWSConfig("
            f"region={self.region!r}, "
            f"table={self.dynamodb_table!r}, "
            f"access_key={key_display}, "
            f"local={self.is_local}, "
            f"dry_run={self.dry_run}"
            f")"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _parse_bool(value: str) -> bool:
    """Parse chuỗi env var thành bool."""
    return value.strip().lower() in ("1", "true", "yes", "on")


@lru_cache(maxsize=1)
def get_aws_config() -> AWSConfig:
    """
    Singleton: đọc biến môi trường và trả về AWSConfig.

    Ưu tiên:
      1. Biến hệ thống (export AWS_ACCESS_KEY_ID=...)
      2. File .env ở root dự án
      3. Mặc định trong AWSConfig

    Sau khi gọi lần đầu, kết quả được cache – gọi
    ``get_aws_config.cache_clear()`` để reset (chủ yếu dùng trong tests).
    """
    return AWSConfig(
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
        profile=os.getenv("AWS_PROFILE") or None,
        region=(
            os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "ap-southeast-1"
        ),
        dynamodb_table=os.getenv("DYNAMODB_TABLE_NAME", "AutoTrade"),
        dynamodb_endpoint_url=os.getenv("DYNAMODB_ENDPOINT_URL") or None,
        s3_bucket=os.getenv("S3_BUCKET_NAME") or None,
        s3_model_prefix=os.getenv("S3_MODEL_PREFIX", "models/xgboost_LTS/"),
        lambda_function_name=os.getenv("LAMBDA_FUNCTION_NAME") or None,
        dry_run=_parse_bool(os.getenv("DRY_RUN", "True")),
    )
