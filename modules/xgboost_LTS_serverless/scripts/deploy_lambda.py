#!/usr/bin/env python3
"""
AWS Lambda Deployment Script for XGBoost Serverless

This script automates the build and deployment process for the XGBoost Serverless module.
It handles:
1. Environment verification (Rust, Cargo Lambda, AWS CLI, MSVC tools / Zig)
2. AWS Infrastructure setup (IAM Role, S3 Model Bucket, SQS Queue) - optional
3. Building the Lambda function (using cargo-lambda)
4. Deploying the function to AWS

Usage:
    python scripts/deploy_lambda.py [--region us-east-1] [--profile default]
    python scripts/deploy_lambda.py --skip-infra --skip-build   # redeploy only
    python scripts/deploy_lambda.py --skip-build                 # infra + deploy only
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("deploy")

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    logger.error("boto3 is required. Please install it with: pip install boto3")
    sys.exit(1)

# ─────────────────────────── Constants ────────────────────────────────────────
MODULE_DIR = Path(__file__).resolve().parent.parent
LAMBDA_DIR = MODULE_DIR / "lambda"

# Cargo package name defined in lambda/Cargo.toml [[bin]] name
BINARY_NAME = "bootstrap"
# AWS Lambda function name (shown in the AWS console)
LAMBDA_FUNCTION_NAME = "xgboost-serverless-predict"
# IAM role to create / reuse
IAM_ROLE_NAME = "XGBoost-Lambda-ExecutionRole"
# S3 bucket for storing trained XGBoost models
S3_BUCKET_NAME = os.getenv("MODEL_BUCKET", "xgboost-models-store")
# SQS queue for receiving raw predictions
SQS_QUEUE_NAME = os.getenv("SQS_QUEUE_NAME", "xgboost-predictions")

# Lambda resource settings (mirrors template.yaml)
LAMBDA_MEMORY_MB = "3008"  # 3 vCPUs — sufficient for AVX2 + model inference
LAMBDA_TIMEOUT_S = "30"  # 30 s is generous for single-request inference

# Default folder where trained XGBoost JSON models are kept locally.
# Models MUST follow the naming convention:  {SYMBOL}_{TIMEFRAME}_{VERSION}.json
# Examples:  BTCUSDT_15m_v1.json   ETHUSDT_1h_v2.json   SOLUSDT_4h_v1.json
# The filename (without .json) becomes the S3 key used by the Lambda handler.
DEFAULT_MODELS_DIR = MODULE_DIR / "models"
# ──────────────────────────────────────────────────────────────────────────────


def run_command(command, cwd=None, capture_output=True, check=True, env=None):
    """Run a shell command and return the CompletedProcess result."""
    try:
        logger.debug(f"Running command: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=cwd,
            check=check,
            text=True,
            capture_output=capture_output,
            env=env,
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(command)}")
        if capture_output:
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
        raise


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Dependency checks                                                          │
# └─────────────────────────────────────────────────────────────────────────────┘


def check_dependencies():
    """Verify all required tools are installed."""
    logger.info("Checking dependencies...")

    # Rust / Cargo
    if not shutil.which("cargo"):
        logger.error("Rust (cargo) is not installed. Please install it from https://rustup.rs/")
        return False

    # cargo-lambda
    try:
        run_command(["cargo", "lambda", "--version"])
    except subprocess.CalledProcessError:
        logger.warning("cargo-lambda not found. Attempting to install...")

        choco_success = False
        if os.name == "nt" and shutil.which("choco"):
            try:
                logger.info("Installing cargo-lambda via Chocolatey...")
                run_command(["choco", "install", "cargo-lambda", "-y"], check=True, capture_output=False)
                run_command(["cargo", "lambda", "--version"])
                choco_success = True
            except Exception as e:
                logger.warning(f"Chocolatey install failed: {e}. Falling back to cargo install.")

        if not choco_success:
            try:
                logger.info("Installing cargo-lambda via cargo (compiling from source – this may take a while)...")
                run_command(["cargo", "install", "cargo-lambda"], check=True, capture_output=False)
            except Exception as e:
                logger.error(f"Failed to install cargo-lambda: {e}")
                logger.error("Please install manually:  cargo install cargo-lambda")
                if os.name == "nt" and shutil.which("choco"):
                    logger.error("Or via Chocolatey:  choco install cargo-lambda")
                return False

    # AWS CLI (optional – main operations use boto3)
    if not shutil.which("aws"):
        logger.warning("AWS CLI not found. Operations will rely solely on boto3.")

    logger.info("Dependencies OK.")
    return True


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  AWS Infrastructure                                                         │
# └─────────────────────────────────────────────────────────────────────────────┘


def setup_iam_role(iam_client):
    """Create or retrieve the IAM execution role for XGBoost Lambda."""
    logger.info(f"Checking IAM Role: {IAM_ROLE_NAME} ...")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        role = iam_client.get_role(RoleName=IAM_ROLE_NAME)
        role_arn = role["Role"]["Arn"]
        logger.info(f"Role already exists: {role_arn}")
        return role_arn
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise

    logger.info("Role not found – creating ...")
    role = iam_client.create_role(
        RoleName=IAM_ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Execution role for XGBoost Serverless Lambda",
    )
    role_arn = role["Role"]["Arn"]
    logger.info(f"Created role: {role_arn}")

    managed_policies = [
        "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",  # read trained model files
    ]
    logger.info("Attaching managed policies ...")
    for policy_arn in managed_policies:
        iam_client.attach_role_policy(RoleName=IAM_ROLE_NAME, PolicyArn=policy_arn)
        logger.info(f"  ✓ {policy_arn}")

    # Scoped SQS inline policy — SendMessage only on the prediction queue
    sqs_inline_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "sqs:SendMessage",
                "Resource": f"arn:aws:sqs:*:*:{SQS_QUEUE_NAME}",
            }
        ],
    }
    iam_client.put_role_policy(
        RoleName=IAM_ROLE_NAME,
        PolicyName="XGBoostSQSSendMessage",
        PolicyDocument=json.dumps(sqs_inline_policy),
    )
    logger.info("  ✓ Inline policy XGBoostSQSSendMessage (sqs:SendMessage scoped)")

    logger.info("Waiting for IAM role propagation (15 s) ...")
    time.sleep(15)
    return role_arn


def setup_s3_bucket(s3_client, region):
    """Create the S3 model bucket if it does not exist."""
    logger.info(f"Checking S3 model bucket: {S3_BUCKET_NAME} ...")

    try:
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
        logger.info(f"Bucket already exists: s3://{S3_BUCKET_NAME}")
        return S3_BUCKET_NAME
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code not in ("404", "NoSuchBucket"):
            raise

    logger.info(f"Bucket not found – creating s3://{S3_BUCKET_NAME} in {region} ...")
    kwargs = {"Bucket": S3_BUCKET_NAME}
    # us-east-1 must NOT include a LocationConstraint
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}

    s3_client.create_bucket(**kwargs)

    # Enable versioning so model files are auditable
    s3_client.put_bucket_versioning(
        Bucket=S3_BUCKET_NAME,
        VersioningConfiguration={"Status": "Enabled"},
    )
    logger.info(f"Created bucket s3://{S3_BUCKET_NAME} (versioning enabled).")
    return S3_BUCKET_NAME


def setup_sqs_queue(sqs_client):
    """Create or retrieve the SQS prediction results queue."""
    logger.info(f"Checking SQS Queue: {SQS_QUEUE_NAME} ...")

    try:
        response = sqs_client.get_queue_url(QueueName=SQS_QUEUE_NAME)
        queue_url = response["QueueUrl"]
        logger.info(f"Queue already exists: {queue_url}")
        return queue_url
    except ClientError as e:
        if e.response["Error"]["Code"] != "AWS.SimpleQueueService.NonExistentQueue":
            raise

    logger.info("Queue not found – creating ...")
    response = sqs_client.create_queue(
        QueueName=SQS_QUEUE_NAME,
        Attributes={
            "VisibilityTimeout": "300",
            "MessageRetentionPeriod": "86400",  # 1 day
        },
    )
    queue_url = response["QueueUrl"]
    logger.info(f"Created queue: {queue_url}")
    return queue_url


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Zig path helper (required by cargo-lambda on Windows)                      │
# └─────────────────────────────────────────────────────────────────────────────┘


def _ensure_zig_on_path():
    """Auto-detect Zig binary from the ziglang pip package and add it to PATH.

    cargo-lambda installs Zig via pip3 but does NOT add it to PATH automatically,
    which causes 'Failed to find zig: cannot find binary path' on Windows.

    Strategy (tried in order):
      1. zig already on PATH → done
      2. Import ziglang from current Python env (venv)
      3. Filesystem glob of system Python site-packages (LOCALAPPDATA/Programs/Python)
      4. Ask system 'python' process to locate ziglang

    Returns:
        str | None: The Zig directory added to PATH, or None if not found/already set.
    """
    zig_bin = "zig.exe" if os.name == "nt" else "zig"

    def _inject(zig_dir: Path) -> str:
        zig_dir_str = str(zig_dir)
        os.environ["PATH"] = zig_dir_str + os.pathsep + os.environ.get("PATH", "")
        try:
            ver = subprocess.run([str(zig_dir / zig_bin), "version"], capture_output=True, text=True).stdout.strip()
            logger.info(f"Zig injected into PATH: {zig_dir / zig_bin} (v{ver})")
        except Exception:
            pass
        return zig_dir_str

    # 1. Already on PATH
    if shutil.which("zig"):
        ver = subprocess.run(["zig", "version"], capture_output=True, text=True).stdout.strip()
        logger.info(f"Zig already on PATH: v{ver}")
        return None

    # 2. Try importing ziglang from the current Python environment (venv)
    try:
        import ziglang  # type: ignore[import-untyped]

        zig_dir = Path(ziglang.__file__).parent
        if (zig_dir / zig_bin).exists():
            return _inject(zig_dir)
    except ImportError:
        pass

    # 3. Search system Python site-packages via glob.
    if os.name == "nt":
        local_app = Path(os.environ.get("LOCALAPPDATA", ""))
        search_roots = [
            local_app / "Programs" / "Python",
            Path("C:/Python312"),
            Path("C:/Python311"),
            Path("C:/Python310"),
        ]
        for root in search_roots:
            if not root.exists():
                continue
            for zig_exe in root.glob(f"**/ziglang/{zig_bin}"):
                logger.info(f"Found zig via filesystem scan: {zig_exe}")
                return _inject(zig_exe.parent)

    # 4. Ask the system 'python' (not the venv) to locate ziglang
    for py_cmd in ("python", "python3"):
        if not shutil.which(py_cmd):
            continue
        try:
            result = subprocess.run(
                [py_cmd, "-c", "import ziglang, os; print(os.path.dirname(ziglang.__file__))"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                zig_dir = Path(result.stdout.strip())
                if (zig_dir / zig_bin).exists():
                    logger.info(f"ziglang located via system {py_cmd}: {zig_dir}")
                    return _inject(zig_dir)
        except Exception:
            continue

    logger.warning(
        "Could not locate Zig. cargo-lambda will attempt its own install. "
        "If the build fails, run:  pip install ziglang  "
        "or download from https://ziglang.org/download/"
    )
    return None


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Build & Deploy                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘


def build_lambda():
    """Build the XGBoost Lambda function using cargo-lambda."""
    logger.info("Building XGBoost Lambda function ...")

    # Informational MSVC linker check (Windows only)
    if os.name == "nt" and not shutil.which("link.exe"):
        if not any("Visual Studio" in v for v in os.environ.values()):
            logger.warning("WARNING: MSVC Linker (link.exe) not found/configured.")
            logger.warning(
                "If the build fails, install 'Desktop development with C++' "
                "via Visual Studio Installer, or rely on Zig as the linker (auto-configured below)."
            )

    # Ensure Zig is on PATH so cargo-lambda can cross-compile for Linux
    _ensure_zig_on_path()

    cmd = ["cargo", "lambda", "build", "--release", "--target", "x86_64-unknown-linux-gnu"]

    # Enable AVX2 SIMD optimizations (Lambda runs on Haswell/Broadwell x86_64)
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=haswell -C target-feature=+avx2"
    logger.info("Building with SIMD optimizations (AVX2 enabled) ...")

    try:
        run_command(cmd, cwd=LAMBDA_DIR, capture_output=False, env=env)
        logger.info("Build successful.")
    except subprocess.CalledProcessError:
        logger.error("\nBuild failed.")
        if os.name == "nt":
            logger.error("NOTE: On Windows, cross-compilation requires Zig as a linker.")
            logger.error("Ensure 'ziglang' is installed:  pip install ziglang")
            logger.error("Or install Visual Studio Build Tools with 'Desktop development with C++' workload.")
        sys.exit(1)


def deploy_lambda(role_arn, bucket_name, queue_url, region):
    """Deploy the built XGBoost Lambda using cargo-lambda.

    Uses the cargo-lambda deploy syntax documented at:
      https://www.cargo-lambda.info/commands/deploy.html

    Because the Cargo [[bin]] name is 'bootstrap' (which matches the AWS
    Lambda custom runtime convention), we do NOT need --binary-name here.
    The positional argument sets the AWS function name.
    """
    logger.info(f"Deploying '{BINARY_NAME}' as Lambda function '{LAMBDA_FUNCTION_NAME}' to {region} ...")

    # Resolve the binary path — cargo-lambda build emits into <workspace>/target/lambda/<bin>/
    binary_path = MODULE_DIR / "target" / "lambda" / "bootstrap" / "bootstrap"
    if not binary_path.exists():
        logger.error(f"Built binary not found at {binary_path}. Run without --skip-build first.")
        sys.exit(1)

    cmd = [
        "cargo",
        "lambda",
        "deploy",
        # Use explicit binary path so deploy works from Cargo workspace root
        "--binary-path",
        str(binary_path),
        "--iam-role",
        role_arn,
        # ── Environment variables ──────────────────────────────────────────
        "--env-var",
        "RUST_LOG=info",
        "--env-var",
        f"MODEL_BUCKET={bucket_name}",
        "--env-var",
        f"PREDICTION_QUEUE_URL={queue_url}",
        # NOTE: AWS_REGION is a Lambda-reserved env-var and must NOT be set here.
        # ── Resource settings (match template.yaml) ────────────────────────
        "--memory",
        LAMBDA_MEMORY_MB,
        "--timeout",
        LAMBDA_TIMEOUT_S,
        # ── AWS function name (positional, must be last) ───────────────────
        LAMBDA_FUNCTION_NAME,
    ]

    try:
        # cargo-lambda deploy must be run from the workspace root (where
        # target/lambda/bootstrap/ lives), NOT from the lambda sub-crate dir.
        run_command(cmd, cwd=MODULE_DIR, capture_output=False)
        logger.info("Deployment successful.")
        logger.info(f"Function ARN: arn:aws:lambda:{region}:<account_id>:function:{LAMBDA_FUNCTION_NAME}")
        logger.info(f"Model Bucket: s3://{bucket_name}")
        logger.info(f"Prediction Queue: {queue_url}")
    except subprocess.CalledProcessError:
        logger.error("Deployment failed.")
        sys.exit(1)


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Model upload                                                               │
# └─────────────────────────────────────────────────────────────────────────────┘

import re

# Valid timeframe identifiers accepted by the Lambda handler
_VALID_TIMEFRAMES = {
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}


def parse_model_filename(path: Path) -> tuple[str, str, str] | None:
    """
    Parse a model file name into (symbol, timeframe, version).

    Expected format:  {SYMBOL}_{TIMEFRAME}_{VERSION}.json

    Examples
    --------
    BTCUSDT_15m_v1.json   -> ("BTCUSDT", "15m", "v1")
    ETHUSDT_1h_v2.json    -> ("ETHUSDT", "1h", "v2")

    Returns None and logs a warning if the filename does not match.
    """
    stem = path.stem  # filename without extension
    # Pattern: <SYMBOL (uppercase+digits)>_<TIMEFRAME>_<VERSION>
    pattern = re.compile(
        r"^([A-Z0-9]+)_"
        r"(\d+[mhdwM])_"
        r"(v\d+(?:[._]\d+)*)$"
    )
    m = pattern.match(stem)
    if not m:
        logger.warning(
            f"  Skipping {path.name}: does not match {{SYMBOL}}_{{TIMEFRAME}}_{{VERSION}}.json "
            f"(e.g. BTCUSDT_15m_v1.json)"
        )
        return None

    symbol, timeframe, version = m.group(1), m.group(2), m.group(3)
    if timeframe not in _VALID_TIMEFRAMES:
        logger.warning(
            f"  Skipping {path.name}: unrecognised timeframe '{timeframe}'. Valid: {sorted(_VALID_TIMEFRAMES)}"
        )
        return None

    return symbol, timeframe, version


def upload_models_to_s3(
    s3_client,
    models_dir: Path,
    bucket_name: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> tuple[int, int, int]:
    """
    Scan *models_dir* for ``*.json`` files and upload each valid model to S3.

    S3 key = ``{stem}.json``  (e.g. ``BTCUSDT_15m_v1.json``).
    The Lambda handler resolves models by that same key when ``model_s3_key``
    is provided in the request payload.

    Parameters
    ----------
    s3_client   : boto3 S3 client
    models_dir  : local directory to scan
    bucket_name : destination S3 bucket
    dry_run     : if True, print what would be uploaded without uploading
    force       : if True, overwrite existing S3 objects; otherwise skip them

    Returns
    -------
    (uploaded, skipped, failed) counts
    """
    if not models_dir.exists():
        logger.error(
            f"Models directory not found: {models_dir}  "
            f"Create it and place your trained .json model files there, "
            f"or pass --models-dir <path>."
        )
        return 0, 0, 0

    model_files = sorted(models_dir.glob("*.json"))
    if not model_files:
        logger.warning(f"No *.json model files found in {models_dir}")
        return 0, 0, 0

    logger.info(
        f"{'[DRY RUN] ' if dry_run else ''}Uploading {len(model_files)} model file(s) "
        f"from {models_dir} → s3://{bucket_name}/"
    )

    uploaded = skipped = failed = 0

    for model_path in model_files:
        parsed = parse_model_filename(model_path)
        if parsed is None:
            failed += 1
            continue

        symbol, timeframe, version = parsed
        s3_key = model_path.name  # e.g. BTCUSDT_15m_v1.json
        size_kb = model_path.stat().st_size / 1024

        # ── Skip-if-exists check (unless force) ───────────────────────────
        if not force and not dry_run:
            try:
                s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                logger.info(f"  ⤸  {s3_key} already exists in S3 (use --force-upload to overwrite)")
                skipped += 1
                continue
            except ClientError as e:
                if e.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                    logger.error(f"  ✗  {s3_key}: S3 head_object error: {e}")
                    failed += 1
                    continue
                # 404 → does not exist → proceed with upload

        if dry_run:
            logger.info(
                f"  [dry-run] would upload: {model_path.name} "
                f"({size_kb:.1f} KB) → s3://{bucket_name}/{s3_key}  "
                f"[{symbol} / {timeframe} / {version}]"
            )
            uploaded += 1
            continue

        # ── Actual upload ─────────────────────────────────────────────────
        try:
            s3_client.upload_file(
                str(model_path),
                bucket_name,
                s3_key,
                ExtraArgs={"ContentType": "application/json"},
            )
            logger.info(
                f"  ✓  {s3_key} ({size_kb:.1f} KB) → s3://{bucket_name}/{s3_key}  [{symbol} / {timeframe} / {version}]"
            )
            uploaded += 1
        except Exception as exc:
            logger.error(f"  ✗  Failed to upload {s3_key}: {exc}")
            failed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(f"Model upload complete — uploaded: {uploaded}, skipped: {skipped}, failed: {failed}")
    return uploaded, skipped, failed


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Entry-point                                                                │
# └─────────────────────────────────────────────────────────────────────────────┘


def main():
    parser = argparse.ArgumentParser(
        description="Deploy XGBoost Serverless to AWS Lambda",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full deploy (build + infra + upload models + deploy)
  python scripts/deploy_lambda.py --upload-models

  # Re-deploy only (infra + binary already exist)
  python scripts/deploy_lambda.py --skip-infra --skip-build --upload-models

  # Upload models only (no Lambda rebuild/deploy)
  python scripts/deploy_lambda.py --skip-infra --skip-build --skip-deploy --upload-models

  # Dry-run: see what would be uploaded without touching AWS
  python scripts/deploy_lambda.py --upload-models --dry-run

  # Force-overwrite existing models in S3
  python scripts/deploy_lambda.py --upload-models --force-upload

Model file naming convention (required):
  {SYMBOL}_{TIMEFRAME}_{VERSION}.json
  Examples:  BTCUSDT_15m_v1.json   ETHUSDT_1h_v2.json
  Place files in <module>/models/ or override with --models-dir.
""",
    )
    parser.add_argument("--region", default="us-east-1", help="AWS Region (default: us-east-1)")
    parser.add_argument("--profile", default=None, help="AWS CLI named profile")
    parser.add_argument("--skip-build", action="store_true", help="Skip the cargo-lambda build step")
    parser.add_argument(
        "--skip-infra",
        action="store_true",
        help="Skip infrastructure setup (IAM Role / S3 / SQS) – reuse existing resources",
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Skip the Lambda function deployment step (useful when only uploading models)",
    )
    # ── Model upload args ─────────────────────────────────────────────────────
    parser.add_argument(
        "--upload-models",
        action="store_true",
        help="Upload *.json model files from --models-dir to the S3 model bucket after deploy",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help=(
            f"Directory containing trained XGBoost .json models to upload "
            f"(default: {DEFAULT_MODELS_DIR}). "
            f"Files must follow the naming convention: {{SYMBOL}}_{{TIMEFRAME}}_{{VERSION}}.json"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading (implies --upload-models)",
    )
    parser.add_argument(
        "--force-upload",
        action="store_true",
        help="Overwrite existing model files in S3 (default: skip if already exists)",
    )

    args = parser.parse_args()

    # --dry-run implies --upload-models
    if args.dry_run:
        args.upload_models = True

    # ── AWS session ───────────────────────────────────────────────────────────
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    iam = session.client("iam")
    s3 = session.client("s3", region_name=args.region)
    sqs = session.client("sqs", region_name=args.region)

    # ── Dependency check ──────────────────────────────────────────────────────
    if not check_dependencies():
        sys.exit(1)

    # ── Infrastructure ────────────────────────────────────────────────────────
    role_arn = None
    bucket_name = None
    queue_url = None

    if not args.skip_infra:
        try:
            role_arn = setup_iam_role(iam)
            bucket_name = setup_s3_bucket(s3, args.region)
            queue_url = setup_sqs_queue(sqs)
        except Exception as e:
            logger.error(f"Infrastructure setup failed: {e}")
            sys.exit(1)
    else:
        # Attempt to fetch existing infra so we can pass env-vars to deploy
        logger.info("Skipping infra setup – fetching existing resource details ...")
        try:
            role = iam.get_role(RoleName=IAM_ROLE_NAME)
            role_arn = role["Role"]["Arn"]
            logger.info(f"  Role ARN:    {role_arn}")
        except Exception:
            logger.warning("  Could not retrieve IAM role. You may need to set it up first.")

        bucket_name = S3_BUCKET_NAME
        logger.info(f"  Model Bucket: s3://{bucket_name}")

        try:
            response = sqs.get_queue_url(QueueName=SQS_QUEUE_NAME)
            queue_url = response["QueueUrl"]
            logger.info(f"  Queue URL:   {queue_url}")
        except Exception:
            logger.warning("  Could not retrieve SQS queue URL.")

    # ── Build ─────────────────────────────────────────────────────────────────
    if not args.skip_build:
        build_lambda()

    # ── Deploy ────────────────────────────────────────────────────────────────
    if not args.skip_deploy:
        if not (role_arn and bucket_name and queue_url):
            logger.error(
                "Cannot deploy: missing Role ARN, Model Bucket, or Queue URL. "
                "Run without --skip-infra to create them automatically."
            )
            sys.exit(1)
        deploy_lambda(role_arn, bucket_name, queue_url, args.region)

    # ── Upload models ─────────────────────────────────────────────────────────
    if args.upload_models or args.dry_run:
        if not bucket_name:
            logger.error("Cannot upload models: S3 bucket name is unknown. Run without --skip-infra.")
            sys.exit(1)
        models_dir = Path(args.models_dir)
        upload_models_to_s3(
            s3,
            models_dir,
            bucket_name,
            dry_run=args.dry_run,
            force=args.force_upload,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("===========================================")
    logger.info("DEPLOYMENT COMPLETE")
    logger.info("===========================================")
    logger.info("Next steps:")
    if not args.upload_models:
        logger.info("  1. Upload trained models to S3:")
        logger.info("       python scripts/deploy_lambda.py --skip-infra --skip-build --skip-deploy --upload-models")
        logger.info(f"       (place *.json files in {DEFAULT_MODELS_DIR} first)")
    logger.info("  2. Invoke the Lambda function with a test payload:")
    logger.info(f"       python scripts/binance_lambda_demo.py --region {args.region}")
    logger.info("  3. View CloudWatch logs:")
    logger.info(f"       aws logs tail /aws/lambda/{LAMBDA_FUNCTION_NAME} --follow --region {args.region}")


if __name__ == "__main__":
    main()
