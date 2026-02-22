#!/usr/bin/env python3
"""
AWS Lambda Deployment Script for ATC Serverless

This script automates the build and deployment process for the ATC Serverless module.
It handles:
1. Environment verification (Rust, Cargo Lambda, AWS CLI, MSVC tools)
2. AWS Infrastructure setup (IAM Role, SQS Queue) - optional
3. Building the Lambda function (using cargo-lambda)
4. Deploying the function to AWS

Usage:
    python scripts/deploy_lambda.py [--region us-east-1] [--profile default]
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
    import boto3  # type: ignore[import-untyped]
    from botocore.exceptions import ClientError  # type: ignore[import-untyped]
except ImportError:
    logger.error("boto3 is required. Please install it with: pip install boto3")
    sys.exit(1)

# Constants
MODULE_DIR = Path(__file__).resolve().parent.parent
LAMBDA_DIR = MODULE_DIR / "lambda"
# Binary name = package name in lambda/Cargo.toml (uses underscores)
BINARY_NAME = "atc_lambda"
# AWS Lambda function name (displayed in AWS console, can use hyphens)
LAMBDA_FUNCTION_NAME = "atc-serverless"
IAM_ROLE_NAME = "ATC-Lambda-ExecutionRole"
SQS_QUEUE_NAME = "atc-results"
DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "AutoTrade")


def run_command(command, cwd=None, capture_output=True, check=True, env=None):
    """Run a shell command and return the result."""
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


def check_dependencies():
    """Verify all required tools are installed."""
    logger.info("Checking dependencies...")

    # Check Rust
    if not shutil.which("cargo"):
        logger.error("Rust (cargo) is not installed. Please install it from https://rustup.rs/")
        return False

    # Check cargo-lambda
    try:
        run_command(["cargo", "lambda", "--version"])
    except subprocess.CalledProcessError:
        logger.warning("cargo-lambda not found. Attempting to install...")

        # Try Chocolatey on Windows first (faster, prebuilt binaries)
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
                logger.info("Installing cargo-lambda via cargo (this may take a while, compiling from source)...")
                run_command(["cargo", "install", "cargo-lambda"], check=True, capture_output=False)
            except Exception as e:
                logger.error(f"Failed to install cargo-lambda: {e}")
                logger.error("Please install manually: cargo install cargo-lambda")
                if os.name == "nt" and shutil.which("choco"):
                    logger.error("Or via Chocolatey: choco install cargo-lambda")
                return False

    # Check AWS CLI (optional but good for some ops, mainly we use boto3)
    if not shutil.which("aws"):
        logger.warning("AWS CLI not found. Operations will rely solely on boto3.")

    logger.info("Dependencies checked.")
    return True


def setup_iam_role(iam_client):
    """Create or retrieve the IAM role for Lambda."""
    logger.info(f"Checking IAM Role: {IAM_ROLE_NAME}...")

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
        logger.info(f"Role exists: {role_arn}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            logger.info("Role not found. Creating...")
            role = iam_client.create_role(
                RoleName=IAM_ROLE_NAME,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Role for ATC Serverless Lambda",
            )
            role_arn = role["Role"]["Arn"]
            logger.info(f"Created role: {role_arn}")

            # Attach policies
            logger.info("Attaching policies...")
            policies = [
                "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
                "arn:aws:iam::aws:policy/AmazonSQSFullAccess",
            ]
            for policy_arn in policies:
                iam_client.attach_role_policy(RoleName=IAM_ROLE_NAME, PolicyArn=policy_arn)

            # Wait for propagation
            logger.info("Waiting for role propagation (10s)...")
            time.sleep(10)
        else:
            raise

    return role_arn


def setup_sqs_queue(sqs_client):
    """Create or retrieve the SQS queue."""
    logger.info(f"Checking SQS Queue: {SQS_QUEUE_NAME}...")

    try:
        response = sqs_client.get_queue_url(QueueName=SQS_QUEUE_NAME)
        queue_url = response["QueueUrl"]
        logger.info(f"Queue exists: {queue_url}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "AWS.SimpleQueueService.NonExistentQueue":
            logger.info("Queue not found. Creating...")
            response = sqs_client.create_queue(QueueName=SQS_QUEUE_NAME)
            queue_url = response["QueueUrl"]
            logger.info(f"Created queue: {queue_url}")
        else:
            raise

    return queue_url


def _ensure_zig_on_path():
    """Auto-detect Zig binary from the ziglang pip package and add it to PATH.

    cargo-lambda installs Zig via pip3 but does NOT add it to PATH automatically,
    which causes 'Failed to find zig: cannot find binary path' on Windows.

    This happens especially when running from a venv because cargo-lambda installs
    ziglang into the SYSTEM Python, not the active virtual environment.

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
        """Prepend zig_dir to os.environ PATH and log the result."""
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
        import ziglang  # type: ignore[import-not-found,import-untyped]

        zig_dir = Path(ziglang.__file__).parent  # type: ignore
        if (zig_dir / zig_bin).exists():
            return _inject(zig_dir)
    except ImportError:
        pass

    # 3. Search system Python site-packages via glob.
    #    cargo-lambda installs ziglang into the SYSTEM Python, not the active venv.
    if os.name == "nt":
        local_app = Path(os.environ.get("LOCALAPPDATA", ""))
        search_roots = [
            local_app / "Programs" / "Python",  # %LOCALAPPDATA%\Programs\Python
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
        "If build fails, run: pip install ziglang  "
        "or download from https://ziglang.org/download/"
    )
    return None


def build_lambda():
    """Build the Lambda function using cargo-lambda."""
    logger.info("Building Lambda function...")

    # Warn about missing MSVC linker on Windows (informational only)
    if os.name == "nt" and not shutil.which("link.exe"):
        if not any("Visual Studio" in v for v in os.environ.values()):
            logger.warning("WARNING: MSVC Linker (link.exe) not found/configured.")
            logger.warning(
                "If the build fails, install 'Desktop development with C++' "
                "via Visual Studio Installer, or use Zig as the linker (auto-configured below)."
            )

    # Ensure Zig is available on PATH (needed by cargo-lambda for cross-compilation).
    # cargo-lambda installs Zig via pip3 but the binary dir is NOT added to PATH on Windows.
    _ensure_zig_on_path()

    # cargo lambda build --release --target x86_64-unknown-linux-gnu
    # This cross-compiles for the AWS Lambda Linux runtime using Zig as the linker.
    cmd = ["cargo", "lambda", "build", "--release", "--target", "x86_64-unknown-linux-gnu"]

    # Enable AVX2 SIMD optimizations for AWS Lambda (Haswell/Broadwell architecture)
    # This significantly improves vector math performance for ATC indicators.
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=haswell -C target-feature=+avx2"
    logger.info("Building with SIMD optimizations (AVX2 enabled)...")

    try:
        run_command(cmd, cwd=LAMBDA_DIR, capture_output=False, env=env)
        logger.info("Build successful.")
    except subprocess.CalledProcessError:
        logger.error("\nBuild failed.")
        if os.name == "nt":
            logger.error("NOTE: On Windows, cross-compilation requires Zig as a linker.")
            logger.error("Ensure 'ziglang' is installed: pip install ziglang")
            logger.error("Or install Visual Studio Build Tools with 'Desktop development with C++' workload.")
        sys.exit(1)


def deploy_lambda(role_arn, queue_url, region):
    """Deploy the Lambda function using cargo-lambda.

    Uses the correct cargo-lambda deploy syntax per official docs:
      https://www.cargo-lambda.info/commands/deploy.html

    Syntax for deploying with a different name than the Cargo package:
      cargo lambda deploy --binary-name <PACKAGE_NAME> <FUNCTION_NAME>

    Where:
      FUNCTION_NAME (positional) = AWS Lambda function name shown in console
      --binary-name              = Cargo package name from lambda/Cargo.toml
    """
    logger.info(f"Deploying '{BINARY_NAME}' as Lambda function '{LAMBDA_FUNCTION_NAME}' to {region}...")

    cmd = [
        "cargo",
        "lambda",
        "deploy",
        "--binary-name",
        BINARY_NAME,  # Cargo package name (atc_lambda)
        "--iam-role",
        role_arn,
        "--env-var",
        "RUST_LOG=info",
        "--env-var",
        f"SQS_QUEUE_URL={queue_url}",
        "--env-var",
        "DB_BACKEND=dynamodb",
        "--env-var",
        f"DYNAMODB_TABLE_NAME={DYNAMODB_TABLE_NAME}",
        "--memory",
        "1769",  # 1769MB = exactly 1 full vCPU on AWS Lambda x86_64 (required for SIMD + Rayon)
        "--timeout",
        "120",  # 2 min — enough for large batches after SIMD speedup
        LAMBDA_FUNCTION_NAME,  # AWS function name (positional, last)
    ]

    try:
        run_command(cmd, cwd=LAMBDA_DIR, capture_output=False)
        logger.info("Deployment successful.")
        logger.info(f"Function ARN: arn:aws:lambda:{region}:...:function:{LAMBDA_FUNCTION_NAME}")
        logger.info(f"SQS Queue: {queue_url}")
    except subprocess.CalledProcessError:
        logger.error("Deployment failed.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Deploy ATC Serverless to AWS")
    parser.add_argument("--region", default="us-east-1", help="AWS Region")
    parser.add_argument("--profile", default=None, help="AWS CLI Profile")
    parser.add_argument("--skip-build", action="store_true", help="Skip build step")
    parser.add_argument("--skip-infra", action="store_true", help="Skip infrastructure setup (Role/SQS)")

    args = parser.parse_args()

    # Setup AWS session
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    iam = session.client("iam")
    sqs = session.client("sqs")

    # Check Environment
    if not check_dependencies():
        sys.exit(1)

    # Setup Infrastructure
    role_arn = None
    queue_url = None

    if not args.skip_infra:
        try:
            role_arn = setup_iam_role(iam)
            queue_url = setup_sqs_queue(sqs)
        except Exception as e:
            logger.error(f"Infrastructure setup failed: {e}")
            sys.exit(1)
    else:
        # Fetch existing infra details
        try:
            response = sqs.get_queue_url(QueueName=SQS_QUEUE_NAME)
            queue_url = response["QueueUrl"]
            role = iam.get_role(RoleName=IAM_ROLE_NAME)
            role_arn = role["Role"]["Arn"]
        except Exception:
            logger.warning("Could not automatically retrieve Infra details. Deployment might require manual inputs.")

    # Build
    if not args.skip_build:
        build_lambda()

    # Deploy
    if role_arn and queue_url:
        deploy_lambda(role_arn, queue_url, args.region)
    else:
        logger.error("Cannot deploy without Role ARN and Queue URL.")
        sys.exit(1)

    logger.info("===========================================")
    logger.info("DEPLOYMENT COMPLETE")
    logger.info("===========================================")
    logger.info("To test your function:")
    logger.info(f"python scripts/binance_lambda_demo.py --endpoint https://lambda.{args.region}.amazonaws.com/...")
    logger.info("Note: You might need to setup Function URL or API Gateway for HTTP access.")


if __name__ == "__main__":
    main()
