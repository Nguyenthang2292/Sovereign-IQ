from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FALLBACK_WARNING = (
    "[WARN] Rust serverless parity runner is unavailable in this environment; "
    "falling back to Python serverless emulation."
)


def test_serverless_matches_source_on_parity_fixtures() -> None:
    cmd = [
        sys.executable,
        "scripts/run_adaptive_trend_parity_harness.py",
        "--fixtures-dir",
        "tests/parity_fixtures",
        "--impl",
        "serverless",
        "--strict",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    full_output = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 0, (
        "Serverless parity harness failed.\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
    assert FALLBACK_WARNING not in full_output, (
        "Serverless parity test ran with Python fallback instead of Rust runner.\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
