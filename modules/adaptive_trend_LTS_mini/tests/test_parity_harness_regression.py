from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_mini_matches_source_on_parity_fixtures() -> None:
    cmd = [
        sys.executable,
        "scripts/run_adaptive_trend_parity_harness.py",
        "--fixtures-dir",
        "tests/parity_fixtures",
        "--impl",
        "mini",
        "--strict",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, (
        "Mini parity harness failed.\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
