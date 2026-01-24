#!/usr/bin/env python3
"""Verify CUDA development environment for Phase 4 (adaptive_trend_LTS GPU kernels).

Checks: nvidia-smi, nvcc, CuPy, PyCUDA, compute capability.
Run from project root: python scripts/verify_cuda_env.py
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], check: bool = False) -> tuple[int, str]:
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = (r.stdout or "").strip() + "\n" + (r.stderr or "").strip()
        return r.returncode, out.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))

    print("=" * 60)
    print("Phase 4 CUDA environment verification")
    print("=" * 60)
    ok = True

    # 1. nvidia-smi
    print("\n[1] nvidia-smi (GPU)")
    code, out = _run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"])
    if code != 0:
        print("    FAIL: nvidia-smi not found or error. Install NVIDIA drivers + CUDA.")
        ok = False
    else:
        print("    OK")
        for line in out.splitlines():
            print(f"      {line.strip()}")
        # Parse compute cap (e.g. "8.9") and check >= 6.0
        try:
            parts = out.split(",")
            if len(parts) >= 3:
                cap = parts[2].strip()
                v = float(cap)
                if v < 6.0:
                    print(f"    WARN: Compute capability {v} < 6.0; >= 6.0 recommended.")
        except Exception:
            pass

    # 2. nvcc (CUDA Toolkit)
    print("\n[2] nvcc (CUDA Toolkit)")
    nvcc = "nvcc"
    if platform.system() == "Windows" and os.environ.get("CUDA_PATH"):
        nvcc = os.path.join(os.environ["CUDA_PATH"], "bin", "nvcc.exe")
    code, out = _run([nvcc, "--version"])
    if code != 0:
        print("    FAIL: nvcc not found. Install CUDA Toolkit 12.x and set CUDA_PATH / PATH.")
        ok = False
    else:
        print("    OK")
        first = out.split("\n")[0] if out else ""
        print(f"      {first}")

    # 3. CuPy
    print("\n[3] CuPy")
    try:
        import cupy as cp

        print(f"    OK (version {cp.__version__})")
        try:
            dev = cp.cuda.Device(0)
            print(f"    GPU 0: {dev.name.decode() if isinstance(dev.name, bytes) else dev.name}")
        except Exception as e:
            print(f"    WARN: Could not get GPU 0: {e}")
    except ImportError as e:
        print("    FAIL: CuPy not installed.")
        print("    Install: pip install cupy-cuda12x  (or cupy-cuda11x depending on CUDA)")
        ok = False

    # 4. PyCUDA
    print("\n[4] PyCUDA")
    try:
        import pycuda.driver as cuda

        cuda.init()
        n = cuda.Device.count()
        print(f"    OK (PyCUDA), {n} GPU(s)")
        for i in range(n):
            d = cuda.Device(i)
            print(f"    GPU {i}: {d.name()}, compute {d.compute_capability()}")
    except ImportError as e:
        print("    FAIL: PyCUDA not installed.")
        print("    Install: pip install pycuda")
        ok = False
    except Exception as e:
        print(f"    WARN: PyCUDA import OK but init failed: {e}")

    # 5. Env vars
    print("\n[5] Environment (CUDA_PATH, etc.)")
    cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_home:
        print(f"    CUDA_PATH/CUDA_HOME: {cuda_home}")
    else:
        print("    CUDA_PATH/CUDA_HOME not set (optional if nvcc in PATH)")

    print("\n" + "=" * 60)
    if ok:
        print("All checks passed. CUDA development environment ready.")
    else:
        print("Some checks failed. See Phase 4 setup: modules/adaptive_trend_LTS/docs/phase4_cuda_setup.md")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
