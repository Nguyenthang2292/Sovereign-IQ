# -*- coding: utf-8 -*-
"""
Local smoke test cho handler.py (bypass /tmp version).
Chay toan bo pipeline thuc: fetch -> indicators -> labels -> train -> serialize.
Chi mock buoc S3 put_object de khong can AWS credentials.

PHAI chay voi: python modules/xgboost_LTS_serverless/scripts/_local_handler_test.py
(Khong dung python -c hay import truc tiep -- Windows multiprocessing spawn yeu cau __main__)

Run:
    python modules/xgboost_LTS_serverless/scripts/_local_handler_test.py
"""

import importlib.util as _ilu
import multiprocessing
import os
import sys
import time
import types
from unittest.mock import MagicMock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODULE_ROOT, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    uploaded_call = {}

    def _fake_put_object(**kwargs):
        uploaded_call.update(kwargs)
        body = kwargs.get("Body", b"")
        print(f"[mock-s3] put_object -> Bucket={kwargs.get('Bucket')} Key={kwargs.get('Key')} size={len(body)} bytes")

    fake_s3_client = MagicMock()
    fake_s3_client.put_object.side_effect = _fake_put_object

    fake_boto3 = types.ModuleType("boto3")
    setattr(fake_boto3, "client", MagicMock(return_value=fake_s3_client))
    sys.modules["boto3"] = fake_boto3
    sys.modules["botocore"] = types.ModuleType("botocore")
    sys.modules["botocore.exceptions"] = types.ModuleType("botocore.exceptions")

    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(REPO_ROOT, "modules", "auto_trade", ".env"), override=True)
        print("[setup] .env loaded")
    except Exception as e:
        print(f"[setup] .env load skipped: {e}")

    _handler_path = os.path.join(MODULE_ROOT, "lambda", "trainer", "handler.py")
    _spec = _ilu.spec_from_file_location("trainer_handler", _handler_path)
    assert _spec is not None, f"Cannot find handler at: {_handler_path}"
    _mod = _ilu.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)
    handler = _mod.handler

    print("=" * 60)
    print("LOCAL SMOKE TEST -- xgboost_LTS_serverless handler")
    print("=" * 60)

    event = {
        "symbol": "BTC/USDT",
        "timeframe": "15m",
        "model_version": "v1",
        "s3_bucket": "xgboost-models-store-local-test",
        "fetch_limit": 2000,
    }

    t0 = time.perf_counter()
    try:
        result = handler(event, context=None)
        elapsed = time.perf_counter() - t0

        print()
        print("=" * 60)
        print(f"PASS -- handler returned: {result}")
        print(f"   Total elapsed: {elapsed:.1f}s")
        print(f"   S3 put_object called: {fake_s3_client.put_object.call_count} time(s)")

        assert result["status"] == "ok", f"Expected status=ok, got {result['status']}"
        assert uploaded_call.get("Key", "").endswith(".json"), "S3 key must end with .json"
        assert len(uploaded_call.get("Body", b"")) > 0, "Uploaded model bytes must be non-empty"
        print("PASS: All assertions passed")

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print()
        print("=" * 60)
        print(f"FAIL after {elapsed:.1f}s -- {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
