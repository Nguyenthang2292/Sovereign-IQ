from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from openai import OpenAI

from config.config_api import get_dashscope_api_key

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def _mask_api_key(api_key: str) -> str:
    if len(api_key) <= 10:
        return "***"
    return f"{api_key[:6]}...{api_key[-4:]}"


def _load_project_env() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)
    load_dotenv(project_root / "modules" / "auto_trade" / ".env", override=False)


def _read_key_from_env_files() -> str | None:
    project_root = Path(__file__).resolve().parents[1]
    env_candidates = [
        project_root / "modules" / "auto_trade" / ".env",
        project_root / ".env",
    ]
    for env_file in env_candidates:
        if not env_file.exists():
            continue
        value = dotenv_values(env_file).get("DASHSCOPE_API_KEY")
        if isinstance(value, str):
            cleaned = value.strip().strip("\"'")
            if cleaned:
                return cleaned
    return None


def _resolve_dashscope_key() -> str | None:
    key = get_dashscope_api_key()
    if key:
        return key.strip().strip("\"'")
    env_key = os.getenv("DASHSCOPE_API_KEY")
    if env_key:
        return env_key.strip().strip("\"'")
    file_key = _read_key_from_env_files()
    if file_key:
        return file_key
    return None


def _collect_model_ids(client: OpenAI) -> list[str]:
    models = client.models.list()
    model_ids = [model.id for model in models.data if getattr(model, "id", None)]
    return sorted(set(model_ids))


def _filter_vision_models(model_ids: list[str]) -> list[str]:
    filtered = []
    for model_id in model_ids:
        lowered = model_id.lower()
        if "-vl" in lowered and "edit" not in lowered and "wan" not in lowered:
            filtered.append(model_id)
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(description="Test DashScope list models API")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print all models instead of only vision (-vl) models",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="DashScope OpenAI-compatible base URL",
    )
    args = parser.parse_args()

    _load_project_env()
    api_key = _resolve_dashscope_key()

    if not api_key:
        print("❌ DASHSCOPE_API_KEY not found in env/config")
        print("Set key in .env or modules/auto_trade/.env and rerun.")
        return 1

    print(f"✅ Using key: {_mask_api_key(api_key)}")
    print(f"🌐 Base URL: {args.base_url}")

    try:
        client = OpenAI(api_key=api_key, base_url=args.base_url)
        all_model_ids = _collect_model_ids(client)

        if args.all:
            output_model_ids = all_model_ids
            print(f"\nFound {len(output_model_ids)} total model(s):")
        else:
            output_model_ids = _filter_vision_models(all_model_ids)
            print(f"\nFound {len(output_model_ids)} vision model(s):")

        for model_id in output_model_ids:
            print(f"- {model_id}")

        if not output_model_ids:
            print("(No models matched current filter)")

        return 0
    except Exception as exc:
        print(f"❌ API call failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
