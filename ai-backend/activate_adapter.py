"""Explicitly promote or disable a QLoRA adapter after manual evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from peft import PeftConfig

ROOT = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ACTIVE_ADAPTER_FILE = ROOT / "active_adapter.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote an evaluated English QLoRA candidate, or switch back to the base model."
    )
    parser.add_argument("--adapter", type=Path, help="Candidate adapter directory to activate")
    parser.add_argument("--disable", action="store_true", help="Disable the active adapter")
    args = parser.parse_args()

    if args.disable == bool(args.adapter):
        raise SystemExit("Use exactly one of --adapter or --disable.")

    if args.disable:
        if ACTIVE_ADAPTER_FILE.exists():
            ACTIVE_ADAPTER_FILE.unlink()
        print("Adapter disabled. Restart the API to use the frozen base model.")
        return

    adapter = args.adapter.resolve()
    config_path = adapter / "adapter_config.json"
    if not config_path.exists():
        raise SystemExit(f"Not an adapter directory: {adapter}")
    config = PeftConfig.from_pretrained(str(adapter))
    if config.base_model_name_or_path != BASE_MODEL:
        raise SystemExit(
            f"Adapter base is {config.base_model_name_or_path}, but this application uses {BASE_MODEL}."
        )
    manifest_path = adapter / "training-manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Missing training-manifest.json; only activate a candidate produced by train_qlora.py.")

    ACTIVE_ADAPTER_FILE.write_text(
        json.dumps(
            {
                "adapter_path": str(adapter),
                "base_model": BASE_MODEL,
                "status": "manually activated after evaluation",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Activated {adapter}. Restart the API to load it.")


if __name__ == "__main__":
    main()
