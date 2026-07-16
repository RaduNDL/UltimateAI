"""Optional FP16 export. Normal API inference loads the 4-bit base plus LoRA directly."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=ROOT / "coder-mentor-lora")
    parser.add_argument("--output", type=Path, default=ROOT / "ultimateai-coder-model")
    args = parser.parse_args()

    if not (args.adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"LoRA adapter not found: {args.adapter}")

    config = PeftConfig.from_pretrained(args.adapter)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    print("Loading base model and merging adapter. This export is optional.")
    base = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map={"": 0} if torch.cuda.is_available() else {"": "cpu"},
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload().to(torch.float16)
    merged.config.torch_dtype = torch.float16
    merged.config.use_cache = True
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(args.output)
    print(f"Saved optional FP16 export to {args.output}")


if __name__ == "__main__":
    main()
