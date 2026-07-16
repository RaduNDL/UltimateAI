"""Build a small, English-only SFT dataset for a frozen-base QLoRA adapter."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_SEED_DATA = DATA_DIR / "english_seed.jsonl"
DEFAULT_TRAIN_OUTPUT = DATA_DIR / "app-train.jsonl"
DEFAULT_EVAL_OUTPUT = DATA_DIR / "app-eval.jsonl"
DEFAULT_MANIFEST = DATA_DIR / "app-data-manifest.json"
SEED = 42
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are UltimateAI, a senior software engineering assistant. Respond only in "
    "clear English. Give direct, technically accurate answers. Do not invent APIs, "
    "files, test results, or project facts. State assumptions when requirements are "
    "incomplete, then provide a practical next step."
)
BAD_PHRASES = (
    "as an ai language model",
    "i cannot assist",
    "i'm unable to assist",
    "i am unable to assist",
    "i do not have enough information",
)
WHITESPACE = re.compile(r"\s+")


def record(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


def normalized(value: str) -> str:
    return WHITESPACE.sub(" ", value.strip().casefold())


def is_english_only(value: str) -> bool:
    """The training corpus intentionally stays ASCII English and source code."""
    return value.isascii()


def fits_token_budget(user: str, assistant: str, tokenizer, max_length: int) -> bool:
    messages = record(user, assistant)["messages"]
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    return len(full_ids) <= max_length and (len(full_ids) - len(prompt_ids)) >= 32


def is_good_pair(
    user: str,
    assistant: str,
    max_answer_chars: int,
    tokenizer,
    max_length: int,
) -> bool:
    if not user or not assistant:
        return False
    if len(user) < 8 or len(assistant) < 60 or len(assistant) > max_answer_chars:
        return False
    if not is_english_only(user) or not is_english_only(assistant):
        return False
    answer_lower = assistant.casefold()
    return not any(phrase in answer_lower for phrase in BAD_PHRASES) and fits_token_budget(
        user, assistant, tokenizer, max_length
    )


def load_seed_records(path: Path, max_answer_chars: int, tokenizer, max_length: int) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Seed dataset not found: {path}")
    records: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            messages = payload["messages"]
            user = str(messages[-2]["content"]).strip()
            assistant = str(messages[-1]["content"]).strip()
            if messages[-2]["role"] != "user" or messages[-1]["role"] != "assistant":
                raise ValueError("last two messages must be user and assistant")
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid seed example at {path}:{line_number}: {error}") from error
        if is_good_pair(user, assistant, max_answer_chars, tokenizer, max_length):
            records.append(record(user, assistant))
    return records


def streamed_pairs(
    dataset_id: str,
    user_field: str,
    answer_field: str,
    limit: int,
    max_answer_chars: int,
    tokenizer,
    max_length: int,
) -> list[dict]:
    if limit == 0:
        return []
    print(f"Streaming {limit} filtered examples from {dataset_id}...")
    try:
        dataset = load_dataset(dataset_id, split="train", streaming=True).shuffle(
            seed=SEED, buffer_size=10_000
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not access {dataset_id}. Check your connection or set its sample count to 0."
        ) from error

    records: list[dict] = []
    for row in dataset:
        user = str(row.get(user_field, "")).strip()
        assistant = str(row.get(answer_field, "")).strip()
        if not is_good_pair(user, assistant, max_answer_chars, tokenizer, max_length):
            continue
        records.append(record(user, assistant))
        if len(records) >= limit:
            break
    if len(records) < limit:
        raise RuntimeError(f"Only found {len(records)} usable records in {dataset_id}.")
    return records


def deduplicate(records: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for item in records:
        user = str(item["messages"][1]["content"])
        assistant = str(item["messages"][2]["content"])
        fingerprint = f"{normalized(user)}\n{normalized(assistant)}"
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(item)
    return unique


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a small, high-signal English programming SFT dataset."
    )
    parser.add_argument("--seed-data", type=Path, default=DEFAULT_SEED_DATA)
    parser.add_argument("--codefeedback-samples", type=int, default=160)
    parser.add_argument("--magicoder-samples", type=int, default=80)
    parser.add_argument("--max-answer-chars", type=int, default=4_000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--minimum-examples", type=int, default=180)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    if args.codefeedback_samples < 0 or args.magicoder_samples < 0:
        raise ValueError("Sample counts cannot be negative.")
    if not 0.05 <= args.validation_ratio <= 0.30:
        raise ValueError("validation-ratio must be between 0.05 and 0.30.")
    if args.base_model != BASE_MODEL or args.max_length not in {384, 512}:
        raise ValueError(f"Use {BASE_MODEL} with a max length of 384 or 512.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=True)

    seed_records = load_seed_records(
        args.seed_data, args.max_answer_chars, tokenizer, args.max_length
    )
    codefeedback_records = streamed_pairs(
        "m-a-p/CodeFeedback-Filtered-Instruction",
        "query",
        "answer",
        args.codefeedback_samples,
        args.max_answer_chars,
        tokenizer,
        args.max_length,
    )
    magicoder_records = streamed_pairs(
        "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "instruction",
        "response",
        args.magicoder_samples,
        args.max_answer_chars,
        tokenizer,
        args.max_length,
    )
    records = deduplicate(seed_records + codefeedback_records + magicoder_records)
    if len(records) < args.minimum_examples:
        raise RuntimeError(
            f"Only {len(records)} usable examples. Add more seed examples or increase dataset samples."
        )

    random.Random(SEED).shuffle(records)
    eval_count = max(24, round(len(records) * args.validation_ratio))
    eval_records = records[:eval_count]
    train_records = records[eval_count:]
    write_jsonl(args.train_output, train_records)
    write_jsonl(args.eval_output, eval_records)
    manifest = {
        "language": "English only",
        "seed_examples": len(seed_records),
        "codefeedback_examples": len(codefeedback_records),
        "magicoder_examples": len(magicoder_records),
        "unique_examples": len(records),
        "train_examples": len(train_records),
        "eval_examples": len(eval_records),
        "base_model": args.base_model,
        "max_length": args.max_length,
        "seed": SEED,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Training data: {args.train_output}")
    print(f"Validation data: {args.eval_output}")


if __name__ == "__main__":
    main()
