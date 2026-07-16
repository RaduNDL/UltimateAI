"""Train a candidate English-only QLoRA adapter while keeping the base frozen."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TRAIN_DATA = ROOT / "data" / "app-train.jsonl"
DEFAULT_EVAL_DATA = ROOT / "data" / "app-eval.jsonl"
DEFAULT_OUTPUT = ROOT / "adapters" / "english-app-candidate"


class TimeLimitCallback(TrainerCallback):
    """Stop cleanly after the user-defined wall-clock limit."""

    def __init__(self, minutes: int) -> None:
        self.limit_seconds = minutes * 60
        self.started_at = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = time.monotonic()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() - self.started_at >= self.limit_seconds:
            print(f"Time limit of {self.limit_seconds // 60} minutes reached; stopping cleanly.")
            control.should_training_stop = True
        return control


def encode_example(example: dict, tokenizer, max_length: int) -> dict:
    """Mask the system and user messages so loss is applied only to the answer."""
    messages = example["messages"]
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer chat template is incompatible with assistant-only masking.")
    completion_tokens = len(full_ids) - len(prompt_ids)
    valid = len(full_ids) <= max_length and completion_tokens >= 32
    return {
        "input_ids": full_ids[:max_length],
        "attention_mask": [1] * min(len(full_ids), max_length),
        "labels": [-100] * min(len(prompt_ids), max_length)
        + full_ids[min(len(prompt_ids), max_length):max_length],
        "valid": valid,
    }


def encode_dataset(raw: Dataset, tokenizer, max_length: int) -> Dataset:
    encoded = raw.map(
        lambda example: encode_example(example, tokenizer, max_length),
        remove_columns=raw.column_names,
        desc="Tokenizing examples",
    ).filter(lambda row: row["valid"], desc="Removing truncated or short answers")
    return encoded.remove_columns("valid")


def assert_frozen_backbone(model) -> None:
    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if unexpected_trainable:
        raise RuntimeError(
            "Backbone is not frozen. Unexpected trainable parameters: "
            f"{unexpected_trainable[:5]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an English-only candidate adapter without changing base weights."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=44)
    parser.add_argument("--time-limit-minutes", type=int, default=24)
    parser.add_argument("--overwrite-output", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA training run.")
    if args.base_model != BASE_MODEL:
        raise ValueError(
            f"This pipeline is locked to {BASE_MODEL}; serving and training bases must match."
        )
    if not args.data.exists() or not args.eval_data.exists():
        raise FileNotFoundError("Run prepare_data.py before training.")
    if not 1 <= args.max_steps <= 60:
        raise ValueError("max-steps must be between 1 and 60 for this short adaptation.")
    if not 1 <= args.time_limit_minutes <= 30:
        raise ValueError("time-limit-minutes must be between 1 and 30.")
    if args.max_length not in {384, 512}:
        raise ValueError("Use max-length 384 or 512 on an 8 GB GPU.")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite_output:
        raise FileExistsError(
            f"Candidate output already exists: {args.output}. Use a new path or --overwrite-output."
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Frozen base model: {args.base_model}")
    print(f"Candidate output: {args.output}")
    print(f"Hard limits: {args.max_steps} steps, {args.time_limit_minutes} minutes")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_train = load_dataset("json", data_files=str(args.data), split="train")
    raw_eval = load_dataset("json", data_files=str(args.eval_data), split="train")
    train_dataset = encode_dataset(raw_train, tokenizer, args.max_length)
    eval_dataset = encode_dataset(raw_eval, tokenizer, args.max_length)
    if len(train_dataset) < 100 or len(eval_dataset) < 24:
        raise RuntimeError(
            f"Need at least 100 train and 24 validation examples after filtering; got "
            f"{len(train_dataset)} train and {len(eval_dataset)} validation."
        )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model_config = AutoConfig.from_pretrained(args.base_model)
    if hasattr(model_config, "use_sliding_window"):
        model_config.use_sliding_window = False
    if hasattr(model_config, "sliding_window"):
        model_config.sliding_window = None
    if hasattr(model_config, "max_window_layers"):
        model_config.max_window_layers = 0
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=model_config,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        ),
    )
    assert_frozen_backbone(model)
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    run_dir = ROOT / "runs" / "english-app-candidate"
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        gradient_checkpointing=True,
        fp16=compute_dtype == torch.float16,
        bf16=compute_dtype == torch.bfloat16,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        logging_steps=2,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=min(10, args.max_steps),
        save_strategy="no",
        report_to="none",
        seed=42,
        data_seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[TimeLimitCallback(args.time_limit_minutes)],
    )
    train_result = trainer.train()
    eval_result = trainer.evaluate()

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    manifest = {
        "base_model": args.base_model,
        "language": "English only",
        "train_examples": len(train_dataset),
        "validation_examples": len(eval_dataset),
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "time_limit_minutes": args.time_limit_minutes,
        "learning_rate": args.learning_rate,
        "training_loss": train_result.training_loss,
        "validation_loss": eval_result.get("eval_loss"),
        "status": "candidate - evaluate before activation",
    }
    (args.output / "training-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print("Candidate saved. It is NOT active until activate_adapter.py is run after evaluation.")


if __name__ == "__main__":
    main()
