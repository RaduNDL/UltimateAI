import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =========================================================
# TURBO CONFIG (fast on 8GB VRAM)
# =========================================================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "llama-coder-lora"

MAX_SEQ_LEN = 256
MICRO_BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 1
LEARNING_RATE = 3e-4
WARMUP_STEPS = 30
SAVE_STEPS = 1000
LOGGING_STEPS = 50
SEED = 42

# =========================================================
# CUSTOM COLLATOR (fixes label padding issues)
# =========================================================
class CausalLMCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch_inputs = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len = batch_inputs["input_ids"].shape[1]
        padded_labels = []
        for l in labels:
            pad_len = max_len - len(l)
            padded_labels.append(l + [-100] * pad_len)

        batch_inputs["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch_inputs

def main():
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(SEED)
    if use_cuda:
        torch.cuda.manual_seed_all(SEED)

    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM GB: {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)}")
    else:
        print("WARNING: CUDA is not available. Running on CPU (very slow).")

    compute_dtype = (
        torch.bfloat16
        if (use_cuda and torch.cuda.is_bf16_supported())
        else (torch.float16 if use_cuda else torch.float32)
    )
    print(f"Compute dtype: {compute_dtype}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")
    if os.path.getsize(DATA_PATH) == 0:
        raise RuntimeError(f"{DATA_PATH} is empty. Run: python prepare_coding_data.py")

    # 1) Load dataset
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after load_dataset.")

    # 2) Format examples
    def format_example(example):
        messages = example.get("messages", [])
        chunks = []
        for m in messages:
            role = str(m.get("role", "")).strip().lower()
            content = str(m.get("content", "")).strip()
            if role in {"system", "user", "assistant"} and content:
                chunks.append(f"<|{role}|>\n{content}")

        if not chunks:
            return {"text": ""}

        return {"text": "\n\n".join(chunks) + "\n\n<|assistant|>\n"}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    print(f"Samples after formatting: {len(dataset)}")

    # Turbo subset for speed
    dataset = dataset.shuffle(seed=SEED).select(range(min(4000, len(dataset))))
    print(f"Samples after turbo limit: {len(dataset)}")

    # 3) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4) Tokenize
    def tokenize_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )
        out["labels"] = [ids.copy() for ids in out["input_ids"]]
        return out

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    print(f"Training samples: {len(tokenized)}")

    # 5) Model + quantization
    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()
    else:
        # CPU fallback (no 4-bit bitsandbytes)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
        )

    model.config.use_cache = False

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 6) Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE if use_cuda else 1,
        gradient_accumulation_steps=GRAD_ACCUM if use_cuda else 1,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=(use_cuda and torch.cuda.is_bf16_supported()),
        fp16=(use_cuda and not torch.cuda.is_bf16_supported()),
        optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=1.0,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=use_cuda,
        tf32=use_cuda,
    )

    collator = CausalLMCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    # 7) Train
    print("Starting fine-tuning...")
    trainer.train()

    # 8) Save
    print("Saving LoRA adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done. LoRA saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()