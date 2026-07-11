import json
import os
from datasets import load_dataset

OUT_PATH = "data/train.jsonl"

SYSTEM_PROMPT = (
    "You are an expert programming assistant. "
    "Give correct, concise, production-ready answers with code when needed."
)

def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def to_record(user_text: str, assistant_text: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text.strip()},
            {"role": "assistant", "content": assistant_text.strip()},
        ]
    }

def main():
    ensure_dir(OUT_PATH)

    # Public dataset (no gated access required)
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

    written = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in ds:
            instruction = str(row.get("instruction", "")).strip()
            inp = str(row.get("input", "")).strip()
            output = str(row.get("output", "")).strip()

            if not instruction or not output:
                continue

            if inp:
                user_text = f"{instruction}\n\nContext:\n{inp}"
            else:
                user_text = instruction

            rec = to_record(user_text, output)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. Wrote {written} examples to: {OUT_PATH}")

if __name__ == "__main__":
    main()