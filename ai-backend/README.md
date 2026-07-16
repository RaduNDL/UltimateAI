# UltimateAI English coding assistant

This project serves `Qwen/Qwen2.5-1.5B-Instruct` locally in 4-bit NF4 on an
8 GB RTX laptop GPU. It responds in English only. The base model remains frozen;
training creates a small QLoRA adapter rather than changing the base weights.

## Run the stable base model

```powershell
cd C:\Users\Administrator\Documents\WebSites\ai-backend
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The server does not load a new training candidate automatically. This prevents a
bad adapter from degrading the stable base model.

## Build English-only training data

Edit `data/english_seed.jsonl` with examples for the stack you actually use.
Then build a small mixed dataset:

```powershell
.\.venv\Scripts\python.exe prepare_data.py
```

The default build contains hand-reviewed application examples plus a small
streamed sample from CodeFeedback and Magicoder. It creates:

- `data/app-train.jsonl`
- `data/app-eval.jsonl`
- `data/app-data-manifest.json`

## Train a candidate adapter

Stop the API before training because inference and training share the GPU.

```powershell
.\.venv\Scripts\python.exe train_qlora.py --max-steps 44 --time-limit-minutes 24
```

This uses 4-bit QLoRA, rank 8 adapters on `q_proj` and `v_proj`, gradient
checkpointing, and an assertion that rejects any trainable base-model parameter.
The adapter is written to `adapters/english-app-candidate` and stays inactive.

## Evaluate before activation

First evaluate the frozen base model and keep the report:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe evaluate_model.py --output base-report.json
```

Stop the API, start it once with the candidate adapter, and evaluate again:

```powershell
$env:ULTIMATEAI_ADAPTER_PATH = "C:\Users\Administrator\Documents\WebSites\ai-backend\adapters\english-app-candidate"
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe evaluate_model.py --output candidate-report.json
```

Compare the reports. If the candidate is not clearly better, keep the base model.
Close that terminal after testing so the temporary environment variable is gone.

## Activate or disable a good candidate

```powershell
.\.venv\Scripts\python.exe activate_adapter.py --adapter .\adapters\english-app-candidate
```

Restart the API. To return to the base model:

```powershell
.\.venv\Scripts\python.exe activate_adapter.py --disable
```

## Retrieval rules

Project context is off by default. Enable it only after selecting a project and
asking an explicitly project-specific question, such as a question about a file,
component, endpoint, or pasted code. Generic programming questions should not
receive unrelated source files as context.

## Important limitation

A 24-minute QLoRA run cannot teach a 1.5B model all world knowledge. It teaches
response style and recurring software tasks. Use source retrieval and curated
documentation for project-specific or current knowledge.
