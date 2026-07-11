# UltimateAI

UltimateAI is a local AI coding assistant project with:

- **Backend**: FastAPI + Hugging Face Transformers + PEFT (LoRA fine-tuning)
- **Model**: TinyLlama-1.1B-Chat (fine-tuned with LoRA)
- **Frontend**: Angular chat UI
- **Training Pipeline**: dataset preparation + fast LoRA training script for 8GB VRAM GPUs

---

## Features

- Local `/chat` API powered by a fine-tuned LLM
- Lightweight LoRA fine-tuning flow for consumer GPUs (e.g., laptop RTX 3070 Ti 8GB)
- Dataset generator for coding instruction data
- Frontend chat interface (Angular)
- Fast “turbo” training profile for quick iteration

---

## Tech Stack

- Python 3.10+
- FastAPI
- Transformers
- PEFT
- Datasets
- BitsAndBytes (4-bit quantization)
- Angular

---

## Project Structure

```bash
UltimateAI/
├─ ai-backend/
│  ├─ app_llama.py
│  ├─ finetune_llama.py
│  ├─ prepare_coding_data.py
│  ├─ requirements.txt
│  ├─ data/
│  │  └─ train.jsonl
│  └─ llama-coder-lora/   # generated after training
└─ ai-frontend/
   └─ src/
```

---

## Backend Setup

1. Go to backend folder:
```bash
cd ai-backend
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Prepare Dataset

Generate `data/train.jsonl` automatically:

```bash
python prepare_coding_data.py
```

---

## Fine-tune Model (LoRA)

Run turbo fine-tuning:

```bash
python finetune_llama.py
```

Output adapter folder:
- `llama-coder-lora/`

---

## Run API

Start FastAPI server:

```bash
python app_llama.py
```

Default endpoint:
- `POST http://127.0.0.1:8000/chat`

Example payload:
```json
{
  "prompt": "Write a Python function for quicksort.",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9
}
```

---

## Frontend Setup (Angular)

```bash
cd ai-frontend
npm install
ng serve
```

Make sure frontend calls backend:
- `http://127.0.0.1:8000/chat`

---

## Notes

- If `CUDA is not available`, install CUDA-enabled PyTorch wheel in your venv.
- For Windows, `dataloader_num_workers=0` is more stable.
- Fine-tuning speed/quality trade-off can be tuned in `finetune_llama.py`.

---

## Roadmap

- [ ] Streaming token responses
- [ ] Conversation memory
- [ ] Better eval metrics for coding tasks
- [ ] Dockerized full-stack deployment
- [ ] Model switching UI

---

## License

MIT (or your preferred license).
