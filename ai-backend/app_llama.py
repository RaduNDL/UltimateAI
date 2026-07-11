from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = FastAPI(title="Llama Coding API")

# IMPORTANT: use the same base model used during fine-tuning
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "llama-coder-lora"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=6000)
    max_new_tokens: int = Field(256, ge=1, le=512)
    temperature: float = Field(0.7, gt=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)

@app.post("/chat")
def chat(req: ChatRequest):
    prompt = (
        "<|system|>\nYou are an expert programming assistant. "
        "Give concise and correct code.\n\n"
        f"<|user|>\n{req.prompt}\n\n<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_llama:app", host="127.0.0.1", port=8000, reload=False)