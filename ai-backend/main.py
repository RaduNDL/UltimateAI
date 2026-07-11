import os
import math
import torch
from model import GPTLanguageModel, encode, device, block_size, AUTOCAST_DTYPE

batch_size = 4
accumulation_steps = 8
max_iters = 15000
learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 200
eval_interval = 500
eval_iters = 20
grad_clip = 1.0

checkpoint_path = "checkpoint.pth"
model_output_path = "chatbot_model.pth"

print(f"Training on: {device.type.upper()}", flush=True)
print(f"Autocast dtype: {AUTOCAST_DTYPE}", flush=True)

if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {gpu_name} | VRAM: {total_vram_gb:.2f} GB", flush=True)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    source = train_data if split == "train" else val_data
    if len(source) <= block_size + 1:
        raise RuntimeError(
            f"Dataset too small for block_size={block_size}. "
            f"Got {len(source)} tokens on split={split}."
        )

    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def save_checkpoint(
    base_model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    iteration: int,
) -> None:
    torch.save(
        {
            "iter": iteration,
            "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        checkpoint_path,
    )
    print(f"[CKPT] Saved at iter {iteration}", flush=True)


def load_checkpoint(
    base_model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
) -> int:
    if not os.path.exists(checkpoint_path):
        return 0

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start = int(ckpt["iter"]) + 1
    print(f"[CKPT] Resuming from iter {start}", flush=True)
    return start


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    b, t, c = logits.shape
    pred_ids = torch.argmax(logits.reshape(b * t, c), dim=-1)
    return (pred_ids == targets.reshape(b * t)).float().mean().item() * 100.0


@torch.no_grad()
def estimate_loss_and_accuracy(model_ref: torch.nn.Module) -> dict:
    out = {}
    model_ref.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        accs = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            with torch.amp.autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                logits, loss = model_ref(x_batch, y_batch)
            losses[k] = float(loss.item())
            accs[k] = calculate_accuracy(logits, y_batch)

        out[split] = {
            "loss": float(losses.mean().item()),
            "accuracy": float(accs.mean().item()),
        }

    model_ref.train()
    return out


def lr_lambda(it: int) -> float:
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    progress = min((it - warmup_iters) / max(1, max_iters - warmup_iters), 1.0)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    min_ratio = min_lr / learning_rate
    return min_ratio + (1 - min_ratio) * cosine_decay


if __name__ == "__main__":
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    base_model = GPTLanguageModel().to(device)
    train_model: torch.nn.Module = base_model

    if os.name != "nt" and hasattr(torch, "compile"):
        try:
            train_model = torch.compile(base_model, mode="reduce-overhead")
            print("torch.compile(reduce-overhead) enabled", flush=True)
        except Exception as compile_error:
            print(f"torch.compile disabled: {compile_error}", flush=True)
            train_model = base_model
    else:
        print("Skipping torch.compile on Windows.", flush=True)

    try:
        optimizer = torch.optim.AdamW(
            train_model.parameters(),
            lr=learning_rate,
            fused=(device.type == "cuda"),
        )
    except Exception:
        optimizer = torch.optim.AdamW(train_model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(AUTOCAST_DTYPE == torch.float16))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(AUTOCAST_DTYPE == torch.float16))

    start_iter = load_checkpoint(base_model, optimizer, scheduler, scaler)
    current_iter = start_iter

    try:
        for current_iter in range(start_iter, max_iters):
            if current_iter % eval_interval == 0 or current_iter == max_iters - 1:
                metrics = estimate_loss_and_accuracy(train_model)
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Step {current_iter:5d} | LR {lr:.2e} | "
                    f"Train Loss {metrics['train']['loss']:.4f} Acc {metrics['train']['accuracy']:.2f}% | "
                    f"Val Loss {metrics['val']['loss']:.4f} Acc {metrics['val']['accuracy']:.2f}%",
                    flush=True,
                )
                save_checkpoint(base_model, optimizer, scheduler, scaler, current_iter)

            optimizer.zero_grad(set_to_none=True)

            for _ in range(accumulation_steps):
                xb, yb = get_batch("train")
                with torch.amp.autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                    _, loss = train_model(xb, yb)
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if device.type == "cuda" and current_iter % 100 == 0:
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                print(f"[VRAM] allocated={allocated:.2f} GB | reserved={reserved:.2f} GB", flush=True)

    except RuntimeError as runtime_error:
        if "out of memory" in str(runtime_error).lower():
            print("\n[OOM] CUDA out of memory. Saving checkpoint and exiting...", flush=True)
            save_checkpoint(base_model, optimizer, scheduler, scaler, current_iter)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise SystemExit(1)
        raise
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl+C caught - saving checkpoint...", flush=True)
        save_checkpoint(base_model, optimizer, scheduler, scaler, current_iter)
        raise SystemExit(0)

    torch.save(base_model.state_dict(), model_output_path)
    print("Training complete.", flush=True)