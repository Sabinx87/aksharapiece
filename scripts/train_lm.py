"""Train one small language model on pre-tokenized data and report bits per byte.

Every run sees the same TEXT (same bytes); token counts differ by tokenizer, which is part of
what is being compared. One epoch = every training token seen once, in shuffled windows.

    python scripts/train_lm.py --tokens data/tokens/akshara_bpe_16k --arch mamba --out results/lm

Bits per byte (the headline metric):
    BPB = (sum of next-token losses in nats over the validation stream) / ln(2) / (validation UTF-8 bytes)
It is comparable across tokenizers because the denominator is the text, not the tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aksharapiece.models import LM, ModelConfig  # noqa: E402


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True, help="folder with train.bin, val.bin, meta.json")
    ap.add_argument("--arch", choices=["transformer", "mamba"], required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "lm"))
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=None, help="default: 8 transformer / 16 mamba")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.03, help="fraction of steps")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None, help="cap, for quick tests")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-tokens", type=int, default=200_000, help="tokens for periodic evals (final eval uses all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-model", action="store_true", help="save float16 weights to --model-dir")
    ap.add_argument("--model-dir", default=None, help="where to save weights (default: --out)")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, val, seq_len, batch, device, amp, max_tokens=None, sep_id=None):
    """Sum of next-token NLL (nats) over the val stream, in non-overlapping windows.

    With sep_id, also returns the NLL summed per text line (a token belongs to the line it is in;
    each line's separator belongs to that line). Lines are identical across tokenizers, so per-line
    sums allow paired significance tests between tokenizers.
    """
    model.eval()
    n = len(val) if max_tokens is None else min(len(val), max_tokens + 1)
    starts = list(range(0, n - 1, seq_len))
    total_nll, total_tok = 0.0, 0
    per_line = None
    if sep_id is not None:
        is_sep = np.asarray(val[:n], dtype=np.int64) == sep_id
        line_of = np.concatenate([[0], np.cumsum(is_sep)[:-1]])      # line index of each position
        per_line = np.zeros(int(line_of[-1]) + 1, dtype=np.float64)
    for i in range(0, len(starts), batch):
        xs, ys = [], []
        for s in starts[i:i + batch]:
            e = min(s + seq_len, n - 1)
            x = np.asarray(val[s:e], dtype=np.int64)
            y = np.asarray(val[s + 1:e + 1], dtype=np.int64)
            pad = seq_len - len(x)
            xs.append(np.pad(x, (0, pad)))
            ys.append(np.pad(y, (0, pad), constant_values=-100))  # ignore padding
        x = torch.from_numpy(np.stack(xs)).to(device)
        y = torch.from_numpy(np.stack(ys)).to(device)
        with amp:
            logits = model(x)
        tok_nll = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                                                    ignore_index=-100, reduction="none")
        total_nll += tok_nll.sum().item()
        total_tok += int((y != -100).sum())
        if per_line is not None:
            tok_nll = tok_nll.view(len(xs), seq_len).cpu().numpy()
            for row, s in enumerate(starts[i:i + batch]):
                e = min(s + seq_len, n - 1)
                np.add.at(per_line, line_of[s + 1:e + 1], tok_nll[row, :e - s])
    model.train()
    if per_line is not None:
        return total_nll, total_tok, per_line
    return total_nll, total_tok


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tok_dir = Path(a.tokens)
    meta = json.loads((tok_dir / "meta.json").read_text())
    train = np.memmap(tok_dir / "train.bin", dtype=np.uint32, mode="r")
    val = np.memmap(tok_dir / "val.bin", dtype=np.uint32, mode="r")
    run_name = f"{meta['name']}__{a.arch}__s{a.seed}"
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    if device.type == "cuda":
        # bfloat16 only on GPUs with native support (compute capability 8.0+, e.g. A100/L4/H100).
        # T4 and P100 report bf16 as "supported" via slow emulation, so they get float16 + loss scaling.
        native_bf16 = torch.cuda.get_device_capability()[0] >= 8
        dtype = torch.bfloat16 if native_bf16 else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        dtype = torch.float32
    amp = torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))

    layers = a.layers or (8 if a.arch == "transformer" else 16)
    cfg = ModelConfig(arch=a.arch, vocab_size=meta["model_vocab"], d_model=a.d_model, n_layers=layers,
                      n_heads=a.heads, max_seq_len=a.seq_len)
    model = LM(cfg).to(device)
    counts = model.param_counts()

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    try:
        opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), fused=(device.type == "cuda"))
    except (RuntimeError, TypeError):  # older GPUs / PyTorch builds without the fused kernel
        opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95))

    # one epoch = every non-overlapping window once, in shuffled order
    n_windows = (len(train) - 1) // a.seq_len
    tokens_per_step = a.batch * a.seq_len * a.grad_accum
    steps = int(a.epochs * n_windows * a.seq_len / tokens_per_step)
    if a.max_steps:
        steps = min(steps, a.max_steps)
    order = np.concatenate([np.random.permutation(n_windows) for _ in range(math.ceil(a.epochs) or 1)])
    warm = max(1, int(a.warmup * steps))

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        prog = (s - warm) / max(1, steps - warm)
        return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))

    print(f"[{run_name}] device {device} ({dtype}) | params {counts['non_embedding'] / 1e6:.1f}M non-embedding "
          f"+ {counts['embedding'] / 1e6:.1f}M embedding | {len(train):,} train tokens | {steps:,} steps "
          f"x {tokens_per_step:,} tokens", flush=True)

    history, w = [], 0
    t0 = time.time()
    tokens_seen = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss_acc = 0.0
        for _ in range(a.grad_accum):
            idx = order[w:w + a.batch] * a.seq_len
            w += a.batch
            xb = np.stack([np.asarray(train[i:i + a.seq_len], dtype=np.int64) for i in idx])
            yb = np.stack([np.asarray(train[i + 1:i + 1 + a.seq_len], dtype=np.int64) for i in idx])
            x = torch.from_numpy(xb).to(device, non_blocking=True)
            y = torch.from_numpy(yb).to(device, non_blocking=True)
            with amp:
                _, loss_sum = model(x, y)
            loss = loss_sum / y.numel() / a.grad_accum
            scaler.scale(loss).backward()
            loss_acc += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        tokens_seen += tokens_per_step

        if step % 50 == 0 or step == steps - 1:
            el = time.time() - t0
            print(f"  step {step:6d}/{steps}  loss {loss_acc:.3f}  lr {lr_at(step):.2e}  "
                  f"{tokens_seen / el:,.0f} tok/s  {el / 60:.1f} min", flush=True)
        if (a.eval_every and step > 0 and step % a.eval_every == 0) or step == steps - 1:
            nll, ntok = evaluate(model, val, a.seq_len, a.batch, device, amp, a.eval_tokens)
            covered_bytes = meta["val_bytes"] * ntok / max(1, meta["val_tokens"] - 1)
            bpb = nll / math.log(2) / covered_bytes
            history.append({"step": step, "tokens": tokens_seen, "train_loss": loss_acc, "val_bpb_est": bpb})
            print(f"  eval step {step}: val loss/token {nll / ntok:.3f}  ~BPB {bpb:.4f}", flush=True)

    train_time = time.time() - t0
    nll, ntok, per_line = evaluate(model, val, a.seq_len, a.batch, device, amp, sep_id=meta["sep_id"])
    np.save(out_dir / f"{run_name}.lines.npy", per_line)   # NLL (nats) per validation line
    bpb = nll / math.log(2) / meta["val_bytes"]
    result = {
        "run": run_name, "tokenizer": meta["name"], "arch": a.arch, "seed": a.seed, "bpb": bpb,
        "val_loss_per_token": nll / ntok, "val_tokens": ntok, "val_bytes": meta["val_bytes"],
        "train_tokens_seen": tokens_seen, "train_bytes": meta["train_bytes"], "steps": steps,
        "tokenizer_vocab": meta["tokenizer_vocab"], "bytes_per_token": meta["bytes_per_token"],
        "val_unk_rate": meta["val_unk_rate"], "params": counts, "config": model.config_dict(),
        "train_minutes": train_time / 60, "train_tokens_per_sec": tokens_seen / train_time,
        "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
        "device": str(torch.cuda.get_device_name(0) if device.type == "cuda" else device), "dtype": str(dtype),
        "args": vars(a), "history": history,
    }
    (out_dir / f"{run_name}.json").write_text(json.dumps(result, indent=2))
    if a.save_model:
        mdir = Path(a.model_dir or out_dir)
        mdir.mkdir(parents=True, exist_ok=True)
        state = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
        torch.save({"config": model.config_dict(), "state_dict": state, "tokenizer": meta["name"],
                    "spec": meta["spec"], "bpb": bpb}, mdir / f"{run_name}.pt")
    print(f"[{run_name}] DONE  BPB {bpb:.4f}  (loss/token {nll / ntok:.3f}, {train_time / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
