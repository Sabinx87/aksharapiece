"""Downstream test: fine-tune each pretrained model on Nepali news topic classification (np20ng).

Every model sees exactly the same documents, cut to the same number of characters (so no tokenizer
gets more text than another), the same label split and the same training recipe:
  mean-pooled final hidden states -> linear head, full fine-tuning, AdamW, a few epochs.
Reports accuracy and macro-F1 per model and the mean over pretraining seeds.

    # from Hugging Face (gated: accept the terms on the dataset page once; HF_TOKEN = a Read token)
    python scripts/finetune_cls.py --models "models/*.pt" --tokenizers tokenizers --out results/cls
    # or from a local CSV with columns text,label (or headline,content,category)
    python scripts/finetune_cls.py --models "models/*.pt" --tokenizers tokenizers --csv news.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from aksharapiece.akshara import normalize  # noqa: E402
from aksharapiece.codec import Codec  # noqa: E402
from generate_eval import load_model  # noqa: E402


# ----------------------------------------------------------------------------- data
def _doc(row, max_chars=4000):
    if row.get("text"):
        return row["text"][:max_chars], row.get("label")
    head = (row.get("headline") or "").strip()
    body = (row.get("content") or "").strip()[:max_chars]
    return (head + " । " + body if head else body), row.get("category") or row.get("label")


def load_rows(args):
    path = args.csv
    if not path:   # np20ng is one CSV file on the Hub (gated: accept its terms once, then use a Read token)
        from huggingface_hub import hf_hub_download
        print(f"downloading {args.hf_dataset}/{args.hf_file} ...", flush=True)
        path = hf_hub_download(args.hf_dataset, args.hf_file, repo_type="dataset",
                               token=os.environ.get("HF_TOKEN") or None, cache_dir=args.cache_dir)
    csv.field_size_limit(10**9)
    with open(path, encoding="utf-8", newline="") as f:
        return [_doc(r, args.max_chars * 2) for r in csv.DictReader(f)]


def make_split(rows, n_train, n_test, max_chars, seed=0):
    """Balanced per-class sample; identical for every model."""
    by = defaultdict(list)
    for text, label in rows:
        text = normalize(str(text or ""))
        if label is None or len(text) < 50:
            continue
        by[str(label)].append(text[:max_chars])
    rng = random.Random(seed)
    labels = sorted(by)
    train, test = [], []
    for li, lab in enumerate(labels):
        docs = list(dict.fromkeys(by[lab]))           # drop exact duplicates
        rng.shuffle(docs)
        nt = min(n_test, max(1, len(docs) // 5))
        test += [(d, li) for d in docs[:nt]]
        train += [(d, li) for d in docs[nt:nt + n_train]]
    rng.shuffle(train)
    return train, test, labels


# ----------------------------------------------------------------------------- model
class Classifier(nn.Module):
    def __init__(self, lm, n_classes, dropout=0.1):
        super().__init__()
        self.lm = lm
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(lm.c.d_model, n_classes)

    def forward(self, idx, mask):
        h = self.lm.hidden(idx).float()                                  # (b, l, d)
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(pooled))


def batches(ids, labels, bs, pad, device, shuffle, rng=None):
    order = list(range(len(ids)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), bs):
        sel = order[i:i + bs]
        L = max(1, max(len(ids[j]) for j in sel))
        x = torch.full((len(sel), L), pad, dtype=torch.long)
        mask = torch.zeros(len(sel), L, dtype=torch.bool)
        for r, j in enumerate(sel):                                       # right padding: causal models ignore it
            x[r, :len(ids[j])] = torch.tensor(ids[j])
            mask[r, :len(ids[j])] = True
        yield x.to(device), mask.to(device), torch.tensor([labels[j] for j in sel], device=device)


def macro_f1(y, p, k):
    f1s = []
    for c in range(k):
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def run_one(path, train, test, n_classes, codecs, args, device, amp_dtype):
    lm, ck = load_model(path, device)
    tname = ck["tokenizer"]
    if tname not in codecs:
        cands = glob.glob(str(Path(args.tokenizers) / f"{tname}.aksharapiece.json")) + \
            glob.glob(str(Path(args.tokenizers) / f"{tname}.json"))
        if not cands:
            print(f"  skip {path}: tokenizer {tname} not found")
            return None
        codecs[tname] = Codec(cands[0])
    codec = codecs[tname]
    max_tok = min(args.max_tokens, lm.c.max_seq_len)
    tr_ids = [t[:max_tok] for t in codec.encode_batch([d for d, _ in train])]
    te_ids = [t[:max_tok] for t in codec.encode_batch([d for d, _ in test])]
    tr_y, te_y = [y for _, y in train], [y for _, y in test]
    truncated = sum(len(t) >= max_tok for t in tr_ids) / max(1, len(tr_ids))

    seed = int(Path(path).stem.split("__s")[-1]) if "__s" in Path(path).stem else 0
    torch.manual_seed(1000 + seed)
    model = Classifier(lm, n_classes).to(device)
    head_p = list(model.head.parameters())
    body_p = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    opt = torch.optim.AdamW([{"params": body_p, "lr": args.lr}, {"params": head_p, "lr": args.lr * 10}],
                            weight_decay=0.01, betas=(0.9, 0.98))
    steps = args.epochs * ((len(tr_ids) + args.batch - 1) // args.batch)
    warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    rng = random.Random(seed)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for x, m, y in batches(tr_ids, tr_y, args.batch, codec.sep_id, device, True, rng):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = F.cross_entropy(model(x, m), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        print(f"    epoch {ep + 1}: train loss {tot / max(1, n):.3f}", flush=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for x, m, _ in batches(te_ids, te_y, args.batch * 2, codec.sep_id, device, False):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                preds += model(x, m).argmax(-1).tolist()
    acc = float(np.mean([a == b for a, b in zip(te_y, preds)]))
    name = Path(path).stem
    parts = name.split("__")
    r = {"model": name, "tokenizer": tname, "arch": parts[1] if len(parts) > 1 else "",
         "seed": parts[2] if len(parts) > 2 else "", "accuracy": acc, "macro_f1": macro_f1(te_y, preds, n_classes),
         "avg_tokens": float(np.mean([len(t) for t in tr_ids])), "truncated_%": 100 * truncated,
         "minutes": (time.time() - t0) / 60}
    del model, lm, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--tokenizers", required=True)
    ap.add_argument("--csv", default=None, help="local CSV (text,label) or (headline,content,category)")
    ap.add_argument("--hf-dataset", default="Suyogyart/np20ng")
    ap.add_argument("--hf-file", default="np20ng.csv")
    ap.add_argument("--cache-dir", default=None, help="where to keep the downloaded CSV")
    ap.add_argument("--train-per-class", type=int, default=1000)
    ap.add_argument("--test-per-class", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=600, help="same characters of text for every tokenizer")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default=str(ROOT / "results" / "classification"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        rows = load_rows(args)
    except Exception as e:  # gated dataset not accepted, no token, no internet ...
        msg = f"Could not load the classification data: {type(e).__name__}: {e}"
        print(msg)
        (out / "SKIPPED.txt").write_text(msg + "\n")
        return
    train, test, labels = make_split(rows, args.train_per_class, args.test_per_class, args.max_chars)
    print(f"{len(rows):,} documents -> {len(labels)} classes, {len(train):,} train / {len(test):,} test")
    print("  classes:", ", ".join(f"{l} ({c})" for l, c in sorted(Counter(labels[y] for _, y in test).items())))
    (out / "split_info.json").write_text(json.dumps(
        {"labels": labels, "n_train": len(train), "n_test": len(test), "args": vars(args)}, indent=2, ensure_ascii=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    codecs, results = {}, []
    for path in sorted(glob.glob(args.models)):
        print(f"  {Path(path).stem}", flush=True)
        r = run_one(path, train, test, len(labels), codecs, args, device, amp_dtype)
        if r:
            results.append(r)
            print(f"    accuracy {100 * r['accuracy']:.2f}%  macro-F1 {100 * r['macro_f1']:.2f}  "
                  f"({r['avg_tokens']:.0f} tokens/doc, {r['minutes']:.1f} min)", flush=True)
            (out / "cls_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    head = "| Model | Accuracy | Macro-F1 | Tokens per doc | Truncated |\n|---|---|---|---|---|"
    body = [f"| {r['model']} | {100 * r['accuracy']:.2f}% | {100 * r['macro_f1']:.2f} | {r['avg_tokens']:.0f} | {r['truncated_%']:.1f}% |"
            for r in results]
    agg = defaultdict(list)
    for r in results:
        agg[(r["tokenizer"], r["arch"])].append(r)
    summ = ["", "**Mean ± sd over pretraining seeds**", "",
            "| Tokenizer | Arch | Seeds | Accuracy | Macro-F1 |", "|---|---|---|---|---|"]
    for (t, a), rs in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        acc = [100 * x["accuracy"] for x in rs]
        f1 = [100 * x["macro_f1"] for x in rs]
        summ.append(f"| {t} | {a} | {len(rs)} | {np.mean(acc):.2f} ± {np.std(acc, ddof=1) if len(rs) > 1 else 0:.2f} | "
                    f"{np.mean(f1):.2f} ± {np.std(f1, ddof=1) if len(rs) > 1 else 0:.2f} |")
    md = f"{len(labels)} classes, {len(train):,} train / {len(test):,} test documents, first {args.max_chars} characters.\n\n" \
         + "\n".join([head] + body + summ) + "\n"
    (out / "cls_metrics.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
