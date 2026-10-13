"""Collect language-model runs into a results table and charts (mean ± std over seeds).

    python scripts/summarize_lm.py --runs results/lm --out results/lm_summary
Writes <out>.md, <out>.csv (one row per run) and <out>.png (BPB by tokenizer/architecture + learning curves).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/lm")
    ap.add_argument("--out", default="results/lm_summary")
    args = ap.parse_args()

    runs = [json.loads(p.read_text()) for p in sorted(Path(args.runs).glob("*.json"))]
    if not runs:
        print("no finished runs yet")
        return
    for r in runs:
        r.setdefault("seed", 0)
    rows = []
    for r in sorted(runs, key=lambda r: (r["arch"], r["tokenizer"], r["seed"])):
        rows.append({
            "tokenizer": r["tokenizer"], "arch": r["arch"], "seed": r["seed"], "bpb": round(r["bpb"], 5),
            "val_loss_per_token": round(r["val_loss_per_token"], 4), "bytes_per_token": round(r["bytes_per_token"], 3),
            "train_tokens_M": round(r["train_tokens_seen"] / 1e6, 2),
            "non_embedding_M": round(r["params"]["non_embedding"] / 1e6, 2),
            "embedding_M": round(r["params"]["embedding"] / 1e6, 2), "train_min": round(r["train_minutes"], 1),
            "tokens_per_sec": round(r["train_tokens_per_sec"]), "dtype": r.get("dtype", ""), "device": r["device"],
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for r in runs:
        groups[(r["arch"], r["tokenizer"])].append(r)
    head = ["Architecture", "Tokenizer", "Seeds", "Bits per byte (mean ± sd)", "Per seed", "Bytes/token",
            "Minutes/run", "Tokens/s"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (arch, tok), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], np.mean([r["bpb"] for r in kv[1]]))):
        b = np.array([r["bpb"] for r in rs])
        sd = f" ± {b.std(ddof=1):.4f}" if len(b) > 1 else ""
        per = ", ".join(f"s{r['seed']}: {r['bpb']:.4f}" for r in sorted(rs, key=lambda r: r["seed"]))
        lines.append(f"| {arch} | {tok} | {len(rs)} | **{b.mean():.4f}**{sd} | {per} | {rs[0]['bytes_per_token']:.2f} | "
                     f"{np.mean([r['train_minutes'] for r in rs]):.0f} | {np.mean([r['train_tokens_per_sec'] for r in rs]):,.0f} |")
    Path(f"{out}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    archs = sorted({r["arch"] for r in runs})
    toks = sorted({r["tokenizer"] for r in runs})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    width = 0.8 / max(1, len(archs))
    allv = []
    for i, arch in enumerate(archs):
        for j, t in enumerate(toks):
            rs = groups.get((arch, t), [])
            if not rs:
                continue
            b = np.array([r["bpb"] for r in rs])
            x = j + i * width
            allv += list(b)
            axes[0].bar(x, b.mean(), width, color=f"C{i}", label=arch if j == 0 else None,
                        yerr=b.std(ddof=1) if len(b) > 1 else None, capsize=4)
            axes[0].scatter([x] * len(b), b, color="black", s=10, zorder=3)
            axes[0].text(x, b.max(), f"{b.mean():.4f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks([j + width * (len(archs) - 1) / 2 for j in range(len(toks))], toks, rotation=15)
    lo, hi = min(allv), max(allv)
    axes[0].set_ylim(lo - 0.2 * (hi - lo + 1e-3), hi + 0.25 * (hi - lo + 1e-3))
    axes[0].set(title="Validation bits per byte (lower is better; dots = seeds)", ylabel="BPB")
    axes[0].legend()
    for r in runs:
        h = r.get("history", [])
        if h and r["seed"] == min(x["seed"] for x in groups[(r["arch"], r["tokenizer"])]):
            mb = [x["tokens"] * r["bytes_per_token"] / 1e6 for x in h]
            axes[1].plot(mb, [x["val_bpb_est"] for x in h], marker=".",
                         linestyle="-" if r["arch"] == "transformer" else "--", label=f"{r['tokenizer']} / {r['arch']}")
    axes[1].set(title="Learning curves (first seed)", xlabel="Training text seen (MB)", ylabel="BPB (subset estimate)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out}.png", dpi=150)
    print("saved", f"{out}.png")


if __name__ == "__main__":
    main()
