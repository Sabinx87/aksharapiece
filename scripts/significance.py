"""Is AksharaPiece really better than plain BPE? Paired tests over validation lines and seeds.

For each architecture and seed, the two models were scored on the SAME validation lines, so we can
resample lines (paired bootstrap) to get a 95% confidence interval for the difference in bits per byte.
Across seeds, we report the mean difference and how many seeds agree.

    python scripts/significance.py --runs results/lm --line-bytes data/tokens/plain_bpe_16k/val_line_bytes.bin \
        --a plain_bpe_16k --b akshara_bpe_16k --out results/significance_16k
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def bootstrap_diff(nll_a, nll_b, line_bytes, n_boot=2000, seed=0):
    """Relative BPB difference (b vs a, in %) with a 95% paired-bootstrap interval."""
    rng = np.random.default_rng(seed)
    n = len(line_bytes)
    point = 100 * (nll_b.sum() / nll_a.sum() - 1)     # same bytes, so the BPB ratio is the NLL ratio
    diffs = np.empty(n_boot)
    for i in range(0, n_boot, 200):
        idx = rng.integers(0, n, size=(min(200, n_boot - i), n))
        diffs[i:i + len(idx)] = 100 * (nll_b[idx].sum(1) / nll_a[idx].sum(1) - 1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_two_sided = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    return point, lo, hi, max(p_two_sided, 1 / n_boot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/lm")
    ap.add_argument("--line-bytes", required=True, help="val_line_bytes.bin written by prepare_tokens.py")
    ap.add_argument("--a", required=True, help="baseline tokenizer name, e.g. plain_bpe_16k")
    ap.add_argument("--b", required=True, help="tokenizer to test, e.g. akshara_bpe_16k")
    ap.add_argument("--out", default="results/significance")
    args = ap.parse_args()

    line_bytes = np.fromfile(args.line_bytes, dtype=np.uint32).astype(np.float64)
    runs = {}
    for p in Path(args.runs).glob("*.json"):
        r = json.loads(p.read_text())
        lines = p.with_name(p.name.replace(".json", ".lines.npy"))
        if lines.exists():
            runs[(r["tokenizer"], r["arch"], r.get("seed", 0))] = (r, np.load(lines))

    rows, lines_md = [], []
    for arch in sorted({k[1] for k in runs}):
        seeds = sorted({k[2] for k in runs if k[1] == arch and (args.a, arch, k[2]) in runs and (args.b, arch, k[2]) in runs})
        per_seed = []
        for s in seeds:
            (ra, la), (rb, lb) = runs[(args.a, arch, s)], runs[(args.b, arch, s)]
            if len(la) != len(line_bytes) or len(lb) != len(line_bytes):
                raise SystemExit(f"line count mismatch for {arch} seed {s}: {len(la)}, {len(lb)} vs {len(line_bytes)}")
            point, lo, hi, p = bootstrap_diff(la, lb, line_bytes, seed=s)
            per_seed.append(point)
            rows.append({"arch": arch, "seed": s, "bpb_a": ra["bpb"], "bpb_b": rb["bpb"],
                         "diff_%": point, "ci_low_%": lo, "ci_high_%": hi, "p_bootstrap": p})
            lines_md.append(f"| {arch} | {s} | {ra['bpb']:.4f} | {rb['bpb']:.4f} | {point:+.2f}% | "
                            f"[{lo:+.2f}%, {hi:+.2f}%] | {p:.3f} |")
        if len(per_seed) >= 2:
            m, sd = float(np.mean(per_seed)), float(np.std(per_seed, ddof=1))
            t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}.get(len(per_seed), 2.0)
            half = t * sd / math.sqrt(len(per_seed))
            agree = sum(1 for d in per_seed if d < 0)
            rows.append({"arch": arch, "seed": "mean", "diff_%": m, "ci_low_%": m - half, "ci_high_%": m + half,
                         "seeds_b_better": f"{agree}/{len(per_seed)}"})
            lines_md.append(f"| {arch} | **mean of {len(per_seed)}** | | | **{m:+.2f}%** | "
                            f"[{m - half:+.2f}%, {m + half:+.2f}%] (t-interval over seeds) | {agree}/{len(per_seed)} seeds favour {args.b} |")

    head = (f"Negative = {args.b} needs fewer bits per byte than {args.a} (better).\n\n"
            f"| Architecture | Seed | BPB {args.a} | BPB {args.b} | Difference | 95% interval | p (bootstrap) |\n"
            "|---|---|---|---|---|---|---|")
    md = head + "\n" + "\n".join(lines_md) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out + ".md").write_text(md, encoding="utf-8")
    Path(args.out + ".json").write_text(json.dumps(rows, indent=2))
    print(md)


if __name__ == "__main__":
    main()
