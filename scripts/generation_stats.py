"""Pooled counts and exact tests for the generated-text validity results.

    python scripts/generation_stats.py results/gen_free/generation_metrics.json results/gen_con/generation_metrics.json \
        --out results/generation_summary
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def binom_two_sided(k, n, p):
    """Exact two-sided binomial test (sum of outcomes no more likely than k)."""
    if n == 0:
        return 1.0
    logpmf = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
              + (i * math.log(p) if i else 0.0) + ((n - i) * math.log(1 - p) if n - i else 0.0) for i in range(n + 1)]
    ref = logpmf[k] + 1e-9
    return min(1.0, sum(math.exp(v) for v in logpmf if v <= ref))


def rate_test(a, ea, b, eb):
    """Is the event rate a/ea different from b/eb? (conditional binomial test for two Poisson counts)"""
    return binom_two_sided(a, a + b, ea / (ea + eb))


def summarize(rows):
    pooled = defaultdict(lambda: {"malformed": 0, "aksharas": 0, "malformed_words": 0, "words": 0,
                                  "nonword_w": 0.0, "models": 0})
    for r in rows:
        if r.get("arch") in ("-", "", None):
            continue
        for key in ((r["tokenizer"], r["arch"], r.get("decoding", "free")), (r["tokenizer"], "all", r.get("decoding", "free"))):
            p = pooled[key]
            p["malformed"] += r["malformed"]
            p["aksharas"] += r["aksharas"]
            p["malformed_words"] += r["malformed_words"]
            p["words"] += r["words"]
            p["nonword_w"] += (r.get("nonword_%") or 0) * r["words"] / 100
            p["models"] += 1
    lines = ["| Tokenizer | Arch | Decoding | Models | Malformed / aksharas | per 1000 | Words with malformed syllable | Non-words |",
             "|---|---|---|---|---|---|---|---|"]
    for (t, a, d), p in sorted(pooled.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        lines.append(f"| {t} | {a} | {d} | {p['models']} | {p['malformed']} / {p['aksharas']:,} | "
                     f"{1000 * p['malformed'] / max(1, p['aksharas']):.3f} | {100 * p['malformed_words'] / max(1, p['words']):.3f}% | "
                     f"{100 * p['nonword_w'] / max(1, p['words']):.2f}% |")
    toks = sorted({k[0] for k in pooled})
    plain = [t for t in toks if t.startswith("plain")]
    aks = [t for t in toks if t.startswith("akshara")]
    tests = ["", "**Exact rate tests (malformed syllables)**", "",
             "| Comparison | Arch | Counts | Rate ratio | p-value |", "|---|---|---|---|---|"]
    out_tests = []

    def add(name, arch, k1, k2):
        if k1 not in pooled or k2 not in pooled:
            return
        a, b = pooled[k1], pooled[k2]
        ratio = (a["malformed"] / a["aksharas"]) / max(1e-12, b["malformed"] / b["aksharas"]) if b["malformed"] else float("nan")
        pv = rate_test(a["malformed"], a["aksharas"], b["malformed"], b["aksharas"])
        tests.append(f"| {name} | {arch} | {a['malformed']} vs {b['malformed']} | {ratio:.2f} | {pv:.2g} |")
        out_tests.append({"comparison": name, "arch": arch, "a": k1, "b": k2, "count_a": a["malformed"],
                          "count_b": b["malformed"], "rate_ratio": ratio, "p": pv})

    for pt in plain:
        for at in aks:
            for arch in ("mamba", "transformer", "all"):
                add("plain vs AksharaPiece (free decoding)", arch, (pt, arch, "free"), (at, arch, "free"))
                add("plain vs AksharaPiece (constrained)", arch, (pt, arch, "constrained"), (at, arch, "constrained"))
                add("plain: free vs constrained", arch, (pt, arch, "free"), (pt, arch, "constrained"))
                add("AksharaPiece: free vs constrained", arch, (at, arch, "free"), (at, arch, "constrained"))
    return "\n".join(lines + tests) + "\n", {"pooled": {"|".join(k): v for k, v in pooled.items()}, "tests": out_tests}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", nargs="+", help="generation_metrics.json files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for f in args.metrics:
        rows += json.load(open(f))
    seen, uniq = set(), []
    for r in rows:                         # the human reference appears in each file: keep one
        if r["model"] in seen:
            continue
        seen.add(r["model"])
        uniq.append(r)
    md, js = summarize(uniq)
    ref = [r for r in uniq if r["arch"] == "-"]
    if ref:
        r = ref[0]
        md = (f"Human-written reference: {r['malformed']} malformed in {r['aksharas']:,} aksharas "
              f"({r['malformed_per_1000']:.3f}/1000), non-words {r['nonword_%']:.2f}%\n\n") + md
    Path(args.out + ".md").write_text(md, encoding="utf-8")
    Path(args.out + ".json").write_text(json.dumps({"rows": uniq, **js}, indent=2, ensure_ascii=False))
    print(md)


if __name__ == "__main__":
    main()
